#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
import rospy
import re
import time
import numpy
import argparse
import math
import underworlds
import tf2_ros
import message_filters
from collections import deque
from underworlds.helpers.geometry import get_world_transform
from underworlds.tools.loader import ModelLoader
from underworlds.helpers.transformations import translation_matrix, quaternion_matrix, euler_matrix, \
    translation_from_matrix, quaternion_from_matrix, quaternion_from_euler, rotation_from_matrix
from multimodal_human_provider.msg import GazeInfoArray, VoiceActivityArray
from underworlds.types import Camera, Mesh, MESH, Situation
from geometry_msgs.msg import Point, PointStamped
from std_srvs.srv import Empty
from nao_interaction_msgs.srv import TrackerLookAt
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage

TF_CACHE_TIME = 5.0
DEFAULT_CLIP_PLANE_NEAR = 0.01
DEFAULT_CLIP_PLANE_FAR = 1000.0
DEFAULT_HORIZONTAL_FOV = 50.0
DEFAULT_ASPECT = 1.33333
LOOK_AT_THRESHOLD = 0.7
MIN_NB_DETECTION = 5
MIN_DIST_DETECTION = 0.2
MAX_HEIGHT = 2.3
MAX_SPEED_LOOKAT = 0.1


# just for convenience
def strip_leading_slash(s):
    return s[1:] if s.startswith("/") else s


# just for convenience
def transformation_matrix(t, q):
    translation_mat = translation_matrix(t)
    rotation_mat = quaternion_matrix(q)
    return numpy.dot(translation_mat, rotation_mat)

class MultimodalHumanMonitor(object):
    def __init__(self, ctx, output_world, mesh_dir, reference_frame):
        self.human_cameras_ids = {}
        self.ctx = ctx
        self.human_bodies = {}
        self.target = ctx.worlds[output_world]
        self.target_world_name = output_world
        self.reference_frame = reference_frame
        self.mesh_dir = mesh_dir
        self.human_meshes = {}
        self.human_aabb = {}

        self.nb_gaze_detected = {}
        self.human_perception_to_uwds_ids = {}

        self.detection_time = None
        self.reco_durations = []
        self.record_time = False

        self.robot_name = rospy.get_param("robot_name", "pepper")
        self.is_robot_moving = rospy.get_param("is_robot_moving", False)

        self.ros_pub = {"voice": rospy.Publisher("voice_attention_point", PointStamped, queue_size=5),
                        "gaze": rospy.Publisher("gaze_attention_point", PointStamped, queue_size=5),
                        "reactive": rospy.Publisher("reactive_attention_point", PointStamped, queue_size=5),
                        "effective": rospy.Publisher("effective_attention_point", PointStamped, queue_size=5),
                        "goal_oriented": rospy.Publisher("goal_oriented_attention_point", PointStamped, queue_size=5)}

        self.ros_sub = {"gaze": message_filters.Subscriber("wp2/gaze", GazeInfoArray),
                        "voice": message_filters.Subscriber("wp2/voice", VoiceActivityArray)}

        self.ts = message_filters.TimeSynchronizer([self.ros_sub["gaze"], self.ros_sub["voice"]], 10)

        self.ts.registerCallback(self.callback)

        self.human_distances = {}

        self.head_signal_dq = deque()

        self.already_removed_nodes = []

        nodes_loaded = []

        try:
            nodes_loaded = ModelLoader().load(self.mesh_dir + "face.blend", self.ctx,
                                              world=output_world, root=None, only_meshes=True,
                                              scale=1.0)
        except Exception as e:
            rospy.logwarn("[multimodal_human_provider] Exception occurred with %s : %s" % (self.mesh_dir + "face.blend", str(e)))

        for n in nodes_loaded:
            if n.type == MESH:
                self.human_meshes["face"] = n.properties["mesh_ids"]
                self.human_aabb["face"] = n.properties["aabb"]

        self.tfBuffer = tf2_ros.Buffer(rospy.Duration(TF_CACHE_TIME), debug=False)
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

    def create_human_pov(self, id):
        new_node = Camera(name="human-" + str(id))
        new_node.properties["clipplanenear"] = DEFAULT_CLIP_PLANE_NEAR
        new_node.properties["clipplanefar"] = DEFAULT_CLIP_PLANE_FAR
        new_node.properties["horizontalfov"] = math.radians(DEFAULT_HORIZONTAL_FOV)
        new_node.properties["aspect"] = DEFAULT_ASPECT
        new_node.parent = self.target.scene.rootnode.id
        return new_node

    def start_lookat_situation(self, subject_name, object_name):
        sit = Situation(desc="lookat("+subject_name+","+object_name+")")
        self.target.timeline.start(sit)
        return sit.id

    def start_speaking_situation(self, subject_name):
        sit = Situation(desc="speaking("+subject_name+")")
        self.target.timeline.start(sit)
        return sit.id

    def start_speaking_to_situation(self, subject_name, object_name):
        sit = Situation(desc="speakingto("+subject_name+","+object_name+")")
        self.target.timeline.start(sit)
        return sit.id

    def look_at(self, stamped_point):
        self.ros_pub["effective"].publish(stamped_point)
        rospy.wait_for_service("/naoqi_driver/tracker/stop_tracker")
        rospy.ServiceProxy("/naoqi_driver/tracker/stop_tracker", Empty)
        t = [stamped_point.point.x, stamped_point.point.y, stamped_point.point.z]
        q = [0, 0, 0, 1]

        msg = self.tfBuffer.lookup_transform("base_link", stamped_point.header.frame_id, rospy.Time(0))
        trans = [msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z]
        rot = [msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z, msg.transform.rotation.w]

        point = translation_from_matrix(numpy.dot(transformation_matrix(trans, rot), transformation_matrix(t, q)))
        point_to_look = Point(point[0], point[1], point[2])
        rospy.logwarn("[human_monitor] pepper lookat human at : %s" % str(point_to_look))
        rospy.wait_for_service("/naoqi_driver/tracker/look_at")
        try:
            look_at = rospy.ServiceProxy("/naoqi_driver/tracker/look_at", TrackerLookAt)
            resp = look_at(point_to_look, 0, MAX_SPEED_LOOKAT, False)
            return resp
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e

    def callback(self, gaze_msg, voice_msg):
        nodes_to_update = []
        gaze_attention_point = None
        voice_attention_point = None
        reactive_attention_point = None

        # VOICE
        min_dist = 10000
        for j, voice in enumerate(voice_msg.data):
            if voice.person_id in self.human_cameras_ids:
                if voice.is_speaking:
                    human_node = self.target.scene.nodes[self.human_cameras_ids[voice.person_id]]
                    self.start_speaking_situation(human_node.name)
                    if self.human_distances[voice.person_id] < min_dist:
                        min_dist = self.human_distances[voice.person_id]
                        point = translation_from_matrix(human_node.transformation)
                        voice_attention_point = PointStamped()
                        voice_attention_point.header.frame_id = self.reference_frame
                        voice_attention_point.point = Point(point[0], point[1], point[2])
        # GAZE
        min_dist = 10000
        if len(gaze_msg.data) > 0:
            rospy.logwarn("[human_monitor] pepper see %s human" % len(gaze_msg.data))
            for i, gaze in enumerate(gaze_msg.data):
                human_id = gaze.person_id

                if human_id not in self.nb_gaze_detected:
                    self.nb_gaze_detected[human_id] = 0
                else:
                    self.nb_gaze_detected[human_id] += 1

                if gaze.head_gaze_available:
                    new_node = self.create_human_pov(human_id)
                    if human_id in self.human_cameras_ids:
                        new_node.id = self.human_cameras_ids[human_id]

                    t = [gaze.head_gaze.position.x, gaze.head_gaze.position.y, gaze.head_gaze.position.z]
                    q = [gaze.head_gaze.orientation.x, gaze.head_gaze.orientation.y, gaze.head_gaze.orientation.z, gaze.head_gaze.orientation.w]

                    dist = math.sqrt(t[0] * t[0] + t[1] * t[1] + t[2] * t[2])

                    if dist < MIN_DIST_DETECTION:
                        continue

                    msg = self.tfBuffer.lookup_transform(self.reference_frame, gaze_msg.header.frame_id, rospy.Time(0))
                    trans = [msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z]
                    rot = [msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z, msg.transform.rotation.w]

                    offset = euler_matrix(0, math.radians(90), math.radians(90), "rxyz")

                    transform = numpy.dot(transformation_matrix(trans, rot), transformation_matrix(t, q))

                    new_node.transformation = numpy.dot(transform, offset)

                    if translation_from_matrix(new_node.transformation)[2] > MAX_HEIGHT:
                        continue

                    if dist < min_dist:
                        min_dist = dist
                        gaze_attention_point = PointStamped()
                        gaze_attention_point.header.frame_id = gaze_msg.header.frame_id
                        gaze_attention_point.header.stamp = rospy.Time.now()
                        gaze_attention_point.point = Point(t[0], t[1], t[2])

                    if self.nb_gaze_detected[human_id] > MIN_NB_DETECTION:
                        self.human_distances[human_id] = dist
                        self.human_cameras_ids[human_id] = new_node.id
                        nodes_to_update.append(new_node)

                        if human_id not in self.human_bodies:
                            rospy.logwarn("[human_monitor] add human-"+str(human_id))
                            self.human_bodies[human_id] = {}

                        if "face" not in self.human_bodies[human_id]:
                            new_node = Mesh(name="human_face-"+str(human_id))
                            new_node.properties["mesh_ids"] = self.human_meshes["face"]
                            new_node.properties["aabb"] = self.human_aabb["face"]
                            new_node.parent = self.human_cameras_ids[human_id]
                            offset = euler_matrix(math.radians(90), math.radians(0), math.radians(90), 'rxyz')
                            new_node.transformation = numpy.dot(new_node.transformation, offset)
                            self.human_bodies[human_id]["face"] = new_node.id
                            nodes_to_update.append(new_node)

            if gaze_attention_point:
                self.ros_pub["gaze"].publish(gaze_attention_point)
                reactive_attention_point = gaze_attention_point
            if voice_attention_point:
                self.ros_pub["voice"].publish(voice_attention_point)
                reactive_attention_point = voice_attention_point
            if reactive_attention_point:
                self.ros_pub["reactive"].publish(reactive_attention_point)

            if nodes_to_update:
                self.target.scene.nodes.update(nodes_to_update)

    def publish_human_tf_frames(self):
        for node in self.target.scene.nodes:
            if re.match("human-", node.name):
                human_id = node.name.split("-")[1]

                t = TransformStamped()
                t.header.frame_id = "map"
                t.header.stamp = rospy.Time.now()
                t.child_frame_id = node.name
                position = translation_from_matrix(get_world_transform(self.target, node))
                t.transform.translation.x = position[0]
                t.transform.translation.y = position[1]
                t.transform.translation.z = position[2]
                orientation = quaternion_from_matrix(get_world_transform(self.target, node))
                t.transform.rotation.x = orientation[0]
                t.transform.rotation.y = orientation[1]
                t.transform.rotation.z = orientation[2]
                t.transform.rotation.w = orientation[3]

                tfm = TFMessage([t])
                self.ros_pub["tf"].publish(tfm)

                t = TransformStamped()
                t.header.frame_id = "map"
                t.header.stamp = rospy.Time.now()
                t.child_frame_id = "human_footprint-"+human_id
                position = translation_from_matrix(node.transformation)
                t.transform.translation.x = position[0]
                t.transform.translation.y = position[1]
                t.transform.translation.z = 0
                rotation = rotation_from_matrix(node.transformation)
                rotation = [0, 0, rotation[2]]
                orientation = quaternion_from_euler(rotation, "rxyz")
                t.transform.rotation.x = orientation[0]
                t.transform.rotation.y = orientation[1]
                t.transform.rotation.z = orientation[2]
                t.transform.rotation.w = orientation[3]

                tfm = TFMessage([t])
                self.ros_pub["tf"].publish(tfm)

    def clean_humans(self):
        nodes_to_remove = []

        for node in self.target.scene.nodes:
            if node not in self.already_removed_nodes:
                if re.match("human-", node.name):
                    if time.time() - node.last_update > 10.0:
                        nodes_to_remove.append(node)
                        for child in node.children:
                            nodes_to_remove.append(self.target.scene.nodes[child])

        if nodes_to_remove:
            rospy.logwarn(nodes_to_remove)
            self.already_removed_nodes = nodes_to_remove
            self.target.scene.nodes.remove(nodes_to_remove)

    def run(self):
        while not rospy.is_shutdown():
            self.publish_human_tf_frames()


if __name__ == "__main__":

    sys.argv = [arg for arg in sys.argv if "__name" not in arg and "__log" not in arg]
    sys.argc = len(sys.argv)

    parser = argparse.ArgumentParser(description="Add in the given output world, the nodes from input "
                                                 "world and the robot agent from ROS")
    parser.add_argument("output_world", help="Underworlds output world")
    parser.add_argument("mesh_dir", help="The path used to localize the human meshes")
    parser.add_argument("--reference", default="map", help="The reference frame")
    args = parser.parse_args()

    rospy.init_node('multimodal_human_provider', anonymous=False)

    with underworlds.Context("Multimodal human monitor") as ctx:
        MultimodalHumanMonitor(ctx, args.output_world, args.mesh_dir, args.reference).run()




