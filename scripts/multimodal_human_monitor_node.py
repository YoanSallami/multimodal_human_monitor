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
from perception_msgs.msg import GazeInfoArray, VoiceActivityArray, TrackedPersonArray
from underworlds.types import Camera, Mesh, MESH, Situation
from geometry_msgs.msg import Point, PointStamped
from std_srvs.srv import Empty
from std_msgs.msg import String
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

INTIMATE_MAX_DISTANCE = 0.45
PERSONAL_MAX_DISTANCE = 1.2
SOCIAL_MAX_DISTANCE = 3.6
PUBLIC_MAX_DISTANCE = 7.6

HYST_DELTA = 0.1


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

        self.ros_pub = {"voice": rospy.Publisher("human_monitor/voice_attention_point", PointStamped, queue_size=5),
                        "gaze": rospy.Publisher("human_monitor/gaze_attention_point", PointStamped, queue_size=5),
                        "reactive": rospy.Publisher("human_monitor/reactive_attention_point", PointStamped, queue_size=5),
                        "effective": rospy.Publisher("human_monitor/effective_attention_point", PointStamped, queue_size=5),
                        "goal_oriented": rospy.Publisher("human_monitor/goal_oriented_attention_point", PointStamped, queue_size=5),
                        "situation_log": rospy.Publisher("human_monitor/log", String, queue_size=5),
                        "tf": rospy.Publisher("/tf", TFMessage, queue_size=10)}

        self.ros_sub = {"gaze": message_filters.Subscriber("wp2/gaze", GazeInfoArray),
                        "voice": message_filters.Subscriber("wp2/voice", VoiceActivityArray),
                        "person": message_filters.Subscriber("wp2/track", TrackedPersonArray)}

        self.ts = message_filters.TimeSynchronizer([self.ros_sub["gaze"], self.ros_sub["voice"], self.ros_sub["person"]], 50)

        self.ts.registerCallback(self.callback)

        self.head_signal_dq = deque()

        self.already_removed_nodes = []

        self.current_situations_map = {}

        self.human_seen = []
        self.previous_human_seen = []

        self.human_speaking = []
        self.previous_human_speaking = []

        self.human_lookat = {}
        self.previous_human_lookat = {}

        self.human_distances = {}
        self.human_intimate = []
        self.previous_human_intimate = []
        self.human_personal = []
        self.previous_human_personal = []
        self.human_social = []
        self.previous_human_social = []
        self.human_public = []
        self.previous_human_public = []

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

    def start_n2_situation(self, predicate, subject_name, object_name):
        description = predicate+"("+subject_name+","+object_name+")"
        sit = Situation(desc=description)
        sit.starttime = time.time()
        self.current_situations_map[description] = sit
        self.ros_pub["situation_log"].publish("START " + description)
        self.target.timeline.update(sit)
        return sit.id

    def start_n1_situation(self, predicate, subject_name):
        description = predicate+"("+subject_name+")"
        sit = Situation(desc=description)
        sit.starttime = time.time()
        self.current_situations_map[description] = sit
        self.ros_pub["situation_log"].publish("START " + description)
        self.target.timeline.update(sit)
        return sit.id

    def end_n1_situation(self, predicate, subject_name):
        description = predicate+"("+subject_name+")"
        sit = self.current_situations_map[description]
        self.ros_pub["situation_log"].publish("END "+description)
        try:
            self.target.timeline.end(sit)
        except Exception as e:
            rospy.logwarn("[robot_monitor] Exception occurred : "+str(e))

    def end_n2_situation(self, predicate, subject_name, object_name):
        description = predicate+"("+subject_name+","+object_name+")"
        sit = self.current_situations_map[description]
        self.ros_pub["situation_log"].publish("END "+description)
        try:
            self.target.timeline.end(sit)
        except Exception as e:
            rospy.logwarn("[robot_monitor] Exception occurred : "+str(e))

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

    def callback(self, gaze_msg, voice_msg, person_msg):
        nodes_to_update = []
        gaze_attention_point = None
        voice_attention_point = None
        reactive_attention_point = None
        self.previous_human_seen = self.human_seen
        self.human_seen = []
        self.previous_human_speaking = self.human_speaking
        self.human_speaking = []
        self.previous_human_lookat = self.human_lookat
        self.human_lookat = {}
        self.previous_human_intimate = self.human_intimate
        self.previous_human_personal = self.human_personal
        self.previous_human_social = self.human_social
        self.previous_human_public = self.human_public
        self.human_intimate = []
        self.human_personal = []
        self.human_social = []
        self.human_public = []
        # VOICE
        min_dist = 10000
        for j, voice in enumerate(voice_msg.data):
            if voice.person_id in self.human_cameras_ids:
                if voice.is_speaking:
                    human_node = self.target.scene.nodes[self.human_cameras_ids[voice.person_id]]

                    self.human_speaking.append("human-"+str(voice.person_id))

                    if self.human_distances["human-"+str(voice.person_id)] < min_dist:
                        min_dist = self.human_distances["human-"+str(voice.person_id)]
                        point = translation_from_matrix(human_node.transformation)
                        voice_attention_point = PointStamped()
                        voice_attention_point.header.frame_id = self.reference_frame
                        voice_attention_point.point = Point(point[0], point[1], point[2])

        # PERSON
        min_dist = 10000
        if len(person_msg.data) > 0:

            for i, person in enumerate(person_msg.data):
                human_id = person.person_id

                if human_id not in self.nb_gaze_detected:
                    self.nb_gaze_detected[human_id] = 0
                else:
                    self.nb_gaze_detected[human_id] += 1

                if person.is_identified > 4:
                    new_node = self.create_human_pov(human_id)
                    if human_id in self.human_cameras_ids:
                        new_node.id = self.human_cameras_ids[human_id]

                    t = [person.head_pose.position.x, person.head_pose.position.y, person.head_pose.position.z]
                    q = [person.head_pose.orientation.x, person.head_pose.orientation.y, person.head_pose.orientation.z, person.head_pose.orientation.w]

                    dist = math.sqrt(t[0] * t[0] + t[1] * t[1] + t[2] * t[2])

                    msg = self.tfBuffer.lookup_transform(self.reference_frame, gaze_msg.header.frame_id, rospy.Time(0))
                    trans = [msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z]
                    rot = [msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z, msg.transform.rotation.w]

                    offset = euler_matrix(0, math.radians(90), math.radians(90), "rxyz")

                    transform = numpy.dot(transformation_matrix(trans, rot), transformation_matrix(t, q))

                    new_node.transformation = numpy.dot(transform, offset)

                    if person.head_distance < min_dist:
                        min_dist = person.head_distance
                        gaze_attention_point = PointStamped()
                        gaze_attention_point.header.frame_id = gaze_msg.header.frame_id
                        gaze_attention_point.header.stamp = rospy.Time.now()
                        gaze_attention_point.point = Point(t[0], t[1], t[2])

                    self.human_distances["human-"+str(human_id)] = dist
                    self.human_cameras_ids[human_id] = new_node.id
                    nodes_to_update.append(new_node)

                    self.human_seen.append("human-"+str(human_id))

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

            #GAZE
            for j, gaze in enumerate(gaze_msg.data):
                if gaze.person_id in self.human_cameras_ids:
                    if gaze.probability_looking_at_robot > LOOK_AT_THRESHOLD:
                        self.human_lookat["human-" + str(gaze.person_id)] = "robot"
                    else:
                        if gaze.probability_looking_at_screen > LOOK_AT_THRESHOLD:
                            self.human_lookat["human-" + str(gaze.person_id)] = "screen"
                        else:
                            for attention in gaze.attentions:
                                if attention.target_id in self.human_cameras_ids:
                                    if attention.probability_looking_at_target > LOOK_AT_THRESHOLD:
                                        self.human_lookat["human-"+str(gaze.person_id)] = "human-"+str(attention.target_id)

            for human, dist in self.human_distances.items():
                if dist < PUBLIC_MAX_DISTANCE:
                    if dist < SOCIAL_MAX_DISTANCE:
                        if dist < PERSONAL_MAX_DISTANCE:
                            if dist < INTIMATE_MAX_DISTANCE:
                                self.human_intimate.append(human)
                            else:
                                self.human_personal.append(human)
                        else:
                            self.human_social.append(human)
                    else:
                        self.human_public.append(human)

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

            self.compute_situations()

    def compute_situations(self):
        # for human in self.human_intimate:
        #     if human not in self.previous_human_intimate:
        #         self.start_n2_situation("inIntimateSpace", human, "robot")
        # for human in self.previous_human_intimate:
        #     if human not in self.human_intimate:
        #         self.end_n2_situation("inIntimateSpace", human, "robot")
        #
        # for human in self.human_personal:
        #     if human not in self.previous_human_personal:
        #         self.start_n2_situation("inPersonalSpace", human, "robot")
        # for human in self.previous_human_personal:
        #     if human not in self.human_personal:
        #         self.end_n2_situation("inPersonalSpace", human, "robot")
        #
        # for human in self.human_social:
        #     if human not in self.previous_human_social:
        #         self.start_n2_situation("inSocialSpace", human, "robot")
        # for human in self.previous_human_social:
        #     if human not in self.human_social:
        #         self.end_n2_situation("inSocialSpace", human, "robot")
        #
        # for human in self.human_public:
        #     if human not in self.previous_human_intimate:
        #         self.start_n2_situation("inPublicSpace", human, "robot")
        # for human in self.previous_human_social:
        #     if human not in self.human_social:
        #         self.end_n2_situation("inPublicSpace", human, "robot")

        for human, target in self.human_lookat.values():
            if human not in self.previous_human_lookat:
                self.start_n2_situation("lookAt", human, target)
            else:
                if target != self.previous_human_lookat[human]:
                    self.end_n2_situation("lookAt", human, self.previous_human_lookat[human])
                    self.start_n2_situation("lookAt", human, target)

        for human, target in self.previous_human_lookat.values():
            if human not in self.human_lookat:
                self.end_n2_situation("lookAt", human, target)

        for human in self.human_speaking:
            if human not in self.previous_human_speaking:
                self.start_n1_situation("speaking", human)

        for human in self.previous_human_speaking:
            if human not in self.human_speaking:
                self.end_n1_situation("speaking", human)

        for human in self.human_seen:
            if human not in self.previous_human_seen:
                self.start_n2_situation("perceive", "robot", human)

        for human in self.previous_human_seen:
            if human not in self.human_seen:
                self.end_n2_situation("perceive", "robot", human)

    def publish_human_tf_frames(self):
        for node in self.target.scene.nodes:
            if re.match("human-", node.name):
                human_id = node.name.split("-")[1]

                t = TransformStamped()
                t.header.frame_id = "map"
                t.header.stamp = rospy.Time.now()
                t.child_frame_id = node.name
                position = translation_from_matrix(get_world_transform(self.target.scene, node))
                t.transform.translation.x = position[0]
                t.transform.translation.y = position[1]
                t.transform.translation.z = position[2]
                orientation = quaternion_from_matrix(get_world_transform(self.target.scene, node))
                t.transform.rotation.x = orientation[0]
                t.transform.rotation.y = orientation[1]
                t.transform.rotation.z = orientation[2]
                t.transform.rotation.w = orientation[3]

                tfm = TFMessage([t])
                self.ros_pub["tf"].publish(tfm)

                # t = TransformStamped()
                # t.header.frame_id = "map"
                # t.header.stamp = rospy.Time.now()
                # t.child_frame_id = "human_footprint-"+human_id
                # position = translation_from_matrix(node.transformation)
                # t.transform.translation.x = position[0]
                # t.transform.translation.y = position[1]
                # t.transform.translation.z = 0
                # rospy.logwarn(node.transformation)
                # q = quaternion_from_matrix(node.transformation)
                # rotation = [0, 0, rotation[2]]
                # orientation = euler_from_quaternion(rotation, "rxyz")
                # t.transform.rotation.x = orientation[0]
                # t.transform.rotation.y = orientation[1]
                # t.transform.rotation.z = orientation[2]
                # t.transform.rotation.w = orientation[3]
                #
                # tfm = TFMessage([t])
                # self.ros_pub["tf"].publish(tfm)

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




