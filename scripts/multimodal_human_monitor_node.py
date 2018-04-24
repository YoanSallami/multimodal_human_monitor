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
    translation_from_matrix, quaternion_from_matrix, quaternion_from_euler
from perception_msgs.msg import GazeInfoArray, VoiceActivityArray, TrackedPersonArray
from underworlds.types import Camera, Mesh, MESH, Situation, Entity, CAMERA, ENTITY
from geometry_msgs.msg import Point, PointStamped
from std_msgs.msg import String
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage
from head_manager.msg import TargetWithPriority
from multimodal_human_monitor.srv import MonitorHumans

TF_CACHE_TIME = 5.0
DEFAULT_CLIP_PLANE_NEAR = 0.01
DEFAULT_CLIP_PLANE_FAR = 1000.0
DEFAULT_HORIZONTAL_FOV = 50.0
DEFAULT_ASPECT = 1.33333
LOOK_AT_THRESHOLD = 0.6
MIN_NB_DETECTION = 6
MIN_DIST_DETECTION = 0.2
MAX_HEIGHT = 2.3
MAX_SPEED_LOOKAT = 0.1

MIN_UPDATE_TIME_BEFORE_CLEAN = 3.0
MAX_UPDATE_TIME_BEFORE_CLEAN = 10.0

CLOSE_MAX_DISTANCE = 1.0
NEAR_MAX_DISTANCE = 2.0

JOINT_ATTENTION_MIN_DURATION = 1.5

MONITORING_DEFAULT_PRIORITY = 100
JOINT_ATTENTION_PRIORITY = 150
VOICE_ATTENTION_PRIORITY = 200

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

        self.ros_pub = {"voice": rospy.Publisher("multimodal_human_monitor/voice_attention_point", PointStamped, queue_size=5),
                        "gaze": rospy.Publisher("multimodal_human_monitor/gaze_attention_point", PointStamped, queue_size=5),
                        "reactive": rospy.Publisher("multimodal_human_monitor/monitoring_attention_point", PointStamped, queue_size=5),
                        "monitoring_attention_point": rospy.Publisher("head_manager/head_monitoring_target", TargetWithPriority, queue_size=5),
                        "tf": rospy.Publisher("/tf", TFMessage, queue_size=10)}

        self.log_pub = {"isLookingAt": rospy.Publisher("predicates_log/lookingat", String, queue_size=5),
                        "isPerceiving": rospy.Publisher("predicates_log/perceiving", String, queue_size=5),
                        "isSpeaking": rospy.Publisher("predicates_log/speak", String, queue_size=5),
                        "isSpeakingTo": rospy.Publisher("predicates_log/speakingto", String, queue_size=5),
                        "isNear": rospy.Publisher("predicates_log/near", String, queue_size=5),
                        "isClose": rospy.Publisher("predicates_log/close", String, queue_size=5),
                        "isMonitoring": rospy.Publisher("predicates_log/monitoring", String, queue_size=5)}

        self.ros_sub = {"gaze": message_filters.Subscriber("wp2/gaze", GazeInfoArray),
                        "voice": message_filters.Subscriber("wp2/voice", VoiceActivityArray),
                        "person": message_filters.Subscriber("wp2/track", TrackedPersonArray)}

        self.ros_services = {"monitor_humans": rospy.Service("multimodal_human_monitor/monitor_humans", MonitorHumans, self.handle_monitor_humans)}

        self.ts = message_filters.TimeSynchronizer([self.ros_sub["gaze"], self.ros_sub["voice"], self.ros_sub["person"]], 50)

        self.ts.registerCallback(self.callback)

        self.head_signal_dq = deque()

        self.already_removed_nodes = []

        self.current_situations_map = {}

        self.humans_to_monitor = []

        self.human_perceived = []
        self.previous_human_perceived = []

        self.human_speaking = []
        self.previous_human_speaking = []

        self.human_speakingto = {}
        self.previous_human_speakingto = {}

        self.human_lookat = {}
        self.previous_human_lookat = {}

        self.human_distances = {}
        self.human_close = []
        self.previous_human_close = []
        self.human_near = []
        self.previous_human_near = []

        nodes_loaded = []

        self.tfBuffer = tf2_ros.Buffer(rospy.Duration(TF_CACHE_TIME), debug=False)
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        try:
            nodes_loaded = ModelLoader().load(self.mesh_dir + "face.blend", self.ctx,
                                              world=output_world, root=None, only_meshes=True,
                                              scale=1.0)
            for n in nodes_loaded:
                if n.type == MESH:
                    self.human_meshes["face"] = n.properties["mesh_ids"]
                    self.human_aabb["face"] = n.properties["aabb"]
        except Exception as e:
            rospy.logwarn("[multimodal_human_provider] Exception occurred with %s : %s" % (self.mesh_dir + "face.blend", str(e)))

    def parse_situation_desc(self, description):
        if "," in description:
            predicate = description.split("(")[0]
            subject = description.split("(")[1].split(")")[0].split(",")[0]
            obj = description.split("(")[1].split(")")[0].split(",")[1]
        else:
            predicate = description.split("(")[0]
            subject = description.split("(")[1].split(")")[0]
            obj = ""
        return predicate, subject, obj

    def create_human_pov(self, id):
        new_node = Camera(name="human-" + str(id))
        new_node.properties["clipplanenear"] = DEFAULT_CLIP_PLANE_NEAR
        new_node.properties["clipplanefar"] = DEFAULT_CLIP_PLANE_FAR
        new_node.properties["horizontalfov"] = math.radians(DEFAULT_HORIZONTAL_FOV)
        new_node.properties["aspect"] = DEFAULT_ASPECT
        new_node.parent = self.target.scene.rootnode.id
        return new_node

    def handle_monitor_humans(self, req):
        if req.action == "ADD":
            for human in req.humans_to_monitor:
                if human not in self.humans_to_monitor:
                    self.start_predicate(self.target.timeline, "isMonitoring", "robot", object_name=human)
                    self.humans_to_monitor.append(human)
            return True
        else:
            if req.action == "REMOVE":
                for human in req.humans_to_monitor:
                    if human in self.humans_to_monitor:
                        self.humans_to_monitor.remove(human)
                        self.end_predicate(self.target.timeline, "isMonitoring", "robot", object_name=human)
                return True
        return False

    def start_predicate(self, timeline, predicate, subject_name, object_name=None, isevent=False):
        if object_name is None:
            description = predicate + "(" + subject_name + ")"
        else:
            description = predicate + "(" + subject_name + "," + object_name + ")"
        sit = Situation(desc=description)
        sit.starttime = time.time()
        if isevent:
            sit.endtime = sit.starttime
        self.current_situations_map[description] = sit
        self.log_pub[predicate].publish("START " + description)
        timeline.update(sit)
        return sit.id

    def end_predicate(self, timeline, predicate, subject_name, object_name=None):
        if object_name is None:
            description = predicate + "(" + subject_name + ")"
        else:
            description = predicate + "(" + subject_name + "," + object_name + ")"
        try:
            sit = self.current_situations_map[description]
            self.log_pub[predicate].publish("END " + description)
            timeline.end(sit)
        except Exception as e:
            rospy.logwarn("[multimodal_human_monitor] Exception occurred : " + str(e))

    def callback(self, gaze_msg, voice_msg, person_msg):
        nodes_to_update = []
        gaze_attention_point = None
        voice_attention_point = None
        reactive_attention_point = None
        self.human_distances = {}

        self.human_speaking = []
        self.human_speakingto = {}
        self.human_lookat = {}
        self.human_perceived = []
        self.human_close = []
        self.human_near = []

        # PERSON
        min_dist = 10000
        if len(person_msg.data) > 0:

            for i, person in enumerate(person_msg.data):
                human_id = person.person_id

                if human_id not in self.nb_gaze_detected:
                    self.nb_gaze_detected[human_id] = 0
                else:
                    self.nb_gaze_detected[human_id] += 1

                if person.is_identified > MIN_NB_DETECTION:
                    try:
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
                            gaze_attention_point.header.frame_id = "human-"+str(person.person_id)
                            gaze_attention_point.header.stamp = rospy.Time.now()
                            gaze_attention_point.point = Point(0, 0, 0)

                        self.human_distances["human-"+str(human_id)] = dist
                        self.human_cameras_ids[human_id] = new_node.id
                        nodes_to_update.append(new_node)

                        self.human_perceived.append("human-"+str(human_id))

                        if human_id not in self.human_bodies:
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
                        else:
                            node_id = self.human_bodies[human_id]["face"]
                            if self.target.scene.nodes[node_id].type == ENTITY:
                                new_node = Mesh(name="human_face-" + str(human_id))
                                new_node.properties["mesh_ids"] = self.human_meshes["face"]
                                new_node.properties["aabb"] = self.human_aabb["face"]
                                new_node.parent = self.human_cameras_ids[human_id]
                                offset = euler_matrix(math.radians(90), math.radians(0), math.radians(90), 'rxyz')
                                new_node.transformation = numpy.dot(new_node.transformation, offset)
                                new_node.id = node_id
                                nodes_to_update.append(new_node)

                    except (tf2_ros.TransformException, tf2_ros.LookupException, tf2_ros.ConnectivityException,
                            tf2_ros.ExtrapolationException) as e:
                            rospy.logwarn("[multimodal_human_monitor] Exception occurred : %s" % str(e))

            # VOICE
            min_dist = 10000
            for j, voice in enumerate(voice_msg.data):
                if voice.person_id in self.human_cameras_ids:
                    if voice.is_speaking:
                        try:
                            self.human_speaking.append("human-" + str(voice.person_id))
                            if "human-" + str(voice.person_id) in self.human_distances:
                                if self.human_distances["human-" + str(voice.person_id)] < min_dist:
                                    min_dist = self.human_distances["human-" + str(voice.person_id)]
                                    voice_attention_point = PointStamped()
                                    voice_attention_point.header.frame_id = "human-" + str(voice.person_id)
                                    voice_attention_point.point = Point(0, 0, 0)
                        except:
                            pass
            #GAZE
            for j, gaze in enumerate(gaze_msg.data):
                if gaze.person_id in self.human_cameras_ids:
                    if gaze.probability_looking_at_robot > LOOK_AT_THRESHOLD:
                        self.human_lookat["human-" + str(gaze.person_id)] = "robot"
                    else:
                        if gaze.probability_looking_at_screen > LOOK_AT_THRESHOLD:
                            self.human_lookat["human-" + str(gaze.person_id)] = "robot"
                        else:
                            for attention in gaze.attentions:
                                if attention.target_id in self.human_cameras_ids:
                                    if attention.probability_looking_at_target > LOOK_AT_THRESHOLD:
                                        self.human_lookat["human-"+str(gaze.person_id)] = "human-"+str(attention.target_id)

            for human, dist in self.human_distances.items():
                if dist < NEAR_MAX_DISTANCE:
                    if dist < CLOSE_MAX_DISTANCE:
                        self.human_close.append(human)
                    else:
                        self.human_near.append(human)

        #computing speaking to
        for human in self.human_speaking:
            if human in self.human_lookat:
                self.human_speakingto[human] = self.human_lookat[human]

        if nodes_to_update:
            self.target.scene.nodes.update(nodes_to_update)

        self.compute_situations()

        priority = MONITORING_DEFAULT_PRIORITY
        if gaze_attention_point:
            sit_regex = "^isLookingAt\(%s" % gaze_attention_point.header.frame_id
            for sit in self.current_situations_map.values():
                if re.match(sit_regex, sit.desc):
                    if sit.endtime == 0:
                        if time.time() - sit.starttime > JOINT_ATTENTION_MIN_DURATION:
                            _,_,obj = self.parse_situation_desc(sit.desc)
                            if obj != "screen" and obj != "robot":
                                gaze_attention_point.header.frame_id = obj
                                gaze_attention_point.point = Point()
                                priority = JOINT_ATTENTION_PRIORITY
            self.ros_pub["gaze"].publish(gaze_attention_point)
            reactive_attention_point = gaze_attention_point
        if voice_attention_point:
            self.ros_pub["voice"].publish(voice_attention_point)
            reactive_attention_point = voice_attention_point
            priority = VOICE_ATTENTION_PRIORITY
        if reactive_attention_point:
            self.ros_pub["reactive"].publish(reactive_attention_point)

        if reactive_attention_point:
            target_with_priority = TargetWithPriority()
            target_with_priority.target = reactive_attention_point
            target_with_priority.priority = priority
            self.ros_pub["monitoring_attention_point"].publish(target_with_priority)

        self.previous_human_perceived = self.human_perceived
        self.previous_human_speaking = self.human_speaking
        self.previous_human_speakingto = self.human_speakingto
        self.previous_human_lookat = self.human_lookat
        self.previous_human_close = self.human_close
        self.previous_human_near = self.human_near

    def compute_situations(self):
        for human in self.human_near:
            if human not in self.previous_human_near:
                self.start_predicate(self.target.timeline, "isNear", human, object_name="robot")

        for human in self.previous_human_near:
            if human not in self.human_near:
                self.end_predicate(self.target.timeline, "isNear", human, object_name="robot")

        for human in self.human_close:
            if human not in self.previous_human_close:
                self.start_predicate(self.target.timeline, "isClose", human, object_name="robot")

        for human in self.previous_human_close:
            if human not in self.human_close:
                self.end_predicate(self.target.timeline, "isClose", human, object_name="robot")

        for human, target in self.human_lookat.items():
            if human not in self.previous_human_lookat:
                self.start_predicate(self.target.timeline, "isLookingAt", human, object_name=target)
            else:
                if target != self.previous_human_lookat[human]:
                    self.end_predicate(self.target.timeline, "isLookingAt", human, object_name=self.previous_human_lookat[human])
                    self.start_predicate(self.target.timeline, "isLookingAt", human, object_name=target)

        for human, target in self.previous_human_lookat.items():
            if human not in self.human_lookat:
                self.end_predicate(self.target.timeline, "isLookingAt", human, object_name=target)

        for human, target in self.human_speakingto.items():
            if human not in self.previous_human_speakingto:
                self.start_predicate(self.target.timeline, "isSpeakingTo", human, object_name=target)
            else:
                if target != self.previous_human_speakingto[human]:
                    self.end_predicate(self.target.timeline, "isSpeakingTo", human, object_name=self.previous_human_speakingto[human])
                    self.start_predicate(self.target.timeline, "isSpeakingTo", human, object_name=target)

        for human, target in self.previous_human_speakingto.items():
            if human not in self.human_speakingto:
                self.end_predicate(self.target.timeline, "isSpeakingTo", human, object_name=target)

        for human in self.human_speaking:
            if human not in self.previous_human_speaking:
                self.start_predicate(self.target.timeline, "isSpeaking", human)

        for human in self.previous_human_speaking:
            if human not in self.human_speaking:
                self.end_predicate(self.target.timeline, "isSpeaking", human)

        for human in self.human_perceived:
            if human not in self.previous_human_perceived:
                self.start_predicate(self.target.timeline, "isPerceiving", "robot", object_name=human)

        for human in self.previous_human_perceived:
            if human not in self.human_perceived:
                self.end_predicate(self.target.timeline, "isPerceiving", "robot", object_name=human)

    def publish_human_tf_frames(self):
        for node in self.target.scene.nodes:
            if re.match("human-", node.name) and node.type == CAMERA:
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

                # Publish a human footprint
                t_footprint = TransformStamped()
                t_footprint.header.frame_id = "map"
                t_footprint.header.stamp = rospy.Time.now()
                t_footprint.child_frame_id = node.name + "_footprint"
                t_footprint.transform.translation.x = t.transform.translation.x
                t_footprint.transform.translation.y = t.transform.translation.y
                t_footprint.transform.translation.z = 0
                map_z = numpy.transpose(numpy.atleast_2d(numpy.array([0, 0, 1, 1])))
                human_z = numpy.dot(get_world_transform(self.target.scene, node), map_z)
                human_z_proj = numpy.array(human_z[0], human_z[1], 0)
                yaw = math.acos(human_z_proj.dot(numpy.array([1, 0, 0])) / math.hypot(human_z_proj[0], human_z_proj[1]))
                rx, ry, rz, rw = quaternion_from_euler(0, 0, yaw)
                t_footprint.transform.rotation.x = rx
                t_footprint.transform.rotation.y = ry
                t_footprint.transform.rotation.z = rz
                t_footprint.transform.rotation.w = rw


                tfm = TFMessage([t, t_footprint])
                self.ros_pub["tf"].publish(tfm)



    def clean_humans(self):
        nodes_to_update = []

        for node in self.target.scene.nodes:
            if re.match("^human-", node.name) and node.type == CAMERA and node.name not in self.humans_to_monitor:
                if time.time() - node.last_update > MIN_UPDATE_TIME_BEFORE_CLEAN:
                    entity = Entity()
                    entity.id = node.id
                    entity.name = node.name
                    entity.parent = node.parent
                    entity.transformation = node.transformation
                    entity.properties = node.properties
                    nodes_to_update.append(entity)
                    for child in node.children:
                        n = self.target.scene.nodes[child]
                        entity = Entity()
                        entity.id = n.id
                        entity.name = n.name
                        entity.parent = n.parent
                        entity.transformation = n.transformation
                        nodes_to_update.append(entity)
                        #self.human_bodies[node.name] = {}

        if nodes_to_update:
            self.target.scene.nodes.update(nodes_to_update)

    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            self.clean_humans()
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




