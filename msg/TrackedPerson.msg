# ID of the person
uint32 person_id

# ID of the tracklet that is associated to person_id
uint32 track_id

# Time since the tracklet has existed
#
# NOT USED YET
duration track_age

# A person may be given several IDs as new tracks are created. When we
# realise that 2 IDs are actually the same person, we store the other
# IDs in alternate_ids. The rule to set the person_id given the
# several IDs is
#
#   person_id = min alternate_ids
uint32[] alternate_ids

# Whether reidentification is performed.
#
# If is_identified > 0, it means that data is accumulated to performed
# re-identification.
#
# If is_identified = 0, it means that no data (i.e. face features) is
# accumulated yet to perform re-identification. It may be because the
# person is too far, moving too fast, profile, etc.
uint32 is_identified

float64  head_distance
bool     is_in_fov
bool     is_occluded
bool     is_face_detected
bool     is_head_detected



# NOT USED YET
geometry_msgs/Pose2D body_pose_gp

# Head orientation
geometry_msgs/Pose head_pose

# Body joints
BodyJoints3D body_joints_3d

# Location of face in the 2D image (debugging)
BoundingBox box

# Location of body joints in the 2D image (debugging)
BodyJoints2D body_joints_2d
