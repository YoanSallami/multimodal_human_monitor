uint32 person_id

# NOT USED IN V0.0
uint32   track_id
duration track_age

# A person may be given several IDs as new tracks are created. When we
# realise that 2 IDs are actually the same person, we store the other
# IDs in alternate_ids. The rule to set the person_id given the
# several IDs is
#
#   person_id = min alternate_ids
uint32[] alternate_ids

float64  head_distance
bool     is_in_fov
bool     is_occluded
bool     is_face_detected
bool     is_head_detected

# NOT USED IN V0.0
geometry_msgs/Pose2D body_pose_gp

# NOT USED IN V0.0
geometry_msgs/Pose head_pose

# Body joints
BodyJoints3D body_joints
