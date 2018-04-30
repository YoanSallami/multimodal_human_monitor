uint32 tracklet_id

# 3D pose in the frame of the input image
geometry_msgs/Pose head

# Bounding box in the frame of the image plane
BoundingBox box

# The body joints output by the CPM
BodyJoints2D joints

# Where the person is looking at
TargetNameWithProbability[] targets
