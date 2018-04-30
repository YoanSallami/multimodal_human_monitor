uint32 person_id

# When a person reappears, the track ID is different than the previous
# track ID.
#
# NOT USED IN V0.0
uint32 track_id

uint32             head_gaze_available
geometry_msgs/Pose head_gaze

uint32             eye_gaze_available
geometry_msgs/Pose eye_gaze

# PeoplePerception/Person/<ID>/IsLookingAtRobot
float64 probability_looking_at_robot

# NOT USED IN V0.0
float64 probability_looking_at_screen

# The (known) targets of interest a person can be looking at
IdWithProbability[] attentions
