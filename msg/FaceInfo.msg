uint32  person_id

# When a person reappears, the track ID is different than the previous
# track ID.
#
# NOT USED IN V0.0
uint32 track_id

# Read from PeoplePerception/Person/<ID>/AgeProperties
int32   age
float32 age_confidence

# Read from PeoplePerception/Person/<ID>/GenderProperties
int32   gender
float32 gender_confidence

# Read from PeoplePerception/Person/<ID>/SmileProperties
float32 smile_degree
float32 smile_degree_confidence

ExpressionProperties expression_properties
