# personid_and_eyegaze

This repository is a pre-trained model to give the id to a person based on the enrollment video and find where a person is looking at in the classroom environment.

## Requirements
See the requirements.txt file.

## Run 
Call the __init__.py or processRequest function in the reidgaze.py. The input contains 6 parameters:

1. registrations:
A list of enrollments, which includes the id, the path of the enroll video, and the duration of the enrollment video.

2. ReferRateSec:
In the registration video, the rate to get the frames as the embedding.
