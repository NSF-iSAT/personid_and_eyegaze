# personid_and_eyegaze

This repository is a trained model to give the id to a person based on the enrollment video, then find where a person is looking at in the classroom environment.

## Requirements
See the requirements.txt file.

## Run 
Call the __init__.py or processRequest(Params) function in the reidgaze.py. The input contains 6 parameters:

1. registrations:
A list of enrollments, which includes the id, the path of the enroll video, and the duration of the enrollment video.

2. ReferRateSec:
In the registration video, the rate to get the frames. Then we use these frames to get the relevant number of embeddings, compute the mean embedding as the reference embedding for this enrollment person.

3. videoPath:
The path to the input video which we want to do the person re-id/gaze tracking.

4. startTime/endTime:
Two timestamps in the input video. The video clip between the startTime and endTime is the input.

6. TestRateSec
The rate we take a frame from input video to do the person re-id and eyegaze tracking.
