from .reidgaze import processRequest


def get_reidgaze(params):
    frame_json,img_list = processRequest(params)
    return frame_json,img_list

'''
example of params:
params = {
    "registrations":[
        {'id': "214307", 
        'registrationVideoPath':"/scratch/shared/whitehill/recording_id_6425d1ae29b832001e054b16/student_enrollment_214307.webm",
        'length':28},
        {'id': "222826", 
        'registrationVideoPath':"/scratch/shared/whitehill/recording_id_6425d1ae29b832001e054b16/student_enrollment_222826.webm",
        'length':12},
        {'id': "236416", 
        'registrationVideoPath':"/scratch/shared/whitehill/recording_id_6425d1ae29b832001e054b16/student_enrollment_236416.webm",
        'length':13},
        {'id': "253270", 
        'registrationVideoPath':"/scratch/shared/whitehill/recording_id_6425d1ae29b832001e054b16/student_enrollment_253270.webm",
        'length':12},
    ],
    "ReferRateSec":4,
    "videoPath":"Chunk0001.webm",
    "startTime":0, 
    "endTime":5,
    "TestRateSec":1
}

'''
