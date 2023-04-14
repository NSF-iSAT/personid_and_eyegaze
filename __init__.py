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
    "videoPath":"/scratch/shared/whitehill/recording_id_6425d1ae29b832001e054b16/Terri-Reh_2023-03-30_PComputer-Science-P5-SP-2023---Terri-Reh---Flagstaff_Cx_Lx_video-device-unknown_12.15.09.285_Chunk0001.webm",
    "startTime":0, 
    "endTime":5,
    "TestRateSec":1
}

'''