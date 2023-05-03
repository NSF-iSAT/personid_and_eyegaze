import requests
import json

params = {
    "registrations":[
        {'id': "611e8d985667804b6c3b7db0", 
        'registrationVideoPath':"s3://isat-mmia/reid/student_enrollment_214307.webm",
        'length':28},
        {'id': "611e8d985667804b6c3b6a45a", 
        'registrationVideoPath':"s3://isat-mmia/reid/student_enrollment_222826.webm",
        'length':12},
        {'id': "611e8d985667804b6c3b4a86", 
        'registrationVideoPath':"s3://isat-mmia/reid/student_enrollment_236416.webm",
        'length':13},
        {'id': "611e8d985667804b6c3ba2b4", 
        'registrationVideoPath':"s3://isat-mmia/reid/student_enrollment_253270.webm",
        'length':12},
    ],
    "ReferRateSec": 10,
    "videoPath":"s3://isat-mmia/reid/Terri-Reh_2023-03-30_PComputer-Science-P5-SP-2023---Terri-Reh---Flagstaff_Cx_Lx_video-device-unknown_12.15.09.285_Chunk0267.webm",
    "startTime":0, 
    "endTime":10,
    "TestRateSec":5
}

headers = {'Content-type': 'application/json'}

response = requests.post('http://localhost:5001', data=json.dumps(params), headers=headers)

if response.status_code == 200:
    print('Request processed successfully')
else:
    print('Request failed with status code:', response.status_code)