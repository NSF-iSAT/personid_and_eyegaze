from flask import Flask, request, jsonify
from reidgaze import processRequest, initialize
import os
import boto3 
import urllib.parse
import time
import subprocess
import asyncio
import tempfile
import shutil
import re
from threading import Thread
app = Flask(__name__)
s3 = boto3.client('s3')

@app.route('/health')
def health_check():
    return jsonify({'status': 'ok'})

def process_and_upload_images():
    params = request.json
    thread = Thread(target=process_request_async, args=(params,), name="eyegaze-thread")
    thread.start()

def process_request_async(params):
    # asynchronous processing logic that creates image files in temp_dir
    return process_request(params)

@app.route('/', methods=['POST'])
def process_request_handler():
    params = request.json
    print(params)
    process_and_upload_images()
    return 'OK', 200


def process_request(params):
    # Create a unique temporary directory to store the downloaded files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download each registration video from S3 to the temporary directory
        for registration in params['registrations']:
            s3_url = registration['registrationVideoPath']
            print(s3_url)
            result = urllib.parse.urlparse(s3_url)
            bucket = result.netloc
            key = result.path.lstrip('/')           
            video_name = s3_url.split('/')[-1]
            video_path = os.path.join(temp_dir, video_name)
            s3.download_file(Bucket=bucket, Key=key, Filename=video_path)
            registration['registrationVideoPath'] = video_path

        chunk_s3_url = params['videoPath']
        result = urllib.parse.urlparse(chunk_s3_url)
        key = result.path.lstrip('/')
        bucket = result.netloc
        chunk_video_name = chunk_s3_url.split('/')[-1]
        chunk_video_path = os.path.join(temp_dir, chunk_video_name)
        s3.download_file(Bucket=bucket, Key=key, Filename=chunk_video_path)
        params['videoPath'] = chunk_video_path
        # Process the request using the updated params
        print("Params")
        print(params)
        os.mkdir(temp_dir + '/images')
        processRequest(params, temp_dir + '/images')
        upload_images(chunk_s3_url, temp_dir + '/images') 



def upload_images(chunk_s3_url, temp_dir):
    print("uploading" + temp_dir)
    input_path = chunk_s3_url
    input_path = input_path.replace("fixedChunks", "Chunks")
    folder_path = temp_dir

    # extract the file name from the input path
    file_name = os.path.splitext(os.path.basename(input_path))[0]

    # extract the bucket name and prefix from the input path
    bucket_name = input_path.split("/")[2]
    prefix = "/".join(input_path.split("/")[3:5])

    # generate the output path for each file in the folder
    for file in os.listdir(folder_path):
        if file.endswith(".jpg"):
            output_path = f"s3://{bucket_name}/{prefix}/re-id-eyegaze/{file_name}/{file}"
            print(output_path)

            # upload the file to S3
            s3_client = boto3.client('s3')
            with open(os.path.join(folder_path, file), "rb") as f:
                s3_client.upload_fileobj(f, bucket_name, output_path.split(bucket_name + "/")[1])

if __name__ == '__main__':
    # Run the script and capture its output
    print("Downloading models from aws..")
    output = subprocess.check_output(['sh', 'download_models.sh'])
    # Print the output
    print(output.decode())
    initialize()
    app.run(host= "0.0.0.0", port=5001)