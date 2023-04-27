from flask import Flask, request, jsonify
from reidgaze import processRequest, initialize
import os
import boto3 
import urllib.parse
import time
import subprocess
app = Flask(__name__)
s3 = boto3.client('s3')

@app.route('/health')
def health_check():
    return jsonify({'status': 'ok'})

@app.route('/', methods=['POST'])
def process_request():
    params = request.json
    start = time.time()
    json, img_list = process_request(params)
    #upload_images()
    print("Processing time : " + str(time.time() - start))
    return json



def process_request(params):
    # Create a temporary directory to store the downloaded files
    temp_dir = 'temp_files'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    try:
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
        return processRequest(params)

    finally:
        # Delete the temporary files
        for registration in params['registrations']:
            video_path = registration['registrationVideoPath']
            os.remove(video_path)
        os.remove(chunk_video_path)
        os.rmdir(temp_dir)

def upload_images(params):

    #"media_uri": "s3://aicl-media-stage/Tara-Crewe/2023-04-25_PSTEM-Spring-2023---Tara-Crewe---Denison-Montessori_Cx_Lx_video-device-unknown_15.33.34.032/Chunks/Tara-Crewe_2023-04-25_PSTEM-Spring-2023---Tara-Crewe---Denison-Montessori_Cx_Lx_video-device-unknown_15.33.34.032_Chunk0167.webm"
    media_uri = params['videoPath']
    # Extract the bucket name from the media URI
    bucket_name = media_uri.split("//")[1].split("/")[0]

    # Extract the image URI format from the media URI
    image_uri_format = media_uri.replace("/Chunks/", "/re-id-eyegaze/").replace(".webm", "/re-id-eyegaze-{}").replace("Chunk", "image")

    # Get the base path of the local images folder
    images_path = "output"
    # Replace with the actual path of the local images folder

    # Get the range of image numbers from the local directory
    image_files = os.listdir(images_path)

    # Loop over the image numbers and upload each image to S3
    for i in len(image_files) + 1:
        # Get the local path of the image file
        image_path = os.path.join(images_path, f"re-id-eyegaze-{i}.jpg")
        
        # Generate the S3 URI for the image file
        image_uri = image_uri_format.format(i)
        
        # Upload the image file to S3
        with open(image_path, "rb") as f:
            s3.upload_fileobj(f, Bucket=bucket_name, Key=image_uri)
        
        # Print a message to confirm that the upload was successful
        print(f"Uploaded {image_uri} to S3 bucket {bucket_name}")

if __name__ == '__main__':
    # Run the script and capture its output
    print("Downloading models from aws..")
    output = subprocess.check_output(['sh', 'download_models.sh'])
    # Print the output
    print(output.decode())
    initialize()
    app.run(port=5001)