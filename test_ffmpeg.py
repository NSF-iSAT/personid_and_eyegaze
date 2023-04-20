import numpy as np
import ffmpeg
# from PIL import Image,ImageDraw,ImageFont
import subprocess as sp

def get_frame(video_path,t):
    probe = ffmpeg.probe(video_path)
    # time = float(probe['streams'][0]['duration']) 
    width = probe['streams'][1]['width']
    height = probe['streams'][1]['height']
    command = ["ffmpeg",
               '-ss',str(t),
               '-i',video_path,
               '-vf','scale = %d:%d'%(width,height),
               '-vframes',"1",
                '-f', 'image2pipe',
                '-pix_fmt', 'rgb24',
                '-vcodec', 'rawvideo', '-']
    pipe = sp.Popen(command, stdout = sp.PIPE)
    frame = pipe.stdout.read()
    image = np.fromstring(frame, dtype='uint8')
    image = image.reshape((height,width,3))
    pipe.stdout.flush()
    return image,width,height
  
if __name__ == "__main__" :
    inputVideo = "/home/ubuntu/personid_and_eyegaze/chunk.webm"
    frame,w,h = get_frame(inputVideo,1)
