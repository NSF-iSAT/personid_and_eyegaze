import numpy as np
import pandas as pd
import os, re, random, json, collections,argparse, ffmpeg,cv2,torch
import os.path as osp
from matplotlib import pyplot as plt
from scipy.spatial import distance
import subprocess as sp
from mtcnn import MTCNN
from PIL import Image,ImageDraw,ImageFont
from tensorflow import keras
from tensorflow.keras.metrics import categorical_accuracy
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from GazeNet.Face_Detection import image, load_test
from get_embedding import main as gemb

model_path ="AGW/resnet50_nl_model_18.pth"

def euclideanLoss(y_true, y_pred):
    return K.mean(K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))) 
gaze_model = keras.models.load_model("GazeNet/Model/1", custom_objects={'euclideanLoss': euclideanLoss,
                                                               'categorical_accuracy': categorical_accuracy})

def processRequest (params):
    registrations = params['registrations']
    inputVideo = params['videoPath']
    startTime = params['startTime']
    endTime = params['endTime']
    testratesec = params['TestRateSec']
    refratesec = params['ReferRateSec']
    ref_embs,labels = get_ref_emb(registrations,refratesec)
    frame_json = {}
    img_list = []
    for t in range(startTime, endTime,testratesec):
        # grab frame from video at time t 
        frame,w,h = get_frame(inputVideo,t) #Frame is array
        image = Image.fromarray(frame) #Image is image
        # detect the bounding boxes and faces
        bounding_boxes = detect_boxes(frame)
        face_boxes,faces,heads = detect_face(image,frame)
        # process frame (reid, eye gaze, etc.)
        emb = gemb(model_path,image,bounding_boxes)
        # set an threshhold
        assign_label_list = assign(ref_embs,labels,emb)
        img_resize = image.resize((256, 256))
        img_f = np.array([np.array(img_resize)]) / 255.0
        img_fr = img_f
        for _ in range(len(faces)-1):
            img_fr = np.vstack((img_fr,img_f))
        preds_gaze = gaze_model.predict([img_fr,np.array(faces),np.array(heads)])
        # output the json file and figure
        frame_res = []
        for i,pbox in enumerate(bounding_boxes):
            a = ImageDraw.ImageDraw(image)
            print(pbox)
            person = {}
            person["Id"] = assign_label_list[i]
            x_top_left, y_top_left, x_bottom_right, y_bottom_right = pbox[0],pbox[1],pbox[2],pbox[3]
            pbox_dic = {}
            pbox_dic["x"]=float(x_top_left)
            pbox_dic["y"]=float(y_top_left)
            pbox_dic["width"]=float(x_bottom_right - x_top_left)
            pbox_dic["height"]=float(y_bottom_right - y_top_left)
            person["Personbox"] = pbox_dic
            a.rectangle(pbox,fill=None,outline = "cyan", width=5)
            a.text((x_top_left+10,y_top_left+10),str(person["Id"]),fill = "cyan")
            for j,face in enumerate(face_boxes):
                a.rectangle((face['box'][0],face['box'][1],face['box'][0]+face['box'][2],face['box'][1]+face['box'][3]),fill=None,outline = "magenta", width=3)
                matched_face = "Unknown"
                gaze = "Unknown"
                if x_top_left<face['box'][0] and y_top_left< face['box'][1] and x_bottom_right>face['box'][0]+face['box'][2] and y_bottom_right> face['box'][1]+face['box'][3]:
                    matched_face = {"x":float(face['box'][0]),"y":float(face['box'][1]),"width":float(face['box'][2]),"height":float(face['box'][3])}
                    gaze = {"x":float(preds_gaze[0][j][0])*w,"y":float(preds_gaze[0][j][1])*h}
                    na = np.array(image)
                    start_point = (int(heads[j][0]* w), int(heads[j][1]* h))
                    end_point = (int(preds_gaze[0][j][0]* w),int(preds_gaze[0][j][1]* h))
                    na = cv2.arrowedLine(na,start_point,end_point,(255, 255, 0),5)
                    image = Image.fromarray(na)
                    break
            person["Facebox"] = matched_face
            person["GazeTarget"] = gaze
            frame_res.append(person)
        frame_json[t]=frame_res
        # image.save(f"img5_{}.jpg")
        img_list.append(image)
    # with open(f"example_all.json","w",encoding='utf-8') as f:
    #     json.dump(frame_json,f)
    return frame_json,img_list

def get_ref_emb(registrition,ref_duration):
    ref_embs = []
    labels = []
    for one_ref in registrition:
        ref_boxes = []
        label = one_ref["id"]
        ref_path = one_ref["registrationVideoPath"]
        length = one_ref['length']
        for t in range(0,length,ref_duration):           
            ref_frame,width,height = get_frame(ref_path,t)
            ref_image = Image.fromarray(ref_frame)
            # ref_image.save(f"exam_{label}_{t}.jpg")
            bounding_boxes = detect_boxes(ref_frame)
            pbox = bounding_boxes[np.argmax((bounding_boxes[:,2]-bounding_boxes[:,0])*(bounding_boxes[:,3]-bounding_boxes[:,1]))]
            ref_boxes.append(pbox)
        embs = gemb(model_path,ref_image,ref_boxes)
        ref_emb = np.array(embs).mean(axis=0)
        ref_embs.append(ref_emb)
        labels.append(label)
    return ref_embs,labels

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

def detect_boxes(im):
    im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    # use pre-trained model to detect the things in pic
    cfg = get_cfg()
    cfg.MODEL.DEVICE='cpu'
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)
    people_instances = outputs['instances'][outputs['instances'].pred_classes == 0]
    boxes = np.array(people_instances.pred_boxes.tensor.cpu())
    # masks = np.array(people_instances.pred_masks.cpu())
    size = people_instances._image_size
    # visualize the detection output
    # v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    # out = v.draw_instance_predictions(people_instances.to("cpu"))
    # cv2.imwrite("des_res.jpg",out.get_image()[:, :, ::-1])
    return boxes

def head_cord(face):
    head_x = face['box'][0]+face['box'][2]/2
    head_y = face['box'][1]+face['box'][3]/2
    return (head_x,head_y)

def detect_face(image,img_ar):
    faces_li = []
    faces_boxes = []
    heads = []
    detector = MTCNN()
    faces = detector.detect_faces(img_ar)
    w= np.size(image)[0]
    h= np.size(image)[1]
    for face in faces:
        (head_x,head_y) = head_cord(face)
        heads.append((head_x/w,head_y/h))
        ltx = face['box'][0]
        lty = face['box'][1]
        rbx = ltx + face['box'][2]
        rby = lty +face['box'][3]
        face_ima = image.crop((ltx,lty,rbx,rby))
        face_ima = face_ima.resize((32,32))
        face_ar = np.array(face_ima)/ 255.0
        faces_li.append(face_ar)
    return faces,faces_li,heads

def assign(ref_embs,labels,frame_emb):
    assign_list=[]
    distmat = distance.cdist(np.array(ref_embs), frame_emb, 'euclidean')
    min_list = np.min(distmat,0)
    index_list = np.argmin(distmat,0)
    for ii,i in enumerate(index_list):
        if min_list[ii]<1.2:
            assign_list.append(labels[i])
        else:
            assign_list.append("Unknown")
    return assign_list

if __name__ == "__main__":
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
        "startTime":10, 
        "endTime":20,
        "TestRateSec":1
    }
    processRequest(params)








