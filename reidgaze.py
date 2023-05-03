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
import sys
sys.path.append("./AGW")
from AGW.modeling import build_model
from AGW.configs_emb import _C as cfg
import time
def initialize():
    def euclideanLoss(y_true, y_pred):
        return K.mean(K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))) 
    global gaze_model
    gaze_model = keras.models.load_model("GazeNet/Model/1", custom_objects={'euclideanLoss': euclideanLoss,
                                                                'categorical_accuracy': categorical_accuracy})
    model_path ="AGW/resnet50_nl_model_18.pth"
    global model
    model = build_model(cfg,10)
    model.load_param(model_path)
    global allRegistrationEmbeddings
    allRegistrationEmbeddings = {}
    
 
def processRequest (params, img_dir):
   # initialize()
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
        cnt = 1
        # grab frame from video at time t 
        frame,w,h = get_frame(inputVideo,t) #Frame is array
        image = Image.fromarray(frame) #Image is image
        # detect the bounding boxes and faces
        bounding_boxes = detect_boxes(frame)
        face_boxes,faces,heads = detect_face(image,frame)
        # process frame (reid, eye gaze, etc.)
        emb = gemb(model,image,bounding_boxes)
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
                matched_face = "Unknown"
                gaze = "Unknown"
                if x_top_left<face['box'][0] and y_top_left< face['box'][1] and x_bottom_right>face['box'][0]+face['box'][2] and y_bottom_right> face['box'][1]+face['box'][3]:
                    a.rectangle((face['box'][0],face['box'][1],face['box'][0]+face['box'][2],face['box'][1]+face['box'][3]),fill=None,outline = "magenta", width=3)
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
        output_dir = img_dir
        image.save(output_dir + f"/re-id-eyegaze-{cnt:03}.jpg")
        print( "count: " + str(cnt))
        cnt += 1 
        img_list.append(image)
    with open(f"example_all.json","w",encoding='utf-8') as f:
        json.dump(frame_json,f)
    return frame_json,img_list

def get_ref_emb(registrition,ref_duration):
    start = time.time()
    ref_embs = []
    labels = []
    for one_ref in registrition:
        ref_boxes = []
        label = one_ref["id"]
        ref_path = one_ref["registrationVideoPath"]
        if ref_path in allRegistrationEmbeddings.keys():
            (ref_emb, label) = allRegistrationEmbeddings[ref_path]
        else:
            length = one_ref['length']
            for t in range(0,length,ref_duration):           
                ref_frame,width,height = get_frame(ref_path,t)
                ref_image = Image.fromarray(ref_frame)
                # ref_image.save(f"exam_{label}_{t}.jpg")
                bounding_boxes = detect_boxes(ref_frame)
                pbox = bounding_boxes[np.argmax((bounding_boxes[:,2]-bounding_boxes[:,0])*(bounding_boxes[:,3]-bounding_boxes[:,1]))]
                ref_boxes.append(pbox)
            embs = gemb(model,ref_image,ref_boxes)
            ref_emb = np.array(embs).mean(axis=0)
            allRegistrationEmbeddings[ref_path] = (ref_emb, label)
        # either way, we now have the correct registration embeddings for this video
        ref_embs.append(ref_emb)
        labels.append(label)
    print("Ref embeddings processing time: " + str(time.time() - start))
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
            {'id': "611e8d985667804b6c3b7db0", 
            'registrationVideoPath':"/home/ubuntu/personid_and_eyegaze/files/student_enrollment_214307.webm",
            'length':28},
            {'id': "611e8d985667804b6c3b6a45a", 
            'registrationVideoPath':"/home/ubuntu/personid_and_eyegaze/files/student_enrollment_222826.webm",
            'length':12},
            {'id': "611e8d985667804b6c3b4a86", 
            'registrationVideoPath':"/home/ubuntu/personid_and_eyegaze/files/student_enrollment_236416.webm",
            'length':13},
            {'id': "611e8d985667804b6c3ba2b4", 
            'registrationVideoPath':"/home/ubuntu/personid_and_eyegaze/files/student_enrollment_253270.webm",
            'length':12},
        ],
        "ReferRateSec": 10,
        "videoPath":"/home/ubuntu/personid_and_eyegaze/files/chunk02.webm",
        "startTime":0, 
        "endTime":10,
        "TestRateSec":5
    }
    start = time.time()
    initialize()
    processRequest(params)
    print("Processing time : " + str(time.time() - start))









