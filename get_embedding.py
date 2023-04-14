'''
Get the embedding from the model.
Steps:
1. Define the data use Data class
2. Change to DataLoader
3. Change the Test Pipeline so that the output/ We save the embedding result
'''
import os
import sys
sys.path.append("/scratch/shared/whitehill/ReidGaze/AGW")
sys.path.append("/scratch/shared/whitehill/ReidGaze")
import torch
import logging
from AGW.modeling import build_model
from AGW.configs_emb import _C as cfg
import logging
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as T
from PIL import Image
import numpy as np
from collections import defaultdict


cfg = cfg
class frameData(Dataset):
    def __init__(self,img,boxes):
        self.img = img
        self.boxes = boxes
        # self.labels = labels
        normalize_transform = T.Normalize(mean = cfg.INPUT.PIXEL_MEAN,std = cfg.INPUT.PIXEL_STD)
        self.transform = T.Compose([
            T.Resize(cfg.INPUT.IMG_SIZE),
            T.ToTensor(),
            normalize_transform
            ])
        # self.files_list = sorted(os.listdir(image_dir))    

    def __getitem__(self, index):
        # file_name = self.files_list[index]
        # path = os.path.join(self.img_dir,file_name)
        # image = self.read_image(path)
        # label = file_name[-11:-4]
        box = self.boxes[index]
        image = self.img.crop(box)
        if self.transform:
            image = self.transform(image)
        # return image,label
        # if self.labels:
        #     label = self.labels[index]
        #     return image,label
        return image
    
    def __len__(self,):
        return len(self.boxes)


def main(model_path,img,boxes):
    data = frameData(img,boxes)
    v_dl = DataLoader(data,batch_size=cfg.TEST.IMS_PER_BATCH,shuffle=False)
    model = build_model(cfg,10)
    model.load_param(model_path)
    empty = True
    # for i,(batch,labels) in enumerate(v_dl):
    # if labels:
    #     ref_emb = defaultdict(list)
    #     for i, (batch,label) in enumerate(v_dl):
    #         model.eval()
    #         feat = model(batch)
    #         if empty:
    #             all_emb = feat
    #             empty = False
    #         else:
    #             all_emb = torch.cat((all_emb,feat),dim= 0)
    #         # if i%5 ==0:
    #         print(f"{i}th batch finished. {i*cfg.TEST.IMS_PER_BATCH} samples finished")
    #     all_emb = torch.nn.functional.normalize(all_emb, dim=1, p=2)
    #     all_emb = all_emb.detach().numpy()
    #     for i in range(len(all_emb)):
    #         emb = all_emb[i]
    #         l = label[i]
    #         ref_emb[l].append(emb)
    #     return ref_emb
    for i, batch in enumerate(v_dl):
        model.eval()
        feat = model(batch)
        if empty:
            all_emb = feat
            empty = False
        else:
            all_emb = torch.cat((all_emb,feat),dim= 0)
        # if i%5 ==0:
        print(f"{i}th batch finished. {i*cfg.TEST.IMS_PER_BATCH} samples finished")
    all_emb = torch.nn.functional.normalize(all_emb, dim=1, p=2)
    all_emb =all_emb.detach().numpy()
    return all_emb


    

# if __name__=="__main__":
#     model_path="/scratch/shared/whitehill/AGW/resnet50_nl_model_18.pth"
#     img_path = "/scratch/shared/whitehill/AGW/Data_demo/0env_try/data"
#     boxes = 
#     emb = main(model_path,img_path,boxes)
#     # np.save("example.npy",emb)
#     np.save("/scratch/shared/whitehill/AGW/Data_demo/0env_try/example_emb.npy",emb)



