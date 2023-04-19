
import os
import sys
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
        normalize_transform = T.Normalize(mean = cfg.INPUT.PIXEL_MEAN,std = cfg.INPUT.PIXEL_STD)
        self.transform = T.Compose([
            T.Resize(cfg.INPUT.IMG_SIZE),
            T.ToTensor(),
            normalize_transform
            ])  

    def __getitem__(self, index):
        box = self.boxes[index]
        image = self.img.crop(box)
        if self.transform:
            image = self.transform(image)
        return image
    
    def __len__(self,):
        return len(self.boxes)


def main(model_path,img,boxes):
    data = frameData(img,boxes)
    v_dl = DataLoader(data,batch_size=cfg.TEST.IMS_PER_BATCH,shuffle=False)
    model = build_model(cfg,10)
    model.load_param(model_path)
    empty = True
    for i, batch in enumerate(v_dl):
        model.eval()
        feat = model(batch)
        if empty:
            all_emb = feat
            empty = False
        else:
            all_emb = torch.cat((all_emb,feat),dim= 0)
    all_emb = torch.nn.functional.normalize(all_emb, dim=1, p=2)
    all_emb =all_emb.detach().numpy()
    return all_emb



