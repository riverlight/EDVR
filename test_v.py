# -*- coding: utf-8 -*-

import torch as t
import sys
sys.path.append("../")
import cv2
import numpy as np
import os
import time
from torchvision import transforms



class ToTensor(object):
    def __call__(self, img):
        img_f = img / 255.0
        lr = t.from_numpy(img_f).float()
        return lr.permute(2, 0, 1)

class CEDVR:
    def __init__(self, w, h, weights_file="checkpoints/model_x2_l_best.pth"):
        self._device = 'cuda'
        self._net = t.load(weights_file, map_location='cuda:0').to(self._device)
        self._net.eval()
        self._input_t = t.zeros((1, 5, 3, h, w))
        self._input_t = self._input_t.cuda()
        self._transform = transforms.Compose([ToTensor()])

    def query(self, img, idx):
        # step one : transform
        if img is not None:
            img_t = self._transform(img).unsqueeze(0)
            img_t = img_t.cuda()

        # step two : adjust input
        if idx==0:
            for i in range(5):
                self._input_t[0, i, :, :, :] = img_t  # t h w c
        else:
            for i in range(4):
                self._input_t[0, i, :, :, :] = self._input_t[0, i+1, :, :, :]
            if img is None:
                self._input_t[0, 4, :, :, :] = self._input_t[0, 3, :, :, :]
            else:
                self._input_t[0, 4, :, :, :] = img_t

        # step three : do infer
        if idx<2:
            return None
        else:
            with t.no_grad():
                out = self._net(self._input_t)
            out = out.clone().detach()
            out = out.to(t.device('cpu'))
            out = out.squeeze()
            out = out.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(t.uint8).numpy()
            return out

def main():
    scale = 2
    s_mp4 = "d:/workroom/testroom/156_lr.mp4"
    cap = cv2.VideoCapture(s_mp4)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'I420')
    out = cv2.VideoWriter(s_mp4.replace('.mp4', '_edvr.avi'), fourcc, fps, (width, height))
    net = CEDVR(width//scale, height//scale)

    count = 0
    frame = None
    starttime = time.time()
    while True:
        if count % 25 == 0:
            print('frame id ', count)
        ret, frame = cap.read()
        if ret is not True:
            break

        ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        print('ts :', ts)
        print("cost time : ", time.time() - starttime)
        pred_img = net.query(frame, count)
        if pred_img is not None:
            out.write(pred_img)
        count += 1
        if ts > 30:
            break
    pred_img = net.query(None, count+1)
    out.write(pred_img)
    pred_img = net.query(None, count + 2)
    out.write(pred_img)
    out.release()
    cap.release()

    pass

if __name__=="__main__":
    print("Hi, this is EDVR video input test program")
    main()
    print('done')
