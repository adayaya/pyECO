import numpy as np
import time
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from eco import ECOTracker
import cv2
import glob
import os
from tqdm import tqdm
from PIL import Image
import pandas as pd
import argparse

class Seq:
    def __init__(self, video_dir):
        self.video_dir = video_dir
        self.gt_bboxes = pd.read_csv(os.path.join(video_dir, "groundtruth_rect.txt"), sep='\t|,| ',
            header=None, names=['xmin', 'ymin', 'width', 'height'],
            engine='python')
        self.init_rect = self.gt_bboxes.iloc[0].values  # 格式: [x, y, w, h]
        
        # 自动读取文件夹下的所有图片
        # 假设图片格式是 jpg，按文件名排序
        self.s_frames = sorted(glob.glob(os.path.join(video_dir, "img/*.jpg")),
           key=lambda x: int(os.path.basename(x).split('.')[0]))
        
        # 自动计算帧数
        self.len = len(self.s_frames)

def run_ECO(seq):
    x = seq.init_rect[0]
    y = seq.init_rect[1]
    w = seq.init_rect[2]
    h = seq.init_rect[3]

    frames = [np.array(Image.open(filename)) for filename in seq.s_frames]
    # frames = [cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB) for filename in seq.s_frames]
    if len(frames[0].shape) == 3:
        is_color = True
    else:
        is_color = False
        frames = [frame[:, :, np.newaxis] for frame in frames]
    tic = time.time()
    # starting tracking
    tracker = ECOTracker(is_color)
    res = []
    for idx, frame in enumerate(frames):
        if idx == 0:
            bbox = (x, y, w, h)
            tracker.init(frame, bbox) # 这里的bbox是用来计算初始要识别的位置吧 
            bbox = (bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3])
        elif idx < len(frames) - 1:
            bbox = tracker.update(frame, True)
        else: # last frame
            bbox = tracker.update(frame, False)
        res.append((bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]))
    duration = time.time() - tic
    result = {}
    result['res'] = res
    result['type'] = 'rect'
    result['fps'] = round(seq.len / duration, 3)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, default='./sequences/Crossing/')
    args = parser.parse_args()
    seq = Seq(args.video_dir)
    result = run_ECO(seq)
    print(f"Tracking completed. FPS: {result['fps']}")


