import glob
import os
import pandas as pd
import argparse
import numpy as np
import cv2
import sys
sys.path.append('./')

from eco import ECOTracker
from PIL import Image
import time
import argparse

def main(video_dir):
    # load videos
    filenames = sorted(glob.glob(os.path.join(video_dir, "img/*.jpg")),
           key=lambda x: int(os.path.basename(x).split('.')[0]))
    # frames = [cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB) for filename in filenames]
    frames = [np.array(Image.open(filename)) for filename in filenames]
    print("---------------- DEBUG INFO ----------------")
    print(f"正在读取的文件夹路径: {video_dir}")
    print(f"找到的图片数量: {len(frames) if 'frames' in locals() else '未定义'}")
    print("--------------------------------------------")
    height, width = frames[0].shape[:2]
    if len(frames[0].shape) == 3:
        is_color = True
    else:
        is_color = False
        frames = [frame[:, :, np.newaxis] for frame in frames]
    gt_bboxes = pd.read_csv(os.path.join(video_dir, "groundtruth_rect.txt"), sep='\t|,| ',
            header=None, names=['xmin', 'ymin', 'width', 'height'],
            engine='python')

    title = video_dir.split('/')[-1]
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # img_writer = cv2.VideoWriter(os.path.join('./videos', title+'.avi'),
    #         fourcc, 25, (width, height))
    # starting tracking
    tic = time.time()
    tracker = ECOTracker(is_color) # 跟踪器
    vis = True
    for idx, frame in enumerate(frames):
        if idx == 0:
            bbox = gt_bboxes.iloc[0].values
            tracker.init(frame, bbox)
            bbox = (bbox[0]-1, bbox[1]-1,
                    bbox[0]+bbox[2]-1, bbox[1]+bbox[3]-1)
        elif idx < len(frames) - 1:
            bbox = tracker.update(frame, True, vis)
        else: # last frame
            bbox = tracker.update(frame, False, vis)
        # bbox xmin ymin xmax ymax
        frame = frame.squeeze()
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        frame = cv2.rectangle(frame,
                              (int(bbox[0]), int(bbox[1])),
                              (int(bbox[2]), int(bbox[3])),
                              (0, 255, 255),
                              1)
        gt_bbox = gt_bboxes.iloc[idx].values
        gt_bbox = (gt_bbox[0], gt_bbox[1],
                   gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3])
        frame = frame.squeeze()
        frame = cv2.rectangle(frame,
                              (int(gt_bbox[0]-1), int(gt_bbox[1]-1)), # 0-index
                              (int(gt_bbox[2]-1), int(gt_bbox[3]-1)),
                              (0, 255, 0),
                              1)
        if vis and idx > 0:
            score = tracker.score # 响应图
            size = tuple(tracker.crop_size.astype(np.int32)) # 目标裁剪区域
            score = cv2.resize(score, size) # 将热力图放大到图像块一样大
            score -= score.min() # 最小值归零
            score /= score.max() # 最大值缩放到1
            score = (score * 255).astype(np.uint8) # 映射到0-255 灰度图
            # score = 255 - score
            score = cv2.applyColorMap(score, cv2.COLORMAP_JET) # 伪彩色： 灰度转为彩色热力图
            # 以上是生成热力图的过程，红色表示响应高的区域，蓝色表示响应低的区域
            
            # 以下是将热力图叠加到当前帧的对应位置上
            pos = tracker._pos # 目标中心位置
            pos = (int(pos[0]), int(pos[1])) 
            xmin = pos[1] - size[1]//2 # 计算左上角和右下角坐标
            xmax = pos[1] + size[1]//2 + size[1] % 2
            ymin = pos[0] - size[0] // 2 
            ymax = pos[0] + size[0] // 2 + size[0] % 2
            # 以下是处理边界情况
            # 左边界
            left = abs(xmin) if xmin < 0 else 0 # 跟踪的人出画面了，导致xmin为负
            xmin = 0 if xmin < 0 else xmin # 修正xmin为0
            # 右边界
            right = width - xmax
            xmax = width if right < 0 else xmax # 跟踪的人出画面了，导致xmax超过图像宽度，限制在width
            right = size[1] + right if right < 0 else size[1] # 计算热力图要切掉多少
            top = abs(ymin) if ymin < 0 else 0
            ymin = 0 if ymin < 0 else ymin
            down = height - ymax
            ymax = height if down < 0 else ymax
            down = size[0] + down if down < 0 else size[0]
            score = score[top:down, left:right] # 裁剪热力图
            crop_img = frame[ymin:ymax, xmin:xmax] # 裁剪原图对应区域
            # if crop_img.shape != score.shape:
            #     print(left, right, top, down)
            #     print(xmin, ymin, xmax, ymax)
            score_map = cv2.addWeighted(crop_img, 0.6, score, 0.4, 0) # 融合热力图和原图
            frame[ymin:ymax, xmin:xmax] = score_map # 将融合结果放回原图

        frame = cv2.putText(frame, str(idx), (5, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
        # img_writer.write(frame)
        cv2.imshow(title, frame)
        cv2.waitKey(1)
    duration = time.time() - tic
    print(round(len(frames) / duration, 3), "FPS")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, default='./sequences/Crossing/')
    args = parser.parse_args()
    main(args.video_dir)
