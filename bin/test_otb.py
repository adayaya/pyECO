import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
# 引入你原本的代码 (假设你原本的文件叫 demo.py 或者 eco_main.py)
# 注意：你需要把 Seq 类和 run_ECO 函数所在的文件名替换这里的 'your_script_name'
from run_ECO import Seq, run_ECO 

# OTB-50 序列列表 (标准列表)
OTB50_SEQS = [
    'Basketball', 'Biker', 'Bird1', 'BlurBody', 'BlurCar2', 'BlurFace', 'BlurOwl', 'Bolt', 
    'Box', 'Car1', 'Car4', 'CarDark', 'CarScale', 'ClifBar', 'Couple', 'Crowds', 'David', 
    'Deer', 'Diving', 'DragonBaby', 'Dudek', 'Football', 'Freeman4', 'Girl', 'Human3', 
    'Human4', 'Human6', 'Human9', 'Ironman', 'Jump', 'Jumping', 'Liquor', 'Matrix', 
    'MotorRolling', 'Panda', 'RedTeam', 'Shaking', 'Singer2', 'Skating1', 'Skating1', 
    'Skating2', 'Skiing', 'Soccer', 'Surfer', 'Sylvester', 'Tiger1', 'Tiger2', 'Trellis', 
    'Walking', 'Walking2', 'Woman'
]

def calc_iou(pred_box, gt_box):
    """计算单帧 IoU"""
    x1, y1, w1, h1 = pred_box
    x2, y2, w2, h2 = gt_box
    
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def calc_center_error(pred_box, gt_box):
    """计算中心点欧式距离误差"""
    cp_x = pred_box[0] + pred_box[2] / 2
    cp_y = pred_box[1] + pred_box[3] / 2
    cg_x = gt_box[0] + gt_box[2] / 2
    cg_y = gt_box[1] + gt_box[3] / 2
    
    return np.sqrt((cp_x - cg_x)**2 + (cp_y - cg_y)**2)

def evaluate_otb(data_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_ious = []
    all_dists = []
    
    print(f"Start testing on {len(OTB50_SEQS)} sequences...")

    for seq_name in tqdm(OTB50_SEQS):
        video_path = os.path.join(data_dir, seq_name)
        
        # 1. 检查数据路径是否存在
        if not os.path.exists(video_path):
            # 尝试处理OTB常见的命名不一致问题 (例如 Human4-2)
            # 这里简单跳过，实际使用时请检查你的数据集文件夹名
            print(f"Warning: {seq_name} not found in {data_dir}")
            continue

        try:
            # 2. 初始化序列
            seq = Seq(video_path)
            
            # 3. 运行 ECO
            # 注意：run_ECO 可能会很慢，如果用的是 CPU
            result = run_ECO(seq) 
            pred_bboxes = result['res'] # list of (x,y,w,h)

            # 4. 保存结果到 txt (OTB 格式通常为 x,y,w,h)
            res_path = os.path.join(output_dir, f'{seq_name}.txt')
            np.savetxt(res_path, pred_bboxes, fmt='%.4f', delimiter=',')

            # 5. 立即计算当前视频的指标 (与 GT 对比)
            # 获取 GT (DataFrame转numpy)
            gt_bboxes = seq.gt_bboxes.values 
            
            # 确保长度一致 (有时预测结果可能比GT少一帧或多一帧，取交集长度)
            min_len = min(len(pred_bboxes), len(gt_bboxes))
            
            ious = []
            dists = []
            
            for i in range(min_len):
                # 跳过由 NaN 组成的 GT (OTB中有些帧是遮挡或未标注)
                if np.isnan(gt_bboxes[i]).any():
                    continue
                    
                iou = calc_iou(pred_bboxes[i], gt_bboxes[i])
                dist = calc_center_error(pred_bboxes[i], gt_bboxes[i])
                
                ious.append(iou)
                dists.append(dist)
            
            all_ious.extend(ious)
            all_dists.extend(dists)

        except Exception as e:
            print(f"Error processing {seq_name}: {e}")

    # 6. 汇总计算 OPE (One Pass Evaluation) 指标
    # Precision: 中心误差小于 20 像素的帧的比例
    precision_score = np.mean(np.array(all_dists) < 20)
    
    # Success: 平均 IOU (这里简化为 Mean IOU，标准的Success Plot是AUC)
    success_score = np.mean(all_ious)

    print("-" * 40)
    print(f"OTB-50 Evaluation Results:")
    print(f"Precision (20px): {precision_score:.4f}")
    print(f"Success (Mean IoU): {success_score:.4f}")
    print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 修改这里的默认路径为你存放 OTB 数据集的路径
    parser.add_argument('--data_dir', type=str, default='./pydataset/OTB-dataset/OTB100')
    parser.add_argument('--output_dir', type=str, default='./results/OTB50_ECO')
    args = parser.parse_args()
    
    evaluate_otb(args.data_dir, args.output_dir)