import os
import time
import numpy as np
import cv2
import math
from dnndk import n2cube

KERNEL_CONV = 'tf_yolov3_detect' 

VAL_IMAGE_DIR = 'val_images/'      
VAL_LABEL_DIR = 'val_labels/'    
NPY_DATA_DIR  = 'preprocessed_data/val/' 
CLASSES_PATH  = './model_data/classes_detect.txt'
ANCHORS_PATH  = './model_data/yolo_anchors_detect.txt'

CONF_THRES = 0.001 
NMS_THRES  = 0.45
IOU_THRES  = 0.5    

CONV_INPUT_NODE = "model_2_conv2d_Conv2D"
CONV_OUTPUT_NODE1 = "model_2_conv2d_58_Conv2D"
CONV_OUTPUT_NODE2 = "model_2_conv2d_66_Conv2D"
CONV_OUTPUT_NODE3 = "model_2_conv2d_74_Conv2D"

def compute_iou(b1, b2):
    # b1, b2 format: [x1, y1, x2, y2]
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (area1 + area2 - inter + 1e-6)

def load_classes(path):
    with open(path) as f:
        class_names = f.readlines()
    return [c.strip() for c in class_names]

def load_anchors(path):
    with open(path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

def load_gt_label(path, img_w, img_h):
    if not os.path.exists(path): return []
    gts = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5: continue
            cid = int(parts[0])
            # YOLO format: x_center, y_center, w, h (normalized)
            xc, yc, nw, nh = map(float, parts[1:5])
            
            w = nw * img_w
            h = nh * img_h
            x = xc * img_w
            y = yc * img_h
            
            x1 = x - w/2
            y1 = y - h/2
            x2 = x + w/2
            y2 = y + h/2
            
            # gts format: [x1, y1, x2, y2, class_id]
            gts.append([x1, y1, x2, y2, cid])
    return gts

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def _get_feats(feats, anchors, num_classes, input_shape):
    num_anchors = len(anchors)
    anchors_tensor = np.reshape(np.array(anchors, dtype=np.float32), [1, 1, 1, num_anchors, 2])
    grid_size = np.shape(feats)[1:3]
    nu = num_classes + 5
    
    predictions = np.reshape(feats, [-1, grid_size[0], grid_size[1], num_anchors, nu])
    
    mask = predictions[..., 4] > -7.0 
    
    if not np.any(mask):
        return None, None, None, None

    masked_predictions = predictions[mask]
    
    grid_y = np.tile(np.reshape(np.arange(grid_size[0]), [-1, 1, 1, 1]), [1, grid_size[1], 1, 1])
    grid_x = np.tile(np.reshape(np.arange(grid_size[1]), [1, -1, 1, 1]), [grid_size[0], 1, 1, 1])
    grid = np.concatenate([grid_x, grid_y], axis = -1)
    grid = np.array(grid, dtype=np.float32)
    
    grid = np.expand_dims(grid, 0)
    grid = np.tile(grid, [1, 1, 1, num_anchors, 1])
    masked_grid = grid[mask]
    
    anchors_broadcast = np.tile(anchors_tensor, [1, grid_size[0], grid_size[1], 1, 1])
    masked_anchors = anchors_broadcast[mask]

    box_xy = (sigmoid(masked_predictions[:, :2]) + masked_grid) / np.array(grid_size[::-1], dtype=np.float32)
    box_wh = np.exp(masked_predictions[:, 2:4]) * masked_anchors / np.array(input_shape[::-1], dtype=np.float32)
    box_confidence = sigmoid(masked_predictions[:, 4:5])
    box_class_probs = sigmoid(masked_predictions[:, 5:])

    return box_xy, box_wh, box_confidence, box_class_probs

def correct_boxes(box_xy, box_wh, input_shape, image_shape):
    box_yx = box_xy[:, ::-1]
    box_hw = box_wh[:, ::-1]
    input_shape = np.array(input_shape, dtype = np.float32)
    image_shape = np.array(image_shape, dtype = np.float32)
    new_shape = np.around(image_shape * np.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = np.concatenate([
        box_mins[:, 0:1], box_mins[:, 1:2],
        box_maxes[:, 0:1], box_maxes[:, 1:2]
    ], axis = -1)
    boxes *= np.concatenate([image_shape, image_shape], axis = -1)
    return boxes

def eval_post_process(yolo_outputs, image_shape, class_names, anchors):
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    boxes = []
    box_scores = []
    
    input_shape = np.shape(yolo_outputs[0])[1 : 3]
    input_shape = np.array(input_shape)*32
    
    for i in range(len(yolo_outputs)):
        box_xy, box_wh, box_conf, box_probs = _get_feats(yolo_outputs[i], anchors[anchor_mask[i]], len(class_names), input_shape)
        if box_xy is not None:
            _boxes = correct_boxes(box_xy, box_wh, input_shape, image_shape)
            _box_scores = box_conf * box_probs
            boxes.append(_boxes)
            box_scores.append(_box_scores)
    
    if len(boxes) == 0:
        return []

    boxes = np.concatenate(boxes, axis=0)
    box_scores = np.concatenate(box_scores, axis=0)
    
    max_scores = np.max(box_scores, axis=1)
    max_classes = np.argmax(box_scores, axis=1)
    
    mask = max_scores >= CONF_THRES
    valid_boxes = boxes[mask]
    valid_scores = max_scores[mask]
    valid_classes = max_classes[mask]
    
    if len(valid_boxes) == 0: return []
    
    # NMS
    x_min = valid_boxes[:, 1]
    y_min = valid_boxes[:, 0]
    x_max = valid_boxes[:, 3]
    y_max = valid_boxes[:, 2]
    width = x_max - x_min
    height = y_max - y_min
    
    boxes_for_cv = []
    scores_for_cv = valid_scores.tolist()
    for i in range(len(valid_boxes)):
        boxes_for_cv.append([int(x_min[i]), int(y_min[i]), int(width[i]), int(height[i])])
        
    indices = cv2.dnn.NMSBoxes(boxes_for_cv, scores_for_cv, CONF_THRES, NMS_THRES)
    
    results = []
    if len(indices) > 0:
        indices = indices.flatten()
        selected_boxes = valid_boxes[indices]
        selected_scores = valid_scores[indices]
        selected_classes = valid_classes[indices]
        
        for i in range(len(indices)):
            ymin, xmin, ymax, xmax = selected_boxes[i]
            score = selected_scores[i]
            cid = selected_classes[i]
            results.append([xmin, ymin, xmax, ymax, score, cid])
            
    return results

def compute_map(all_dets, all_gts, class_names, iou_thr=0.5):
    print("\nStarting mAP calculation...")
    results_map = []
    
    for cid, cname in enumerate(class_names):
        dets, gts = [], []
        
        for i, ds in enumerate(all_dets):
            for d in ds:
                if d[5] == cid: 
                    dets.append((i, d[4], d[:4])) # (img_idx, score, box)
        
        for i, gs in enumerate(all_gts):
            for g in gs:
                if g[4] == cid:
                    gts.append((i, g[:4]))
        
        if len(gts) == 0: 
            results_map.append((cname, 0.0, 0.0, 0.0))
            continue
            
        dets.sort(key=lambda x: -x[1]) 
        
        tp = np.zeros(len(dets))
        fp = np.zeros(len(dets))
        matched = set() 
        
        for k, (imgid, conf, box) in enumerate(dets):
            best_iou = 0
            best_gt_idx = -1
            
            for j, (gid, gt_box) in enumerate(gts):
                if gid != imgid: continue 
                if (imgid, j) in matched: continue 
                
                iou = compute_iou(box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_iou >= iou_thr and best_gt_idx >= 0:
                tp[k] = 1
                matched.add((imgid, best_gt_idx))
            else:
                fp[k] = 1
                
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        
        rec = tp_cum / len(gts)
        pre = tp_cum / np.maximum(tp_cum + fp_cum, 1e-6)
        
        ap = 0
        for t in np.linspace(0, 1, 11):
            if np.any(rec >= t):
                ap += np.max(pre[rec >= t])
        ap /= 11
        
        results_map.append((cname, ap, pre[-1] if pre.size else 0, rec[-1] if rec.size else 0))
        
    return results_map

if __name__ == "__main__":

    n2cube.dpuOpen()
    kernel = n2cube.dpuLoadKernel(KERNEL_CONV)
    task = n2cube.dpuCreateTask(kernel, 0)
    
    class_names = load_classes(CLASSES_PATH)
    anchors = load_anchors(ANCHORS_PATH)
    NN_obj_info_dim = 3*(5+len(class_names))
    
    img_files = [f for f in os.listdir(VAL_IMAGE_DIR) if f.endswith('.jpg')]
    img_files.sort()
    
    all_dets = []
    all_gts = []
    
    print(f"Found {len(img_files)} validation images.")
    start_time = time.time()
    
    cnt = 0
    for img_file in img_files:
        cnt += 1
        basename = os.path.splitext(img_file)[0]
        
        npy_path = os.path.join(NPY_DATA_DIR, basename + '.npy')
        label_path = os.path.join(VAL_LABEL_DIR, basename + '.txt')
        img_path = os.path.join(VAL_IMAGE_DIR, img_file)
  
        img = cv2.imread(img_path)
        if img is None: continue
        img_h, img_w = img.shape[:2]
        
        gts = load_gt_label(label_path, img_w, img_h)
        all_gts.append(gts)
        
        if not os.path.exists(npy_path):
            all_dets.append([]) 
            continue
            
        image_data = np.load(npy_path)
        
        input_len = n2cube.dpuGetInputTensorSize(task, CONV_INPUT_NODE)
        n2cube.dpuSetInputTensorInHWCFP32(task, CONV_INPUT_NODE, image_data, input_len)
        n2cube.dpuRunTask(task)
        
        conv_out1 = n2cube.dpuGetOutputTensorInHWCFP32(task, CONV_OUTPUT_NODE1, n2cube.dpuGetOutputTensorSize(task, CONV_OUTPUT_NODE1))
        conv_out2 = n2cube.dpuGetOutputTensorInHWCFP32(task, CONV_OUTPUT_NODE2, n2cube.dpuGetOutputTensorSize(task, CONV_OUTPUT_NODE2))
        conv_out3 = n2cube.dpuGetOutputTensorInHWCFP32(task, CONV_OUTPUT_NODE3, n2cube.dpuGetOutputTensorSize(task, CONV_OUTPUT_NODE3))
        
        conv_out1 = np.reshape(conv_out1, (1, 19, 19, NN_obj_info_dim))
        conv_out2 = np.reshape(conv_out2, (1, 38, 38, NN_obj_info_dim))
        conv_out3 = np.reshape(conv_out3, (1, 76, 76, NN_obj_info_dim))
        
        yolo_outputs = [conv_out1, conv_out2, conv_out3]
        
        dets = eval_post_process(yolo_outputs, (img_h, img_w), class_names, anchors)
        all_dets.append(dets)
        

        if cnt % 100 == 0:
            print(f"\n[Progress] Processing {cnt}/{len(img_files)} ... calculating temp mAP")
            
            temp_results = compute_map(all_dets, all_gts, class_names, IOU_THRES)
            
            t_mAP = sum([r[1] for r in temp_results]) / len(temp_results) if len(temp_results) > 0 else 0
            
            print(f">>> mAP@0.5: {t_mAP:.4f}")
            
            print(f"{'Class':<10} {'AP':<8}")
            for r in temp_results:
                print(f"{r[0]:<10} {r[1]:<8.4f}")
            print("-" * 30)

    n2cube.dpuDestroyTask(task)
    
    print("\n" + "="*50)
    print("FINAL EVALUATION REPORT")
    print("-" * 50)
    final_results = compute_map(all_dets, all_gts, class_names, IOU_THRES)
    
    print(f"{'Class':<12} {'AP':<8} {'Precision':<10} {'Recall':<10}")
    print("-" * 50)
    
    mAP_sum = 0
    for r in final_results:
        print(f"{r[0]:<12} {r[1]:<8.4f} {r[2]:<10.4f} {r[3]:<10.4f}")
        mAP_sum += r[1]
        
    mAP = mAP_sum / len(final_results) if len(final_results) > 0 else 0
    print("-" * 50)
    print(f"Final mAP@0.5 = {mAP:.4f}")
    print("="*50)