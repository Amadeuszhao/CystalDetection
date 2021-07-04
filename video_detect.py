import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
from mmdet.apis import init_detector, inference_detector
from mmdet.core.post_processing import multiclass_nms
import mmcv
import torch
import cv2
import time
import datetime

# Specify the path to model config and checkpoint file
config_file = '/home/zhaowei/seg/mmdetection/configs/mask_rcnn_r50_caffe_fpn_mstrain-poly_1xSF.py'
checkpoint_file = '/home/zhaowei/seg/mmdetection/work_dirs/mask_rcnn_r50_caffe_fpn_mstrain-poly_1xSF/latest.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')


# test a single image and show the results
# img = '/home/zhaowei/seg/first_datset/IMG_11.png'  # or img = mmcv.imread(img), which will only load it once
video = '/home/zhaowei/seg/test.mp4'
cap = cv2.VideoCapture(video)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    start = time.time()
    result = inference_detector(model, frame)

    bbox = result[0]
    seg = result[1]


    multi_scores = []
    multi_bboxes = []


    for c, i in enumerate(bbox):
        # for per class
        if len(i) == 0:
            continue
        # [n, 5]
        multi_scores = []
        multi_bboxes = []
        for b in i:
            multi_scores.append([b[-1], 0])
            multi_bboxes.append([b[0], b[1], b[2], b[3]])

        multi_scores = torch.Tensor(multi_scores)
        multi_bboxes = torch.Tensor(multi_bboxes)


        _, _, keep = multiclass_nms(multi_bboxes=multi_bboxes, multi_scores=multi_scores, score_thr=0.3,
                                     nms_cfg={"iou_threshold": 0.3, "class_agnostic": False}, return_inds=True)

        multi_scores = multi_scores[keep].numpy()
        multi_bboxes = multi_bboxes[keep].numpy()
        i = i[keep]


        seg_c = np.array(result[1][c])
        seg_c = seg_c[keep]
        if len(keep) == 1:
            seg_c = np.array([seg_c])
            i = np.array([i.tolist()])
        seg_c = [s for s in seg_c]
        result[1][c] = seg_c

        result[0][c] = i


    frame = model.show_result(frame, result, show=False)
    end = time.time()
    seconds = end - start
    fps = int(1 / seconds)
    print(fps)
    cv2.putText(frame, str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (230, 60, 50), 2)
    cv2.imshow('Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



