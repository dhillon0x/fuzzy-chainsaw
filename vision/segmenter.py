import cv2
import numpy as np
from ultralytics import FastSAM

class FastSegmenter:
    def __init__(self, model_path="FastSAM-s.pt", conf=0.4):
        self.model = FastSAM(model_path)
        self.conf = conf

    def extract_mask_and_centroid(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        results = self.model(frame, bboxes=[[x1, y1, w, h]], conf=self.conf, verbose=False)

        if not results or results[0].masks is None:
            return None, (int((x1 + x2) / 2), int((y1 + y2) / 2))

        mask = results[0].masks.data[0].cpu().numpy().astype(np.uint8)
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        moments = cv2.moments(mask)

        if moments["m00"] > 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
        else:
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

        return mask, (cx, cy)
