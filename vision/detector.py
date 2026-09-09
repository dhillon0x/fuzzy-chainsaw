from ultralytics import YOLO

class FastDetector:
    def __init__(self, model_path="yolov8n.pt", conf_thresh=0.45):
        self.model = YOLO(model_path)
        self.conf = conf_thresh

    def detect(self, frame):
        results = self.model(frame, conf=self.conf, verbose=False)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            confidence = float(box.conf[0].item())
            cls_id = int(box.cls[0].item())
            detections.append({
                "class": self.model.names[cls_id],
                "confidence": confidence,
                "box": [x1, y1, x2, y2]
            })
        return detections
