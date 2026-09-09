import cv2
from vision.detector import FastDetector
from vision.segmenter import FastSegmenter
from src.kinematics import ArmKinematics
from src.controller import ArmController

def pixel_to_world_coords(cx, cy, frame_shape, z_plane=25.0):
    h, w, _ = frame_shape
    scale_mm_per_px = 0.5
    x_world = 200.0 + (cy - (h / 2)) * scale_mm_per_px
    y_world = (cx - (w / 2)) * scale_mm_per_px
    return x_world, y_world, z_plane

def main():
    detector = FastDetector(conf_thresh=0.5)
    segmenter = FastSegmenter(conf=0.4)
    kinematics = ArmKinematics()
    controller = ArmController(port="COM3")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERR] Camera not found.")
        return

    print("[RUN] Vision pipeline running. Press 'q' to stop.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for det in detector.detect(frame):
            bbox = det["box"]
            cls_name = det["class"]
            mask, (cx, cy) = segmenter.extract_mask_and_centroid(frame, bbox)
            target_xyz = pixel_to_world_coords(cx, cy, frame.shape)
            joint_angles = kinematics.inverse_kinematics(target_xyz)

            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"{cls_name}: {target_xyz[0]:.0f},{target_xyz[1]:.0f},{target_xyz[2]:.0f}mm", 
                        (bbox[0], bbox[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if joint_angles is not None:
                controller.send_joint_targets(joint_angles)

        cv2.imshow("6-Axis Vision Pipeline", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    controller.close()

if __name__ == "__main__":
    main()
