import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import os
import torch
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Initialize YOLOv11-Nano and DeepSORT
model = YOLO('yolo11n.pt')  # Official YOLOv11 Nano model
tracker = DeepSort(
    max_age=50,
    embedder='mobilenet',  # Better for resource-constrained systems
    max_cosine_distance=0.4  # Tighter similarity threshold
)

def main(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    # Video writer setup
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter('output.mp4', 
                         cv2.VideoWriter_fourcc(*'mp4v'), 
                         fps, 
                         (frame_width, frame_height))

    # Warmup model
    model.predict(source=torch.zeros(1,3,640,640), verbose=False)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOv11 inference (optimized for Nano)
        results = model.track(
            frame,
            persist=True,  # Better for video tracking
            imgsz=640,
            conf=0.4,  # Optimal for Nano's precision
            verbose=False
        )

        # Process detections
        detections = []
        if results[0].boxes.id is not None:
            for box, cls, track_id in zip(results[0].boxes.xyxy, 
                                         results[0].boxes.cls,
                                         results[0].boxes.id):
                x1, y1, x2, y2 = map(int, box)
                cls_name = model.names[int(cls)]
                
                # Filter for people only
                if cls_name == "person":
                    detections.append(([x1, y1, x2-x1, y2-y1], float(0.9), cls_name))

        # DeepSORT update
        tracks = tracker.update_tracks(detections, frame=frame)

        # Visualization
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            
            # Draw bounding box and ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Write output
        out.write(frame)
        cv2.imshow('YOLOv11 Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_file = "xyz.mp4"
    if os.path.exists(video_file):
        main(os.path.abspath(video_file))
    else:
        print(f"Error: Video file not found at {os.path.abspath(video_file)}")

