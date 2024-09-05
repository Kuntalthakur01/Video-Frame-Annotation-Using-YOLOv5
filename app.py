import cv2
import pandas as pd
import os
import torch


def load_model():
    # Load YOLOv5 model (pre-trained on COCO dataset)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model


def annotate_frame(frame, model):
    # Perform inference
    results = model(frame)

    # Render results on frame
    frame = results.render()[0]
    labels = results.pandas().xyxy[0]['name'].tolist()
    return frame, labels


def main():
    model = load_model()
    # Update this to your video path if different
    video_path = '/Users/kuntal/Desktop/proj-1/dog-glases.mp4'

    if os.path.exists(video_path):
        try:
            # Try opening the file to read a byte
            with open(video_path, 'rb') as f:
                f.read(1)
            print("File is readable")
            cap = cv2.VideoCapture(video_path)
        except IOError as e:
            print(f"Cannot read file: {e}")
            return
    else:
        print(f"File not found: {video_path}")
        return

    annotations = []
    frame_count = 0
    output_dir = 'annotated_images'

    # Create directory to save annotated images
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to RGB (YOLOv5 expects RGB images)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Annotate frame with YOLOv5
            annotated_frame, detected_labels = annotate_frame(frame_rgb, model)

            # Display the frame
            cv2.imshow('Frame', annotated_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                # Quit the loop if 'q' is pressed
                break

            # Save the annotated frame
            output_path = os.path.join(output_dir, f'frame_{frame_count}.jpg')
            cv2.imwrite(output_path, annotated_frame)

            # Append to CSV data
            annotations.append(
                {'frame': frame_count, 'labels': ', '.join(detected_labels)})
            frame_count += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()

        # Save annotations to a CSV file
        df = pd.DataFrame(annotations)
        df.to_csv('annotations.csv', index=False)
        print("Annotations saved to 'annotations.csv' and images saved to 'annotated_images' folder.")


if __name__ == "__main__":
    main()
