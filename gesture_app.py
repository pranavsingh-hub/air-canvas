import cv2
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


def main():
    print("Loading model and processor from Hugging Face...")
    # Load model and processor (downloads automatically on first run)
    model_name = "dima806/hand_gestures_image_detection"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)

    # Use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"Model loaded successfully on device: {device}")

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\nStarting webcam feed...")
    print("Press 'q' in the camera window to quit.\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip horizontally for natural mirror view
        frame = cv2.flip(frame, 1)

        # 1. Convert OpenCV BGR frame to PIL RGB Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        # 2. Preprocess frame into PyTorch tensors
        inputs = processor(images=pil_image, return_tensors="pt").to(device)

        # 3. Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Extract highest probability predicted class
            predicted_class_id = logits.argmax(-1).item()
            predicted_label = model.config.id2label[predicted_class_id]

        # 4. Display result on webcam feed
        cv2.rectangle(frame, (10, 10), (450, 60), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"Gesture: {predicted_label}",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Live Hand Gesture Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()