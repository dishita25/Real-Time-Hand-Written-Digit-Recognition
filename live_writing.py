import cv2
import numpy as np
import torch
from model import MNIST_CNN
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# Load trained model
def load_model(model_path="mnist_cnn.pth"):
    model = MNIST_CNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_digit(digit_img):
    if len(digit_img.shape) == 3:
        digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)

    digit_img = cv2.bitwise_not(digit_img)  # Invert colors

    _, digit_img = cv2.threshold(digit_img, 127, 255, cv2.THRESH_BINARY)

    # Find contours to crop digit tightly
    contours, _ = cv2.findContours(digit_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        digit_img = digit_img[y:y+h, x:x+w]

    # Ensure the digit is well-centered and aspect ratio is maintained
    h, w = digit_img.shape
    aspect_ratio = h / w

    # Ensure max height and width do not exceed 28
    if h > w:
        new_h = 20  # Keep it slightly smaller to avoid boundary issues
        new_w = int((new_h / h) * w)
    else:
        new_w = 20
        new_h = int((new_w / w) * h)

    # Resize while keeping proportions
    digit_img = cv2.resize(digit_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create a centered 28x28 image
    final_img = np.zeros((28, 28), dtype=np.uint8)

    # Compute offset to center the digit
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    # Ensure sizes match
    final_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = digit_img

    final_img = final_img.astype(np.float32) / 255.0
    final_img = Image.fromarray((final_img * 255).astype(np.uint8))

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    return transform(final_img).unsqueeze(0)


# Predict digit using CNN
def predict_digit(model, img):
    with torch.no_grad():
        output = model(img)
        print("Raw model output:", output.numpy())  # Print confidence scores
        # print("Predicted digit:", torch.argmax(output, 1).item())
        predicted = torch.argmax(output, 1).item()
    return predicted

# Load HSV values
hsv_value = np.load('Real-Time-Hand-Written-Digit-Recognition\hsv_value.npy')

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
kernel = np.ones((5, 5), np.uint8)
canvas = None
x1, y1 = 0, 0
noise_thresh = 800
model = load_model()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_range, upper_range = hsv_value
    mask = cv2.inRange(hsv, lower_range, upper_range)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours and cv2.contourArea(max(contours, key=cv2.contourArea)) > noise_thresh:
        c = max(contours, key=cv2.contourArea)
        x2, y2, w, h = cv2.boundingRect(c)

        if x1 == 0 and y1 == 0:
            x1, y1 = x2, y2
        else:
            cv2.line(canvas, (x1, y1), (x2, y2), (255, 255, 255), 4)
        x1, y1 = x2, y2
    else:
        x1, y1 = 0, 0

    frame = cv2.add(canvas, frame)
    stacked = np.hstack((canvas, frame))
    cv2.imshow('Screen_Pen', cv2.resize(stacked, None, fx=0.6, fy=0.6))

    key = cv2.waitKey(1)
    if key == 13:  # Enter key to break
        break
    elif key == ord('c'):  # Clear canvas
        canvas = np.zeros_like(frame)
    elif key == ord('s'):  # Save & Predict
        gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_canvas, 127, 255, cv2.THRESH_BINARY)
        digit_img = cv2.bitwise_not(thresh)
        digit_img = preprocess_digit(digit_img)
        
        # Show the preprocessed digit
        plt.imshow(digit_img.squeeze(), cmap="gray")
        plt.title("Preprocessed Image")
        plt.show()
        
        prediction = predict_digit(model, digit_img)
        print(f"Predicted Digit: {prediction}")
        canvas = np.zeros_like(frame)  # Reset canvas after prediction

cap.release()
cv2.destroyAllWindows()