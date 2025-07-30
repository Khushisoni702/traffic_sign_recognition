import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model('traffic_sign_model.keras')

# Class indices and names
class_indices = {'0': 0, '1': 1, '10': 2, '11': 3, '12': 4, '13': 5, '14': 6, '15': 7, '16': 8, '17': 9, '18': 10,
                 '19': 11, '2': 12, '20': 13, '21': 14, '22': 15, '23': 16, '24': 17, '25': 18, '26': 19, '27': 20,
                 '28': 21, '29': 22, '3': 23, '30': 24, '31': 25, '32': 26, '33': 27, '34': 28, '35': 29, '36': 30,
                 '37': 31, '38': 32, '39': 33, '4': 34, '40': 35, '41': 36, '42': 37, '5': 38, '6': 39, '7': 40,
                 '8': 41, '9': 42}

index_to_label = {v: k for k, v in class_indices.items()}
class_names = {
    '0': 'Speed limit (20km/h)', '1': 'Speed limit (30km/h)', '2': 'Speed limit (50km/h)', '3': 'Speed limit (60km/h)',
    '4': 'Speed limit (70km/h)', '5': 'Speed limit (80km/h)', '6': 'End of speed limit (80km/h)',
    '7': 'Speed limit (100km/h)', '8': 'Speed limit (120km/h)', '9': 'No passing',
    '10': 'No passing > 3.5 tons', '11': 'Right-of-way', '12': 'Priority road', '13': 'Yield', '14': 'Stop',
    '15': 'No vehicles', '16': 'No > 3.5 tons', '17': 'No entry', '18': 'General caution',
    '19': 'Curve left', '20': 'Curve right', '21': 'Double curve', '22': 'Bumpy road', '23': 'Slippery',
    '24': 'Road narrows', '25': 'Road work', '26': 'Traffic signals', '27': 'Pedestrian', '28': 'Children',
    '29': 'Bicycles', '30': 'Ice/snow', '31': 'Wild animals', '32': 'End limits', '33': 'Turn right',
    '34': 'Turn left', '35': 'Ahead only', '36': 'Straight or right', '37': 'Straight or left',
    '38': 'Keep right', '39': 'Keep left', '40': 'Roundabout', '41': 'End no passing',
    '42': 'End no passing > 3.5 tons'
}

def preprocess(frame):
    img = cv2.resize(frame, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    roi_size = 200
    x1 = width // 2 - roi_size // 2
    y1 = height // 2 - roi_size // 2
    x2 = width // 2 + roi_size // 2
    y2 = height // 2 + roi_size // 2

    # Draw ROI box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    roi = frame[y1:y2, x1:x2]

    processed = preprocess(roi)
    prediction = model.predict(processed)
    pred_index = np.argmax(prediction)
    confidence = prediction[0][pred_index] * 100

    class_id = index_to_label[pred_index]
    class_name = class_names.get(class_id, f'Class {class_id}')
    label = f"{class_name} ({confidence:.2f}%)"

    cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow("Live Traffic Sign Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()