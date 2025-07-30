# 🚦Traffic Sign Recognition using Deep Learning

This project is a Deep Learning-based Traffic Sign Recognition system built using TensorFlow, Keras, OpenCV and Jupyter Notebook. It classifies traffic signs from images into their respective categories using a Convolutional Neural Network (CNN) trained on a traffic sign dataset.

## 🚀 Project Highlights

- 🧠 Built using Convolutional Neural Networks (CNNs)
- 📊 Trained and evaluated using TensorFlow and Scikit-learn
- 🖼 Real-time image preprocessing using OpenCV
- 📁 Dataset preprocessed into .npy files for faster training
- ✅ Achieved high accuracy on validation/test data
- 📦 Model saved in .keras format for future deployment

---

## 🧠 Model Summary

- *Architecture*: Convolutional Neural Network (CNN)
- *Input Shape*: 64x64x3
- *Output*: Softmax layer with 43 traffic sign classes
- *Frameworks Used*: TensorFlow, OpenCV, NumPy, Pandas

---

## 🖥 Live Camera Detection

To run live detection using webcam:
Run this in Terminal

python live_traffic_sign_detection.py

---

## 📦 Installation

1. Clone the repository:
git clone https://github.com/Khushisoni702/traffic_sign_recognition.git

2. Navigate to the folder:
cd traffic_sign_recognition

3. Create a virtual environment and install dependencies:
python -m venv venv
.\venv\Scripts\activate   
pip install -r requirements.txt

4. Open the Jupyter Notebook and by running:
jupyter notebook

5. Load notebooks/train_model.ipynb to explore the training steps or use the pre-trained model for inference.

---

## 📌 Note

Dataset is not uploaded to GitHub due to size limits.
You can download the dataset from: 
🔗 Dataset (GTSRB - German Traffic Sign Recognition Benchmark):
https://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---

## ⭐ Star the Repo

If you found this helpful, consider giving it a ⭐ on GitHub!



