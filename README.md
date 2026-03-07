# Sign Language to Text Translator

A Machine Learning based system that converts **hand sign language gestures into readable text** using computer vision and deep learning.

This project helps bridge the communication gap between **hearing-impaired individuals and non-sign language users**.

---

## Project Overview

The **Sign to Text Translator** captures hand gestures using a webcam and processes the image using computer vision techniques.
A trained deep learning model then predicts the corresponding sign language gesture and converts it into text.

---

## Features

* Real-time hand gesture recognition
* Sign language to text translation
* Image preprocessing for gesture detection
* Trained CNN deep learning models
* Dataset collection for training gestures
* Supports multiple sign gestures

---

## Technologies Used

* Python
* OpenCV
* TensorFlow / Keras
* NumPy
* Computer Vision
* Deep Learning

---

## Project Structure

```
sign-to-text-translator
│
├── app.py
├── collect-data.py
├── preprocessing.py
├── image_processing.py
├── convert_models.py
│
├── data/
│   └── train/
│
├── model/
│   ├── model-bw.h5
│   ├── model-bw.json
│
├── signs.png
└── README.md
```

---

## Installation

### 1. Clone the repository

```
git clone https://github.com/abhisheknj37/sign-to-text-translator.git
```

### 2. Navigate to the project directory

```
cd sign-to-text-translator
```

### 3. Install required libraries

```
pip install opencv-python
pip install tensorflow
pip install numpy
```

---

## How to Run the Project

Run the main application:

```
python app.py
```

The webcam will start and detect hand gestures in real time.

---

## Dataset

The dataset contains images of different hand gestures used for training the deep learning model.

Example gestures:

* Hello
* Good Morning
* Good Night
* Alphabet signs

---

## Project Workflow

1. Collect gesture images using webcam
2. Preprocess images
3. Train CNN model
4. Predict gesture from webcam input
5. Convert gesture to text output

---

## Future Improvements

* Convert text to speech
* Support full sentence recognition
* Mobile application integration
* Increase dataset size for better accuracy

---

## Author

**Abhishek N J**

Machine Learning Enthusiast
Computer Science Student

GitHub:
https://github.com/abhisheknj37

---

## License

This project is open-source and available for educational purposes.
