# Hand Sign Prediction

## Project Description
This project takes an image of a hand sign as input and predicts the corresponding American Sign Language (ASL) character. It utilizes a Convolutional Neural Network (CNN) trained on a dataset of ASL signs (digits 0-9 and letters a-z). The solution is deployed as an interactive web application using Streamlit, allowing users to upload images and receive real-time predictions with confidence scores.

---

## Project Details

### Problem Statement
Sign language is the primary mode of communication for the deaf and hard-of-hearing community. Automating the recognition of sign language characters can bridge communication gaps and serve as an educational tool for learning ASL.

### Data Preprocessing
- **Image Resizing:** All input images are resized to a fixed dimension of 64x64 pixels.
- **Normalization:** Pixel values are normalized to the range [0, 1] by dividing by 255.0.
- **Data Structure:** The dataset is organized into subdirectories for each class (0-9, a-z).

### Model Architecture
- **Type:** Convolutional Neural Network (CNN)
- **Framework:** TensorFlow / Keras
- **Layers:**
  - Convolutional Layers (Conv2D) for feature extraction.
  - Max Pooling Layers (MaxPooling2D) for down-sampling.
  - Flatten Layer to convert 2D feature maps to 1D vectors.
  - Dense Layers for classification (Output layer with Softmax activation).
- **Classes:** 36 classes (10 digits + 26 alphabets).

### Web Application
The Streamlit app provides:
- File uploader for images (JPG, JPEG, PNG).
- Display of the uploaded image.
- "Predict" button to trigger the inference.
- Output display showing the predicted class and prediction confidence.

---

## Tech Stack
- **Languages:** Python 3.x
- **Frameworks:** TensorFlow, Keras
- **Web Interface:** Streamlit
- **Libraries:** NumPy, Pillow (PIL)
- **Tools:** Jupyter Notebook

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/DCode-v05/Hand-Sign-Detection.git
cd "Hand Sign Language"
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
To launch the Streamlit app:
```bash
streamlit run app.py
```

To view the training process (Optional):
```bash
jupyter notebook Hand_Sign_Prediction_SP.ipynb
```

---

## Usage
1. Launch the app using `streamlit run app.py`.
2. Click on "Browse files" to upload an image of a hand sign.
3. Click the "Predict" button.
4. View the predicted character and the model's confidence.

---

## Project Structure
```
Hand Sign Language/
│
├── app.py                         # Streamlit web application
├── Hand_Sign_Prediction_SP.ipynb  # Model training and analysis notebook
├── asl_model.h5                   # Trained model file
├── requirements.txt               # Python dependencies
├── training_history.png           # Training accuracy/loss plot
├── asl_dataset/                   # Dataset directory (Images)
├── Output/                        # Generated outputs (if any)
└── README.md                      # Project documentation
```

---

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your feature"
   ```
4. Push to your branch:
   ```bash
   git push origin feature/your-feature
   ```
5. Open a pull request describing your changes.

---

## Contact
- **GitHub:** [DCode-v05](https://github.com/DCode-v05)
- **Email:** denistanb05@gmail.com
