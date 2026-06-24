# Hand Sign Detection

**Upload a photo of an American Sign Language hand sign and get back the character it spells — all 36 of them, with a confidence score.**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white) ![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

## Overview

This project recognizes American Sign Language (ASL) hand signs from a single image. It covers 36 classes — the digits 0–9 and the letters a–z — and returns the predicted character along with how confident the model is.

The whole thing is two parts. A Jupyter notebook trains a neural network on a folder of cropped hand-sign photos, and a small Streamlit app loads the saved model so anyone can drag in an image and see a prediction. I built it as a computer-vision warm-up — the goal was to take a labelled image dataset end to end: load it, preprocess it, train a classifier, plot how training went, save the model, and put a usable web front-end on top of it.

The trained network reaches about 96% accuracy on the training split and peaks around 90% validation accuracy across 10 epochs. It is a deliberately simple model (a fully-connected network, not a convolutional one), which keeps it small and fast but is also the main thing I'd change if I took it further.

## Key Features

- Classifies hand signs into 36 ASL classes: digits `0`–`9` and letters `a`–`z`.
- Streamlit web app with a file uploader (JPG / JPEG / PNG), an image preview, and a one-click Predict button.
- Returns both the predicted character and a confidence percentage from the softmax output.
- Self-contained training notebook that goes from raw image folders to a saved model, with each stage labelled (loading, visualisation, preprocessing, model build, training, plots, saving, inference).
- Automatic class discovery — the app reads the dataset folder names and sorts them, so the label order always matches what the model was trained on.
- Model and class loading are cached in Streamlit (`@st.cache_resource` / `@st.cache_data`) so the `.h5` file is only read once per session.
- Training/validation accuracy and loss curves saved to `training_history.png` for a quick read on how the run went.
- Sample prediction outputs included under `Output/` so you can see what the app produces without running it.

## How It Works

The pipeline is straightforward: a labelled image dataset goes through preprocessing into a Keras data generator, a small dense network is trained on it, the model is saved to disk, and the Streamlit app loads that model to score new uploads.

### Dataset

The data lives in `asl_dataset/`, organised as one subfolder per class. There are 36 folders (`0`–`9`, `a`–`z`), each holding cropped JPEG photos of a hand making that sign — roughly 70 images per class, 2,515 images in total. Because the folder name is the label, no separate annotation file is needed.

### Preprocessing

Preprocessing is handled by Keras' `ImageDataGenerator`:

- `rescale=1./255` normalises pixel values from the 0–255 range down to 0–1.
- `target_size=(64, 64)` resizes every image to 64×64 pixels.
- `class_mode='categorical'` produces one-hot labels for the 36 classes.
- `validation_split=0.2` carves the data into roughly 80/20 — about 2,013 images for training and 503 for validation — loaded with `flow_from_directory` at a batch size of 32.

### Model

The classifier is a fully-connected (dense) neural network built with the Keras Sequential API. The 64×64×3 image is flattened into a 12,288-value vector and pushed through three hidden layers and a softmax output:

```
Input(64, 64, 3)
  → Flatten            (12,288)
  → Dense(512, relu)
  → Dense(256, relu)
  → Dense(128, relu)
  → Dense(36, softmax)
```

That comes to about 6.46 million trainable parameters (~24.6 MB). It is compiled with the Adam optimiser and `categorical_crossentropy` loss, tracking accuracy, and trained for 10 epochs.

### Training and saving

Training runs over the generator with the validation split monitored each epoch. Accuracy and loss for both train and validation are plotted side by side with matplotlib and exported as `training_history.png`. The trained model is then written to `asl_model.h5` in Keras' HDF5 format (the saved file is around 74 MB).

### Inference (the app)

`app.py` is the Streamlit front-end. On startup it loads `asl_model.h5` and reads the sorted class list from the dataset folder, both cached so they only load once. When you upload an image and hit Predict, it:

1. Resizes the image to 64×64,
2. Converts it to an array and divides by 255 to normalise,
3. Adds a batch dimension and runs `model.predict`,
4. Takes the `argmax` of the output as the predicted class and turns the matching softmax value into a confidence percentage,
5. Shows the predicted character (uppercased) and the confidence.

## Results / Highlights

- 36-class classifier over digits 0–9 and letters a–z.
- ~2,515 images, split 2,013 train / 503 validation (80/20).
- Final epoch: ~95.7% training accuracy, ~87.7% validation accuracy; validation accuracy peaks near 90% (epoch 7).
- Model: dense network, ~6.46M parameters, 64×64×3 input, trained 10 epochs with Adam.
- A spot-check inference on a held-out `d` image predicts `d` correctly.

## Tech Stack

- **Language:** Python 3
- **Deep learning:** TensorFlow / Keras (`Sequential`, `Dense`, `ImageDataGenerator`)
- **Web app:** Streamlit
- **Data / imaging:** NumPy, Pillow (PIL)
- **Notebook / plots:** Jupyter, matplotlib

## Getting Started

### Prerequisites

- Python 3.x
- The trained model `asl_model.h5` and the `asl_dataset/` folder in the project root (both are in the repo; the app reads class names from the dataset folder).

### Installation

```bash
git clone https://github.com/DCode-v05/Hand-Sign-Detection.git
cd Hand-Sign-Detection
pip install -r requirements.txt
```

`requirements.txt` pulls in `streamlit`, `tensorflow`, `numpy`, and `pillow`.

### Running

Launch the web app:

```bash
streamlit run app.py
```

To retrain or step through the training process yourself:

```bash
jupyter notebook Hand_Sign_Prediction_SP.ipynb
```

Note: the notebook paths point at a Google Drive location (it was written in Colab), so update the `asl_dataset` path to your local copy before running it.

## Usage

1. Run `streamlit run app.py` and open the URL it prints.
2. Use the uploader to select a hand-sign image (`.jpg`, `.jpeg`, or `.png`).
3. The image previews at 300px wide; click **Predict**.
4. You get the predicted ASL character and the model's confidence, e.g. `Predicted Class: D` with `Confidence: 97.42%`.

The model expects a reasonably tight crop of a hand on a plain background, similar to the training images — that is what it was trained on.

## Project Structure

```
Hand-Sign-Detection/
├── app.py                         # Streamlit app: upload an image, get a prediction + confidence
├── Hand_Sign_Prediction_SP.ipynb  # Training notebook: load → preprocess → train → plot → save
├── asl_model.h5                   # Trained Keras model (~74 MB, HDF5)
├── requirements.txt               # streamlit, tensorflow, numpy, pillow
├── training_history.png           # Train/validation accuracy & loss curves
├── asl_dataset/                   # 36 class folders (0-9, a-z) of cropped hand images
│   ├── 0/ … 9/
│   └── a/ … z/
├── Output/                        # Sample prediction screenshots
└── README.md
```

---

## Contact

<table>
  <tr><td><b>Portfolio:</b> <a href="https://www.denistan.me">Denistan</a></td><td><b>LinkedIn:</b> <a href="https://www.linkedin.com/in/denistanb">denistanb</a></td></tr>
  <tr><td><b>GitHub:</b> <a href="https://github.com/DCode-v05">DCode-v05</a></td><td><b>LeetCode:</b> <a href="https://leetcode.com/u/Denistan_B">Denistan_B</a></td></tr>
  <tr><td colspan="2" align="center"><b>Email:</b> <a href="mailto:denistanb05@gmail.com">denistanb05@gmail.com</a></td></tr>
</table>

Made with ❤️ by **Denistan B**
