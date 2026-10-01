# Wound Classifier

A deep learning web application that classifies chronic wound images into six clinical categories. Upload a photo (or use your webcam) and the model returns a predicted wound type with a confidence score.

Built with a ResNet50 transfer-learning backbone, served through a Flask API with a browser front end.

## Classes

| Label | Wound type |
|-------|-----------|
| `Background` | No wound present / background image |
| `Diabetic` | Diabetic foot ulcer |
| `Nerves` | Neuropathic wound |
| `Pressure` | Pressure ulcer |
| `Surgical` | Surgical wound |
| `Venous` | Venous leg ulcer |

## Model

- **Backbone:** ResNet50 (265 convolutional layers with residual connections, batch normalization)
- **Head:** Flatten + dense layers, 6-way softmax output
- **Input:** 224 × 224 RGB, raw 0–255 pixel values (no rescaling — the normalization layer is inside the graph)
- **Format:** TensorFlow SavedModel, loaded via `keras.layers.TFSMLayer`

The model was trained in Google Colab; this repository contains the inference and serving code.

## Repository layout

```
deploy.py        Flask API — loads the model, exposes POST /predict
website.html     Front end: file upload
wrbsiteV2.html   Front end: live webcam capture
requirements.txt Pinned dependencies
model/           SavedModel directory (not in git — see Setup)
Train/           Dataset (not in git — see Dataset)
```

## Setup

Requires **Python 3.12**. TensorFlow 2.16 is pinned because `TFSMLayer` — the Keras 3 API used to load the SavedModel — was introduced in that release.

```bash
git clone git@github.com:AniruddhaK23/wound-classifier.git
cd wound-classifier
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Download the trained model from the [Releases](../../releases) page and extract it so the SavedModel sits at `model/` in the project root:

```
model/
├── saved_model.pb
├── keras_metadata.pb
└── variables/
```

To load the model from elsewhere, set `WOUND_MODEL_PATH` to its directory.

## Running

```bash
python3.12 deploy.py
```

The API starts on `http://127.0.0.1:8888`. Open `website.html` in a browser, choose an image, and click **Predict**. For the webcam version, open `wrbsiteV2.html` instead.

## API

**`POST /predict`** — multipart form with an `image` field.

```bash
curl -X POST -F "image=@wound.jpg" http://127.0.0.1:8888/predict
```

```json
{ "class": "Venous", "prediction": 0.9999 }
```

`prediction` is the softmax confidence for the returned class.

## Dataset

Trained on the **AZH Chronic Wound Database**, a clinical dataset of wound photographs from the AZH Wound and Vascular Center (Milwaukee, WI), used in published multi-modal wound classification research. The training set here comprised 695 images across the six classes, with accompanying body-location labels.

**The dataset is not redistributed in this repository.** These are clinical patient images subject to the original dataset's terms of use. Please obtain it from the original authors.

## Limitations

- **No held-out evaluation.** This repository does not include a train/test split or reported test-set metrics. Performance on unseen data is uncharacterized.
- **Class imbalance.** The training set ranged from 75 images (Background, Nerves) to 185 (Venous), which likely biases predictions toward the majority classes.
- **Narrow domain.** Images come from a single wound care center, so lighting, camera, and patient demographics are not diverse.

## Disclaimer

This is a student research project. It is **not a medical device** and must not be used for diagnosis, triage, or clinical decision-making. Consult a qualified healthcare professional for any wound requiring care.
