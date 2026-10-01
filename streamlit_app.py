"""Streamlit front end for the wound classifier.

Weights are pulled from the Hugging Face model repo at startup rather than
committed here, so the GitHub repo stays small.
"""

import keras
import numpy as np
import streamlit as st
from huggingface_hub import snapshot_download
from PIL import Image

HF_REPO = "AniruddhaK23/wound-classifier"

CLASS_NAMES = ["Background", "Diabetic", "Nerves", "Pressure", "Surgical", "Venous"]

CLASS_DESCRIPTIONS = {
    "Background": "No wound detected",
    "Diabetic": "Diabetic foot ulcer",
    "Nerves": "Neuropathic wound",
    "Pressure": "Pressure ulcer",
    "Surgical": "Surgical wound",
    "Venous": "Venous leg ulcer",
}

st.set_page_config(page_title="Wound Classifier", page_icon="🩹", layout="wide")


@st.cache_resource(show_spinner="Downloading model (first run takes a minute)...")
def load_model():
    local_dir = snapshot_download(repo_id=HF_REPO)
    return keras.layers.TFSMLayer(f"{local_dir}/model", call_endpoint="serving_default")


def classify(image: Image.Image) -> dict[str, float]:
    """Resize to 224x224 and return a confidence score per class."""
    # float32 is explicit: TensorFlow 2.21 no longer auto-casts uint8 input.
    # Values stay at 0-255 — rescaling happens inside the saved graph.
    resized = image.convert("RGB").resize((224, 224))
    batch = np.expand_dims(np.array(resized, dtype=np.float32), axis=0)

    # Take the output tensor by position so a retrain that renames the final
    # layer does not break this.
    outputs = load_model()(batch)
    probabilities = np.array(list(outputs.values())[0])[0]

    return dict(zip(CLASS_NAMES, (float(p) for p in probabilities)))


st.title("🩹 Wound Classifier")
st.markdown(
    "Classifies chronic wound photographs into six clinical categories using a "
    "**ResNet50** network fine-tuned on the AZH Chronic Wound Database."
)

left, right = st.columns(2)

with left:
    upload_tab, camera_tab = st.tabs(["Upload", "Camera"])
    with upload_tab:
        uploaded = st.file_uploader("Wound image", type=["jpg", "jpeg", "png"])
    with camera_tab:
        captured = st.camera_input("Take a photo")

    source = uploaded or captured
    if source:
        image = Image.open(source)
        st.image(image, caption="Input", use_container_width=True)

with right:
    if not source:
        st.info("Upload a photo or take one with your camera to get a prediction.")
    else:
        scores = classify(image)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_class, top_score = ranked[0]

        st.metric(
            label=CLASS_DESCRIPTIONS[top_class],
            value=top_class,
            delta=f"{top_score:.1%} confidence",
        )
        st.divider()
        st.caption("All classes")
        for name, score in ranked:
            # Clamp: float error can nudge a softmax score just past 1.0,
            # which st.progress rejects.
            st.progress(min(max(score, 0.0), 1.0), text=f"{name} — {score:.1%}")

st.divider()
st.warning(
    "**Not a medical device.** This is a student research project, not validated "
    "for clinical use. It must not inform diagnosis, triage, or treatment. "
    "Trained on 695 images from a single wound care center with no held-out test "
    "set, so accuracy on unseen images is uncharacterized."
)
st.caption(f"[Source code](https://github.com/AniruddhaK23/wound-classifier) · [Model]({'https://huggingface.co/' + HF_REPO})")
