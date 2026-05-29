
from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageFilter

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "models" / "balanced_alzheimer_cnn.keras"
MAPPING_PATH = PROJECT_DIR / "models" / "class_mapping.json"
IMG_SIZE = (160, 160)
GLCM_LEVELS = 16
MAX_UPLOAD_SIZE = 10 * 1024 * 1024

app = FastAPI(title="Alzheimer MRI Classification API", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8080",
        "http://localhost:8080",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

model = None
index_to_class: dict[str, str] = {}
last_conv_layer_name = None


def load_artifacts() -> None:
    global model, index_to_class, last_conv_layer_name
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not MAPPING_PATH.exists():
        raise FileNotFoundError(f"Class mapping not found: {MAPPING_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)
    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    index_to_class = {str(k): v for k, v in mapping["index_to_class"].items()}

    conv_layers = [layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    if not conv_layers:
        for layer in model.layers:
            if isinstance(layer, tf.keras.Sequential):
                conv_layers.extend(
                    nested.name for nested in layer.layers if isinstance(nested, tf.keras.layers.Conv2D)
                )
    last_conv_layer_name = conv_layers[-1] if conv_layers else None


@app.on_event("startup")
def startup_event() -> None:
    load_artifacts()


def crop_black_background(img: Image.Image) -> Image.Image:
    arr = np.asarray(img, dtype=np.uint8)
    mask = arr > 8
    if not mask.any():
        return img
    ys, xs = np.where(mask)
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    pad_y = max(2, int(0.04 * (y2 - y1 + 1)))
    pad_x = max(2, int(0.04 * (x2 - x1 + 1)))
    y1 = max(0, y1 - pad_y)
    y2 = min(arr.shape[0] - 1, y2 + pad_y)
    x1 = max(0, x1 - pad_x)
    x2 = min(arr.shape[1] - 1, x2 + pad_x)
    return img.crop((x1, y1, x2 + 1, y2 + 1))


def preprocess_pil(image: Image.Image) -> np.ndarray:
    img = image.convert("L")
    img = crop_black_background(img)
    img = img.filter(ImageFilter.MedianFilter(size=3))
    img = img.resize(IMG_SIZE, Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=-1)
    return arr


def image_to_data_url(arr: np.ndarray) -> str:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def make_gradcam(input_batch: np.ndarray, class_index: int) -> np.ndarray | None:
    if last_conv_layer_name is None:
        return None

    try:
        conv_layer = model.get_layer(last_conv_layer_name)
    except ValueError:
        return None

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[conv_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_batch)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        return None

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)
    if max_value == 0:
        return None
    heatmap = heatmap / max_value
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap.numpy())).resize(IMG_SIZE, Image.Resampling.BILINEAR)
    return np.asarray(heatmap_img, dtype=np.float32) / 255.0


def overlay_heatmap(gray_image: np.ndarray, heatmap: np.ndarray | None) -> str | None:
    if heatmap is None:
        return None
    gray = np.squeeze(gray_image)
    gray_rgb = np.stack([gray, gray, gray], axis=-1)
    heat = np.zeros((*heatmap.shape, 3), dtype=np.float32)
    heat[..., 0] = heatmap
    heat[..., 1] = np.clip(1.0 - np.abs(heatmap - 0.5) * 2.0, 0, 1) * 0.45
    heat[..., 2] = 1.0 - heatmap
    overlay = 0.55 * gray_rgb + 0.45 * heat
    return image_to_data_url(np.uint8(np.clip(overlay, 0, 1) * 255))


def compute_glcm_features(gray: np.ndarray) -> dict:
    quantized = np.clip((gray * (GLCM_LEVELS - 1)).astype(np.int32), 0, GLCM_LEVELS - 1)
    directions = {
        "0 deg": (0, 1),
        "45 deg": (-1, 1),
        "90 deg": (-1, 0),
        "135 deg": (-1, -1),
    }
    feature_rows = []
    i_idx = np.arange(GLCM_LEVELS).reshape(-1, 1)
    j_idx = np.arange(GLCM_LEVELS).reshape(1, -1)

    for name, (dy, dx) in directions.items():
        y_start = max(0, -dy)
        y_end = quantized.shape[0] - max(0, dy)
        x_start = max(0, -dx)
        x_end = quantized.shape[1] - max(0, dx)
        a = quantized[y_start:y_end, x_start:x_end]
        b = quantized[y_start + dy:y_end + dy, x_start + dx:x_end + dx]

        glcm = np.zeros((GLCM_LEVELS, GLCM_LEVELS), dtype=np.float64)
        np.add.at(glcm, (a.ravel(), b.ravel()), 1)
        glcm = glcm + glcm.T
        glcm = glcm / max(glcm.sum(), 1.0)

        contrast = np.sum(((i_idx - j_idx) ** 2) * glcm)
        homogeneity = np.sum(glcm / (1.0 + np.abs(i_idx - j_idx)))
        energy = np.sqrt(np.sum(glcm**2))
        entropy = -np.sum(glcm * np.log2(glcm + 1e-12))
        mean_i = np.sum(i_idx * glcm)
        mean_j = np.sum(j_idx * glcm)
        std_i = math.sqrt(np.sum(((i_idx - mean_i) ** 2) * glcm))
        std_j = math.sqrt(np.sum(((j_idx - mean_j) ** 2) * glcm))
        correlation = np.sum((i_idx - mean_i) * (j_idx - mean_j) * glcm) / (std_i * std_j + 1e-12)

        feature_rows.append(
            {
                "direction": name,
                "contrast": float(contrast),
                "homogeneity": float(homogeneity),
                "energy": float(energy),
                "entropy": float(entropy),
                "correlation": float(correlation),
            }
        )

    summary = {
        key: float(np.mean([row[key] for row in feature_rows]))
        for key in ["contrast", "homogeneity", "energy", "entropy", "correlation"]
    }
    return {"summary": summary, "by_direction": feature_rows}


def compute_lbp_features(gray: np.ndarray) -> dict:
    img = np.asarray(gray, dtype=np.float32)
    center = img[1:-1, 1:-1]
    neighbors = [
        img[:-2, :-2], img[:-2, 1:-1], img[:-2, 2:],
        img[1:-1, 2:], img[2:, 2:], img[2:, 1:-1],
        img[2:, :-2], img[1:-1, :-2],
    ]
    codes = np.zeros(center.shape, dtype=np.uint8)
    for bit, neigh in enumerate(neighbors):
        codes |= ((neigh >= center).astype(np.uint8) << bit)
    hist, _ = np.histogram(codes, bins=16, range=(0, 256), density=True)
    return {
        "histogram_16_bins": [float(v) for v in hist],
        "mean_code": float(codes.mean()),
        "std_code": float(codes.std()),
        "texture_uniformity": float(np.sum(hist**2)),
    }


def compute_quality_and_roi(gray: np.ndarray) -> dict:
    gy, gx = np.gradient(gray)
    focus_score = float(np.var(gx) + np.var(gy))
    black_ratio = float(np.mean(gray < 0.08))
    brain_ratio = float(np.mean(gray > 0.08))
    center = gray[gray.shape[0] // 4: 3 * gray.shape[0] // 4, gray.shape[1] // 4: 3 * gray.shape[1] // 4]
    central_dark_ratio = float(np.mean(center < 0.12))
    mean_intensity = float(gray.mean())
    std_intensity = float(gray.std())

    if focus_score < 0.0005:
        quality = "Faible"
    elif focus_score < 0.0015:
        quality = "Acceptable"
    else:
        quality = "Bonne"

    return {
        "quality_label": quality,
        "focus_score": focus_score,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "black_pixel_ratio": black_ratio,
        "brain_occupancy_ratio": brain_ratio,
        "central_dark_ratio": central_dark_ratio,
    }


def class_severity_score(class_name: str) -> int:
    return {
        "NonDemented": 0,
        "VeryMildDemented": 1,
        "MildDemented": 2,
        "ModerateDemented": 3,
    }.get(class_name, 0)


def decision_support(class_name: str, confidence: float, quality_label: str) -> dict:
    severity = class_severity_score(class_name)
    if confidence >= 0.85 and quality_label != "Faible":
        reliability = "Elevee"
    elif confidence >= 0.65:
        reliability = "Moyenne"
    else:
        reliability = "Faible"

    if severity == 0:
        priority = "Routine"
    elif severity == 1:
        priority = "Controle clinique"
    elif severity == 2:
        priority = "Avis specialise recommande"
    else:
        priority = "Prioritaire"

    return {"severity_score": severity, "reliability": reliability, "priority": priority}


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "classes": [index_to_class[str(i)] for i in range(len(index_to_class))],
        "last_conv_layer": last_conv_layer_name,
        "features": ["CNN", "Grad-CAM", "GLCM", "LBP", "image quality"],
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file")

    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="Image file is too large")

    try:
        pil_img = Image.open(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image") from exc

    arr = preprocess_pil(pil_img)
    gray = np.squeeze(arr)
    batch = np.expand_dims(arr, axis=0)
    preds = model.predict(batch, verbose=0)[0]
    pred_idx = int(np.argmax(preds))
    pred_class = index_to_class[str(pred_idx)]
    confidence = float(preds[pred_idx])

    heatmap = make_gradcam(batch, pred_idx)
    gradcam_url = overlay_heatmap(arr, heatmap)
    processed_url = image_to_data_url(gray * 255)

    probabilities = [
        {"class_name": index_to_class[str(i)], "probability": float(preds[i])}
        for i in range(len(preds))
    ]
    probabilities.sort(key=lambda item: item["probability"], reverse=True)

    quality = compute_quality_and_roi(gray)
    decision = decision_support(pred_class, confidence, quality["quality_label"])
    texture = {
        "glcm": compute_glcm_features(gray),
        "lbp": compute_lbp_features(gray),
    }

    return JSONResponse(
        {
            "predicted_class": pred_class,
            "confidence": confidence,
            "probabilities": probabilities,
            "decision_support": decision,
            "quality": quality,
            "texture": texture,
            "processed_image": processed_url,
            "gradcam_image": gradcam_url,
        }
    )
