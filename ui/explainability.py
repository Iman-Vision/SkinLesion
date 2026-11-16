"""
DERMA.AI Explainability Module (Grad-CAM)
Calculates Class Activation Maps (Grad-CAM) using the target convolutional layer
and overlays heatmaps onto original dermoscopic lesions.
"""

from typing import Tuple, Optional, List, Dict, Any
import numpy as np
import tensorflow as tf
from PIL import Image

# Re-export LIME and feature extraction capabilities
from .xai_explainer import (
    generate_lime_explanation,
    extract_visual_features_and_explain,
    DERMOSCOPIC_CRITERIA
)



def generate_gradcam(model, image: Image.Image, img_size: int = None, target_class_idx: int = None) -> Tuple[np.ndarray, Image.Image]:
    """
    Computes Grad-CAM for the given PIL image and model.
    Dynamically respects model input shape (e.g. 300x300 vs 224x224).
    Returns: (raw_heatmap_2d, blended_pil_image)
    """
    # Detect target spatial dimension from model input if available
    if img_size is None or img_size == 224:
        if hasattr(model, 'input_shape') and model.input_shape and len(model.input_shape) >= 3:
            h = model.input_shape[1]
            if h is not None:
                img_size = h
    if img_size is None:
        img_size = 224

    # Find last 4D conv/activation layer
    last_conv_layer = None
    for layer in reversed(model.layers):
        out = getattr(layer, 'output', None)
        if out is not None and not isinstance(out, (list, tuple)) and len(getattr(out, 'shape', [])) == 4:
            if any(k in layer.name.lower() for k in ['conv', 'activation', 'block']):
                last_conv_layer = layer
                break
    if last_conv_layer is None:
        raise ValueError("No suitable 4D convolutional layer found in model for Grad-CAM.")

    # Target predictions layer
    try:
        pred_layer = model.get_layer('predictions')
    except (ValueError, KeyError):
        pred_layer = model.layers[-1]

    # Construct gradient model
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, pred_layer.output]
    )

    # Preprocess image
    resized_img = image.convert('RGB').resize((img_size, img_size))
    arr = np.expand_dims(np.array(resized_img, dtype=np.float32) / 255.0, axis=0)

    # Gradient Tape
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(arr)
        if target_class_idx is None:
            target_class_idx = int(tf.argmax(predictions[0]))
        loss = predictions[:, target_class_idx]

    # Compute gradients with respect to feature maps
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight feature map activations
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU and normalization
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    heatmap_np = heatmap.numpy()

    # Resize heatmap to match original image size using PIL bilinear interpolation
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap_np)).resize(image.size, Image.Resampling.BILINEAR)
    heatmap_scaled = np.array(heatmap_img, dtype=np.float32) / 255.0

    # Apply clinical heatmap colormap (Teal-to-Coral thermal map)
    # 0.0 -> Cool dark teal (0, 71, 68), 1.0 -> High attention bright coral/crimson (220, 60, 60)
    r = np.clip(heatmap_scaled * 2.2, 0, 1)
    g = np.clip(1.0 - np.abs(heatmap_scaled - 0.5) * 1.8, 0, 1)
    b = np.clip((1.0 - heatmap_scaled) * 1.5, 0, 1)

    color_heatmap = np.stack([r, g, b], axis=-1)
    color_heatmap_uint8 = np.uint8(255 * color_heatmap)
    color_img = Image.fromarray(color_heatmap_uint8)

    # Alpha blend: 65% original image + 35% heatmap
    blended = Image.blend(image.convert('RGB'), color_img, alpha=0.38)

    return heatmap_np, blended

