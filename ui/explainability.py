"""
DERMA.AI Explainability Module (Grad-CAM)
Calculates Class Activation Maps (Grad-CAM) using the target convolutional layer
and overlays heatmaps onto original dermoscopic lesions.
"""

from typing import Tuple
import numpy as np
import tensorflow as tf
from PIL import Image


def generate_gradcam(model, image: Image.Image, img_size: int = 224, target_class_idx: int = None) -> Tuple[np.ndarray, Image.Image]:
    """
    Computes Grad-CAM for the given PIL image and model.
    Returns: (raw_heatmap_2d, blended_pil_image)
    """
    # Find last 4D conv layer
    last_conv_layer_name = 'Conv_1'
    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
    except ValueError:
        # Fallback: scan backwards for Conv2D or DepthwiseConv2D layer
        last_conv_layer = None
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower() and len(layer.output.shape) == 4:
                last_conv_layer = layer
                break
        if last_conv_layer is None:
            raise ValueError("No suitable 4D convolutional layer found in model for Grad-CAM.")

    # Target predictions layer
    pred_layer = model.get_layer('predictions')

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

