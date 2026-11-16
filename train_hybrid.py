"""
DERMA.AI — State-of-the-Art Hybrid CNN-Transformer Training Pipeline
======================================================================
Architecture : DenseNet121 + Multi-Head Self-Attention (MHSA) Transformer Block
Input        : 224x224 RGB, [0, 1] normalized (ImageNet normalized internally)
Split        : StratifiedGroupKFold on lesion_id (100% patient/lesion leak-free)
Loss         : Sparse Categorical Focal Loss (gamma=2.0, label_smoothing=0.05)
Optimization : 2-Phase Fine-Tuning (AdamW, Cosine LR Decay)
Explainability: Compatible with DenseNet conv5 / relu layer for Grad-CAM
"""

import os
import sys
import json
import random
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    balanced_accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
)
from sklearn.preprocessing import label_binarize

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training_hybrid.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger("derma_hybrid")

SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "ham10000"
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR = PROJECT_ROOT / ".tf_cache_hybrid"

MODELS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

CLASS_ORDER = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
FULL_CLASS_NAMES = {
    'akiec': 'Actinic keratoses',
    'bcc': 'Basal cell carcinoma',
    'bkl': 'Benign keratosis',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic nevi',
    'vasc': 'Vascular lesions'
}
LABEL2IDX = {c: i for i, c in enumerate(CLASS_ORDER)}


def set_seeds():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def load_dataframe():
    csv_path = DATA_DIR / "HAM10000_metadata.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"HAM10000_metadata.csv not found at {csv_path}")
    df = pd.read_csv(csv_path)

    image_path_map = {}
    for d in [DATA_DIR / "HAM10000_images_part_1", DATA_DIR / "HAM10000_images_part_2"]:
        if not d.exists():
            log.warning(f"Image directory not found: {d}")
            continue
        for p in d.glob("*.[jJ][pP][gG]"):
            image_path_map[p.stem] = str(p)

    df['path'] = df['image_id'].map(image_path_map)
    df = df.dropna(subset=['path']).reset_index(drop=True)
    df['dx'] = df['dx'].str.strip().str.lower()
    df = df[df['dx'].isin(CLASS_ORDER)].copy().reset_index(drop=True)
    df['label'] = df['dx'].map(LABEL2IDX)

    log.info(f"Loaded {len(df)} images across {len(CLASS_ORDER)} classes.")
    return df


def group_aware_split(df):
    """
    Splits into 70% Train, 15% Validation, 15% Test
    using StratifiedGroupKFold on `lesion_id` to prevent data leakage.
    """
    sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED)
    trainval_idx, test_idx = next(sgkf.split(df, df['label'], groups=df['lesion_id']))

    trainval_df = df.iloc[trainval_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    sgkf_val = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=SEED)
    t_idx, v_idx = next(sgkf_val.split(trainval_df, trainval_df['label'], groups=trainval_df['lesion_id']))

    train_df = trainval_df.iloc[t_idx].reset_index(drop=True)
    val_df = trainval_df.iloc[v_idx].reset_index(drop=True)

    # Verify zero leakage
    train_lesions = set(train_df['lesion_id'])
    val_lesions = set(val_df['lesion_id'])
    test_lesions = set(test_df['lesion_id'])

    assert len(train_lesions.intersection(val_lesions)) == 0, "Leakage Train <-> Val!"
    assert len(train_lesions.intersection(test_lesions)) == 0, "Leakage Train <-> Test!"
    assert len(val_lesions.intersection(test_lesions)) == 0, "Leakage Val <-> Test!"

    log.info(f"Split sizes — Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df


def load_and_preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0  # Normalize to [0, 1]
    return img, label


def augment(img, label):
    """Clinically valid dermoscopic augmentations."""
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    img = tf.image.rot90(img, k=k)
    img = tf.image.random_brightness(img, max_delta=0.08)
    img = tf.image.random_contrast(img, lower=0.92, upper=1.08)
    crop_size = tf.random.uniform([], 200, 224, dtype=tf.int32)
    img = tf.image.random_crop(img, [crop_size, crop_size, 3])
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, label


def make_dataset(dataframe, training=False, batch_size=BATCH_SIZE, cache_name=None):
    paths = dataframe['path'].values
    labels = dataframe['label'].values.astype(np.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)

    if cache_name:
        cache_path = str(CACHE_DIR / cache_name)
        ds = ds.cache(cache_path)
    else:
        ds = ds.cache()

    if training:
        ds = ds.shuffle(min(len(dataframe), 2048), seed=SEED)
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


def sparse_categorical_focal_loss(gamma=2.0, label_smoothing=0.05):
    """Sparse Categorical Focal Loss with label smoothing."""
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.float32)
        num_classes = tf.shape(y_pred)[-1]
        y_true_oh = tf.one_hot(y_true, depth=num_classes)
        if label_smoothing > 0:
            y_true_oh = y_true_oh * (1.0 - label_smoothing) + label_smoothing / tf.cast(num_classes, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true_oh * tf.math.log(y_pred)
        weight = y_true_oh * tf.pow(1.0 - y_pred, gamma)
        return tf.reduce_sum(weight * ce, axis=-1)
    return loss_fn


class PositionalEmbedding(layers.Layer):
    """Learnable 1D positional embedding for patch tokens."""
    def __init__(self, num_tokens=49, d_model=256, **kwargs):
        super().__init__(**kwargs)
        self.num_tokens = num_tokens
        self.d_model = d_model
        self.pos_emb = self.add_weight(
            name="pos_emb",
            shape=(1, num_tokens, d_model),
            initializer="glorot_uniform",
            trainable=True
        )

    def call(self, x):
        return x + self.pos_emb

    def get_config(self):
        config = super().get_config()
        config.update({"num_tokens": self.num_tokens, "d_model": self.d_model})
        return config


def build_hybrid_model(num_classes=7, img_size=IMG_SIZE):
    """
    Builds the Hybrid CNN-Transformer architecture:
    1. Pretrained DenseNet121 backbone (dense feature reuse for microscopic textures)
    2. Multi-Head Self-Attention Transformer block (global asymmetry modeling)
    3. Dual feature fusion (CNN + ViT tokens)
    4. Classification head
    """
    inputs = layers.Input(shape=(img_size, img_size, 3), name='input_image')

    # Internal ImageNet Normalization layer (handles [0, 1] inputs)
    norm = layers.Normalization(
        mean=[0.485, 0.456, 0.406],
        variance=[0.229**2, 0.224**2, 0.225**2],
        name='imagenet_norm'
    )(inputs)

    # DenseNet121 Backbone
    backbone = DenseNet121(
        include_top=False,
        weights='imagenet',
        input_tensor=norm,
        pooling=None,
    )
    backbone.trainable = False

    conv_features = backbone.output  # (None, 7, 7, 1024)

    # ── Transformer Branch ───────────────────────────────────────────────────
    # Reshape 7x7 spatial grid into 49 sequence tokens
    tokens = layers.Reshape((49, 1024), name='spatial_tokens')(conv_features)
    tokens_proj = layers.Dense(256, name='token_projection')(tokens)
    tokens_pos = PositionalEmbedding(num_tokens=49, d_model=256, name='pos_embedding')(tokens_proj)

    # Multi-Head Attention (4 heads, key dimension 64)
    attn_out = layers.MultiHeadAttention(num_heads=4, key_dim=64, name='mha')(tokens_pos, tokens_pos)
    x1 = layers.Add(name='attn_residual')([tokens_pos, attn_out])
    x1 = layers.LayerNormalization(name='attn_ln')(x1)

    # Feed-Forward Network (GELU MLP)
    mlp = layers.Dense(512, activation='gelu', name='mlp_dense1')(x1)
    mlp = layers.Dropout(0.2, name='mlp_drop')(mlp)
    mlp = layers.Dense(256, name='mlp_dense2')(mlp)
    x2 = layers.Add(name='mlp_residual')([x1, mlp])
    x2 = layers.LayerNormalization(name='mlp_ln')(x2)

    # Pool Transformer tokens
    trans_features = layers.GlobalAveragePooling1D(name='gap_transformer')(x2)  # (None, 256)

    # ── CNN Direct Branch ────────────────────────────────────────────────────
    cnn_gap = layers.GlobalAveragePooling2D(name='gap_cnn')(conv_features)  # (None, 1024)
    cnn_features = layers.Dense(256, activation='gelu', name='proj_cnn')(cnn_gap)  # (None, 256)

    # ── Hybrid Fusion & Classification Head ──────────────────────────────────
    fused = layers.Concatenate(name='hybrid_fusion')([cnn_features, trans_features])  # (None, 512)
    x = layers.Dropout(0.35, name='head_drop1')(fused)
    x = layers.Dense(256, activation='gelu', name='head_fc')(x)
    x = layers.Dropout(0.25, name='head_drop2')(x)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32', name='predictions')(x)

    model = Model(inputs=inputs, outputs=outputs, name='DermaAI_HybridCNN_ViT')
    return model, backbone


def evaluate_test_set(model, test_ds, test_df, class_names):
    """Thorough evaluation on the held-out test set."""
    log.info("Evaluating on held-out test set...")
    y_true, y_prob = [], []
    for imgs, labels in test_ds:
        preds = model.predict(imgs, verbose=0)
        y_prob.extend(preds)
        y_true.extend(labels.numpy())

    y_true, y_prob = np.array(y_true), np.array(y_prob)
    y_pred = np.argmax(y_prob, axis=1)

    y_true_bin = label_binarize(y_true, classes=np.arange(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_norm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))), normalize='true')
    clf_rep = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)

    per_class = {}
    for c in class_names:
        per_class[c] = {
            'full_name': FULL_CLASS_NAMES.get(c, c),
            'precision': float(clf_rep[c]['precision']),
            'recall': float(clf_rep[c]['recall']),
            'f1_score': float(clf_rep[c]['f1-score']),
            'support': int(clf_rep[c]['support'])
        }

    bkl_idx, nv_idx = LABEL2IDX['bkl'], LABEL2IDX['nv']
    bkl_support = int(np.sum(y_true == bkl_idx))
    bkl_as_nv = int(cm[bkl_idx, nv_idx])

    mel_idx = LABEL2IDX['mel']
    mel_support = int(np.sum(y_true == mel_idx))
    mel_as_nv = int(cm[mel_idx, nv_idx])

    metrics = {
        'model_name': 'Hybrid CNN-Transformer (DenseNet121 + MHSA ViT)',
        'eval_dataset': 'HAM10000 Lesion-Independent Held-Out Test Set',
        'test_samples': int(len(y_true)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'roc_auc_macro': float(roc_auc_score(y_true_bin, y_prob, average='macro', multi_class='ovr')),
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_normalized': cm_norm.round(4).tolist(),
        'class_names': class_names,
        'per_class': per_class,
        'class_distribution': test_df['dx'].value_counts().to_dict(),
        'clinical_confusion': {
            'bkl_as_nv': bkl_as_nv,
            'bkl_support': bkl_support,
            'mel_as_nv': mel_as_nv,
            'mel_support': mel_support,
        },
        'timestamp': datetime.now().isoformat()
    }

    log.info("=" * 60)
    log.info(f"Accuracy         : {metrics['accuracy']:.4f}")
    log.info(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    log.info(f"Macro F1         : {metrics['f1_macro']:.4f}")
    log.info(f"Weighted F1      : {metrics['f1_weighted']:.4f}")
    log.info(f"ROC-AUC (macro)  : {metrics['roc_auc_macro']:.4f}")
    log.info("-" * 60)
    for c in class_names:
        p = per_class[c]
        log.info(f"  {c:6s}  P={p['precision']:.3f}  R={p['recall']:.3f}  F1={p['f1_score']:.3f}  n={p['support']}")
    log.info("-" * 60)
    log.info(f"BKL -> NV confusion: {bkl_as_nv} / {bkl_support} ({bkl_as_nv/max(1,bkl_support)*100:.1f}%)")
    log.info(f"MEL -> NV confusion: {mel_as_nv} / {mel_support} ({mel_as_nv/max(1,mel_support)*100:.1f}%)")
    log.info("=" * 60)

    return metrics


def main():
    set_seeds()
    log.info("=" * 60)
    log.info("  DERMA.AI — Training Hybrid CNN-Transformer")
    log.info(f"  Started: {datetime.now():%Y-%m-%d %H:%M:%S}")
    log.info("=" * 60)

    df = load_dataframe()
    train_df, val_df, test_df = group_aware_split(df)

    # Clear old cache
    for f in CACHE_DIR.glob("*"):
        try:
            f.unlink()
        except Exception:
            pass

    log.info("Building cached datasets...")
    train_ds = make_dataset(train_df, training=True, cache_name="train")
    val_ds = make_dataset(val_df, training=False, cache_name="val")
    test_ds = make_dataset(test_df, training=False, cache_name="test")

    # Balanced class weights with targeted BKL & MEL prioritization
    cw = compute_class_weight('balanced', classes=np.arange(len(CLASS_ORDER)), y=train_df['label'].values)
    class_weights = dict(enumerate(cw))
    # Give BKL and MEL +25% boost to prevent NV confusion while keeping true balance
    class_weights[LABEL2IDX['bkl']] *= 1.25
    class_weights[LABEL2IDX['mel']] *= 1.25

    log.info("Class weights:")
    for i, c in enumerate(CLASS_ORDER):
        log.info(f"  {c:6s}: {class_weights[i]:.3f}")

    # Build model
    log.info("Building Hybrid CNN-Transformer model...")
    model, backbone = build_hybrid_model(len(CLASS_ORDER))
    model.summary(print_fn=log.info, expand_nested=False)

    loss_fn = sparse_categorical_focal_loss(gamma=2.0, label_smoothing=0.05)

    # ── Phase 1: Train Transformer + Fusion Head (8 epochs) ───────────────────
    log.info("\n" + "=" * 60)
    log.info("  PHASE 1: Training Transformer & Fusion Head (8 epochs, lr=1e-3)")
    log.info("=" * 60)

    model.compile(
        optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss=loss_fn,
        metrics=['accuracy']
    )

    cb_p1 = [
        EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    ]

    hist_p1 = model.fit(
        train_ds,
        epochs=8,
        validation_data=val_ds,
        class_weight=class_weights,
        callbacks=cb_p1,
        verbose=1
    )
    log.info(f"Phase 1 best val_accuracy: {max(hist_p1.history['val_accuracy']):.4f}")

    # ── Phase 2: Fine-Tuning Top Dense Block (14 epochs) ──────────────────────
    log.info("\n" + "=" * 60)
    log.info("  PHASE 2: Fine-Tuning Top Dense Block of DenseNet121 (14 epochs, lr=2e-5)")
    log.info("=" * 60)

    # Unfreeze top dense block (conv5_block1 to conv5_block16, layers 312 onwards)
    backbone.trainable = True
    for layer in backbone.layers[:312]:
        layer.trainable = False

    unfrozen_count = sum(1 for l in backbone.layers if l.trainable)
    log.info(f"Unfroze {unfrozen_count} layers in DenseNet121 block 5 for fine-tuning")

    model.compile(
        optimizer=AdamW(learning_rate=2e-5, weight_decay=1e-5),
        loss=loss_fn,
        metrics=['accuracy']
    )

    ckpt_path = str(MODELS_DIR / "hybrid_best.keras")
    cb_p2 = [
        ModelCheckpoint(ckpt_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, verbose=1),
    ]

    hist_p2 = model.fit(
        train_ds,
        epochs=14,
        validation_data=val_ds,
        class_weight=class_weights,
        callbacks=cb_p2,
        verbose=1
    )
    log.info(f"Phase 2 best val_accuracy: {max(hist_p2.history['val_accuracy']):.4f}")

    # ── Save Artifacts ───────────────────────────────────────────────────────
    final_path = str(MODELS_DIR / "hybrid_final.keras")
    model.save(final_path)
    log.info(f"Model saved to {final_path}")

    # Also save to project root
    root_model_path = str(PROJECT_ROOT / "hybrid_final.keras")
    model.save(root_model_path)
    log.info(f"Root model saved to {root_model_path}")

    # Save canonical label map
    label_map_dict = {
        'class_names': CLASS_ORDER,
        'label2idx': LABEL2IDX,
        'full_names': FULL_CLASS_NAMES,
        'img_size': IMG_SIZE,
        'architecture': 'Hybrid CNN-Transformer (DenseNet121 + MHSA ViT)',
        'updated_at': datetime.now().isoformat()
    }
    for p in [MODELS_DIR / "label_map.json", PROJECT_ROOT / "label_map.json"]:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(label_map_dict, f, indent=2)

    # Evaluate
    metrics = evaluate_test_set(model, test_ds, test_df, CLASS_ORDER)

    for p in [MODELS_DIR / "evaluation_results.json", PROJECT_ROOT / "evaluation_results.json"]:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    log.info("Saved evaluation results to models/ and project root.")
    log.info(f"Finished at: {datetime.now():%Y-%m-%d %H:%M:%S}")
    return metrics


if __name__ == '__main__':
    main()

