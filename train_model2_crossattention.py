"""
DERMA.AI — Model 2: EfficientNetV2B3 + Multi-Scale Cross-Attention Fusion
==========================================================================
Replication of Bahria University Deep Learning (AIC-401) Assignment 3:
- Backbone: EfficientNetV2B3 (ICML 2021) at native 300x300 input resolution
- Novel Layer: MultiScaleCrossAttention (CrossViT ICCV 2021 / DualViT NeurIPS 2022)
  Bidirectional interaction between:
    - Early texture tokens: block5g_add (19x19, 136) -> 361 tokens
    - Late semantic tokens: top_activation (10x10, 1536) -> 100 tokens
- Imbalance Strategy: Split first -> Class Weights on train only -> Focal Loss (gamma=2, smoothing=0.1)
- Training Schedule: 2-Phase Cosine Warmup Schedule
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
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetV2B3
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
keras = tf.keras  # Use TF-bundled Keras 2 compat shim throughout

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training_model2.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger("derma_model2")

SEED = 42
IMG_SIZE = 300
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "ham10000"
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR = PROJECT_ROOT / ".tf_cache_model2"

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
    os.environ['PYTHONHASHSEED'] = str(SEED)


# ============================================================================ #
# 1. MULTI-SCALE CROSS-ATTENTION LAYER (FROM REPORT)
# ============================================================================ #

@tf.keras.utils.register_keras_serializable(package="derma_ai")
class MultiScaleCrossAttention(layers.Layer):
    """
    Bidirectional cross-attention between texture tokens (361) and semantic tokens (100).
    Both projected to common_dim=128. Built entirely from scratch.
    Reference: CrossViT (ICCV 2021), DualViT (NeurIPS 2022)
    """
    def __init__(self, common_dim=128, num_heads=8, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.common_dim = common_dim
        self.num_heads = num_heads
        self.dropout = dropout

    def build(self, input_shape):
        d = self.common_dim
        self.proj_tex = layers.Dense(d, name='proj_tex')
        self.proj_sem = layers.Dense(d, name='proj_sem')
        self.proj_drop = layers.Dropout(0.2)

        # Direction 1: Semantic queries Texture
        self.q_sem = layers.Dense(d, use_bias=False)
        self.k_tex = layers.Dense(d, use_bias=False)
        self.v_tex = layers.Dense(d, use_bias=False)
        self.out_sem = layers.Dense(d)

        # Direction 2: Texture queries Semantic
        self.q_tex = layers.Dense(d, use_bias=False)
        self.k_sem = layers.Dense(d, use_bias=False)
        self.v_sem = layers.Dense(d, use_bias=False)
        self.out_tex = layers.Dense(d)

        self.norm_sem = layers.LayerNormalization(epsilon=1e-5)
        self.norm_tex = layers.LayerNormalization(epsilon=1e-5)
        self.attn_drop = layers.Dropout(self.dropout)
        self.scale = (d // self.num_heads) ** -0.5
        super().build(input_shape)

    def _attn(self, Ql, Kl, Vl, Ol, q_src, kv_src, training):
        B = tf.shape(q_src)[0]
        Nq = tf.shape(q_src)[1]
        Nk = tf.shape(kv_src)[1]
        h = self.num_heads
        d = self.common_dim // h
        Q = tf.transpose(tf.reshape(Ql(q_src), (B, Nq, h, d)), (0, 2, 1, 3))
        K = tf.transpose(tf.reshape(Kl(kv_src), (B, Nk, h, d)), (0, 2, 1, 3))
        V = tf.transpose(tf.reshape(Vl(kv_src), (B, Nk, h, d)), (0, 2, 1, 3))
        attn = tf.nn.softmax(tf.matmul(Q, K, transpose_b=True) * self.scale, axis=-1)
        attn = self.attn_drop(attn, training=training)
        out = tf.matmul(attn, V)
        out = tf.reshape(tf.transpose(out, (0, 2, 1, 3)), (B, Nq, self.common_dim))
        return Ol(out)

    def call(self, inputs, training=None):
        texture_tokens, semantic_tokens = inputs
        tex = self.proj_drop(self.proj_tex(texture_tokens), training=training)
        sem = self.proj_drop(self.proj_sem(semantic_tokens), training=training)

        # Direction 1: Semantic queries Texture
        enhanced_sem = self.norm_sem(sem + self._attn(
            self.q_sem, self.k_tex, self.v_tex, self.out_sem, sem, tex, training))

        # Direction 2: Texture queries Semantic
        enhanced_tex = self.norm_tex(tex + self._attn(
            self.q_tex, self.k_sem, self.v_sem, self.out_tex, tex, sem, training))

        return [enhanced_sem, enhanced_tex]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            'common_dim': self.common_dim,
            'num_heads': self.num_heads,
            'dropout': self.dropout
        })
        return cfg


# ============================================================================ #
# 2. FOCAL LOSS & COSINE WARMUP SCHEDULE (FROM REPORT)
# ============================================================================ #

def focal_loss(gamma=2.0, smoothing=0.1):
    cce = keras.losses.CategoricalCrossentropy(label_smoothing=smoothing, reduction='none')
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        ce = cce(y_true, y_pred)
        p_true = tf.clip_by_value(tf.reduce_sum(y_true * y_pred, axis=-1), 1e-6, 1.0 - 1e-6)
        return tf.reduce_mean(tf.pow(1.0 - p_true, gamma) * ce)
    return loss_fn


class CosineWarmupSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, warmup_steps, total_steps):
        super().__init__()
        self.base_lr = base_lr
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.total_steps = tf.cast(total_steps, tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = self.base_lr * (step / self.warmup_steps)
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        cosine = self.base_lr * 0.5 * (1.0 + tf.cos(np.pi * progress))
        return tf.where(step < self.warmup_steps, warmup, cosine)

    def get_config(self):
        return {
            'base_lr': self.base_lr,
            'warmup_steps': int(self.warmup_steps.numpy()),
            'total_steps': int(self.total_steps.numpy())
        }


# ============================================================================ #
# 3. DATA LOADING & PREPROCESSING (CLEAN SPLIT FIRST)
# ============================================================================ #

def load_dataframe():
    csv_path = DATA_DIR / "HAM10000_metadata.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"HAM10000_metadata.csv not found at {csv_path}")
    df = pd.read_csv(csv_path)

    image_path_map = {}
    for d in [DATA_DIR / "HAM10000_images_part_1", DATA_DIR / "HAM10000_images_part_2"]:
        if not d.exists():
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


def load_img(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)  # [0, 255] range for EfficientNetV2 built-in scaling
    return img, tf.one_hot(label, len(CLASS_ORDER))


def augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=0.20 * 255.0)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
    img = tf.image.random_hue(img, max_delta=0.04)
    crop = int(IMG_SIZE * 0.88)
    img = tf.image.random_crop(img, [crop, crop, 3])
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.clip_by_value(img, 0.0, 255.0)
    return img, label


def make_dataset(dataframe, training=False, batch_size=BATCH_SIZE):
    paths = dataframe['path'].values
    labels = dataframe['label'].values.astype(np.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_img, num_parallel_calls=AUTOTUNE)

    if training:
        ds = ds.shuffle(min(len(dataframe), 2048), seed=SEED)
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


# ============================================================================ #
# 4. MODEL 2 ARCHITECTURE BUILDER
# ============================================================================ #

def build_model2(num_classes=7, img_size=IMG_SIZE):
    backbone = EfficientNetV2B3(
        include_top=False,
        weights='imagenet',
        input_shape=(img_size, img_size, 3)
    )
    backbone.trainable = False
    inputs = backbone.input

    # Multi-scale feature extraction
    early_feat = backbone.get_layer('block5g_add').output   # (B, 19, 19, 136)
    late_feat = backbone.get_layer('top_activation').output # (B, 10, 10, 1536)

    # Flatten to token sequences
    texture_tokens = layers.Reshape((19 * 19, 136), name='tex_tokens')(early_feat)
    semantic_tokens = layers.Reshape((10 * 10, 1536), name='sem_tokens')(late_feat)

    # Cross-attention fusion
    ca = MultiScaleCrossAttention(common_dim=128, num_heads=8, dropout=0.1, name='cross_attention')
    enhanced_sem, enhanced_tex = ca([texture_tokens, semantic_tokens])

    # Pooling & projection
    pooled_sem = layers.GlobalAveragePooling1D(name='gap_sem')(enhanced_sem)
    pooled_tex = layers.GlobalAveragePooling1D(name='gap_tex')(enhanced_tex)

    x = layers.Concatenate(name='combine')([pooled_sem, pooled_tex])
    x = layers.LayerNormalization(epsilon=1e-5)(x)
    x = layers.Dense(256, activation='gelu', kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32', name='output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='Model2_EfficientNetV2B3_CrossAttention')
    return model, backbone


# ============================================================================ #
# 5. TRAINING PIPELINE (PHASE 1 & PHASE 2)
# ============================================================================ #

def main():
    set_seeds()
    log.info("=" * 65)
    log.info("  DERMA.AI — Training Model 2 (EfficientNetV2B3 + Cross-Attention)")
    log.info("  Replication of Bahria University DL Assignment 3")
    log.info("=" * 65)

    df = load_dataframe()

    # Split first: 70% Train, 15% Val, 15% Test
    train_orig, temp = train_test_split(df, test_size=0.30, stratify=df['label'], random_state=SEED)
    val_df, test_df = train_test_split(temp, test_size=0.50, stratify=temp['label'], random_state=SEED)
    train_df = train_orig.reset_index(drop=True)

    log.info(f"Split sizes — Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Class weights on train set only (no leakage)
    cw = compute_class_weight('balanced', classes=np.unique(train_df['label']), y=train_df['label'])
    class_weight_dict = dict(enumerate(cw))
    log.info(f"Class weights computed: {class_weight_dict}")

    train_ds = make_dataset(train_df, training=True)
    val_ds = make_dataset(val_df, training=False)
    test_ds = make_dataset(test_df, training=False)

    steps_per_epoch = len(train_df) // BATCH_SIZE

    # Build Model 2
    model2, backbone = build_model2(len(CLASS_ORDER), IMG_SIZE)
    log.info(f"Model 2 constructed! Total parameters: {model2.count_params():,}")

    loss_fn = focal_loss(gamma=2.0, smoothing=0.1)

    # ── Phase 1: Train Head Only (15 epochs) ─────────────────────────────────
    EPOCHS_P1 = 15
    lr_p1 = CosineWarmupSchedule(5e-4, 3 * steps_per_epoch, EPOCHS_P1 * steps_per_epoch)

    model2.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr_p1, weight_decay=1e-4),
        loss=loss_fn,
        metrics=['accuracy']
    )

    ckpt_p1 = str(MODELS_DIR / "model2_p1_best.keras")
    callbacks_p1 = [
        EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True, verbose=1),
        ModelCheckpoint(ckpt_p1, monitor='val_accuracy', save_best_only=True, verbose=1),
        CSVLogger(str(MODELS_DIR / "model2_p1_log.csv"))
    ]

    log.info("\n=== Starting Phase 1: Backbone Frozen (Cross-Attention Head) ===")
    model2.fit(
        train_ds,
        epochs=EPOCHS_P1,
        validation_data=val_ds,
        class_weight=class_weight_dict,
        callbacks=callbacks_p1,
        verbose=1
    )

    # ── Phase 2: Unfreeze Top 30 Backbone Layers (25 epochs) ─────────────────
    backbone.trainable = True
    for layer in backbone.layers[:-30]:
        layer.trainable = False

    unfrozen = sum(1 for l in model2.layers if l.trainable)
    log.info(f"\n=== Starting Phase 2: Unfroze top 30 layers ({unfrozen} trainable layers) ===")

    EPOCHS_P2 = 25
    lr_p2 = CosineWarmupSchedule(1e-4, 2 * steps_per_epoch, EPOCHS_P2 * steps_per_epoch)

    model2.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr_p2, weight_decay=1e-5),
        loss=loss_fn,
        metrics=['accuracy']
    )

    final_model_path = str(PROJECT_ROOT / "model2_final.keras")
    callbacks_p2 = [
        EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1),
        ModelCheckpoint(final_model_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        CSVLogger(str(MODELS_DIR / "model2_p2_log.csv"))
    ]

    model2.fit(
        train_ds,
        epochs=EPOCHS_P2,
        validation_data=val_ds,
        class_weight=class_weight_dict,
        callbacks=callbacks_p2,
        verbose=1
    )

    # Save to models/ as well
    model2.save(str(MODELS_DIR / "model2_final.keras"))
    log.info(f"Model 2 successfully trained and saved to {final_model_path}")

    # ── Evaluation ───────────────────────────────────────────────────────────
    log.info("Evaluating Model 2 on held-out test set...")
    test_loss, test_acc = model2.evaluate(test_ds, verbose=0)
    y_pred_probs = model2.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.concatenate([np.argmax(l.numpy(), axis=1) for _, l in test_ds])

    y_true_bin = label_binarize(y_true, classes=np.arange(len(CLASS_ORDER)))
    macro_auc = roc_auc_score(y_true_bin, y_pred_probs, average='macro', multi_class='ovr')
    clf_rep = classification_report(y_true, y_pred, target_names=CLASS_ORDER, output_dict=True)

    metrics = {
        'model_name': 'Model 2: EfficientNetV2B3 + Multi-Scale Cross-Attention Fusion',
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'macro_auc': float(macro_auc),
        'classification_report': clf_rep,
        'mel_recall': float(clf_rep['mel']['recall']),
        'mel_f1': float(clf_rep['mel']['f1-score']),
        'timestamp': datetime.now().isoformat()
    }

    with open(PROJECT_ROOT / "evaluation_results_model2.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    log.info(f"Test Accuracy: {test_acc*100:.2f}% | Macro AUC: {macro_auc:.4f} | MEL Recall: {clf_rep['mel']['recall']*100:.2f}%")
    return metrics


if __name__ == '__main__':
    main()
