"""
train.py — Production training script for the HAM10000 skin lesion classifier.

Unlike the assignment notebook, this is meant to be run anywhere (local GPU box,
cloud VM, Colab, Kaggle, on-prem) via the command line, with no Google Drive
dependency. It exports a single deployable model in the modern Keras format
(.keras), ready to be picked up by app.py / a serving container.

Usage:
    python train.py \
        --metadata_csv /path/to/HAM10000_metadata.csv \
        --image_dirs /path/to/images_part_1 /path/to/images_part_2 \
        --output_dir ./models \
        --architecture hybrid \
        --epochs_head 10 --epochs_finetune 20

Architectures:
    --architecture mobilenet  -> MobileNetV2 (CPU-friendly; recommended for local machines)
    --architecture convnext   -> ConvNeXt-Tiny only (Model 1 from the assignment)
    --architecture hybrid     -> ConvNeXt-Tiny + Transformer encoder (Model 2, usually best accuracy)
    --architecture both       -> trains ConvNeXt + hybrid and saves both (for ensembling in the app)
"""

import os
import json
import argparse
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ConvNeXtTiny, MobileNetV2
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, f1_score, precision_score, recall_score
)
from sklearn.preprocessing import label_binarize

SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

CLASS_ORDER = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
FULL_CLASS_NAMES = {
    'akiec': 'Actinic keratoses', 'bcc': 'Basal cell carcinoma',
    'bkl': 'Benign keratosis', 'df': 'Dermatofibroma',
    'mel': 'Melanoma', 'nv': 'Melanocytic nevi', 'vasc': 'Vascular lesions'
}


def set_seeds():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_dataframe(metadata_csv, image_dirs):
    df = pd.read_csv(metadata_csv)
    image_path_map = {}
    for d in image_dirs:
        if not os.path.exists(d):
            print(f'WARNING: image dir not found: {d}')
            continue
        for fname in os.listdir(d):
            if fname.lower().endswith(('.jpg', '.jpeg')):
                img_id = os.path.splitext(fname)[0]
                image_path_map[img_id] = os.path.join(d, fname)

    df['path'] = df['image_id'].map(image_path_map)
    df = df.dropna(subset=['path']).reset_index(drop=True)

    class_names = sorted(df['dx'].unique())
    label2idx = {c: i for i, c in enumerate(class_names)}
    df['label'] = df['dx'].map(label2idx)
    return df, class_names, label2idx


def balanced_split(df, max_oversample_ratio=2.0):
    class_counts = df['label'].value_counts()
    target = int(min(class_counts.max(), class_counts.median() * max_oversample_ratio))

    parts = []
    for label_id in sorted(class_counts.index):
        subset = df[df['label'] == label_id]
        if len(subset) < target:
            subset = subset.sample(target, replace=True, random_state=SEED)
        parts.append(subset)
    df_balanced = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)

    train_df, temp_df = train_test_split(df_balanced, test_size=0.30,
                                          stratify=df_balanced['label'], random_state=SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.50,
                                        stratify=temp_df['label'], random_state=SEED)
    return train_df, val_df, test_df


def load_and_preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


def augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=0.15)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
    img = tf.image.random_hue(img, max_delta=0.08)
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    img = tf.image.rot90(img, k=k)
    scale = tf.random.uniform([], 0.85, 1.0)
    new_size = tf.cast(IMG_SIZE * scale, tf.int32)
    img = tf.image.random_crop(img, [new_size, new_size, 3])
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, label


def make_dataset(dataframe, training=False, batch_size=BATCH_SIZE, cache_path=None):
    paths = dataframe['path'].values
    labels = dataframe['label'].values.astype(np.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    # Disk-backed cache: caching this dataset in RAM (the no-arg ds.cache()
    # default) needs several GB for HAM10000-sized splits and thrashes on
    # low-memory machines. A file cache trades a bit of disk I/O for a
    # bounded memory footprint.
    ds = ds.cache(cache_path) if cache_path else ds.cache()
    if training:
        # Buffer holds decoded float32 images (224x224x3 each ~600KB) —
        # kept small since this machine has very little free RAM.
        ds = ds.shuffle(512, seed=SEED)
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #
def sparse_categorical_focal_loss(gamma=2.0, label_smoothing=0.05):
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.float32)
        num_classes = tf.shape(y_pred)[-1]
        y_true_oh = tf.one_hot(y_true, depth=num_classes)
        if label_smoothing > 0:
            y_true_oh = y_true_oh * (1 - label_smoothing) + label_smoothing / tf.cast(num_classes, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true_oh * tf.math.log(y_pred)
        weight = y_true_oh * tf.pow(1 - y_pred, gamma)
        return tf.reduce_sum(weight * ce, axis=-1)
    return loss_fn


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def build_convnext_model(num_classes, img_size=IMG_SIZE, name='convnext_tiny'):
    inputs = layers.Input(shape=(img_size, img_size, 3), name='input_image')
    x = layers.Rescaling(scale=255.0, name='rescale')(inputs)
    backbone = ConvNeXtTiny(include_top=False, weights='imagenet', input_tensor=x, pooling=None)
    backbone.trainable = False

    x = backbone.output
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.LayerNormalization(name='ln_head')(x)
    x = layers.Dense(512, activation='gelu', name='fc1')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='gelu', name='fc2')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32', name='predictions')(x)
    return Model(inputs, outputs, name=name), backbone


def build_mobilenet_model(num_classes, img_size=IMG_SIZE, name='mobilenet_v2'):
    """MobileNetV2 backbone — ~15x fewer FLOPs than ConvNeXt-Tiny at 224x224,
    the practical choice for CPU-only training."""
    inputs = layers.Input(shape=(img_size, img_size, 3), name='input_image')
    # load_and_preprocess yields [0,1]; MobileNetV2 expects [-1,1].
    x = layers.Rescaling(scale=2.0, offset=-1.0, name='rescale')(inputs)
    backbone = MobileNetV2(include_top=False, weights='imagenet', input_tensor=x, pooling=None)
    backbone.trainable = False

    x = backbone.output
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.LayerNormalization(name='ln_head')(x)
    x = layers.Dense(512, activation='gelu', name='fc1')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='gelu', name='fc2')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32', name='predictions')(x)
    return Model(inputs, outputs, name=name), backbone


def transformer_encoder(x, num_heads=4, key_dim=64, ff_dim=512, dropout=0.1, prefix='enc'):
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout,
                                      name=f'{prefix}_mha')(x, x)
    x = layers.Add(name=f'{prefix}_add1')([x, attn])
    x = layers.LayerNormalization(name=f'{prefix}_ln1')(x)
    ff = layers.Dense(ff_dim, activation='gelu', name=f'{prefix}_ff1')(x)
    ff = layers.Dropout(dropout)(ff)
    ff = layers.Dense(x.shape[-1], name=f'{prefix}_ff2')(ff)
    x = layers.Add(name=f'{prefix}_add2')([x, ff])
    x = layers.LayerNormalization(name=f'{prefix}_ln2')(x)
    return x


def build_hybrid_model(num_classes, img_size=IMG_SIZE, embed_dim=256, num_blocks=2, name='hybrid_convnext_transformer'):
    inputs = layers.Input(shape=(img_size, img_size, 3), name='input_image')
    x = layers.Rescaling(scale=255.0, name='rescale')(inputs)
    backbone = ConvNeXtTiny(include_top=False, weights='imagenet', input_tensor=x, pooling=None)
    backbone.trainable = False

    fmap = backbone.output
    h, w, c = fmap.shape[1], fmap.shape[2], fmap.shape[3]
    num_patches = h * w
    tokens = layers.Reshape((num_patches, c), name='to_tokens')(fmap)
    tokens = layers.Dense(embed_dim, name='patch_proj')(tokens)

    pos_embed = layers.Embedding(input_dim=num_patches, output_dim=embed_dim, name='pos_embed')
    positions = tf.range(start=0, limit=num_patches, delta=1)
    tokens = tokens + pos_embed(positions)

    for i in range(num_blocks):
        tokens = transformer_encoder(tokens, prefix=f'tenc{i}')

    x = layers.GlobalAveragePooling1D(name='token_pool')(tokens)
    x = layers.LayerNormalization(name='ln_head')(x)
    x = layers.Dense(256, activation='gelu', name='fc1')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32', name='predictions')(x)
    return Model(inputs, outputs, name=name), backbone


# --------------------------------------------------------------------------- #
# Train / evaluate one model
# --------------------------------------------------------------------------- #
def train_one(model, backbone, train_ds, val_ds, class_weights, focal_loss,
               output_dir, tag, epochs_head, epochs_finetune):
    model.compile(optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-4),
                   loss=focal_loss, metrics=['accuracy'])
    cb = [
        EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ]
    print(f'[{tag}] Phase 1: training head (backbone frozen)...')
    model.fit(train_ds, epochs=epochs_head, validation_data=val_ds,
              class_weight=class_weights, callbacks=cb, verbose=2)

    total_layers = len(backbone.layers)
    unfreeze_from = int(total_layers * 0.60)
    backbone.trainable = True
    for layer in backbone.layers[:unfreeze_from]:
        layer.trainable = False

    model.compile(optimizer=AdamW(learning_rate=2e-5, weight_decay=1e-5),
                   loss=focal_loss, metrics=['accuracy'])
    ckpt_path = os.path.join(output_dir, f'{tag}_best.keras')
    cb2 = [
        EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
        ModelCheckpoint(ckpt_path, monitor='val_accuracy', save_best_only=True, verbose=1),
    ]
    print(f'[{tag}] Phase 2: fine-tuning...')
    model.fit(train_ds, epochs=epochs_finetune, validation_data=val_ds,
              class_weight=class_weights, callbacks=cb2, verbose=2)
    return model


def evaluate(model, test_df, class_names, tag, output_dir):
    cache_dir = os.path.join(output_dir, 'tf_cache')
    os.makedirs(cache_dir, exist_ok=True)
    test_ds = make_dataset(test_df, training=False, cache_path=os.path.join(cache_dir, f'{tag}_test'))
    y_true, y_prob = [], []
    for imgs, labels in test_ds:
        y_prob.extend(model.predict(imgs, verbose=0))
        y_true.extend(labels.numpy())
    y_true, y_prob = np.array(y_true), np.array(y_prob)
    y_pred = np.argmax(y_prob, axis=1)

    y_true_bin = label_binarize(y_true, classes=np.arange(len(class_names)))
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'roc_auc_macro': float(roc_auc_score(y_true_bin, y_prob, average='macro', multi_class='ovr')),
    }
    print(f'\n=== {tag} test metrics ===')
    print(json.dumps(metrics, indent=2))
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    print('Confusion matrix:\n', confusion_matrix(y_true, y_pred))

    with open(os.path.join(output_dir, f'{tag}_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    return metrics


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_csv', required=True)
    parser.add_argument('--image_dirs', nargs='+', required=True)
    parser.add_argument('--output_dir', default='./models')
    parser.add_argument('--architecture', choices=['convnext', 'hybrid', 'mobilenet', 'both'], default='hybrid')
    parser.add_argument('--epochs_head', type=int, default=10)
    parser.add_argument('--epochs_finetune', type=int, default=20)
    parser.add_argument('--max_oversample_ratio', type=float, default=2.0)
    args = parser.parse_args()

    set_seeds()
    os.makedirs(args.output_dir, exist_ok=True)

    gpus = tf.config.list_physical_devices('GPU')
    print(f'GPUs available: {len(gpus)}')
    if gpus:
        keras.mixed_precision.set_global_policy('mixed_float16')

    df, class_names, label2idx = load_dataframe(args.metadata_csv, args.image_dirs)
    print(f'Loaded {len(df)} labeled images across {len(class_names)} classes: {class_names}')

    train_df, val_df, test_df = balanced_split(df, args.max_oversample_ratio)
    print(f'Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}')

    cache_dir = os.path.join(args.output_dir, 'tf_cache')
    os.makedirs(cache_dir, exist_ok=True)
    train_ds = make_dataset(train_df, training=True, cache_path=os.path.join(cache_dir, 'train'))
    val_ds = make_dataset(val_df, training=False, cache_path=os.path.join(cache_dir, 'val'))

    cw = compute_class_weight('balanced', classes=np.arange(len(class_names)), y=train_df['label'].values)
    class_weights = dict(enumerate(cw))
    focal_loss = sparse_categorical_focal_loss()

    # persist the label map so the serving app always agrees with training
    with open(os.path.join(args.output_dir, 'label_map.json'), 'w') as f:
        json.dump({'class_names': class_names, 'label2idx': label2idx,
                   'full_names': FULL_CLASS_NAMES, 'img_size': IMG_SIZE}, f, indent=2)

    architectures = ['convnext', 'hybrid'] if args.architecture == 'both' else [args.architecture]
    all_metrics = {}

    for arch in architectures:
        if arch == 'convnext':
            model, backbone = build_convnext_model(len(class_names))
        elif arch == 'mobilenet':
            model, backbone = build_mobilenet_model(len(class_names))
        else:
            model, backbone = build_hybrid_model(len(class_names))

        model = train_one(model, backbone, train_ds, val_ds, class_weights, focal_loss,
                           args.output_dir, arch, args.epochs_head, args.epochs_finetune)

        final_path = os.path.join(args.output_dir, f'{arch}_final.keras')
        model.save(final_path)
        print(f'Saved deployable model -> {final_path}')

        all_metrics[arch] = evaluate(model, test_df, class_names, arch, args.output_dir)

    with open(os.path.join(args.output_dir, 'all_metrics.json'), 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print('\nTraining complete. Deployable artifacts are in:', args.output_dir)
    print('Copy the .keras file(s) and label_map.json next to app.py to deploy.')


if __name__ == '__main__':
    main()
