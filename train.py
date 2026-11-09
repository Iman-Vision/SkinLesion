"""
train.py — Rigorous, Leak-Free Training Script for DERMA.AI on HAM10000.

Key improvements over legacy script:
1. Strict Group-Aware Stratified Splitting: Uses StratifiedGroupKFold on `lesion_id`
   so no patient or lesion images leak across Train, Validation, or Test splits.
2. NO Oversampling before splitting: Test and Validation sets remain 100% held-out
   and representative of clinical distribution.
3. Class Imbalance Mitigation: Addressed strictly via computed class weights
   and Sparse Categorical Focal Loss (\(\gamma=2.0\)) with label smoothing.
4. Consistent Preprocessing: [0, 1] float32 RGB, identical to inference pipeline.
5. Saves clean model artifacts, label map, and evaluation metrics.
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
from tensorflow.keras.applications import MobileNetV2, ConvNeXtTiny
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    balanced_accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
)
from sklearn.preprocessing import label_binarize

SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

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


def set_seeds():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def load_dataframe(metadata_csv, image_dirs):
    df = pd.read_csv(metadata_csv)
    image_path_map = {}
    for d in image_dirs:
        if not os.path.exists(d):
            print(f'WARNING: Image directory not found: {d}')
            continue
        for fname in os.listdir(d):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_id = os.path.splitext(fname)[0]
                image_path_map[img_id] = os.path.join(d, fname)

    df['path'] = df['image_id'].map(image_path_map)
    df = df.dropna(subset=['path']).reset_index(drop=True)

    class_names = sorted(df['dx'].unique())
    label2idx = {c: i for i, c in enumerate(class_names)}
    df['label'] = df['dx'].map(label2idx)
    return df, class_names, label2idx


def group_aware_split(df):
    """
    Splits dataset into 70% Train, 15% Validation, 15% Test
    using StratifiedGroupKFold on `lesion_id` to prevent data leakage.
    """
    # 5 folds -> 80% train_val, 20% test
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_val_idx, test_idx = next(sgkf.split(df, df['label'], groups=df['lesion_id']))

    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # Split train_val into train (80% of train_val = 64% total) and val (20% of train_val = 16% total)
    sgkf_val = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    t_idx, v_idx = next(sgkf_val.split(train_val_df, train_val_df['label'], groups=train_val_df['lesion_id']))

    train_df = train_val_df.iloc[t_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[v_idx].reset_index(drop=True)

    # Verify zero leakage
    train_lesions = set(train_df['lesion_id'])
    val_lesions = set(val_df['lesion_id'])
    test_lesions = set(test_df['lesion_id'])

    assert len(train_lesions.intersection(val_lesions)) == 0, "Lesion leakage between Train and Val!"
    assert len(train_lesions.intersection(test_lesions)) == 0, "Lesion leakage between Train and Test!"
    assert len(val_lesions.intersection(test_lesions)) == 0, "Lesion leakage between Val and Test!"

    return train_df, val_df, test_df


def load_and_preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


def augment(img, label):
    """Conservative, clinically defensible dermoscopic augmentations."""
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=0.10)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    img = tf.image.rot90(img, k=k)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, label


def make_dataset(dataframe, training=False, batch_size=BATCH_SIZE, cache_path=None):
    paths = dataframe['path'].values
    labels = dataframe['label'].values.astype(np.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    if cache_path:
        ds = ds.cache(cache_path)
    else:
        ds = ds.cache()

    if training:
        ds = ds.shuffle(512, seed=SEED)
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


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


def build_mobilenet_model(num_classes, img_size=IMG_SIZE, name='mobilenet_v2'):
    inputs = layers.Input(shape=(img_size, img_size, 3), name='input_image')
    # Preprocessing layer: maps [0, 1] to [-1, 1] for MobileNetV2
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

    metrics = {
        'model_name': tag,
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'roc_auc_macro': float(roc_auc_score(y_true_bin, y_prob, average='macro', multi_class='ovr')),
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_normalized': cm_norm.round(4).tolist(),
        'per_class': per_class
    }

    print(f'\n=== {tag} Held-Out Test Evaluation ===')
    print(json.dumps(metrics, indent=2))
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    with open(os.path.join(output_dir, f'{tag}_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_csv', default='data/ham10000/HAM10000_metadata.csv')
    parser.add_argument('--image_dirs', nargs='+', default=['data/ham10000/HAM10000_images_part_1', 'data/ham10000/HAM10000_images_part_2'])
    parser.add_argument('--output_dir', default='./models')
    parser.add_argument('--epochs_head', type=int, default=10)
    parser.add_argument('--epochs_finetune', type=int, default=20)
    args = parser.parse_args()

    set_seeds()
    os.makedirs(args.output_dir, exist_ok=True)

    df, class_names, label2idx = load_dataframe(args.metadata_csv, args.image_dirs)
    print(f'Loaded {len(df)} images across {len(class_names)} classes.')

    train_df, val_df, test_df = group_aware_split(df)
    print(f'Lesion-independent Split — Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}')

    train_ds = make_dataset(train_df, training=True)
    val_ds = make_dataset(val_df, training=False)

    cw = compute_class_weight('balanced', classes=np.arange(len(class_names)), y=train_df['label'].values)
    class_weights = dict(enumerate(cw))
    focal_loss = sparse_categorical_focal_loss()

    # Save canonical label map
    with open(os.path.join(args.output_dir, 'label_map.json'), 'w') as f:
        json.dump({
            'class_names': class_names,
            'label2idx': label2idx,
            'full_names': FULL_CLASS_NAMES,
            'img_size': IMG_SIZE
        }, f, indent=2)

    model, backbone = build_mobilenet_model(len(class_names))

    # Phase 1: Train Head
    model.compile(optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-4), loss=focal_loss, metrics=['accuracy'])
    cb1 = [
        EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    ]
    print("\nPhase 1: Training Classification Head...")
    model.fit(train_ds, epochs=args.epochs_head, validation_data=val_ds, class_weight=class_weights, callbacks=cb1, verbose=1)

    # Phase 2: Fine-Tuning Top 40% of Backbone
    unfreeze_from = int(len(backbone.layers) * 0.60)
    backbone.trainable = True
    for layer in backbone.layers[:unfreeze_from]:
        layer.trainable = False

    model.compile(optimizer=AdamW(learning_rate=2e-5, weight_decay=1e-5), loss=focal_loss, metrics=['accuracy'])
    ckpt_path = os.path.join(args.output_dir, 'mobilenet_best.keras')
    cb2 = [
        EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
        ModelCheckpoint(ckpt_path, monitor='val_accuracy', save_best_only=True, verbose=1),
    ]
    print("\nPhase 2: Fine-tuning MobileNetV2...")
    model.fit(train_ds, epochs=args.epochs_finetune, validation_data=val_ds, class_weight=class_weights, callbacks=cb2, verbose=1)

    final_path = os.path.join(args.output_dir, 'mobilenet_final.keras')
    model.save(final_path)
    print(f'Saved deployable model -> {final_path}')

    metrics = evaluate(model, test_df, class_names, 'mobilenet', args.output_dir)
    with open(os.path.join(args.output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    main()
