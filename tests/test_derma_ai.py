"""
DERMA.AI Complete Functional Test Suite
Covers image intake validation, model preprocessing & inference, Grad-CAM generation,
PDF clinical report generation, security defenses, and application state integrity.
"""

import io
import json
import os
import pytest
import numpy as np
from PIL import Image
import tensorflow as tf

from ui.security import (
    validate_image_bytes,
    sanitize_text,
    sanitize_filename,
    MAX_FILE_SIZE_BYTES,
    MIN_IMAGE_DIMENSION,
    MAX_IMAGE_DIMENSION,
    JPEG_MAGIC,
    PNG_MAGIC
)
from ui.explainability import generate_gradcam
from ui.report_generator import generate_pdf_report
from app import preprocess, predict, load_models, _init_state


# ============================================================================ #
# 1. IMAGE INPUT & VALIDATION TESTS
# ============================================================================ #

class TestImageInput:

    def test_valid_jpeg(self):
        """Tests that a standard valid JPEG image passes validation."""
        img = Image.new('RGB', (300, 300), color=(180, 120, 90))
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        raw = buf.getvalue()

        is_valid, err, pil_img = validate_image_bytes(raw, "test.jpg")
        assert is_valid is True
        assert err is None
        assert pil_img is not None
        assert pil_img.size == (300, 300)
        assert pil_img.mode == 'RGB'

    def test_valid_png(self):
        """Tests that a valid PNG image passes validation."""
        img = Image.new('RGB', (224, 224), color=(60, 90, 140))
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        raw = buf.getvalue()

        is_valid, err, pil_img = validate_image_bytes(raw, "test.png")
        assert is_valid is True
        assert err is None
        assert pil_img is not None
        assert pil_img.size == (224, 224)

    def test_rgba_image_converted_to_rgb(self):
        """Tests that 4-channel RGBA images are converted to 3-channel RGB cleanly."""
        img = Image.new('RGBA', (200, 200), color=(100, 150, 200, 255))
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        raw = buf.getvalue()

        is_valid, err, pil_img = validate_image_bytes(raw, "transparent.png")
        assert is_valid is True
        assert pil_img.mode == 'RGB'

    def test_grayscale_image_converted_to_rgb(self):
        """Tests that 1-channel Grayscale images are properly converted to 3-channel RGB."""
        img = Image.new('L', (200, 200), color=128)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        raw = buf.getvalue()

        is_valid, err, pil_img = validate_image_bytes(raw, "gray.png")
        assert is_valid is True
        assert pil_img.mode == 'RGB'

    def test_empty_image(self):
        """Tests that 0-byte files are rejected."""
        is_valid, err, pil_img = validate_image_bytes(b'', "empty.jpg")
        assert is_valid is False
        assert "empty" in err.lower()
        assert pil_img is None

    def test_corrupted_image(self):
        """Tests that malformed or truncated byte streams are safely rejected."""
        # Starts with valid JPEG header but has corrupted random payload
        corrupt_bytes = JPEG_MAGIC + b"\x00\x11\x22\x33\x44corrupted_stream_data"
        is_valid, err, pil_img = validate_image_bytes(corrupt_bytes, "corrupt.jpg")
        assert is_valid is False
        assert "corrupted" in err.lower() or "malformed" in err.lower()

    def test_oversized_file(self):
        """Tests that files exceeding the maximum file size limit are rejected."""
        oversized_bytes = b"0" * (MAX_FILE_SIZE_BYTES + 1024)
        is_valid, err, pil_img = validate_image_bytes(oversized_bytes, "huge.jpg")
        assert is_valid is False
        assert "exceeds" in err.lower()

    def test_unsupported_format(self):
        """Tests that non-JPEG/PNG files (e.g., text, executable, GIF) are rejected."""
        fake_file = b"%PDF-1.4 file content disguised as an image"
        is_valid, err, pil_img = validate_image_bytes(fake_file, "fake.jpg")
        assert is_valid is False
        assert "invalid image format" in err.lower()

    def test_image_dimensions_too_small(self):
        """Tests that images below the minimum resolution limit are rejected."""
        tiny_img = Image.new('RGB', (10, 10), color='white')
        buf = io.BytesIO()
        tiny_img.save(buf, format='JPEG')
        is_valid, err, pil_img = validate_image_bytes(buf.getvalue(), "tiny.jpg")
        assert is_valid is False
        assert "too small" in err.lower()


# ============================================================================ #
# 2. MODEL PREPROCESSING & INFERENCE TESTS
# ============================================================================ #

class TestModelPipeline:

    @pytest.fixture(scope="module")
    def loaded_model_suite(self):
        models, label_map, error = load_models()
        assert error is None, f"Model loading failed: {error}"
        assert models is not None and len(models) > 0
        assert label_map is not None
        return models, label_map

    def test_model_loading_and_input_shape(self, loaded_model_suite):
        models, label_map = loaded_model_suite
        model = next(iter(models.values()))
        assert model.input_shape == (None, 224, 224, 3)
        assert model.output_shape == (None, 7)

    def test_class_mapping_consistency(self, loaded_model_suite):
        _, label_map = loaded_model_suite
        class_names = label_map['class_names']
        label2idx = label_map['label2idx']

        assert len(class_names) == 7
        assert class_names == ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
        for i, c in enumerate(class_names):
            assert label2idx[c] == i

    def test_preprocessing_output_tensor(self):
        img = Image.new('RGB', (500, 350), color=(200, 100, 50))
        tensor = preprocess(img, img_size=224)

        assert isinstance(tensor, np.ndarray)
        assert tensor.shape == (1, 224, 224, 3)
        assert tensor.dtype == np.float32
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_inference_probability_distribution(self, loaded_model_suite):
        models, label_map = loaded_model_suite
        test_img = Image.new('RGB', (224, 224), color=(120, 80, 60))
        tensor = preprocess(test_img, img_size=224)

        probs = predict(models, tensor, use_tta=True)

        assert isinstance(probs, np.ndarray)
        assert len(probs) == 7
        assert np.isclose(np.sum(probs), 1.0, atol=1e-4)
        assert all(0.0 <= p <= 1.0 for p in probs)

    def test_multiple_consecutive_predictions_determinism(self, loaded_model_suite):
        models, _ = loaded_model_suite
        img = Image.new('RGB', (224, 224), color=(90, 70, 50))
        tensor = preprocess(img, img_size=224)

        p1 = predict(models, tensor, use_tta=False)
        p2 = predict(models, tensor, use_tta=False)
        np.testing.assert_allclose(p1, p2, rtol=1e-5)


# ============================================================================ #
# 3. EXPLAINABILITY (GRAD-CAM) TESTS
# ============================================================================ #

class TestExplainability:

    def test_gradcam_generation(self):
        models, label_map, _ = load_models()
        model = next(iter(models.values()))
        img = Image.new('RGB', (224, 224), color=(140, 95, 75))

        heatmap, blended = generate_gradcam(model, img, img_size=224, target_class_idx=5)

        assert isinstance(heatmap, np.ndarray)
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0
        assert isinstance(blended, Image.Image)
        assert blended.size == img.size


# ============================================================================ #
# 4. REPORT GENERATION (PDF) TESTS
# ============================================================================ #

class TestReportGeneration:

    def test_pdf_report_creation(self):
        img = Image.new('RGB', (200, 200), color=(170, 110, 85))
        class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
        full_names = {
            'akiec': 'Actinic keratoses', 'bcc': 'Basal cell carcinoma',
            'bkl': 'Benign keratosis', 'df': 'Dermatofibroma',
            'mel': 'Melanoma', 'nv': 'Melanocytic nevi', 'vasc': 'Vascular lesions'
        }
        probs = [0.02, 0.03, 0.70, 0.01, 0.04, 0.18, 0.02]

        pdf_bytes = generate_pdf_report(
            analysis_id="DA-TEST01",
            timestamp="2026-11-09 12:00:00",
            filename="sample_lesion.jpg",
            image=img,
            top_code="bkl",
            top_name="Benign keratosis",
            top_conf=0.70,
            is_high_risk=False,
            class_probs=probs,
            class_names=class_names,
            full_names=full_names
        )

        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 1000
        assert pdf_bytes.startswith(b'%PDF-')

    def test_pdf_report_with_special_characters(self):
        """Tests that filenames or descriptions with &, <, > do not crash ReportLab XML parser."""
        img = Image.new('RGB', (100, 100), color=(100, 100, 100))
        class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
        full_names = {c: f"Diagnosis & Category <{c}>" for c in class_names}
        probs = [0.1, 0.1, 0.1, 0.1, 0.1, 0.4, 0.1]

        pdf_bytes = generate_pdf_report(
            analysis_id="DA-<REF>&01",
            timestamp="2026-11-09 12:00:00 & UTC",
            filename="../../malicious & <test>.jpg",
            image=img,
            top_code="nv",
            top_name="Melanocytic & Nevi <Benign>",
            top_conf=0.40,
            is_high_risk=False,
            class_probs=probs,
            class_names=class_names,
            full_names=full_names
        )
        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes.startswith(b'%PDF-')


# ============================================================================ #
# 5. SECURITY & INPUT SANITIZATION TESTS
# ============================================================================ #

class TestSecurity:

    def test_xss_sanitization(self):
        malicious_input = "<script>alert('XSS')</script>lesion.jpg"
        clean = sanitize_text(malicious_input)
        assert "<script>" not in clean
        assert "&lt;script&gt;" in clean

    def test_magic_bytes_detection(self):
        fake_jpeg = b"GIF89a Fake GIF disguised as JPEG"
        assert not fake_jpeg.startswith(JPEG_MAGIC)
        assert not fake_jpeg.startswith(PNG_MAGIC)
        is_valid, _, _ = validate_image_bytes(fake_jpeg, "fake.jpg")
        assert is_valid is False

    def test_filename_path_traversal_sanitization(self):
        """Tests that relative and absolute path attempts are stripped from filenames."""
        dirty_path1 = "../../../etc/passwd.jpg"
        clean1 = sanitize_filename(dirty_path1)
        assert ".." not in clean1
        assert "/" not in clean1
        assert clean1 == "passwd.jpg"

        dirty_path2 = "C:\\Windows\\System32\\cmd.exe.png"
        clean2 = sanitize_filename(dirty_path2)
        assert "C:" not in clean2
        assert "\\" not in clean2
        assert clean2 == "cmd.exe.png"

        null_byte_path = "harmless.jpg\x00<script>.png"
        clean3 = sanitize_filename(null_byte_path)
        assert "\x00" not in clean3
        assert "<script>" not in clean3
