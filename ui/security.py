"""
DERMA.AI Security Module
Validates uploaded image files, prevents decompression bombs, verifies magic bytes,
and sanitizes user inputs to prevent injection, path traversal, and XSS.
"""

import html
import io
import os
from typing import Tuple, Optional
from PIL import Image

# Enforce reasonable file size limit: 10 MB
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024

# Enforce image dimensions to prevent memory exhaustion / decompression bombs
MAX_IMAGE_DIMENSION = 8192
MIN_IMAGE_DIMENSION = 32

# Set PIL safety limit against decompression bombs
Image.MAX_IMAGE_PIXELS = 20_000_000

# Magic bytes signatures
JPEG_MAGIC = b'\xFF\xD8\xFF'
PNG_MAGIC = b'\x89PNG\r\n\x1a\n'


def sanitize_text(text: str) -> str:
    """Sanitize any user-controlled string to prevent HTML/XSS injection."""
    if not isinstance(text, str):
        text = str(text)
    return html.escape(text.strip())


def sanitize_filename(filename: str) -> str:
    """
    Sanitizes an uploaded filename to prevent path traversal (../, ..\\)
    and script/control character injection.
    """
    if not filename:
        return "lesion_image.jpg"
    # Normalize slashes and extract pure basename
    basename = os.path.basename(filename.replace('\\', '/'))
    # Filter printable ASCII/Unicode only, eliminate control chars and null bytes
    clean_basename = "".join(ch for ch in basename if ch.isprintable() and ch != '\x00')
    # Escape HTML
    sanitized = html.escape(clean_basename.strip())
    return sanitized if sanitized else "lesion_image.jpg"


def validate_image_bytes(raw_bytes: bytes, filename: str = '') -> Tuple[bool, Optional[str], Optional[Image.Image]]:
    """
    Rigorously validates uploaded file bytes:
    1. Checks file size (<= 10MB)
    2. Validates magic bytes (JPEG or PNG)
    3. Verifies image integrity using PIL
    4. Enforces dimension limits
    Returns: (is_valid, error_message, PIL.Image)
    """
    if not raw_bytes or len(raw_bytes) == 0:
        return False, "The uploaded file is empty.", None

    if len(raw_bytes) > MAX_FILE_SIZE_BYTES:
        max_mb = MAX_FILE_SIZE_BYTES // (1024 * 1024)
        return False, f"File size exceeds the maximum allowed limit of {max_mb} MB.", None

    # Magic byte validation
    is_jpeg = raw_bytes.startswith(JPEG_MAGIC)
    is_png = raw_bytes.startswith(PNG_MAGIC)

    if not (is_jpeg or is_png):
        return False, "Invalid image format. Only authentic JPEG and PNG files are accepted.", None

    # Decode image using PIL
    try:
        img_buffer = io.BytesIO(raw_bytes)
        img = Image.open(img_buffer)
        img.verify()  # Verify image integrity
    except Exception as e:
        return False, f"Corrupted or malformed image data: {str(e)}", None

    # Re-open after verify() (PIL requirement)
    try:
        img_buffer.seek(0)
        img = Image.open(img_buffer)
        width, height = img.size

        if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
            return False, f"Image dimensions too small ({width}×{height}). Minimum required: {MIN_IMAGE_DIMENSION}×{MIN_IMAGE_DIMENSION}.", None

        if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
            return False, f"Image dimensions too large ({width}×{height}). Maximum allowed: {MAX_IMAGE_DIMENSION}×{MAX_IMAGE_DIMENSION}.", None

        # Convert to RGB (handles RGBA, grayscale, CMYK, Palette)
        rgb_img = img.convert('RGB')
        return True, None, rgb_img

    except Exception as e:
        return False, f"Unable to process image dimensions or channels: {str(e)}", None
