"""
DERMA.AI Explainability & Clinical Feature Extraction Module (XAI)
===================================================================
Translates complex deep learning attributions (LIME & Grad-CAM) into:
1. Layman-Friendly Plain-English explanations (What is it? Why was it flagged? What should you do?)
2. Doctor-Level Clinical Dermoscopy breakdowns (ABCDE criteria, histological biomarkers, differential rationale)
3. Zero-repetition, highly scannable UI summaries.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from PIL import Image
import tensorflow as tf
from skimage.segmentation import mark_boundaries, quickshift
from lime import lime_image


# ============================================================================ #
# 1. DUAL-AUDIENCE TAXONOMY: LAYMAN & DOCTOR KNOWLEDGE BASE
# ============================================================================ #

DIAGNOSTIC_PROFILES: Dict[str, Dict[str, Any]] = {
    'mel': {
        'code': 'MEL',
        'clinical_name': 'Melanoma',
        'layman_name': 'Melanoma (Serious Skin Cancer Risk)',
        'risk_tier': 'High Risk / Immediate Attention',
        'badge_color': '#e11d48',
        'layman': {
            'what_is_it': 'Melanoma is a serious type of skin cancer that develops in pigment-producing cells. It often starts as an unusual or changing mole.',
            'what_ai_saw': 'The AI detected uneven borders, irregular color variations, and asymmetric dark pigment clusters across the spot.',
            'action_advice': '🚨 **Immediate Action Recommended:** Schedule an in-person examination with a dermatologist as soon as possible. When caught early, melanoma is highly treatable. Avoid scratching or picking at the spot.'
        },
        'doctor': {
            'biomarkers': [
                'Abrupt, irregular peripheral border cutoff with focal notched margins',
                'Marked chromophore variegation (slate-gray, dark brown, black, and erythematous hues)',
                'Atypical, thickened reticular pigment network with irregular mesh distribution',
                'Architectural and structural asymmetry across orthogonal poles'
            ],
            'differential_driver': 'High spatial irregularity and pigment heterogeneity distinguish it from benign melanocytic nevi.',
            'clinical_action': 'Urgent dermatological dermoscopy and full-thickness excisional biopsy recommended.'
        }
    },
    'nv': {
        'code': 'NV',
        'clinical_name': 'Melanocytic Nevus',
        'layman_name': 'Normal Mole (Harmless)',
        'risk_tier': 'Low Risk / Benign',
        'badge_color': '#10b981',
        'layman': {
            'what_is_it': 'A common, non-cancerous skin mole. Most adults have between 10 and 40 normal moles across their body.',
            'what_ai_saw': 'The AI observed a balanced, symmetric shape with uniform brown color and smooth borders that fade gently into normal skin.',
            'action_advice': '✅ **Standard Self-Care:** This looks like a harmless mole. Continue regular skin checks and consult a doctor if it ever changes size, shape, or color.'
        },
        'doctor': {
            'biomarkers': [
                'Regular, delicate pigment network with uniform gridline spacing',
                'High architectural and chromophore symmetry across orthogonal axes',
                'Smooth peripheral border with gradual reticular fading into surrounding skin',
                'Evenly dispersed central pigment globules without focal clustering'
            ],
            'differential_driver': 'Absence of border notching, eccentric hyperpigmentation, or multi-chromophore variegation.',
            'clinical_action': 'Routine clinical observation; educate patient on standard ABCDE self-monitoring.'
        }
    },
    'bkl': {
        'code': 'BKL',
        'clinical_name': 'Benign Keratosis',
        'layman_name': 'Benign Keratosis (Harmless Age Spot)',
        'risk_tier': 'Low Risk / Harmless Growth',
        'badge_color': '#0ea5e9',
        'layman': {
            'what_is_it': 'A very common, harmless non-cancerous growth (often called a seborrheic keratosis or age spot). They are not contagious and do not turn into cancer.',
            'what_ai_saw': 'The AI noticed a distinct "stuck-on" waxy border and surface texture typical of non-cancerous keratosis.',
            'action_advice': '✅ **Generally Harmless:** No treatment is medically necessary unless the spot becomes irritated by clothing, itches, or bleeds. If it changes rapidly, have a doctor inspect it.'
        },
        'doctor': {
            'biomarkers': [
                'Sharply demarcated, well-circumscribed "stuck-on" epidermal border morphology',
                'Verrucous / cerebriform surface texture pattern with fingerprint-like ridges',
                'Clustered keratin aggregates and comedo-like follicular pseudocomedones',
                'Uniform yellowish-tan to dull dark-brown keratinous pigment density'
            ],
            'differential_driver': 'Presence of stuck-on epidermal borders and follicular keratin openings, lacking a true melanocytic network.',
            'clinical_action': 'Benign finding; reassurance provided. Cryotherapy or curettage only if symptomatic.'
        }
    },
    'bcc': {
        'code': 'BCC',
        'clinical_name': 'Basal Cell Carcinoma',
        'layman_name': 'Basal Cell Carcinoma (Common Skin Cancer)',
        'risk_tier': 'High Risk / Medical Care Needed',
        'badge_color': '#f59e0b',
        'layman': {
            'what_is_it': 'The most common form of skin cancer. It grows slowly over months or years and rarely spreads to other organs, but requires medical treatment to remove completely.',
            'what_ai_saw': 'The AI detected shiny, translucent skin areas and tiny branching red blood vessels characteristic of basal cell growths.',
            'action_advice': '🩺 **Doctor Consultation Needed:** Schedule an appointment with a dermatologist. A minor in-office procedure or topical treatment can remove it effectively.'
        },
        'doctor': {
            'biomarkers': [
                'Prominent focal arborizing (treelike branching) telangiectatic blood vessels',
                'Translucent pearly stromal background with micro-ulceration',
                'Non-reticular globular blue-gray ovoid nests',
                'Circumscribed focal architecture lacking melanocytic lines'
            ],
            'differential_driver': 'Dominance of arborizing microvasculature and absence of melanocytic pigment network.',
            'clinical_action': 'Biopsy recommended (shave or punch) followed by excision, Mohs surgery, or electrodesiccation.'
        }
    },
    'akiec': {
        'code': 'AKIEC',
        'clinical_name': 'Actinic Keratosis / Bowen Disease',
        'layman_name': 'Actinic Keratosis (Pre-Cancerous Sun Spot)',
        'risk_tier': 'Moderate Risk / Treatment Recommended',
        'badge_color': '#f97316',
        'layman': {
            'what_is_it': 'A rough, dry, or scaly patch caused by long-term sun exposure. It is considered a pre-cancerous spot that should be treated so it does not turn into skin cancer.',
            'what_ai_saw': 'The AI identified rough surface scaling over a reddish, sun-damaged skin background.',
            'action_advice': '🩺 **Doctor Visit Advised:** A dermatologist can easily treat pre-cancerous sun spots with cryotherapy (freezing) or prescription creams.'
        },
        'doctor': {
            'biomarkers': [
                'Hyperkeratotic surface scaling on an erythematous base',
                'Strawberry pattern (erythematous pseudo-network around follicular openings)',
                'Ill-defined peripheral border blending into actinic background skin',
                'Targetoid follicular keratin plugs'
            ],
            'differential_driver': 'Diffuse surface scale heterogeneity and erythema lacking a melanocytic pigment network.',
            'clinical_action': 'Lesion-directed cryotherapy or field therapy (5-fluorouracil, imiquimod) recommended.'
        }
    },
    'vasc': {
        'code': 'VASC',
        'clinical_name': 'Vascular Lesion',
        'layman_name': 'Vascular Lesion / Blood Spot (Harmless)',
        'risk_tier': 'Low Risk / Benign Blood Vessel',
        'badge_color': '#8b5cf6',
        'layman': {
            'what_is_it': 'A harmless collection of small blood vessels under the skin (such as a cherry angioma or blood spot). They are completely non-cancerous.',
            'what_ai_saw': 'The AI recognized clear red-to-purple blood pools with well-defined borders and no dark pigment network.',
            'action_advice': '✅ **Harmless Blood Spot:** No medical action is needed. Consult a physician if the spot bleeds frequently or becomes painful.'
        },
        'doctor': {
            'biomarkers': [
                'Well-circumscribed red, purple, or blue-black round-to-oval lacunae (lagoons)',
                'Absence of melanocytic pigment network and follicular structures',
                'Homogeneous vascular luminal pooling with high hemoglobin contrast',
                'Clean circumscription without inflammatory epidermal scale'
            ],
            'differential_driver': 'Pure vascular lacunar pooling with complete absence of melanocytic pigment structures.',
            'clinical_action': 'Benign vascular malformation; clinical reassurance. Electrocautery or pulsed-dye laser if cosmetic removal requested.'
        }
    },
    'df': {
        'code': 'DF',
        'clinical_name': 'Dermatofibroma',
        'layman_name': 'Dermatofibroma (Harmless Skin Bump)',
        'risk_tier': 'Low Risk / Benign Firm Nodule',
        'badge_color': '#64748b',
        'layman': {
            'what_is_it': 'A firm, harmless bump under the skin that often develops after a minor injury like an insect bite or ingrown hair.',
            'what_ai_saw': 'The AI detected a pale central scar-like area surrounded by a thin, delicate pigment ring.',
            'action_advice': '✅ **Safe Skin Nodule:** Dermatofibromas are completely harmless and rarely require removal unless they cause irritation.'
        },
        'doctor': {
            'biomarkers': [
                'Central hypopigmented or white scar-like fibrous patch',
                'Delicate, faint pigmentary network restricted strictly to the outer periphery',
                'Compact, symmetric architectural circumscription',
                'Positive clinical pinch/dimple sign correlation'
            ],
            'differential_driver': 'Pathognomonic central white patch enclosed by a delicate peripheral pigment collar.',
            'clinical_action': 'Benign fibrous histiocytoma; clinical reassurance. Simple excision only if painful or repeatedly traumatized.'
        }
    }
}


# Alias for backward compatibility
DERMOSCOPIC_CRITERIA = DIAGNOSTIC_PROFILES


def generate_lime_explanation(
    model: tf.keras.Model,
    image: Image.Image,
    target_class_idx: int,
    img_size: int = None,
    num_samples: int = 100,
    num_features: int = 4,
    random_seed: int = 42
) -> Tuple[Image.Image, List[Tuple[int, float]], np.ndarray]:
    """
    Computes Local Interpretable Model-agnostic Explanations (LIME).
    Returns superpixel visualization highlighting positive (green) and negative (coral) areas.
    """
    if img_size is None or img_size == 224:
        if hasattr(model, 'input_shape') and model.input_shape and len(model.input_shape) >= 3:
            h = model.input_shape[1]
            if h is not None:
                img_size = h
    if img_size is None:
        img_size = 224

    resized_pil = image.convert('RGB').resize((img_size, img_size), Image.Resampling.BILINEAR)
    img_arr = np.array(resized_pil, dtype=np.float32) / 255.0

    def predict_fn(images_batch: np.ndarray) -> np.ndarray:
        if hasattr(model, 'input_shape') and model.input_shape and len(model.input_shape) >= 3:
            h = model.input_shape[1]
            if h is not None and (images_batch.shape[1] != h or images_batch.shape[2] != h):
                images_batch = tf.image.resize(images_batch, [h, h]).numpy()
        return model.predict(images_batch, verbose=0)

    explainer = lime_image.LimeImageExplainer(random_state=random_seed)
    segmenter = lambda x: quickshift(x, kernel_size=4, max_dist=12, ratio=0.2)

    explanation = explainer.explain_instance(
        img_arr,
        predict_fn,
        labels=[target_class_idx],
        top_labels=None,
        hide_color=0.5,
        num_samples=num_samples,
        segmentation_fn=segmenter,
        random_seed=random_seed
    )

    top_features = explanation.local_exp.get(target_class_idx, [])[:num_features]
    segments = explanation.segments

    overlay = np.array(resized_pil).copy()
    pos_mask = np.zeros(segments.shape, dtype=bool)
    neg_mask = np.zeros(segments.shape, dtype=bool)

    for seg_id, weight in top_features:
        if weight > 0:
            pos_mask[segments == seg_id] = True
        elif weight < 0:
            neg_mask[segments == seg_id] = True

    overlay_float = overlay.astype(np.float32)
    green_tint = np.array([30, 190, 110], dtype=np.float32)
    coral_tint = np.array([230, 70, 70], dtype=np.float32)

    overlay_float[pos_mask] = 0.55 * overlay_float[pos_mask] + 0.45 * green_tint
    overlay_float[neg_mask] = 0.55 * overlay_float[neg_mask] + 0.45 * coral_tint

    boundaries = mark_boundaries(
        overlay_float / 255.0,
        segments,
        color=(1.0, 0.95, 0.2),
        mode='subpixel'
    )

    lime_pil = Image.fromarray(np.uint8(np.clip(boundaries * 255.0, 0, 255)))
    lime_pil_matched = lime_pil.resize(image.size, Image.Resampling.BILINEAR)

    return lime_pil_matched, top_features, segments


# ============================================================================ #
# 3. CLINICAL & LAYMAN FEATURE EXTRACTION REPORT
# ============================================================================ #

def extract_visual_features_and_explain(
    image: Image.Image,
    top_code: str,
    top_conf: float,
    probs: List[float],
    class_names: List[str],
    full_names: Dict[str, str],
    top_features: Optional[List[Tuple[int, float]]] = None,
    segments: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Synthesizes two clear, non-repetitive explanations:
    1. Plain-English guide for the patient/layman
    2. Professional dermoscopic review for the doctor
    """
    code = top_code.lower()
    profile = DIAGNOSTIC_PROFILES.get(code, DIAGNOSTIC_PROFILES['nv'])

    # 1. Identify runner-up class for differential diagnosis
    sorted_indices = np.argsort(probs)[::-1]
    runner_up_idx = sorted_indices[1] if len(sorted_indices) > 1 else 0
    runner_up_code = class_names[runner_up_idx]
    runner_up_name = full_names.get(runner_up_code, runner_up_code)
    runner_up_conf = float(probs[runner_up_idx])

    # 2. Determine Primary Salience Focus Area (concise, human-readable)
    primary_location = "Central body and lesion borders"
    if segments is not None and top_features:
        pos_superpixels = [seg_id for seg_id, weight in top_features if weight > 0]
        if pos_superpixels:
            y_indices, x_indices = np.where(np.isin(segments, pos_superpixels))
            if len(x_indices) > 0:
                mean_x = np.mean(x_indices) / segments.shape[1]
                mean_y = np.mean(y_indices) / segments.shape[0]
                dx = mean_x - 0.5
                dy = mean_y - 0.5
                radius = np.sqrt(dx**2 + dy**2)

                if radius > 0.30:
                    primary_location = "Outer peripheral border"
                elif radius < 0.18:
                    primary_location = "Central lesion core"
                else:
                    primary_location = "Paracentral transition zone"

                # Add simple direction
                direction = ""
                if dy < -0.10:
                    direction += "upper"
                elif dy > 0.10:
                    direction += "lower"

                if dx < -0.10:
                    direction += (" " if direction else "") + "left"
                elif dx > 0.10:
                    direction += (" " if direction else "") + "right"

                if direction:
                    primary_location = f"{primary_location} ({direction} side)"

    # 3. Clean, crisp differential text (no repetition)
    if code in ['mel', 'nv'] and runner_up_code in ['mel', 'nv']:
        differential_text = (
            f"The primary secondary consideration was **{runner_up_name}** ({runner_up_conf*100:.1f}%). "
            f"The AI prioritized **{profile['clinical_name']}** based on "
            f"{'evident architectural asymmetry and abrupt border cutoffs' if code == 'mel' else 'consistent pigment symmetry and gradual peripheral fading'}."
        )
    elif code in ['bkl', 'nv'] and runner_up_code in ['bkl', 'nv']:
        differential_text = (
            f"There is optical overlap with **{runner_up_name}** ({runner_up_conf*100:.1f}%). "
            f"The model prioritized **{profile['clinical_name']}** due to "
            f"{'distinct stuck-on surface borders and keratin cyst patterns' if code == 'bkl' else 'a uniform reticular pigment network'}."
        )
    else:
        differential_text = (
            f"The secondary differential was **{runner_up_name}** ({runner_up_conf*100:.1f}%). "
            f"The decisive factor favoring **{profile['clinical_name']}** was the strong alignment with "
            f"{profile['doctor']['differential_driver']}."
        )

    # 4. Synthesize backward-compatible summary paragraph (clean & direct)
    biomarkers = profile['doctor']['biomarkers']
    summary_paragraph = (
        f"The neural network classified this lesion as **{profile['clinical_name']}** with **{top_conf*100:.1f}% confidence**, "
        f"identifying key visual features at the **{primary_location}**. "
        f"Primary findings include {biomarkers[0].lower()} and {biomarkers[1].lower()}. {differential_text}"
    )

    # 5. Check Melanoma secondary suspicion level
    mel_idx = class_names.index('mel') if 'mel' in class_names else -1
    mel_risk = float(probs[mel_idx]) if mel_idx >= 0 else 0.0
    mel_alert = mel_risk >= 0.15 and code != 'mel'

    return {
        # Backward-compatible fields
        'primary_location': primary_location,
        'extracted_biomarkers': biomarkers,
        'differential_text': differential_text,
        'summary_paragraph': summary_paragraph,
        'runner_up_name': runner_up_name,
        'runner_up_conf': runner_up_conf,
        'clinical_driver': profile['doctor']['differential_driver'],
        # New Layman & Doctor-friendly sections
        'layman_name': profile['layman_name'],
        'clinical_name': profile['clinical_name'],
        'risk_tier': profile['risk_tier'],
        'badge_color': profile['badge_color'],
        'mel_risk': mel_risk,
        'mel_alert': mel_alert,
        'layman_guide': profile['layman'],
        'doctor_guide': {
            'biomarkers': biomarkers,
            'clinical_action': profile['doctor']['clinical_action'],
            'differential_driver': profile['doctor']['differential_driver'],
            'focus_area': primary_location
        }
    }
