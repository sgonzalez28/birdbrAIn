"""
inference_pipeline.py

Ensemble trait-based species classification pipeline.

This script:
1. Loads exported fastai trait models (.pkl)
2. Removes image backgrounds (optional)
3. Predicts trait-level classes
4. Aggregates predictions via majority voting
5. Applies rule-based ensemble scoring
6. Outputs species and sex classification

Example:
    python inference_pipeline.py \
        --model_dir models/ \
        --input_dir path/to/specimen_folder/ \
        --config configs/inference_config.yaml
"""

# -----------------------------
# 1. Imports
# -----------------------------

import os
import argparse
import yaml
import numpy as np
from collections import Counter
from pathlib import Path
from fastai.vision.all import load_learner, Learner
from PIL import Image
from rembg import remove


# -----------------------------
# 2. Argument Parsing
# -----------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True,
                        help="Directory containing exported .pkl models.")
    parser.add_argument("--input_dir", required=True,
                        help="Directory containing specimen subfolders.")
    parser.add_argument("--config", required=True,
                        help="Path to inference configuration YAML.")
    return parser.parse_args()


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# -----------------------------
# 3. Utility Functions
# -----------------------------

def safe_load_learner(path):
    """
    Load exported fastai learner and validate type.
    """
    model = load_learner(path, cpu=False)
    if not isinstance(model, Learner):
        raise TypeError(
            f"{path} is not a Fastai Learner. "
            "Ensure model was exported with learn.export()."
        )
    return model


def remove_background(pil_img):
    """
    Remove background using rembg and convert to RGB.
    """
    output = remove(pil_img)

    if output.mode == "RGBA":
        bg = Image.new("RGB", output.size, (255, 255, 255))
        bg.paste(output, mask=output.split()[3])
        return bg

    return output.convert("RGB")


def predict_trait(model, pil_img):
    """
    Return predicted class label and raw output.
    """
    pred_class, pred_idx, probs = model.predict(pil_img)
    return str(pred_class), probs


# -----------------------------
# 4. Ensemble Aggregation
# -----------------------------

def compute_mode(predictions):
    """
    Majority vote with custom tie-breaking rule.
    If mode is '0' but class '1' also present,
    select '1' (paper-specific rule).
    """
    counter = Counter(predictions)
    mode = counter.most_common(1)[0][0]

    if mode == "0" and "1" in counter:
        mode = "1"

    return mode


def ensemble_species_score(trait_modes, low_thresh, high_thresh):
    """
    Convert trait modes into final species classification.
    """
    trait_scores = [int(v) for k, v in trait_modes.items()
                    if k != "Sex"]

    final_score = sum(trait_scores)

    if final_score <= low_thresh:
        classification = "MEXICAN DUCK"
    elif final_score >= high_thresh:
        classification = "MALLARD"
    else:
        classification = "UNKNOWN"

    return classification, final_score


def interpret_sex(sex_value):
    return "MALE" if int(sex_value) == 1 else "FEMALE"


# -----------------------------
# 5. Main Inference Pipeline
# -----------------------------

def run_inference(model_dir, input_dir, config):

    # -------------------------
    # Load Models
    # -------------------------

    model_files = config["models"]

    models = {
        name: safe_load_learner(Path(model_dir) / filename)
        for name, filename in model_files.items()
    }

    # -------------------------
    # Folder-to-Model Mapping
    # -------------------------

    folder_models = config["folder_models"]

    # Storage
    model_predictions = {name: [] for name in models}
    model_confidences = {name: [] for name in models}

    # -------------------------
    # Loop Through Specimen
    # -------------------------

    for folder_name, model_names in folder_models.items():

        folder_path = Path(input_dir) / folder_name
        if not folder_path.exists():
            continue

        for img_file in os.listdir(folder_path):

            img_path = folder_path / img_file
            pil_img = Image.open(img_path).convert("RGB")

            if config["remove_background"]:
                pil_img = remove_background(pil_img)

            for model_name in model_names:

                model = models[model_name]
                pred_class, probs = predict_trait(model, pil_img)

                model_predictions[model_name].append(pred_class)
                model_confidences[model_name].append(
                    float(max(probs))
                )

    # -------------------------
    # Aggregate Modes
    # -------------------------

    trait_modes = {
        name: compute_mode(preds)
        for name, preds in model_predictions.items()
        if preds
    }

    # -------------------------
    # Species & Sex
    # -------------------------

    classification, final_score = ensemble_species_score(
        trait_modes,
        config["species_threshold_low"],
        config["species_threshold_high"]
    )

    sex = interpret_sex(trait_modes["Sex"])

    average_confidence = np.mean([
        conf for conf_list in model_confidences.values()
        for conf in conf_list
    ])

    # -------------------------
    # Output
    # -------------------------

    print("\n===== FINAL CLASSIFICATION =====")
    print(f"Species: {classification}")
    print(f"Final Score: {final_score}")
    print(f"Sex: {sex}")
    print(f"Mean Confidence: {average_confidence:.2%}")
    print("\nTrait Modes:", trait_modes)

    return classification, final_score, sex


# -----------------------------
# 6. Entry Point
# -----------------------------

if __name__ == "__main__":

    args = parse_args()
    config = load_config(args.config)

    run_inference(
        model_dir=args.model_dir,
        input_dir=args.input_dir,
        config=config
    )