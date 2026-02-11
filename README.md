# birdbrAIn
Modular trait-based deep learning framework for fine-grained species identification using ensemble CNN classification.

# Trait-Based Ensemble Classification Framework
Supplementary Code for: _Interpretable Wildlife Classification by Coupling Genetics, Scoring Systems, and Computer Vision_

This repository contains the full training and inference pipeline used in:

Gonzalez, S et al. 2026. Interpretable Wildlife Classification by Coupling Genetics, Scoring Systems, and Computer Vision. (Journal)

It implements a modular, trait-based convolutional neural network (CNN) framework for fine-grained species identification using morphological subregions.

# Overview

## Repository Structure
```
species-id-supplement/
│
├── src/
│   ├── trait_training.py
│   ├── inference_pipeline.py
│   ├── explain_model.py
│
├── configs/
│   ├── train_*.yaml
│
├── requirements.txt
└── README.md
```
Trained model .pkl files are not included in this repository but can be regenerated using the training scripts.

## Computational Environment
All models were trained using:
> Python 3.10

> fastai

> PyTorch

> Albumentations

> scikit-learn

> rembg (for background removal)

> Captum (for attribution visualization)

Package details and requirements are available in *requirements.txt*.

GPU acceleration (CUDA) was used during training but is not required for inference.

## Part I — Training Trait Models

Each morphological trait was trained independently using transfer learning with a ResNet34 backbone.

## Training Procedure
Metadata for this project uses the following formatting: 
### CSV File 1 (Species 1):
```
File Name (without file type) - Trait 1 - Trait 2 - Trait 3 ...
SAMPLE1_00                         0          1        0
SAMPLE1_01                         0          1        0
SAMPLE2_00                         1          1        0
```
### CSV File 2 (Species 2):
```
File Name (without file type) - Trait 1 - Trait 2 - Trait 3 ...
SAMPLE3_00                         1          0        1
SAMPLE3_01                         1          0        1
SAMPLE4_00                         0          1        0
```
Where 0 is the absence of a trait and 1 is the presence of the trait. 



The pipeline for the trait models achieves the following: 
Metadata from both species were merged.
Image IDs were pruned to match available image files.
Albumentations was applied for data augmentation.

### Models were trained in two stages:
Frozen backbone
Unfrozen fine-tuning
Models were exported using learn.export().

_Example_

```
python src/trait_training.py --config configs/train_config_template.yaml
```

**Each configuration file specifies:**
Image directory
Metadata paths
Trait column
Image size
Epochs
Output model name

To reproduce the full ensemble, repeat training for each trait described in the manuscript.

## Part II — Ensemble Inference Pipeline

The inference pipeline performs:
Background removal using rembg
Trait-level model predictions
Per-trait majority voting (mode)
Rule-based ensemble scoring
Final species and sex classification

### Running Inference
python src/inference_pipeline.py \
    --model_dir path/to/models \
    --input_dir path/to/specimen_folder

**The input directory must contain subfolders corresponding to body regions, or location to traits being trained:**
```
input_dir/
├── head/
├── wing/
├── back/
├── belly/
└── tail/
```
Each subfolder contains one or more images of that morphological region.

### Ensemble Scoring Method

For each trait model:
Predictions across all images of a region are aggregated.
The modal class is selected.
If both classes occur and mode = 0, class 1 is favored (tie-breaking rule described in manuscript).

**Final species classification example:**
```
Final Score = sum of trait-level binary outputs

≤ 4 → Mexican Duck

≥ 8 → Mallard

Otherwise → Unknown / Intermediate
```
Sex classification is computed independently using a dedicated model.

## Output

The pipeline produces:
Final species classification
Final trait score
Sex classification
Mean confidence score
Per-trait modal predictions
Per-image predictions

_Example:_
```
Classification: MALLARD
Final Score: 9
Sex: MALE
Confidence: 92%
Explainability Analyses
```

**Attribution maps were generated using Captum:**
Integrated Gradients
Occlusion sensitivity

These analyses validate morphological feature localization.

**To generate attribution maps:**
```
python src/explain_model.py --model path/to/model.pkl --image path/to/image.jpg
```
## Reproducibility Notes

All randomness during training was controlled using default fastai seed behavior.

Validation splits were generated using valid_pct.

Background removal was applied consistently during inference.

Exact hyperparameters for each trait model are provided in the configuration files.

## Data Availability

Due to data usage restrictions, training images are not included in this repository.
Metadata structure is described in the manuscript and configuration examples.

## Contact

For questions regarding implementation details or reproduction of results, please contact:

Dr. Sara Gonzalez

University of Texas at El Paso

sgonzalez28@miners.utep.edu
