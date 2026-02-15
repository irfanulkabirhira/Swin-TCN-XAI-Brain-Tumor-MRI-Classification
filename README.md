# Swin-TCN-XAI-Brain-Tumor-MRI-Classification
Official implementation of "Swin-TCN-XAI: A Hybrid Transformer-Convolutional Framework for Explainable Brain Tumor MRI Classification". This repository includes model training, evaluation, Grad-CAM, SHAP, and LIME explainability, and a reproducible experimental pipeline.

## Repository Structure

```text
Swin-TCN-XAI-Brain-Tumor-MRI-Classification/
│
├── README.md
├── requirements.txt
├── train.py
├── test.py
│
├── model/
│   ├── swin_tcn_model.py
│   └── tcn.py
│
├── explainability/
│   ├── gradcam.py
│   ├── shap_analysis.py
│   └── lime_analysis.py
│
├── preprocessing/
│   ├── preprocessing.py
│   └── augmentation.py
│
├── notebooks/
│   ├── training.ipynb
│   └── shap_analysis.ipynb
│
├── results/
│   ├── confusion_matrix.png
│   └── gradcam_results.png
│
└── LICENSE

# Swin-TCN-XAI: Explainable Brain Tumor MRI Classification

Official implementation of the paper:

"Swin-TCN-XAI: A Hybrid Transformer-Convolutional Framework for Explainable Brain Tumor MRI Classification"

## Overview

This repository provides the implementation of a hybrid Swin Transformer and Temporal Convolutional Network (TCN) architecture for accurate and explainable brain tumor MRI classification.

The framework integrates:

- Swin Transformer for hierarchical spatial feature extraction
- Temporal Convolutional Network (TCN) for inter-slice dependency modeling
- Grad-CAM for spatial explainability
- SHAP for global feature attribution
- LIME for instance-level explainability

## Dataset

Brain Tumor MRI Dataset available at:

https://data.mendeley.com/datasets/zwr4ntf94j/5

Classes:
- Glioma
- Meningioma
- Pituitary tumor
- No tumor

## Model Architecture

Swin Transformer → TCN → Classification Head → Explainability Module

## Requirements

Python 3.9+

Install dependencies:

```bash
pip install -r requirements.txt
