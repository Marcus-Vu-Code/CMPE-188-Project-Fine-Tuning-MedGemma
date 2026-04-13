# Clinical Decision Support System via Parameter-Efficient Fine-Tuning of MedGemma

## Team Members
- Marcus Vu (marcus.vu@sjsu.edu)

## Project Title
Clinical Decision Support System for Colorectal Tissue Classification via Parameter-Efficient Fine-Tuning (QLoRA) of MedGemma

## Problem Statement
Histopathology interpretation is time-intensive and requires expert review. This project aims to support clinical decision-making by classifying colorectal tissue image patches into diagnostically relevant classes, helping identify suspicious or malignant regions faster and more consistently.

## Dataset / Data Source
- Primary dataset: NCT-CRC-HE-100K
- Source: https://zenodo.org/records/1214456
- Size: 100,000 RGB histology image patches (224x224)
- Classes (9): Adipose, Background, Debris, Lymphocytes, Mucus, Smooth Muscle, Normal Colon Mucosa, Cancer-Associated Stroma, Colorectal Adenocarcinoma Epithelium
- Labels: integer class IDs for supervised classification

## Platform / Development Environment
This baseline implementation was developed and run in Google Colab using GPU runtime.

## Planned Model / System Approach
1. Establish a strong benchmark with a ViT image-classification baseline.
2. Migrate to MedGemma and apply QLoRA/PEFT for efficient fine-tuning.
3. Compare baseline vs fine-tuned model performance.
4. Deliver a clinician-facing upload demo with prediction confidence scores.

## Current Implementation Progress (Based on Notebook)
Notebook file: `clinical_decision_support_baseline_notebook (2).ipynb`

Completed in the notebook:
1. Installed required ML libraries in Colab (`transformers`, `datasets`, `peft`, `trl`, `bitsandbytes`, `torchvision`, `gradio`, `scikit-learn`, `evaluate`).
2. Verified compute environment and GPU availability.
3. Prepared dataset directories and loaded NCT-CRC-HE-100K using Hugging Face `imagefolder`.
4. Created train/test split (80/20) with fixed seed for reproducibility.
5. Explored class labels and visualized sample images.
6. Loaded ViT baseline model (`google/vit-base-patch16-224`) for 9-class classification.
7. Implemented preprocessing transforms and training pipeline with Hugging Face `Trainer`.
8. Trained baseline model and evaluated accuracy.
9. Generated prediction outputs, confusion matrix, and classification report.
10. Saved trained baseline model artifacts.
11. Built single-image inference helper.
12. Built a lightweight Gradio demo interface for image upload and confidence-score display.

## Current Status
- End-to-end baseline pipeline is implemented in the notebook.
- Baseline benchmark and demo components are in place.
- Next step is MedGemma + QLoRA fine-tuning and metric comparison against this baseline.

## How to Run
Open and run all cells in Google Colab:
1. Upload or connect dataset path for NCT-CRC-HE-100K.
2. Execute cells in order from dependency installation to evaluation/demo.
3. Optionally uncomment `demo.launch(share=True)` in the Gradio section to launch a shareable demo.

## References
- MedGemma Official Notebooks: Google Health - MedGemma GitHub
- PEFT: https://github.com/huggingface/peft
- TRL: https://github.com/huggingface/trl
- Dataset source: https://zenodo.org/records/1214456
