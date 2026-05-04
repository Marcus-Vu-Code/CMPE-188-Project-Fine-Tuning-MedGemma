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
- Final evaluation dataset: CRC-VAL-HE-7K

## Platform / Development Environment
This project was developed and run in Google Colab using a GPU runtime.

## Planned Model / System Approach
1. Establish a strong benchmark with a ViT image-classification baseline.
2. Fine-tune MedGemma with QLoRA/PEFT for parameter-efficient adaptation.
3. Compare baseline vs fine-tuned model performance.
4. Deliver a clinician-facing upload demo with prediction confidence scores.

## Current Implementation Progress
Notebook files:
- `clinical_decision_support_baseline_notebook (2).ipynb`
- `clinical_decision_support_full_project_medgemmaV2.ipynb`
- `improved_medgemma_qlora_histopathology_colabV3.ipynb`

Completed in the project:
1. Installed required ML libraries in Colab (`transformers`, `datasets`, `peft`, `trl`, `bitsandbytes`, `torchvision`, `gradio`, `scikit-learn`, `evaluate`).
2. Verified compute environment and GPU availability.
3. Prepared dataset directories and loaded NCT-CRC-HE-100K using Hugging Face `imagefolder`.
4. Created train/test split (80/20) with fixed seed for reproducibility.
5. Explored class labels and visualized sample images.
6. Trained a ViT baseline model (`google/vit-base-patch16-224`) for 9-class classification.
7. Fine-tuned MedGemma with QLoRA/PEFT and saved the resulting adapter artifacts.
8. Evaluated both models on the CRC-VAL-HE-7K validation set.
9. Generated prediction outputs, confusion matrices, and classification reports for both models.
10. Saved baseline and MedGemma model artifacts.
11. Built single-image inference helpers and a lightweight Gradio demo interface for image upload and confidence-score display.

## V3 Improvements Over V2
The v3 notebook (`improved_medgemma_qlora_histopathology_colabV3.ipynb`) adds an improved MedGemma training/evaluation pipeline on top of v2:

1. Stratified dataset splitting to preserve class representation in train/validation/test.
2. Balanced per-class training subset with oversampling for underrepresented classes.
3. Histopathology-friendly image augmentation (flip/rotation/color/contrast/brightness and mild blur).
4. Stronger QLoRA adapter configuration (higher-rank LoRA and expanded target modules, including MLP blocks).
5. Assistant-only loss masking so the model learns the output label instead of copying prompt text.
6. More robust label extraction/parsing during evaluation.
7. Added balanced-accuracy and macro-F1 tracking, plus invalid-prediction ratio.
8. Added row-normalized confusion matrix and error-analysis visualization for misclassifications.

These changes were introduced specifically to reduce class-collapse behavior observed in earlier runs.

## Current Status
- Baseline and MedGemma QLoRA fine-tuning workflows are both implemented.
- Final comparison results are available under `MedGemma_Colorectal_Project/results/final_results/` and include `metrics.json`, `comparison_summary.csv`, `predictions.csv`, and the saved reports.
- V3 improved training workflow is implemented, but full v3 end-to-end reruns were limited by Colab resource constraints.
- Next step is packaging the demo and documenting deployment or sharing instructions if needed.

## Compute Limitation Note
The project hit Google Colab compute-unit limits during v3 experimentation. As a result, some long v3 reruns/evaluations may be partial or pending until additional compute is available.

## Final Comparison
| Model | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| ViT | 0.1155 | 0.1327 | 0.1155 | 0.1192 |
| MedGemma | 0.1228 | 0.1408 | 0.1228 | 0.1269 |

MedGemma slightly outperformed the ViT baseline across all reported metrics in the saved results.

## How to Run
Open and run the relevant notebook in Google Colab:
1. Use `clinical_decision_support_baseline_notebook (2).ipynb` for the baseline-only workflow.
2. Use `clinical_decision_support_full_project_medgemmaV2.ipynb` for the full baseline + MedGemma workflow.
3. Connect or upload the NCT-CRC-HE-100K dataset path.
4. Execute the cells in order from dependency installation to evaluation and demo.
5. Optionally enable `demo.launch(share=True)` in the Gradio section to launch a shareable interface.

## Project Artifacts
- Baseline model artifacts: `MedGemma_Colorectal_Project/models/vit-baseline/`
- MedGemma adapter artifacts: `MedGemma_Colorectal_Project/models/medgemma-crc-qlora/`
- Baseline outputs: `MedGemma_Colorectal_Project/results/baseline/`
- Final comparison outputs: `MedGemma_Colorectal_Project/results/final_results/`
- Notebook-backed backups and checkpoints: `MedGemma_Colorectal_Project/outputs/`

## References
- MedGemma Official Notebooks: Google Health MedGemma GitHub
- PEFT: https://github.com/huggingface/peft
- TRL: https://github.com/huggingface/trl
- Dataset source: https://zenodo.org/records/1214456
