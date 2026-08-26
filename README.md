# Predicting Persistent Fever in Patients with Cancer  
*A Machine Learning Approach Using Electronic Health Records*

## Overview
This repository contains the code used for the study **“Predicting Persistent Fever in Patients with Cancer Receiving Antibiotics: A Machine Learning Approach.”**  
The project develops and externally validates machine-learning models to predict whether hospitalized patients with cancer will experience **persistent fever 48–72 hours after initiation of intravenous antibiotic therapy**, a clinically important reassessment time point that frequently determines antimicrobial escalation and diagnostic imaging.

The overarching aim is to evaluate whether routinely collected electronic health record (EHR) data can support **more targeted reassessment decisions**, with potential implications for antimicrobial stewardship and reduction of unnecessary diagnostic procedures in immunocompromised patients.

---

## Study Summary
- **Population:** Adult hospitalized patients with hematologic and solid malignancies receiving IV antibiotics  
- **Outcome:** Persistent fever at 48–72 hours after antibiotic initiation  
- **Data sources:**  
  - University Hospital Essen (Germany): model development and internal validation  
  - Brigham and Women’s Hospital (USA): external validation  
- **Methods:**  
  - Feature engineering from demographics, comorbidities, laboratory values, vital signs, and temperature trajectories  
  - Machine-learning models including gradient boosting, random forest, regularized logistic regression, and tabular foundation models  
  - Hybrid modeling incorporating short-term temperature forecasting  
  - Nested cross-validation, external validation, calibration assessment, and decision curve analysis  

---

## Repository Structure
```text
├── data-processing/          # Feature extraction and preprocessing pipelines
├── training-validation/      # Model training and evaluation
└── requirements.txt          # Python dependencies
```


> **Note:** Patient-level data are not included in this repository.

---

## Data Availability
Due to legal and ethical restrictions, **raw patient-level data cannot be shared publicly**.  
A **pseudo-anonymized dataset** may be made available for validation purposes upon reasonable request and subject to institutional approvals.

For data access inquiries, please contact the corresponding author.

---

## Reproducibility
This repository reflects the complete analytical pipeline used in the manuscript, including:
- Feature engineering  
- Model training and hyperparameter optimization  
- Internal and external validation  
- Decision curve analysis  

All analyses were performed in **Python 3.11**. Core dependencies include:
- `scikit-learn`
- `xgboost`
- `catboost`
- `chronos-forecasting`
- `tabpfn-extensions`
- `torch`
- `shap`
- `optuna`

Exact package versions are specified in `requirements.txt`.

---

## Intended Use
This code is provided **for research and reproducibility purposes only**.  
The models are **not intended for direct clinical deployment** without prospective validation, local recalibration, and appropriate clinical governance.

---

## Citation
If you use this code, please cite the corresponding manuscript:

> Pucher G, et al. *Predicting Persistent Fever in Patients with Cancer Receiving Antibiotics: A Machine Learning Approach.* (in review)


---

## Contact
For questions regarding the code or the study:

**Christopher M. Sauer, MD**  
Laboratory for Clinical Research and Real-World Evidence  
Department of Hematology & Stem Cell Transplantation  
University Hospital Essen, Germany  
📧 christopher.sauer@uk-essen.de

**Gernot Pucher**  
Laboratory for Clinical Research and Real-World Evidence  
Department of Hematology & Stem Cell Transplantation  
University Hospital Essen, Germany  
📧 gernot.pucher@uk-essen.de
