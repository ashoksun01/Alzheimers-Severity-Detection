# Detecting Alzheimer’s Severity in MRI Scans

## Project Summary
This project aims to detect and classify the severity of Alzheimer’s disease using machine learning models on MRI brain scans. Given that Alzheimer’s is a neurodegenerative disease affecting millions, early and accurate diagnosis can significantly impact patient outcomes. Our approach involves building, training, and evaluating multiple machine learning models on two distinct datasets, leveraging both 2D and 3D image data.

- **Problem:** Accurate diagnosis of Alzheimer's severity is challenging and often requires expert interpretation of MRI scans.
- **Solution:** Our models classify MRI scans into four severity levels of Alzheimer's: non-demented, very mildly demented, mildly demented, and moderately demented. We experimented with 21 models, including Logistic Regression, Decision Trees, Random Forests, Convolutional Neural Networks (CNNs), and Transfer Learning models.
- **Target Users:** Radiologists, neurologists, and healthcare providers seeking automated support in Alzheimer's diagnosis.

---

## Repository Structure

- **`dataset1_models.ipynb`**: Jupyter notebook containing 11 models trained on the first dataset (2D MRI scans).
- **`dataset2_2d_models.ipynb`**: Jupyter notebook containing 6 models trained on the second dataset’s 2D pipeline (binary and multiclass classification).
- **`dataset2_3d_augmented_models.ipynb`**: Jupyter notebook containing models for the 3D pipeline of the second dataset (augmented and class-merged).
- **`dataset2_3d_models.ipynb`**: Jupyter notebook containing models for the 3D pipeline of the second dataset (3D brain volumes).
- **`alzheimers_severity_detection_paper.pdf`**: Research paper detailing the project’s objective, data pre-processing, model development, validation, and evaluation.
- **`alzheimers_severity_detection_slides.pdf`**: Presentation slides summarizing the project.

### What This Repository Contains
- **Data Preprocessing and Augmentation:** Pipelines for resizing, normalizing, and augmenting MRI images.
- **Model Training and Evaluation:** Various machine learning models, including CNNs and transfer learning, trained on 2D and 3D image data.
- **Research Paper and Slides:** A detailed report and visual presentation of the project.

### What This Repository Does Not Contain
- **Raw Image Data:** The MRI datasets are not included for privacy and storage reasons.

---

## How It Works
1. **Data Loading:** MRI images are loaded from two datasets with distinct structures:
   - Dataset 1: 6,400 preprocessed 128x128 pixel 2D brain scan images.
   - Dataset 2: 86,000 unprocessed 244x488 pixel images, which are further processed into 2D and 3D pipelines.
2. **Data Preprocessing:** The images are resized, normalized, and augmented (including 3D stacking for the second dataset).
3. **Model Training:** 21 models are trained and optimized across three pipelines:
   - 2D Pipeline (Dataset 1)
   - 2D Pipeline (Dataset 2)
   - 3D Pipeline (Dataset 2)
4. **Model Evaluation:** Performance is assessed using accuracy.
   
---

## Data Source

The image data used in this project came from the following two sources:

- **Alzheimer's MRI Pre-processed Dataset**  
  - [Kaggle: Alzheimer's MRI Dataset](https://www.kaggle.com/datasets/sachinkumar413/alzheimer-mri-dataset)  
  - Contains 6,400 preprocessed MRI images classified into four classes: "Mild Demented," "Moderate Demented," "Non-Demented," and "Very Mild Demented."

- **OASIS Alzheimer's Detection Dataset**  
  - [Kaggle: OASIS Alzheimer's Detection](https://www.kaggle.com/datasets/ninadaithal/imagesoasis)  
  - Contains 86,000 unprocessed MRI images from 461 patients, classified into the same four categories.

### Data Cleaning & Preparation
- For Dataset 1:
  - Images were balanced using augmentation until classes were evenly distributed.
  - Images were resized and normalized for model input.
- For Dataset 2:
  - Two pipelines were created:
    - **2D Pipeline:** Images were resized, normalized, and class-balanced using undersampling.
    - **3D Pipeline:** Images were stacked to form 3D brain volumes, and data augmentation was applied to mitigate class imbalance.

---

## Model Performance
Our best-performing models achieved the following metrics:

| Model                    | Dataset | Type         | Accuracy |
|--------------------------|----------|---------------|-----------|
| Logistic Regression       | Dataset 1 | Multiclass    | 86.09%   
| Improved CNN (2D)         | Dataset 1 | Multiclass    | 92.97%   
| Improved VGG16 (2D)       | Dataset 2 | Binary        | 98.62%   
| CNN (3D Pipeline)         | Dataset 2 | Binary        | 76.47%    

---

## Key Learnings & Impact
- **2D vs. 3D Pipelines:** 2D models consistently outperformed 3D models due to class balancing and computational efficiency.
- **Model Selection:** CNNs and Transfer Learning models (VGG16) demonstrated superior performance, with VGG16 providing the highest test accuracy.
- **Data Augmentation:** Augmenting data helped balance classes, significantly improving model performance.

---

## Future Improvements
- **Advanced 3D Modeling:** Refine the 3D pipeline with more advanced 3D architectures (e.g., 3D CNNs, Vision Transformers).
- **Enhanced Preprocessing:** Implement more robust data augmentation techniques.
- **Deployment:** Develop an API for easy input of MRI scans and real-time inference.
