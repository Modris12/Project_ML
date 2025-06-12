# Career Path Prediction:Insights from the Big 5 Personality Test

## Overview

This project aims to predict career categories based on personality trait scores using a variety of machine learning models. The workflow includes data preprocessing, synthetic data generation (using SMOTE and RandomOverSampler), supervised and unsupervised learning, model stacking, and ensemble methods. The notebook is structured to allow for both model evaluation and practical prediction on new data.

## Key Features

- **Data Augmentation:** Uses SMOTE and RandomOverSampler to balance classes and generate synthetic samples for minority career categories.
- **Multiple Models:** Implements RandomForest, KNN, SVM, XGBoost, KMeans (unsupervised), and a meta-model (stacking classifier).
- **Single-Feature Ensemble:** Combines predictions from single-feature RandomForest models for robust classification.
- **Agreement Analysis:** Checks for consensus among models and uses majority voting or meta-model fallback.
- **Visualization:** Includes plots for class distribution and cluster assignments.
- **Extensible:** Modular code structure allows easy addition of new models or features.

## Workflow

1. **Data Preparation:** Cleans and preprocesses the input data, including renaming and dropping columns as needed.
2. **Synthetic Data Generation:** For each career category, creates a binary classification problem (category vs. other), applies oversampling and SMOTE, and generates synthetic samples with feature bounds.
3. **Model Training:** Trains individual classifiers for each category and evaluates their performance.
4. **Synthetic Dataset Assembly:** Combines all synthetic samples into a single DataFrame for downstream modeling.
5. **Unsupervised Learning:** Applies KMeans clustering to the synthetic dataset and maps clusters to careers.
6. **Supervised Learning:** Trains RandomForest, KNN, SVM, and XGBoost models on the synthetic data.
7. **Prediction on Real Data:** Applies trained models to the original dataset and compares predictions.
8. **Ensemble and Meta-Model:** Uses stacking and single-feature ensemble approaches for improved accuracy.
9. **Agreement Analysis:** Identifies rows where models agree or disagree, and applies majority voting or meta-model fallback.
10. **Visualization and Reporting:** Plots class distributions and prints model performance metrics.

## File Structure

- **Jupyter Notebook:** Contains all code, results, and visualizations.
- **README.md:** Project overview and instructions (this file).
- **Model Artifacts:** Trained models can be saved/loaded using joblib for deployment or further analysis.

## Requirements

- Python 3.x
- pandas, numpy, scikit-learn, imbalanced-learn, xgboost, matplotlib, seaborn, joblib

## Usage

1. **Run the Notebook:** Execute all cells in order for a full pipeline run.
2. **Model Evaluation:** Review printed classification reports and accuracy metrics.
3. **Prediction:** Use the provided code to predict careers for new samples.
4. **Visualization:** Inspect plots for class distribution and clustering.

## Customization

- **Feature Engineering:** Update feature bounds and feature lists as needed for your data.
- **Model Tuning:** Adjust hyperparameters for each model for improved performance.
- **New Categories:** Add new career categories by replicating the synthetic data generation and model training steps.

## License

This project is for educational and research purposes. Please cite appropriately if used in academic work.

## Acknowledgements

- scikit-learn, imbalanced-learn, and xgboost for machine learning tools.
- All contributors to open-source data science libraries.
