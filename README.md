# ![](UTA-DataScience-Logo.png)

# Player Position Classification

This repository contains a supervised machine learning project focused on predicting a soccer player’s primary position (Defender, Midfielder, Forward, Goalkeeper) using skill-based features from a publicly available player dataset.

## Overview

The task is to build a classification model that accurately predicts a player's primary position using a variety of physical, skill-based, and mental attributes. The dataset was cleaned and transformed to prepare it for classification modeling. Two models were tested: Random Forest and Logistic Regression. Model evaluation was conducted using classification reports, confusion matrices, and validation/test accuracy. Both models achieved strong performance, with Random Forest slightly outperforming Logistic Regression in macro-average F1-score.

## Summary of Work Done

### Data

* **Input**: CSV file containing player attributes (physical, mental, technical) and position labels.
* **Output**: Position group label (Defender, Midfielder, Forward, Goalkeeper)
* **Size**: Approximately 17,000 players
* **Train/Test/Validation Split**: 70% training, 15% validation, 15% test (200 players used in each evaluation set)

#### Preprocessing / Clean up

* Removed irrelevant or redundant columns
* Standardized numerical columns
* Encoded categorical columns (e.g., preferred foot)
* Combined original position labels into 4 broader groups for classification

### Data Visualization

* A bar chart displayed the distribution of player positions to reveal class imbalance.
* Feature importance from the Random Forest model identified the top 10 predictive attributes.
* A correlation heatmap of top features was generated to understand multicollinearity.

### Problem Formulation

* **Input**: Processed player features (numeric and categorical)
* **Output**: One of four class labels representing player position
* **Models**: Random Forest Classifier and Logistic Regression
* **Loss Function & Optimization**: Models trained using scikit-learn defaults; no deep learning used, so standard sklearn fit procedures applied.

### Training

* Conducted in Google Colab with scikit-learn, pandas, and seaborn libraries
* Dataset scaled and split using stratified splitting to maintain class balance
* Training was fast due to dataset size and model type
* Hyperparameters kept simple and default for comparison

### Performance Comparison

| Model               | Validation Accuracy | Test Accuracy |
| ------------------- | ------------------- | ------------- |
| Random Forest       | 0.89                | 0.880         |
| Logistic Regression | 0.88                | 0.875         |

* Random Forest achieved a slightly higher macro F1-score and accuracy
* Confusion matrices and classification heatmaps used for interpretability
* Goalkeeper classification was perfect in both models; Forward class was the hardest to classify accurately

### Conclusions

* Random Forest classifier performed slightly better than Logistic Regression
* Technical and defending attributes are most important in determining position
* Class imbalance (fewer Goalkeepers) did not affect performance negatively due to stratified splits

### Future Work

* Explore other models like XGBoost or LightGBM for potential improvements
* Implement hyperparameter tuning (e.g., GridSearchCV)
* Evaluate performance with cross-validation
* Use additional metadata such as club level or country if available

## How to Reproduce Results

To replicate the analysis:

1. Clone the repository
2. Open the `Zewdie_PlayerPosition.ipynb` notebook
3. Run all cells in sequence
4. Ensure required packages are installed in Colab or local environment:

   * `pandas`
   * `scikit-learn`
   * `matplotlib`
   * `seaborn`

## Overview of Files in Repository

* `Zewdie_PlayerPosition.ipynb`: Full notebook with preprocessing, modeling, and evaluation steps
* `fifa_cleaned_model_data.csv`: Final cleaned dataset used for modeling
* `random_forest_fifa_model.pkl`: Saved Random Forest model
* `README.md`: Project summary and documentation (this file)

## Software Setup

* Standard Python 3.10+
* Run using Google Colab or Jupyter Notebook
* Key packages: `pandas`, `sklearn`, `matplotlib`, `seaborn`

## Data

* Dataset originally obtained from a publicly shared Kaggle FIFA dataset
* Preprocessing steps outlined in notebook

## Training

* Run notebook cells from top to bottom
* Model training included both Logistic Regression and Random Forest

### Performance Evaluation

* Confusion matrices and classification reports were generated using scikit-learn
* Heatmaps and charts used to interpret results

## Citations

* FIFA dataset: [Kaggle Dataset](https://www.kaggle.com/datasets)
* scikit-learn documentation
* matplotlib and seaborn for plotting

---

This README is structured for presentation and project reproducibility, based on class submission guidelines.
