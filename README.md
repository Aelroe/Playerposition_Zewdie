![UTA-DataScience-Logo](UTA-DataScience-Logo.png)

# Player Position Classification

This repository presents a machine learning solution for classifying soccer players into one of four primary position groups (Goalkeeper, Defender, Midfielder, Forward) based on their skill attributes.

## Overview

The objective is to build a supervised classification model that predicts a player's position group based on numerical performance features. The dataset was cleaned and processed by handling missing values, encoding categorical variables, and standardizing numerical features. Feature importance was assessed, and two models were compared: Random Forest and Logistic Regression. Evaluation was performed using accuracy, confusion matrices, and classification reports. Random Forest achieved the highest overall accuracy.

## Summary of Workdone

### Data

* **Type**: CSV file of player features; output is position group (Defender, Midfielder, Forward, Goalkeeper).
* **Size**: \~18,000 players and 84 features.
* **Split**: 60% Training, 20% Validation, 20% Testing (\~10,800 train / \~3,600 val / \~3,600 test)

#### Preprocessing / Clean Up

* Missing values were removed.
* Unnecessary and redundant columns were dropped.
* Categorical variables were encoded (e.g., preferred foot, work rate).
* Numerical features were scaled for the Logistic Regression model.

#### Data Visualization

* Distribution of players by position group was plotted.
* Top 10 important features identified by Random Forest were visualized.
* A correlation heatmap showed redundancy among top features.

### Problem Formulation

* **Input**: Selected numerical features representing technical, physical, and mental skills.
* **Output**: Predicted player position group.
* **Models**:

  * Random Forest (default scikit-learn hyperparameters)
  * Logistic Regression (scaled inputs, `max_iter=1000`)
* **Metrics**: Accuracy, precision, recall, f1-score, and confusion matrix.

### Training

* **Software**: Python with pandas, scikit-learn, seaborn, matplotlib.
* **Hardware**: Local development on Jupyter Notebook (MacBook).
* **Training Duration**: Less than 2 minutes per model.
* Models were evaluated using validation and test splits.

### Performance Comparison

| Model               | Validation Accuracy | Test Accuracy |
| ------------------- | ------------------- | ------------- |
| Random Forest       | 0.89                | 0.88          |
| Logistic Regression | 0.88                | 0.875         |

* Classification reports and confusion matrices were used to assess per-class performance.
* Random Forest achieved perfect classification for Goalkeepers and the highest f1-scores overall.

### Conclusions

Random Forest provided the best generalization and most consistent classification across all position groups. Logistic Regression performed similarly but slightly underperformed on the Forward class.

### Future Work

* Try XGBoost or LightGBM for improved gradient boosting performance.
* Perform more detailed hyperparameter tuning.
* Incorporate additional contextual features such as age, height, or club ranking.

## How to Reproduce Results

### Environment Setup

Install the required packages:

```bash
pip install pandas scikit-learn matplotlib seaborn
```

### Data

The dataset is derived from EA FC / FIFA 23 public datasets. After cleaning, the final file used was `fifa_cleaned_model_data.csv`.

### Training

Run the notebook `Zewdie_PlayerPosition.ipynb` to reproduce the entire preprocessing, modeling, and evaluation pipeline.

### Evaluation

Evaluation results including confusion matrices, classification reports, and heatmaps are visualized and interpreted within the notebook.

## Overview of Files in Repository

* `Zewdie_PlayerPosition.ipynb`: Main notebook with end-to-end modeling and analysis.
* `fifa_cleaned_model_data.csv`: Cleaned and processed dataset used for modeling.
* `random_forest_fifa_model.pkl`: Trained Random Forest model saved for reuse.
* `README.md`: Project overview and instructions (this file).

## Citations

* EA Sports FIFA / Kaggle datasets.
* scikit-learn documentation.
* UTA Data Science course materials.
