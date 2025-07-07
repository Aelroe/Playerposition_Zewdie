![UTA-DataScience-Logo](UTA-DataScience-Logo.png)

# Player Position Classification

This repository presents a machine learning solution for classifying soccer players into one of four primary position groups (Goalkeeper, Defender, Midfielder, Forward) based on their in-game attributes.

## Overview

The objective is to build a supervised classification model that predicts a player's position group using numerical and categorical features representing technical, physical, and mental skills. The dataset was cleaned and processed by removing missing values, encoding categorical features, and standardizing numerical columns. Two classification models were implemented and evaluated — Random Forest and Logistic Regression. Model performance was assessed using accuracy, confusion matrices, and classification reports. Random Forest yielded the highest classification accuracy across all player roles.

## Summary of Work Done

### Data

- **Type**: Tabular CSV dataset with player statistics.
  - **Input**: 83 numerical and categorical features (e.g., short passing, standing tackle, vision, work rate, preferred foot).
  - **Target**: One of four position groups — Goalkeeper, Defender, Midfielder, Forward.
- **Size**: ~18,000 player entries.
- **Split**:
  - Training set: ~10,800 players (60%)
  - Validation set: ~3,600 players (20%)
  - Test set: ~3,600 players (20%)

### Preprocessing / Clean-Up

- Columns with high missing values (e.g., goalkeeper-specific stats for non-GKs) were dropped.
- Categorical columns such as `preferred_foot` and `work_rate` were one-hot encoded.
- Numerical features were scaled using `StandardScaler` for Logistic Regression.
- Final dataset used 84 features with no missing values.

### Data Visualization

**1. Distribution of Position Groups**

![position_distribution](visuals/position_distribution.png)

**2. Top 10 Most Important Features (Random Forest)**

![top10_features](visuals/top10_features.png)

**3. Feature Correlation Heatmap (Among Important Features)**

![correlation_heatmap](visuals/correlation_heatmap.png)

### Problem Formulation

- **Input**: Player performance attributes (e.g., passing accuracy, agility, strength).
- **Output**: Predicted player position group.
- **Models**:
  - **Random Forest Classifier** (default scikit-learn parameters)
  - **Logistic Regression** (with scaled features, `max_iter=1000`)
- **Evaluation Metrics**:
  - Accuracy
  - Precision, Recall, F1-score (per class)
  - Confusion Matrix

### Training

- **Environment**:
  - Python 3, Jupyter Notebook
  - Libraries: `pandas`, `scikit-learn`, `matplotlib`, `seaborn`
- **Hardware**: MacBook, local environment
- **Training Duration**: Under 2 minutes per model

- Models were trained using the training set, validated on the validation set, and evaluated on the test set to ensure generalizability.

### Performance Comparison

| Model               | Validation Accuracy | Test Accuracy |
|---------------------|---------------------|---------------|
| Random Forest       | 0.89                | 0.88          |
| Logistic Regression | 0.88                | 0.875         |

**Confusion Matrix (Random Forest Test Set)**

![confusion_rf](visuals/confusion_matrix_rf.png)

**Classification Report (Random Forest)**

- Goalkeepers: Perfect classification (Precision = 1.0)
- Highest F1-scores overall compared to Logistic Regression
- Midfielders had the most confusion, but overall still high metrics

### Conclusions

Random Forest outperformed Logistic Regression slightly, particularly in classifying Goalkeepers and Forwards. The model generalizes well to unseen data and is robust with minimal tuning. Logistic Regression performed reasonably well, especially for defenders and midfielders.

### Future Work

- Test with advanced models like **XGBoost** or **LightGBM** to improve accuracy further.
- Tune hyperparameters (e.g., `max_depth`, `n_estimators`) to optimize Random Forest performance.
- Explore player-specific or contextual features such as club level, nationality, or transfer value.
- Consider multi-label approaches for players with dual roles (e.g., Midfielder/Defender).

---

## How to Reproduce Results

### Data

The cleaned dataset is embedded within the notebook. It is a subset and processed version of the original public FIFA/EA FC dataset.

### Training and Evaluation

Run `Zewdie_PlayerPosition.ipynb` to reproduce the full pipeline including:

- Data Cleaning and Encoding
- Exploratory Visualizations
- Model Training (Random Forest, Logistic Regression)
- Evaluation using confusion matrices and classification reports

---

## Overview of Files in Repository

- `Zewdie_PlayerPosition.ipynb`: Complete project notebook with all steps and outputs.
- `README.md`: Project overview and instructions (this file).
- `visuals/`: Folder containing visualization PNGs included above.

---

## Citations

- EA Sports / FIFA Dataset via Kaggle  
- scikit-learn Documentation  
- UTA Data Science Course Materials
