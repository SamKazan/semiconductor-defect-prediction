# Semiconductor Manufacturing Defect Prediction

## Overview
In this project, we aim to develop machine learning models to predict defects in semiconductor manufacturing processes. Semiconductor manufacturing involves complex processes where defects can have significant impacts on product quality and yield. By accurately predicting defects, manufacturers can implement proactive measures to improve production efficiency and reduce costs.

## Dataset
- **Source**: The dataset used in this project was obtained from https://www.kaggle.com/datasets/paresh2047/uci-semcom/data.
- **Features**: The dataset contains various process parameters and measurements from semiconductor manufacturing processes.
- **Target Variable**: The target variable indicates whether a product was defective or not.
- **Data Preprocessing**: Preprocessing steps include handling missing values, removing features with zero variance, and scaling numerical features.

## Exploratory Data Analysis (EDA)
Conducted exploratory data analysis to gain insights into the distribution of features and the relationship between features and the target variable. Visualized distributions, correlations, and class imbalances in the dataset using plots and charts.

## Model Development
### Gaussian Naive Bayes (GNB)
Implemented Gaussian Naive Bayes classifier for defect prediction. Conducted feature selection using stepwise selection to identify the most relevant features. Evaluated model performance using metrics such as accuracy, precision, recall, and F1-score. Explored cross-validation to assess model generalization.

### K-Nearest Neighbors (KNN)
Utilized K-Nearest Neighbors algorithm for defect prediction. Tuned hyperparameters such as the number of neighbors (k) to optimize model performance. Evaluated model performance using cross-validation and compared results with GNB.

## Results and Evaluation
Compared the performance of GNB and KNN models based on metrics such as accuracy, precision, recall, AUC and F1-score. Analyzed confusion matrices to understand the strengths and weaknesses of each model. Provided insights into the bias-variance trade-off observed in the models and their implications for real-world applications.

## Conclusion and Future Work
Summarized key findings and insights from the project.
Discussed potential areas for further improvement, such as feature engineering, model selection, and ensemble methods.
Outlined future directions for research and development in semiconductor defect prediction.

## Technologies Used
- Python
- Scikit-learn
- Pandas
- Matplotlib
- Seaborn
- MLxtend
