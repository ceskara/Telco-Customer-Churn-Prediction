# Telco Customer Churn Prediction

This project aims to predict customer churn using a supervised machine learning approach, specifically a **Random Forest Classifier**, applied to the popular Telco Customer Churn dataset.

## 📊 Dataset

The dataset contains information about a telecom company's customers and whether or not they have churned. It includes customer demographic info, services they’ve subscribed to, and billing/payment information.

- 📌 **Source on Kaggle**: [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data)  
- 🏢 **Originally published by IBM**: [IBM Community Blog](https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113)

## 🧠 Project Overview

The key steps in this project are:

1. **Data Preprocessing**  
   - Handling missing values  
   - Encoding categorical variables  
   - Feature scaling

2. **Modeling with Random Forest**  
   - Training a baseline model  
   - Evaluating with confusion matrix, classification report, and ROC-AUC

3. **Hyperparameter Optimization**  
   - Grid search over multiple parameters  
   - Best model selection

4. **Feature Importance Analysis**  
   - Visualizing top features influencing churn

## 📁 Repository Contents

- `churn_rf_model.py`: Full pipeline in a Python script  
- `churn_analysis.ipynb`: Interactive Jupyter notebook with exploratory analysis and model implementation  
- `WA_Fn-UseC_-Telco-Customer-Churn.csv`: The dataset file  
- `README.md`: This file

## ▶️ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/ceskara/Telco-Customer-Churn-Prediction.git
   cd Telco-Customer-Churn-Prediction

2. Install Dependencies:
   pip install pandas numpy matplotlib seaborn scikit-learn

3. Run the Python script:
   python churn_rf_model.py

   
Or open the churn_analysis.ipynb notebook in JupyterLab or VS Code for an interactive experience.

📈 Sample Outputs
Confusion Matrix and Classification Report

ROC-AUC Score

Top 15 Most Important Features Visualization

🧑‍💻 Author

https://github.com/ceskara
