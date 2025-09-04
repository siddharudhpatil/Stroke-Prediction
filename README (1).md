# 🧠 Stroke Prediction

This project focuses on predicting the likelihood of a patient suffering a **stroke** using machine learning models trained on healthcare data. The goal is to assist in early detection and prevention by analyzing medical and lifestyle attributes.

## 📂 Dataset
The dataset used is **`healthcare-dataset-stroke-data.csv`**, which contains patient records with features such as:
- Gender
- Age
- Hypertension
- Heart disease
- Marital status
- Work type
- Residence type
- Average glucose level
- BMI
- Smoking status

The target variable is whether the patient has experienced a **stroke (1 = Yes, 0 = No)**.

## ⚙️ Project Workflow
1. **Data Preprocessing**
   - Handling missing values
   - Encoding categorical variables
   - Normalization/Scaling

2. **Exploratory Data Analysis (EDA)**
   - Distribution of attributes
   - Correlation heatmaps
   - Visualization of stroke prevalence by different factors

3. **Model Development**
   - Implemented and compared various models including:
     - Logistic Regression
     - Decision Tree
     - Random Forest
     - Support Vector Machine (SVM)
   - Hyperparameter tuning for optimized results

4. **Model Evaluation**
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - ROC-AUC curve

5. **Explainability**
   - Used **SHAP** (SHapley Additive Explanations) to interpret feature importance.

6. **Deployment**
   - Built a simple **Flask API** to serve the trained model and make predictions on new data.

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- SHAP
- Flask
- Requests, Joblib

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/stroke-prediction.git
   cd stroke-prediction
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook "Stroke Prediction.ipynb"
   ```

4. (Optional) Start the Flask app:
   ```bash
   python app.py
   ```

## 🎯 Objective
To build a machine learning solution that can assist healthcare professionals in predicting the risk of stroke, thereby enabling **timely medical intervention and preventive care**.
