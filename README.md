#  Cervical Cancer Risk Analysis — Machine Learning Web App

This project builds a **machine learning-powered web application** for predicting **cervical cancer risk** based on personal health and lifestyle factors.  
It is designed to **assist awareness and research**, not to replace professional medical diagnosis.

 **Live Web App**: [Cervical Cancer Risk Prediction](https://cervicalcancerriskanalysis-pkcmcgg4lbyhsdgmdee7hd.streamlit.app/)  
 **GitHub Repository**: [Cervical_Cancer_Risk_Analysis](https://github.com/Sanjai-Chippukutty/Cervical_Cancer_Risk_Analysis)

---

##  Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Data Preprocessing](#data-preprocessing)
4. [Feature Selection](#feature-selection)
5. [Model Training](#model-training)
6. [Model Evaluation](#model-evaluation)
7. [Web App Development](#web-app-development)
8. [Deployment on Streamlit](#deployment-on-streamlit)
9. [Installation & Usage](#installation--usage)
10. [Technologies Used](#technologies-used)
11. [Results & Insights](#results--insights)

---

## 1️ Overview <a name="overview"></a>
Cervical cancer is a preventable disease if detected early.  
This project aims to:
- Predict **risk level** of cervical cancer using **ML algorithms**.
- Provide an **interactive web-based tool** for risk awareness.
- Demonstrate **data science workflows** in healthcare applications.

---

## 2️⃣ Dataset <a name="dataset"></a>
We used a publicly available cervical cancer dataset from the **UCI Machine Learning Repository**.

**Key Dataset Information:**
- Contains **biomedical, demographic, and lifestyle variables**.
- Target variable: **Biopsy** (1 = cancer detected, 0 = no cancer).
- Missing values handled with imputation and data cleaning.

---

## 3️⃣ Data Preprocessing <a name="data-preprocessing"></a>
Steps taken:
1. **Load the dataset** and inspect basic statistics.
2. **Handle missing values** (`?` replaced with `NaN`, imputation applied).
3. **Convert categorical values** (e.g., Yes/No → 1/0).
4. **Scale numeric features** using `StandardScaler`.

---

## 4️⃣ Feature Selection <a name="feature-selection"></a>
We identified the Top 10 most important features using:

**Random Forest feature importance**

**Model interpretability methods**

**Selected Features:**

1.Age

2.Number of sexual partners

3.Age at first sexual intercourse

4.Number of pregnancies

5.Smokes (packs/year)

6.Years smoked

7.Hormonal contraceptives (years)

8.IUD use (years)

9.Number of STDs

10.STD: HPV (Yes/No)

---

## 5️⃣ Model Training <a name="model-training"></a>
We trained a Random Forest Classifier:
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)
Chosen for its high accuracy and robustness with tabular data.

---

## 6️⃣ Model Evaluation <a name="model-evaluation"></a>
Metrics used:

1.Accuracy

2.Precision

3.Recall

4.ROC-AUC

Example:
from sklearn.metrics import classification_report, roc_auc_score
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))

---

## 7️⃣ Web App Development <a name="web-app-development"></a>
We built a Streamlit app that:

Takes user input through form fields.

Scales the input using the trained scaler.

Uses the trained Random Forest model to predict risk.

Displays probability-based results.

Core app snippet:
if prediction == 1:
    st.error(f"High Risk — Probability: {probability:.2%}")
else:
    st.success(f"Low Risk — Probability: {probability:.2%}")

---

## 9️⃣ Installation & Usage <a name="installation--usage"></a>
Local Setup:
git clone https://github.com/Sanjai-Chippukutty/Cervical_Cancer_Risk_Analysis.git
cd Cervical_Cancer_Risk_Analysis/streamlit_app
pip install -r requirements.txt
streamlit run app.py
Then open: http://localhost:8501


---

## 🔟 Technologies Used <a name="technologies-used"></a>
Python (Pandas, NumPy, Scikit-learn)

Streamlit (Web app)

Joblib (Model persistence)

Matplotlib, Seaborn (EDA & visualization)

GitHub (Version control & hosting)

Streamlit Cloud (Deployment)

---

## 1️⃣1️⃣ Results & Insights <a name="results--insights"></a>
Top 10 features improve model interpretability.

Model is highly sensitive, reducing false negatives.

Web deployment makes it accessible without coding skills.






    




