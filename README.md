# Data-Driven Student Dropout Prediction and Performance Monitoring
A smart solution combining Machine Learning and Business Intelligence to detect students at risk of dropping out and monitor their academic performance. Includes a predictive Streamlit app and a Metabase dashboard for actionable insights.

---

## 📚 Contents
- [🧠 Business Understanding](#-business-understanding)
  - [🔧 Business Problems](#-business-problems)
  - [🔍 Project Scope](#-project-scope)
- [📦 Preparation](#-preparation)
- [📊 Business Dashboard](#-business-dashboard)
  - [🔍 Key Visualizations](#-key-visualizations)
  - [📌 Insight Highlights](#-insight-highlights)
- [🚀 Running Metabase via Docker](#-running-metabase-via-docker)
- [🤖 Machine Learning System](#-machine-learning-system)
  - [🧠 Selected Features](#-selected-features)
  - [🔄 Preprocessing](#-preprocessing)
  - [📈 Evaluation](#-evaluation)
- [🌐 Streamlit Prototype](#-streamlit-prototype)
- [✅ Conclusion](#-conclusion)
  - [🔧 Recommendations](#-recommendations)
- [📎 Appendix - Project Files Overview](#-appendix---project-files-overview)



## 🧠 Business Understanding

Jaya Jaya Institut, a respected higher education institution, faces a significant challenge—high student dropout rates. Many students fail to complete their studies, affecting the institution's academic reputation and financial sustainability.

To address this, the institution leveraged **Machine Learning** and **Business Intelligence** to identify at-risk students early and enable proactive intervention.

### 🔧 Business Problems
- **High Dropout Rates**  
- **No Early Warning System**  
- **Lack of Data-Driven Insight**

### 🔍 Project Scope
- Data collection & exploration  
- ML model development (Random Forest)  
- Metabase dashboard for visualization  
- Streamlit prototype for predictions

---

## 📦 Preparation

**Data Source**:  
[Students' Performance Dataset (Dicoding)](https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/data.csv)

**Environment Setup**:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn streamlit sqlalchemy python-dotenv psycopg2-binary
````

---

## 📊 Business Dashboard
![Metabase Dashboard](./hanugafc-dashboard.png)
**Tool**: Metabase
**Database**: PostgreSQL

### 🔍 Key Visualizations:

1. **Dropout vs Graduate Distribution**
2. **Status by Gender**
3. **Admission Grade vs Status**
4. **Age at Enrollment vs Status**
5. **Scholarship vs Status**
6. **Debtor vs Status**

### 📌 Insight Highlights:

* Lower admission grades and older age → higher dropout risk
* Scholarship holders & non-debtors → higher graduation rates

---

## 🚀 Running Metabase via Docker

```bash
docker pull metabase/metabase:latest
docker run -d -p 3000:3000 \
  -v /path/to/metabase.db.mv.db:/metabase.db/metabase.db.mv.db \
  --name metabase-jaya-jaya-institut \
  metabase/metabase:latest
```

**Access:**
`http://localhost:3000`
**Login:**

* Email: `johndhoe123@mail.com`
* Password: `metabase-root123-john`

> Note: You need to connect the metabase with your own database.
---

## 🤖 Machine Learning System

**Model**: Random Forest Classifier

### 🧠 Selected Features:

* Marital status
* Application order
* Admission grade
* Debtor
* Displaced
* Gender
* Scholarship holder
* Age at enrollment

### 🔄 Preprocessing:

* Label encoding
* SelectKBest for feature selection
* StandardScaler
* SMOTE for class balancing

### 📈 Evaluation:

| Metric    | Dropout | Graduate |
| --------- | ------- | -------- |
| Precision | 0.87    | 0.87     |
| Recall    | 0.87    | 0.87     |
| F1-Score  | 0.87    | 0.87     |

---

## 🌐 Streamlit Prototype
![Streamlit Prototype](./prototype-overview.png)

**Live Demo**: [Streamlit App](https://jayajayainstitut-studentdropout-h7v5lauua6ybavmvped2xd.streamlit.app/)

**Run Locally**:

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## ✅ Conclusion

This project successfully integrates machine learning and business intelligence to combat dropout at Jaya Jaya Institut.

**Deliverables**:

* Predictive ML model with 87% accuracy
* Metabase dashboard with dropout insights
* Streamlit app for real-time prediction

### 🔧 Recommendations:

* Deploy early warning system institution-wide
* Expand scholarship programs
* Provide support for low-grade students
* Offer flexible learning for older students
* Launch financial counseling to help debtor students

---

## 📎 Appendix - Project Files Overview

| **Filename**             | **Description**                                                                                |
|--------------------------|------------------------------------------------------------------------------------------------|
| `.env.example`           | A template for environment variables, useful for setting up secrets like database credentials. |
| `.gitignore`             | Specifies which files/folders should be ignored by Git (e.g., virtual envs, cache files).      |
| `documentation.md`       | Main documentation file describing project objectives, setup, and results in detail.           |
| `app.py`                 | Streamlit script to run the interactive prediction web application.                            |
| `cleaned_data.csv`       | Preprocessed dataset used for training and evaluation.                                         |
| `confusion-matrix.png`   | Visualization showing model performance through a confusion matrix.                            |
| `features.pkl`           | Pickled file containing the list of selected features used in the final model.                 |
| `hanugafc-dashboard.png` | Screenshot of the Metabase dashboard for visual dropout analysis.                              |
| `label_encoder.pkl`      | Stores the label encoder used to convert target classes into numeric values.                   |
| `metabase.db.mv.db`      | Internal H2 database file used by Metabase to store questions, dashboards, and metadata.       |
| `model.pkl`              | The trained machine learning model (Random Forest) used for dropout prediction.                |
| `notebook.ipynb`         | Jupyter Notebook containing full analysis: EDA, modeling, feature selection, and evaluation.   |
| `prototype-overview.png` | Screenshot of the Streamlit prototype interface.                                               |
| `requirements.txt`       | List of Python dependencies needed to run the project locally.                                 |
| `scaler.pkl`             | StandardScaler object used to normalize numerical input features before prediction.            |