# Reader Sentiment Classifier for Book Reviews 📚🧠

This project builds a machine learning pipeline to classify book reviews as positive or negative, simulating a sentiment detection engine for book recommendation platforms.

## 📌 Project Overview

Originally developed in 2023 and refined for public release in **June 2025**, this project leverages **natural language processing (NLP)** and **logistic regression** to evaluate 1,973 real book reviews. The final model achieved **AUC scores up to 0.89**.

## 🔍 Problem Statement

How can we automatically detect reader sentiment in text reviews to improve book recommendation systems and content moderation tools?

## 💡 Solution

- Applied **TF-IDF vectorization** with n-gram feature engineering to extract signals from review text
- Trained a **logistic regression model** using Scikit-learn
- Evaluated model performance using **ROC-AUC**, and runtime analysis
- Tuned document frequency thresholds (`min_df`) to optimize the tradeoff between feature space size and classification performance

## 📈 Key Results

- **AUC Score:** Up to 0.92 on test data
- **Data:** 1,973 labeled book reviews (binary: positive/negative)
- **Vectorizer:** TF-IDF with n-gram range (1,2)
- **Classifier:** Logistic Regression with `max_iter=200`

## 🧰 Technologies Used

- **Python**, **Pandas**, **NumPy**
- **Scikit-learn** (TF-IDF, LogisticRegression, GridSearchCV, roc_auc_score)
- **Matplotlib**, **Seaborn**

## 📊 Visualizations

Model performance was visualized using ROC curves. ![ROC_curve](https://github.com/user-attachments/assets/3ae53e58-d28b-4997-826b-69f3c7d74cd4)


## 📁 Repository Structure

- `ImplementMLProjectPlan.ipynb` – End-to-end Jupyter notebook
- `bookReviewsData.csv` – Input dataset (text + sentiment labels)
- `README.md` – Project overview

## 🚀 Next Steps

- Deploy as a **Streamlit app** to allow live sentiment prediction
- Compare performance with transformer models like **DistilBERT**
- Use SHAP or LIME for interpretability and trust

## 👤 Author

**Noah Dufresne**  
📍 Brooklyn, NY | 🎓 Fordham University  
[LinkedIn](https://linkedin.com/in/noahdufresne)

---

*This project was originally created in 2023 and finalized for public release in June 2025.*
