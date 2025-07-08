# Appointment No-Show Predictor

This project builds a high-accuracy machine learning model to predict which patients are likely to **miss their medical appointments**. It uses appointment history, temporal patterns, and patient demographics to provide actionable intervention recommendations.

---

## 🔍 Problem Statement

Missed appointments result in wasted resources and delayed care. This system predicts high-risk no-shows and recommends strategies like reminders or transport aid to improve attendance rates.

---

## 🚀 Features

- ✅ Preprocesses & engineers features from raw appointment data  
- ✅ Handles class imbalance using SMOTE  
- ✅ Stacking Ensemble Model: XGBoost, Random Forest, Gradient Boosting + Logistic Regression  
- ✅ Achieves high prediction accuracy (~90%+)  
- ✅ Generates risk scores and intervention suggestions  

---

## 📊 Dataset

Expected file: `appointments.csv`  
**Required Columns:**
- `ScheduledDay`
- `AppointmentDay`
- `No-show` (`Yes`/`No`)
- `PatientId`
- `AppointmentID`
- `Gender`
- `Neighbourhood`

---

## 🧠 Model Architecture

- **Preprocessing:** Date features, patient history, categorical encoding
- **Balancing:** `SMOTE` for oversampling no-show cases
- **Ensemble:**  
  - Base Models: `XGBoost`, `RandomForest`, `GradientBoosting`  
  - Meta Model: `LogisticRegression`  

---

## 📦 Requirements

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn
