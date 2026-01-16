#  Breast Cancer Prediction using Logistic Regression

## Project Overview
This project is a Machine Learning application designed to classify breast tumors as either **Benign (non-cancerous)** or **Malignant (cancerous)** based on cell attributes. 

Using the **Breast Cancer Wisconsin (Original) Dataset**, the model analyzes cytological features such as Clump Thickness, Cell Size, and Adhesion to provide accurate medical predictions. The system is built using **Logistic Regression**, a powerful binary classification algorithm.

## Key Features
* **Data Preprocessing:** Automatically handles missing data (replacing '?' with mean values) and drops irrelevant columns like ID.
* **Exploratory Data Analysis (EDA):** Includes comprehensive visualizations:
    * **Countplots** to visualize the balance of Benign vs. Malignant cases.
    * **Histograms** to understand feature distribution.
* **Predictive Modeling:** Trains a Logistic Regression model with a split of 80% training and 20% testing data.
* **Custom Prediction System:** Allows users to input new patient data and get an instant diagnosis.

## Tech Stack
* **Language:** Python
* **Libraries:** * `pandas` & `numpy` (Data Manipulation)
    * `matplotlib` & `seaborn` (Data Visualization)
    * `sklearn` (Model Training & Evaluation)
    * `pickle` (Model Saving)

## Dataset Details
The model predicts the class based on the following attributes (rated 1-10):
1.  Clump Thickness
2.  Uniformity of Cell Size
3.  Uniformity of Cell Shape
4.  Marginal Adhesion
5.  Single Epithelial Cell Size
6.  Bare Nuclei
7.  Bland Chromatin
8.  Normal Nucleoli
9.  Mitoses

**Target Classes:**
* **2:** Benign
* **4:** Malignant

##Model Performance
**Accuracy Score:** Measures overall correctness.
**Confusion Matrix:** visualizes true positives/negatives vs. false predictions.
**Classification Report:** Detailed precision, recall, and F1-score analysis.

## Outcomes
<img width="1251" height="855" alt="Prediction" src="https://github.com/user-attachments/assets/b45c43e0-7b0e-4fb4-8775-a2b9b0eb2530" />
<img width="1238" height="616" alt="Layout" src="https://github.com/user-attachments/assets/1cda734d-2b0d-4dad-acc5-e23ab7075b7a" />


## Installation & Usage

**1. Clone the Repository**
```bash
git clone [https://github.com/your-username/Breast-Cancer-Prediction.git](https://github.com/your-username/Breast-Cancer-Prediction.git)
cd Breast-Cancer-Prediction
pip install pandas numpy matplotlib seaborn scikit-learnR
**Run the Model** python Breast_Cancer_Prediction.py
