# =============================MINI PROJECT================================= #
# ===Title:Data analysis and predicting when to send mails for effective marketing=== #
# =================== PreProcessing Of Data =================== #
# Importing the necessary libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Warnings
import warnings
warnings.filterwarnings("ignore")
# Read the dataset
df = pd.read_csv("C:/Users/psai5/OneDrive/Desktop/MINIPROJECT/dataset.csv")
# Head of the Data
df.head()
# Tail of the Data
df.tail()
# Shape of the Data 
df.shape
# Info of the Data 
df.info()
# Finding missing value
df.fillna(0, inplace=True)
df.isnull().sum()
# Finding Duplicates
df.drop_duplicates(inplace=True)

#================================= Class Imbalance ===============================#
# Read the cattegorical and numerical columns
cat_cols=['email_text','email_version','weekday','user_country','engagement_status']
num_cols=['hour','user_past_purchases']
# Finding missing data
df.isnull().sum()
# Calculates the percentage of each class in the engagement_status column
df['engagement_status'].value_counts()/len(df)
import seaborn as sys
sys.countplot(x='engagement_status',data=df)
# Merge engagement categories into two classes: Success and Failed
df['engagement_status'] = df['engagement_status'].replace({
    'Clicked and Opened': 'Success',
    'Opened but Not Clicked': 'Success',
    'Not Opened': 'Failed'
})
import seaborn as sns
sns.countplot(x='engagement_status', data=df)
df['engagement_status'].value_counts()
# Encode categorical features and apply SMOTE to balance the engagement_status classes
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter
# Separate features and target
X = df.drop('engagement_status', axis=1)
y = df['engagement_status']
# Encode categorical columns
cat_cols = X.select_dtypes(include=['object']).columns
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
# Apply SMOTE
smote = SMOTE(sampling_strategy=0.81, random_state=52)
X_resampled, y_resampled = smote.fit_resample(X, y)
# Check distribution
print(Counter(y_resampled))
# Visualize class distribution before and after applying SMOTE using bar charts
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
# Before SMOTE
before_counts = Counter(y)
# After SMOTE
after_counts = Counter(y_resampled)
# Convert to lists
labels_before = list(before_counts.keys())
values_before = list(before_counts.values())
labels_after = list(after_counts.keys())
values_after = list(after_counts.values())
# Plot
plt.figure(figsize=(10,5))
# Before SMOTE
plt.subplot(1,2,1)
sns.barplot(x=labels_before, y=values_before)
plt.title("Before SMOTE")
plt.xlabel("Engagement Status")
plt.ylabel("Count")
# After SMOTE
plt.subplot(1,2,2)
sns.barplot(x=labels_after, y=values_after)
plt.title("After SMOTE")
plt.xlabel("Engagement Status")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# ============================= Encoding ============================ #
from sklearn.preprocessing import LabelEncoder
# Separate features and target
X = df.drop('engagement_status', axis=1)
y = df['engagement_status']
encoders = {}
# Encode categorical columns
cat_cols = X.select_dtypes(include=['object']).columns
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le  
# Target encoding
y = y.map({'Success':1, 'Failed':0})

# ============================= HeatMap ============================= #
import matplotlib.pyplot as plt
import seaborn as sns
# Set the size of the plot to make the visualization larger and clearer
plt.figure(figsize=(20,10))
sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap='coolwarm')
plt.show()
df

# =================== Exploratory Data Analysis(EDA) ================ #
# Exploratory Data Analysis (EDA)
# Descriptive Statistics
# It gives the Count, Mean, Std, Min, Max and basic understanding of the data
df.describe().T
df.describe(include='all')
# Histogram to understand the distribution
# we use this for getting the histogram for each numerical columns
for i in df.select_dtypes(include="number").columns:
    sns.histplot(data=df,x=i)
    plt.show()
# Boxplot to identify the outliers
# we use this for getting the Boxplot for each numerical columns
for i in df.select_dtypes(include="number").columns:
    sns.boxplot(data=df,x=i)
    plt.show()
# Scatter plot to understand the relationship
# we use this for getting the Scatterplot for each numerical columns
for i in['email_text','email_version','hour','weekday','user_country','user_past_purchases','engagement_status']:
    sns.scatterplot(data=df,x=i,y='user_past_purchases')
    plt.show()

# ======================= Model Building ============================ #
# By using Three Models : SVC(support vector classifier),Random Forest Classifier,XG Booster
# =======================SVM(support Vector machine) ================ #
# By using four planes : Linear plane,poly plane,rdf plane,sigmoid plane
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Split BEFORE SMOTE
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=52, stratify=y
)

# Apply SMOTE ONLY on training data
smote = SMOTE(sampling_strategy=0.81, random_state=52)
X_train, y_train = smote.fit_resample(X_train, y_train)

# ===================== SCALING ===================== #
# StandardScaler is used to standardize the features
# It converts data into mean = 0 and standard deviation = 1
scaler = StandardScaler()
# Fit on training data and transform
X_train = scaler.fit_transform(X_train)
# Use same scaling on test data (NO fit here to avoid data leakage)
X_test = scaler.transform(X_test)
df

# ======================== Linear plane ===============================#
# SVC model (linear kernel)
model = SVC(kernel='linear', max_iter=1000)
# Train model
model.fit(X_train, y_train)
# Prediction
y_pred = model.predict(X_test)
# Evaluation
print("Kernel: Linear")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ========================= Poly plane ============================= #
# SVC model (Poly kernel)
model = SVC(kernel='poly', max_iter=1000)
# Train model
model.fit(X_train, y_train)
# Prediction
y_pred = model.predict(X_test)
# Evaluation
print("Kernel: poly")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ========================== rbf plane ============================== #
# SVM model (rbf kernel)
model = SVC(kernel='rbf', max_iter=1000)
# Train model
model.fit(X_train, y_train)
# Prediction
y_pred = model.predict(X_test)
# Evaluation
print("Kernel: rbf")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ========================== sigmoid plane ========================= #
# SVM model (sigmoid kernel)
model = SVC(kernel='sigmoid', max_iter=1000)
# Train model
model.fit(X_train, y_train)
# Prediction
y_pred = model.predict(X_test)
# Evaluation
print("Kernel: sigmoid")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ================== comparision between four planes ================ #
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
results = []
for k in kernels:
    model = SVC(kernel=k, max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    results.append([k, acc, prec, rec, f1])
comparison = pd.DataFrame(
    results,
    columns=["Kernel", "Accuracy", "Precision", "Recall", "F1 Score"]
)
print(comparison)
# Find best kernel based on Accuracy
best_kernel = comparison.loc[comparison['Accuracy'].idxmax()]
print("\nBest Kernel Based on Accuracy:")
print(best_kernel)
print("\nThe best kernel for the SVM model is:", best_kernel["Kernel"])
 
# ======================== RANDOM FOREST MODEL ======================== #
from sklearn.ensemble import RandomForestClassifier
# Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=52)
# Train model
rf_model.fit(X_train, y_train)
# Prediction
y_pred_rf = rf_model.predict(X_test)
# Evaluation
print("\nRandom Forest Model")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# ======================== XGBOOST MODEL ======================== #
from xgboost import XGBClassifier
# XGBoost model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=52)
# Train model
xgb_model.fit(X_train, y_train)
# Prediction
y_pred_xgb = xgb_model.predict(X_test)
# Evaluation
print("\nXGBoost Model")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print(confusion_matrix(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))

# ================== MODEL COMPARISON (ALL 3 MODELS) ================== #
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Initialize models
models = {
    "SVM (RBF)": SVC(kernel='rbf', max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=52),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=52)
}
results = []
# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    results.append([name, acc, prec, rec, f1])
# Create comparison table
comparison_df = pd.DataFrame(results, columns=[
    "Model", "Accuracy", "Precision", "Recall", "F1 Score"
])
# Evaluation
print("\n Model Comparison:")
print(comparison_df)
# Find best model
best_model = comparison_df.loc[comparison_df['Accuracy'].idxmax()]
print("\n Best Model Based on Accuracy:")
print(best_model)
print(f"\n The best model is: {best_model['Model']}")

# ================== SAVING MODEL USING PICKLE ================== #
import pickle
from sklearn.svm import SVC

# Re-train BEST SVM (RBF) model to ensure correct model is saved
best_svm_model = SVC(kernel='rbf', max_iter=1000)
best_svm_model.fit(X_train, y_train)

# Save SVM model
with open("svm_model.pkl", "wb") as f:
    pickle.dump(best_svm_model, f)

# Save Random Forest
with open("random_forest_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

# Save XGBoost
with open("xgboost_model.pkl", "wb") as f:
    pickle.dump(xgb_model, f)

# Save Scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save Encoders (VERY IMPORTANT)
with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

print(" Models, scaler, and encoders saved successfully!")

import os
print("Current Working Directory:", os.getcwd())
