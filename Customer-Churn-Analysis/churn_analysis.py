
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ✅ Set the file path for the dataset
file_path = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Step 1: Check if the file exists
if not os.path.exists(file_path):
    print("❌ File not found. Please check the path.")
    exit()

# Step 2: Load the dataset
df = pd.read_csv(file_path)
print("✅ Dataset Loaded Successfully\n")

# Step 3: Clean the data
df.replace(" ", pd.NA, inplace=True)
df.dropna(inplace=True)
df['TotalCharges'] = df['TotalCharges'].astype(float)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Convert categorical variables to dummy/indicator variables
df = pd.get_dummies(df, drop_first=True)

# Step 4: Split the data
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train models

# Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)

# Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Step 6: Evaluation
print("\n📊 Logistic Regression Report:")
print(classification_report(y_test, log_pred))

print("\n🌲 Random Forest Report:")
print(classification_report(y_test, rf_pred))

# Step 7: Visualizations
plt.figure(figsize=(6,4))
sns.countplot(x='Churn', data=df)
plt.title("Churn Count")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(data=df, x='MonthlyCharges', hue='Churn', bins=30, kde=True)
plt.title("Monthly Charges Distribution by Churn")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(data=df, x='TotalCharges', hue='Churn', bins=30, kde=True)
plt.title("Total Charges Distribution by Churn")
plt.tight_layout()
plt.show()
