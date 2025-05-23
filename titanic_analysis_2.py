
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ✅ Set the file path for the dataset
file_path = "/Users/fizasiddiqui/Desktop/titanic/train.csv"

# Step 1: Check if the file exists
if not os.path.exists(file_path):
    print("❌ File not found. Please check the path.")
    exit()

# Step 2: Load the dataset
df = pd.read_csv(file_path)
print("✅ Dataset Loaded Successfully\n")

# Step 3: Basic info
print(df.head())
print("\n🔍 Missing Values:\n", df.isnull().sum())

# Step 4: Data cleaning
df.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True, errors='ignore')
df['Age'].fillna(df['Age'].mean(), inplace=True)

# Step 5: Convert categorical to numeric
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Step 6: Feature selection (more features included)
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y = df['Survived']

# Step 7: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Train Logistic Regression model
log_model = LogisticRegression(max_iter=200)
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)
print("\n🎯 Logistic Regression Accuracy:", accuracy_score(y_test, log_pred))
print("\n📊 Classification Report (Logistic Regression):\n", classification_report(y_test, log_pred))

# Step 9: Train Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print("\n🎯 Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("\n📊 Classification Report (Random Forest):\n", classification_report(y_test, rf_pred))

# Step 10: Feature importance (Random Forest)
print("\n🔍 Feature Importance (Random Forest):")
importance = rf_model.feature_importances_
for feature, score in zip(X.columns, importance):
    print(f"{feature}: {score:.2f}")

# Step 11: Visualizations
sns.countplot(x='Survived', hue='Sex', data=df)
plt.title("Survival Count by Gender")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.legend(title="Gender", labels=['Male', 'Female'])
plt.tight_layout()
plt.show()

sns.boxplot(x='Pclass', y='Age', data=df)
plt.title("Age Distribution by Passenger Class")
plt.tight_layout()
plt.show()

sns.histplot(data=df, x='Age', hue='Survived', bins=30, kde=True)
plt.title("Age Distribution by Survival")
plt.tight_layout()
plt.show()

sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title("Survival by Passenger Class")
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()
