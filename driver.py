import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load the sample dataset into a Pandas DataFrame
data = pd.read_csv('sample_data.csv')

# Assuming 'loan_eligibility' is the column name for the target variable
X = data.drop('loan_eligibility', axis=1)
y = data['loan_eligibility']

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess categorical features (One-hot encoding for 'Position')
categorical_features = ['Position']
numeric_features = ['Number_of_children', 'Years_of_service']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features),
        ('num', StandardScaler(), numeric_features)
    ])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Create and train the Logistic Regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train_processed, y_train)

# Make predictions on the test data using Logistic Regression
y_pred_logistic = logistic_model.predict(X_test_processed)

# Create and train the Random Forest model
random_forest_model = RandomForestClassifier(random_state=42)
random_forest_model.fit(X_train_processed, y_train)

# Make predictions on the test data using Random Forest
y_pred_rf = random_forest_model.predict(X_test_processed)

# Create and train the Gradient Boosting model
gradient_boosting_model = GradientBoostingClassifier(random_state=42)
gradient_boosting_model.fit(X_train_processed, y_train)

# Make predictions on the test data using Gradient Boosting
y_pred_gb = gradient_boosting_model.predict(X_test_processed)

# Create and train the SVM model
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_processed, y_train)

# Make predictions on the test data using SVM
y_pred_svm = svm_model.predict(X_test_processed)

# Evaluate models
def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    print(f"{model_name} Performance:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"AUC-ROC Score: {roc_auc:.2f}")
    print()

print("Evaluating the model.......\n")
evaluate_model(y_test, y_pred_logistic, "Logistic Regression")
evaluate_model(y_test, y_pred_rf, "Random Forest")
evaluate_model(y_test, y_pred_gb, "Gradient Boosting")
evaluate_model(y_test, y_pred_svm, "Support Vector Machine (SVM)")
