from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA

# Load the dataset
file_path = 'E:\\UNI\\Sympto-Scan-main-20240801T060037Z-001\\Sympto-Scan-main\\datasets\\new_dataset.csv'
try:
    data = pd.read_csv(file_path)
    print("File loaded successfully!")
except FileNotFoundError:
    print("File not found. Please check the path.")
    exit()

# Define features and target
X = data[['sudden_fever','headache','mouth_bleed','nose_bleed','muscle_pain','joint_pain','vomiting','rash','diarrhea','pleural_effusion','ascites','swelling','nausea','chills','myalgia','digestion_trouble','fatigue','stomach_pain','orbital_pain','neck_pain','weakness','back_pain','weight_loss','gum_bleed','jaundice','diziness','inflammation','red_eyes','loss_of_appetite','urination_loss','abdominal_pain','yellow_skin','yellow_eyes','rigor','bitter_tongue','convulsion','anemia','cocacola_urine','prostraction','stiff_neck','irritability','lymph_swells','breathing_restriction','itchiness','ulcers','toenail_loss','bullseye_rash']]
y = data['prediction']

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply PCA for feature reduction
pca = PCA(n_components=10)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Initialize Logistic Regression model
log_reg = LogisticRegression(max_iter=1000)

# Hyperparameter tuning using GridSearchCV with StratifiedKFold
param_grid = [
    {'solver': ['newton-cg', 'lbfgs', 'saga'], 'penalty': ['l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100]},
    {'solver': ['liblinear'], 'penalty': ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100]}
]
grid_search = GridSearchCV(estimator=log_reg, param_grid=param_grid, cv=StratifiedKFold(5), scoring='accuracy', error_score='raise')
grid_search.fit(X_train, y_train)

# Best parameters from GridSearchCV
best_params = grid_search.best_params_
print(f"Best parameters for Logistic Regression: {best_params}")

# Train the best Logistic Regression model
log_reg_model = LogisticRegression(**best_params, max_iter=1000)
log_reg_model.fit(X_train, y_train)

# Evaluate Logistic Regression using cross-validation
cv_scores_log_reg = cross_val_score(log_reg_model, X_train, y_train, cv=StratifiedKFold(5), scoring='accuracy')
print(f"Cross-validation accuracy for Logistic Regression: {cv_scores_log_reg.mean():.1f}")

# Predict the labels for the test set using Logistic Regression
y_pred_log_reg = log_reg_model.predict(X_test)
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)

# Confusion Matrix and Metrics for Logistic Regression
conf_matrix_log_reg = confusion_matrix(y_test, y_pred_log_reg)
precision_log_reg = precision_score(y_test, y_pred_log_reg, average='weighted', zero_division=0)
recall_log_reg = recall_score(y_test, y_pred_log_reg, average='weighted', zero_division=0)
f2_score_log_reg = fbeta_score(y_test, y_pred_log_reg, beta=2, average='weighted', zero_division=0)

print(f"Logistic Regression - Accuracy: {accuracy_log_reg:.1f}, Precision: {precision_log_reg:.1f}, Recall: {recall_log_reg:.1f}, F2-Score: {f2_score_log_reg:.1f}")

# Initialize and train the Decision Tree model with pruning
decision_tree_model = DecisionTreeClassifier(random_state=42, max_depth=4, min_samples_split=20, min_samples_leaf=10)
decision_tree_model.fit(X_train, y_train)

# Evaluate Decision Tree using cross-validation
cv_scores_decision_tree = cross_val_score(decision_tree_model, X_train, y_train, cv=StratifiedKFold(5), scoring='accuracy')
print(f"Cross-validation accuracy for Decision Tree: {cv_scores_decision_tree.mean():.1f}")

# Predict the labels for the test set using Decision Tree
y_pred_decision_tree = decision_tree_model.predict(X_test)
accuracy_decision_tree = accuracy_score(y_test, y_pred_decision_tree)

# Confusion Matrix and Metrics for Decision Tree
conf_matrix_decision_tree = confusion_matrix(y_test, y_pred_decision_tree)
precision_decision_tree = precision_score(y_test, y_pred_decision_tree, average='weighted', zero_division=0)
recall_decision_tree = recall_score(y_test, y_pred_decision_tree, average='weighted', zero_division=0)
f2_score_decision_tree = fbeta_score(y_test, y_pred_decision_tree, beta=2, average='weighted', zero_division=0)

print(f"Decision Tree - Accuracy: {accuracy_decision_tree:.1f}, Precision: {precision_decision_tree:.1f}, Recall: {recall_decision_tree:.1f}, F2-Score: {f2_score_decision_tree:.1f}")

# Save both models and scaler
with open('Sympto-Scan-main/datasets/log_reg_model.pkl', 'wb') as model_file:
    pickle.dump(log_reg_model, model_file)

with open('Sympto-Scan-main/datasets/decision_tree_model.pkl', 'wb') as model_file:
    pickle.dump(decision_tree_model, model_file)

with open('Sympto-Scan-main/datasets/scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Models and scaler saved successfully!")

app = Flask(__name__)

# Load the models and scaler
with open('Sympto-Scan-main/datasets/log_reg_model.pkl', 'rb') as model_file:
    log_reg_model = pickle.load(model_file)
with open('Sympto-Scan-main/datasets/decision_tree_model.pkl', 'rb') as model_file:
    decision_tree_model = pickle.load(model_file)
with open('Sympto-Scan-main/datasets/scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/gallery')
def gallery():
    return render_template('gallery.html')

@app.route('/contact_us')
def contact_us():
    return render_template('contactus.html')

@app.route('/check_symptoms')
def check_symptoms():
    return render_template('check_symptoms.html')

@app.route('/symptoms_with_logistic')
def symptoms_with_logistic():
    return render_template('symptoms_with_logistic.html')

@app.route('/symptoms_with_decision_tree')
def symptoms_with_decision_tree():
    return render_template('symptoms_with_decision_tree.html')

@app.route('/check_medical_store')
def check_medical_store():
    return render_template('check_medical_store.html')

@app.route('/appointment')
def appointment():
    return render_template('appointment.html')

@app.route('/affiliation')
def affiliation():
    return render_template('affiliation.html')

@app.route('/result')
def result():
    return render_template('result.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    features = [float(x) for x in request.form.values()]
    features = np.array(features).reshape(1, -1)
    print("feature ", features)
    # Scale features
    features = scaler.transform(features)
    features = pca.transform(features)

    # Predict using both models
    prediction_log_reg = log_reg_model.predict(features)
    prediction_proba_log_reg = log_reg_model.predict_proba(features)
    prediction_decision_tree = decision_tree_model.predict(features)
    prediction_proba_decision_tree = decision_tree_model.predict_proba(features)

    return render_template('result.html',
                           prediction_log_reg=prediction_log_reg[0], 
                           prediction_proba_log_reg=prediction_proba_log_reg,
                           accuracy_log_reg=f"{accuracy_log_reg:.1f}", 
                           precision_log_reg=f"{precision_log_reg:.1f}", 
                           recall_log_reg=f"{recall_log_reg:.1f}", 
                           f2_score_log_reg=f"{f2_score_log_reg:.1f}",
                           prediction_decision_tree=prediction_decision_tree[0], 
                           prediction_proba_decision_tree=prediction_proba_decision_tree,
                           accuracy_decision_tree=f"{accuracy_decision_tree:.1f}", 
                           precision_decision_tree=f"{precision_decision_tree:.1f}", 
                           recall_decision_tree=f"{recall_decision_tree:.1f}", 
                           f2_score_decision_tree=f"{f2_score_decision_tree:.1f}")

if __name__ == "__main__":
    app.run(debug=True)
