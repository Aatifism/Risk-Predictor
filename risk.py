import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb
from flask import Flask, request, jsonify, render_template
import traceback
import joblib
import os
import shap

class ImprovedAllDiseaseRiskPredictionModel:
    def __init__(self, n_splits=5):
        self.models = {}
        self.preprocessor = None
        self.feature_names = None
        self.training_columns = None
        self.target_columns = None
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.6
        }
        self.single_class_diseases = set()
        self.n_splits = n_splits
        self.feature_importances = {}
        self.default_values = {
            'Age': 35, 'Gender': 'Unknown', 'Height': 170, 'Weight': 70, 'BMI': 24.22,
            'Systolic_BP': 120, 'Diastolic_BP': 80, 'Cholesterol_Total': 200,
            'Cholesterol_HDL': 50, 'Cholesterol_LDL': 130, 'Triglycerides': 150,
            'Blood_Glucose_Fasting': 90, 'HbA1c': 5.5, 'Smoking_Status': 'Never',
            'Alcohol_Consumption': 'Moderate', 'Physical_Activity': 'Moderate',
            'Family_History_CVD': 0, 'Family_History_Diabetes': 0, 'Family_History_Cancer': 0,
            'Stress_Level': 'Medium', 'Sleep_Hours': 7, 'Fruits_Veggies_Daily': 3,
            'Creatinine': 1.0, 'eGFR': 90, 'ALT': 25, 'AST': 25, 'TSH': 2.0, 'T4': 1.2,
            'Vitamin_D': 30, 'Calcium': 9.5, 'Hemoglobin': 14, 'White_Blood_Cell_Count': 7.0,
            'Platelet_Count': 250, 'C_Reactive_Protein': 2, 'Vitamin_B12': 400, 'Folate': 10,
            'Ferritin': 100, 'Uric_Acid': 5, 'PSA': 1, 'Bone_Density_T_Score': 0
        }

    def preprocess_data(self, data, is_training=True):
        if is_training:
            self.target_columns = [col for col in data.columns if col.endswith('_Risk') or col in [
                'Hypertension', 'Prehypertension', 'Diabetes', 'Prediabetes', 'Obesity', 'Overweight',
                'Hypercholesterolemia', 'Low_HDL', 'High_LDL', 'High_Triglycerides',
                'Chronic_Kidney_Disease', 'Liver_Disease', 'Hypothyroidism',
                'Hyperthyroidism', 'Vitamin_D_Deficiency', 'Anemia'
            ]]
            X = data.drop(columns=self.target_columns)
            y = data[self.target_columns]
            self.training_columns = X.columns

            numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_features = X.select_dtypes(include=['object']).columns.tolist()

            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ]
            )

            X_processed = self.preprocessor.fit_transform(X)

            self.feature_names = (
                numeric_features +
                list(self.preprocessor.named_transformers_['cat']
                     .named_steps['onehot']
                     .get_feature_names_out(categorical_features))
            )

            return X_processed, y
        else:
            X = data.copy()
            for col in self.training_columns:
                if col not in X.columns:
                    X[col] = self.default_values.get(col, np.nan)
            X = X.reindex(columns=self.training_columns, fill_value=np.nan)
            X_processed = self.preprocessor.transform(X)
            return X_processed

    def build_models(self):
        for column in self.target_columns:
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
            xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
            lr = LogisticRegression(penalty='l2', C=1.0, random_state=42, max_iter=1000)

            self.models[column] = VotingClassifier(
                estimators=[('rf', rf), ('gb', gb), ('xgb', xgb_model), ('lr', lr)],
                voting='soft'
            )

    def train_and_evaluate_models(self, X, y):
        results = {}
        feature_importances = {}

        for column in self.target_columns:
            if len(np.unique(y[column])) == 1:
                print(f"Skipping {column} as it contains only one class.")
                self.single_class_diseases.add(column)
                continue

            print(f"Training and evaluating model for {column}...")
            
            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            fold_results = []
            fold_feature_importances = []

            for fold, (train_index, val_index) in enumerate(skf.split(X, y[column]), 1):
                X_train, X_val = X[train_index], X[val_index]
                y_train, y_val = y[column].iloc[train_index], y[column].iloc[val_index]

                self.models[column].fit(X_train, y_train)
                
                y_pred = self.models[column].predict(X_val)
                y_pred_proba = self.models[column].predict_proba(X_val)[:, 1]

                accuracy = accuracy_score(y_val, y_pred)
                try:
                    auc_roc = roc_auc_score(y_val, y_pred_proba)
                except ValueError:
                    auc_roc = None

                fold_results.append({
                    'fold': fold,
                    'accuracy': accuracy,
                    'auc_roc': auc_roc
                })

                rf_model = self.models[column].estimators_[0]
                fold_feature_importances.append(rf_model.feature_importances_)

            results[column] = fold_results
            feature_importances[column] = np.mean(fold_feature_importances, axis=0)

        return results, feature_importances

    def predict_risks(self, X):
        predictions = {}
        for column in self.target_columns:
            if column in self.single_class_diseases:
                predictions[column] = {
                    'probability': np.zeros(X.shape[0]),
                    'risk_level': np.array(['Low'] * X.shape[0])
                }
            else:
                probabilities = self.models[column].predict_proba(X)[:, 1]
                risk_levels = np.select(
                    [probabilities < self.risk_thresholds['low'],
                     probabilities < self.risk_thresholds['medium']],
                    ['Low', 'Medium'],
                    default='High'
                )
                predictions[column] = {
                    'probability': probabilities,
                    'risk_level': risk_levels
                }
        return predictions

    def get_important_features(self, disease, top_n=5):
        if disease in self.single_class_diseases:
            return [("N/A", 0)] * top_n
        feature_importance = self.feature_importances.get(disease, np.zeros(len(self.feature_names)))
        feature_importance_dict = dict(zip(self.feature_names, feature_importance))
        sorted_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
        return sorted_importance[:top_n]

    def save_model(self, filepath):
        joblib.dump(self, filepath)

    @classmethod
    def load_model(cls, filepath):
        return joblib.load(filepath)

def load_data(file_path):
    return pd.read_csv(file_path)

def summarize_model_performance(results):
    summary = {}
    for disease, fold_results in results.items():
        accuracies = [result['accuracy'] for result in fold_results]
        auc_rocs = [result['auc_roc'] for result in fold_results if result['auc_roc'] is not None]

        summary[disease] = {
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'mean_auc_roc': np.mean(auc_rocs) if auc_rocs else None,
            'std_auc_roc': np.std(auc_rocs) if auc_rocs else None
        }

    return summary

def print_concise_evaluation(summary):
    print("\nModel Performance Summary:")
    for disease, metrics in summary.items():
        print(f"--- {disease} ---")
        print(f"Accuracy: {metrics['mean_accuracy']:.4f} (+/- {metrics['std_accuracy']:.4f})")
        if metrics['mean_auc_roc'] is not None:
            print(f"AUC-ROC: {metrics['mean_auc_roc']:.4f} (+/- {metrics['std_auc_roc']:.4f})")
        else:
            print("AUC-ROC: Not Available")

def explain_prediction(predictions, feature_importances):
    explanations = {}
    for disease, prediction in predictions.items():
        top_features = feature_importances.get(disease, [])
        explanation = "Risk determined by features: " + ", ".join([f"{feat} ({imp:.2f})" for feat, imp in top_features])
        explanations[disease] = explanation

    return explanations

app = Flask(__name__)
model_path = 'disease_risk_model.joblib'

if os.path.exists(model_path):
    global_model = ImprovedAllDiseaseRiskPredictionModel.load_model(model_path)
else:
    global_model = ImprovedAllDiseaseRiskPredictionModel()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    try:
        data_file = request.files['data_file']
        if data_file.filename == '':
            return jsonify({"status": "error", "message": "No file selected"})
        
        if data_file:
            data = pd.read_csv(data_file)
            X, y = global_model.preprocess_data(data, is_training=True)
            global_model.build_models()
            results, feature_importances = global_model.train_and_evaluate_models(X, y)
            summary = summarize_model_performance(results)
            global_model.feature_importances = feature_importances
            print_concise_evaluation(summary)
            
            global_model.save_model(model_path)
            
            return jsonify({"status": "success", "summary": summary})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        patient_data = request.json
        patient_df = pd.DataFrame([patient_data])
        X_processed = global_model.preprocess_data(patient_df, is_training=False)
        predictions = global_model.predict_risks(X_processed)
        feature_importances = {disease: global_model.get_important_features(disease) for disease in predictions.keys()}
        explanations = explain_prediction(predictions, feature_importances)
        
        results = {}
        for disease, pred in predictions.items():
            results[disease] = {
                "risk_level": pred['risk_level'][0],
                "probability": float(pred['probability'][0]),
                "explanation": explanations[disease]
            }
        
        return jsonify(results)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True)