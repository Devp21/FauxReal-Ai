import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class ModelEvaluator:
    def __init__(self, real_data, synthetic_data, target_column):
        """
        Initializes the evaluator with real and synthetic datasets.
        """
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.target_column = target_column

    def preprocess_data(self, data):
        """
        Splits the dataset into features and target, then into train and test sets.
        """
        X = data.drop(columns=[self.target_column])
        y = data[self.target_column]
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def evaluate_model(self, model, X_train, X_test, y_train, y_test):
        """
        Trains and evaluates the given model, returning metrics.
        """
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        }
        
        if y_pred_proba is not None:
            metrics["ROC-AUC"] = roc_auc_score(y_test, y_pred_proba)
        else:
            metrics["ROC-AUC"] = "N/A"  # Some models like SVM may not support predict_proba
            
        return metrics

    def run_evaluation(self):
        """
        Runs evaluation for Logistic Regression, Random Forest, and XGBoost.
        """
        # Preprocess real and synthetic data
        real_X_train, real_X_test, real_y_train, real_y_test = self.preprocess_data(self.real_data)
        syn_X_train, syn_X_test, syn_y_train, syn_y_test = self.preprocess_data(self.synthetic_data)
        
        # Models to evaluate
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        }
        
        # Results
        evaluation_results = {}
        for name, model in models.items():
            evaluation_results[name] = {
                "Real Data": self.evaluate_model(model, real_X_train, real_X_test, real_y_train, real_y_test),
                "Synthetic Data": self.evaluate_model(model, syn_X_train, syn_X_test, syn_y_train, syn_y_test),
            }
        
        return evaluation_results


# Example Usage
if __name__ == "__main__":
    # Load real and synthetic datasets
    real_data = pd.read_csv("real_data.csv")
    synthetic_data = pd.read_csv("synthetic_data.csv")
    target_column = "target"  # Replace with your actual target column name
    
    evaluator = ModelEvaluator(real_data, synthetic_data, target_column)
    results = evaluator.run_evaluation()
    
    # Display results
    for model_name, metrics in results.items():
        print(f"\nModel: {model_name}")
        print("Real Data Metrics:")
        for metric, value in metrics["Real Data"].items():
            print(f"  {metric}: {value}")
        print("Synthetic Data Metrics:")
        for metric, value in metrics["Synthetic Data"].items():
            print(f"  {metric}: {value}")
