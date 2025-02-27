import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class ModelEvaluator:
    def __init__(self, real_data, synthetic_datasets, target_column):
        """
        Initializes the evaluator with real and synthetic datasets.
        """
        self.real_data = real_data
        self.synthetic_datasets = synthetic_datasets  # Dictionary of synthetic datasets
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

        # Handle predict_proba() safely
        y_pred_proba = None
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)
            if y_pred_proba.shape[1] == 2:  # Ensure there are two probability columns
                y_pred_proba = y_pred_proba[:, 1]
            else:
                y_pred_proba = None  # Avoid IndexError

        # Calculate metrics
        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else "N/A"
        }
        
        return metrics

    def run_evaluation(self):
        """
        Runs evaluation for Random Forest and XGBoost
        on real data and synthetic datasets.
        """
        # Ensure the target column has at least two unique values
        if len(self.real_data[self.target_column].unique()) < 2:
            raise ValueError(f"Target column '{self.target_column}' must have at least two unique values.")

        real_X_train, real_X_test, real_y_train, real_y_test = self.preprocess_data(self.real_data)

        # Models to evaluate
        models = {
            "Random Forest": RandomForestClassifier(random_state=42),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        }

        # Results dictionary
        evaluation_results = {"Real Data": {}}

        # Convert target column to int (if needed)
        if self.real_data[self.target_column].dtype != 'int64':
            self.real_data[self.target_column] = self.real_data[self.target_column].astype('int64')

        # Evaluate on real data
        for name, model in models.items():
            evaluation_results["Real Data"][name] = self.evaluate_model(
                model, real_X_train, real_X_test, real_y_train, real_y_test
            )

        # Evaluate synthetic datasets
        for synthetic_name, synthetic_data in self.synthetic_datasets.items():
            # Skip datasets with only one class
            if len(synthetic_data[self.target_column].unique()) < 2:
                print(f"Skipping {synthetic_name} (only one class present).")
                continue

            # Convert target column to int (if needed)
            if synthetic_data[self.target_column].dtype != 'int64':
                self.synthetic_datasets[synthetic_name][self.target_column] = synthetic_data[self.target_column].astype('int64')

            syn_X_train, syn_X_test, syn_y_train, syn_y_test = self.preprocess_data(synthetic_data)
            evaluation_results[synthetic_name] = {}
            for name, model in models.items():
                evaluation_results[synthetic_name][name] = self.evaluate_model(
                    model, syn_X_train, syn_X_test, syn_y_train, syn_y_test
                )

        return evaluation_results
