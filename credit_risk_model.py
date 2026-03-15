"""
Credit Risk Modeling - Python Implementation

Regression-based models quantifying loan default probabilities for 
allowance for doubtful accounts and risk mitigation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.metrics import (confusion_matrix, roc_curve, auc, 
                             roc_auc_score, classification_report, 
                             accuracy_score, precision_score, recall_score, f1_score)
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')


class CreditRiskModel:
    """
    Credit risk modeling using regression-based approaches.
    
    Attributes:
        df (pd.DataFrame): Original dataset
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable (Default)
        X_train, X_test, y_train, y_test: Train/test splits
        models (dict): Fitted models
        results (dict): Model performance metrics
    """
    
    def __init__(self, df):
        """Initialize with loan dataset."""
        self.df = df.copy()
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.predictions = {}
        self._preprocess()
    
    def _preprocess(self):
        """Preprocess data: encode categoricals, separate features and target."""
        df = self.df.copy()
        
        # Separate target
        self.y = df['Default']
        df = df.drop(['Default', 'LoanID'], axis=1)
        
        # Encode categorical variables
        categorical_cols = df.select_dtypes(include=['object']).columns
        le = LabelEncoder()
        for col in categorical_cols:
            df[col] = le.fit_transform(df[col].astype(str))
        
        self.X = df
        print(f"Data preprocessed: {self.X.shape[0]} samples, {self.X.shape[1]} features")
        print(f"Default rate: {self.y.mean():.2%}")
    
    def train_test_split(self, test_size=0.2, random_state=42):
        """Split data into train and test sets."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Train set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
    
    def fit_logistic_regression(self, max_iter=1000):
        """
        Fit Generalized Linear Model (Logistic Regression).
        
        Baseline model with interpretable coefficients.
        Provides probability estimates directly.
        """
        print("\n" + "="*60)
        print("LOGISTIC REGRESSION (GLM)")
        print("="*60)
        
        model = LogisticRegression(max_iter=max_iter, random_state=42)
        model.fit(self.X_train_scaled, self.y_train)
        
        self.models['Logistic'] = model
        
        # Predictions
        y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
        y_pred = model.predict(self.X_test_scaled)
        
        self.predictions['Logistic'] = y_pred_proba
        
        # Evaluation
        results = self._evaluate_model(y_pred, y_pred_proba, 'Logistic')
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': self.X.columns,
            'Coefficient': model.coef_[0]
        }).sort_values('Coefficient', key=abs, ascending=False)
        
        print("\nTop 10 Features (by coefficient magnitude):")
        print(feature_importance.head(10).to_string(index=False))
        
        return results
    
    def fit_ridge_regression(self, alpha=1.0):
        """
        Fit Ridge Regression (L2 Regularization).
        
        Handles multicollinearity with stable coefficient estimates.
        """
        print("\n" + "="*60)
        print("RIDGE REGRESSION (L2 Regularization)")
        print("="*60)
        
        model = Ridge(alpha=alpha)
        model.fit(self.X_train_scaled, self.y_train)
        
        self.models['Ridge'] = model
        
        # Predictions (ridge returns raw predictions, need to transform to probabilities)
        y_pred_raw = model.predict(self.X_test_scaled)
        # Clip to [0, 1] range for probability interpretation
        y_pred_proba = np.clip(y_pred_raw, 0, 1)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        self.predictions['Ridge'] = y_pred_proba
        
        # Evaluation
        results = self._evaluate_model(y_pred, y_pred_proba, 'Ridge')
        
        return results
    
    def fit_lasso_regression(self, alpha=0.001):
        """
        Fit Lasso Regression (L1 Regularization).
        
        Automatic feature selection with sparse coefficient estimates.
        Prevents overfitting.
        """
        print("\n" + "="*60)
        print("LASSO REGRESSION (L1 Regularization)")
        print("="*60)
        
        model = Lasso(alpha=alpha, max_iter=10000, random_state=42)
        model.fit(self.X_train_scaled, self.y_train)
        
        self.models['Lasso'] = model
        
        # Predictions
        y_pred_raw = model.predict(self.X_test_scaled)
        y_pred_proba = np.clip(y_pred_raw, 0, 1)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        self.predictions['Lasso'] = y_pred_proba
        
        # Evaluation
        results = self._evaluate_model(y_pred, y_pred_proba, 'Lasso')
        
        # Feature selection info
        selected_features = self.X.columns[model.coef_ != 0].tolist()
        print(f"\nSelected features: {len(selected_features)} / {self.X.shape[1]}")
        print(f"Non-zero coefficients: {(model.coef_ != 0).sum()}")
        
        return results
    
    def _evaluate_model(self, y_pred, y_pred_proba, model_name):
        """Evaluate model performance with comprehensive metrics."""
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        auc_score = roc_auc_score(self.y_test, y_pred_proba)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = recall  # Same as recall
        
        results = {
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'Specificity': specificity,
            'F1-Score': f1,
            'AUC': auc_score,
            'TP': tp,
            'FP': fp,
            'TN': tn,
            'FN': fn
        }
        
        self.results[model_name] = results
        
        # Print results
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives:  {tn:6d}")
        print(f"  False Positives: {fp:6d}")
        print(f"  False Negatives: {fn:6d}")
        print(f"  True Positives:  {tp:6d}")
        
        print(f"\nPerformance Metrics:")
        print(f"  Accuracy:    {accuracy:.4f}")
        print(f"  Precision:   {precision:.4f}")
        print(f"  Recall:      {recall:.4f}")
        print(f"  Specificity: {specificity:.4f}")
        print(f"  F1-Score:    {f1:.4f}")
        print(f"  AUC:         {auc_score:.4f}")
        
        return results
    
    def compare_models(self):
        """Compare performance across all fitted models."""
        print("\n" + "="*80)
        print("MODEL COMPARISON SUMMARY")
        print("="*80)
        
        results_df = pd.DataFrame(list(self.results.values()))
        print(results_df.to_string(index=False))
        
        # Identify best model by AUC
        best_model = max(self.results, key=lambda x: self.results[x]['AUC'])
        print(f"\nBest Model (by AUC): {best_model}")
        print(f"AUC Score: {self.results[best_model]['AUC']:.4f}")
        
        return results_df
    
    def plot_roc_curves(self, figsize=(12, 6)):
        """Plot ROC curves for all models."""
        plt.figure(figsize=figsize)
        
        for model_name, y_pred_proba in self.predictions.items():
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})', linewidth=2)
        
        # Plot random classifier
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.500)', linewidth=1)
        
        plt.xlabel('False Positive Rate', fontsize=11)
        plt.ylabel('True Positive Rate', fontsize=11)
        plt.title('ROC Curves - Model Comparison', fontsize=13, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curves.png', dpi=300)
        print("ROC curves saved to 'roc_curves.png'")
        plt.show()
    
    def predict_probability(self, new_data):
        """
        Predict default probability for new loans.
        
        Args:
            new_data (pd.DataFrame): New loan data with same features
            
        Returns:
            pd.DataFrame: Default probabilities and risk classification
        """
        # Preprocess new data
        df_new = new_data.copy()
        if 'LoanID' in df_new.columns:
            loan_ids = df_new['LoanID']
            df_new = df_new.drop('LoanID', axis=1)
        
        # Drop target column if present
        if 'Default' in df_new.columns:
            df_new = df_new.drop('Default', axis=1)
        
        # Encode categoricals
        categorical_cols = df_new.select_dtypes(include=['object']).columns
        le = LabelEncoder()
        for col in categorical_cols:
            df_new[col] = le.fit_transform(df_new[col].astype(str))
        
        # Scale
        df_new_scaled = self.scaler.transform(df_new)
        
        # Get best model
        best_model_name = max(self.results, key=lambda x: self.results[x]['AUC'])
        best_model = self.models[best_model_name]
        
        # Predict probability
        if best_model_name == 'Logistic':
            proba = best_model.predict_proba(df_new_scaled)[:, 1]
        else:
            proba = np.clip(best_model.predict(df_new_scaled), 0, 1)
        
        # Classify risk
        risk_class = pd.cut(proba, bins=[0, 0.33, 0.67, 1.0], 
                           labels=['Low', 'Medium', 'High'])
        
        results_df = pd.DataFrame({
            'Probability_of_Default': proba,
            'Risk_Classification': risk_class
        })
        
        if 'loan_ids' in locals():
            results_df.insert(0, 'LoanID', loan_ids.values)
        
        return results_df
    
    def calculate_expected_loss(self, exposure_at_default, loss_given_default):
        """
        Calculate Expected Loss (EL) for allowance for doubtful accounts.
        
        EL = Probability of Default × Exposure at Default × Loss Given Default
        
        Args:
            exposure_at_default (float or array): Exposure amount(s)
            loss_given_default (float or array): Loss rate(s), typically 0.3-0.5
            
        Returns:
            pd.DataFrame: Expected loss calculations
        """
        best_model_name = max(self.results, key=lambda x: self.results[x]['AUC'])
        best_model = self.models[best_model_name]
        
        # Get default probabilities for test set
        if best_model_name == 'Logistic':
            pd_probabilities = best_model.predict_proba(self.X_test_scaled)[:, 1]
        else:
            pd_probabilities = np.clip(best_model.predict(self.X_test_scaled), 0, 1)
        
        # Calculate expected loss
        el = pd_probabilities * exposure_at_default * loss_given_default
        
        results_df = pd.DataFrame({
            'Probability_of_Default': pd_probabilities,
            'Exposure_at_Default': exposure_at_default,
            'Loss_Given_Default': loss_given_default,
            'Expected_Loss': el
        })
        
        print("\n" + "="*60)
        print("EXPECTED LOSS CALCULATIONS (ALDA Provisioning)")
        print("="*60)
        print(f"Total Expected Loss: ${results_df['Expected_Loss'].sum():,.2f}")
        print(f"Average EL per Loan: ${results_df['Expected_Loss'].mean():,.2f}")
        print(f"Min EL: ${results_df['Expected_Loss'].min():,.2f}")
        print(f"Max EL: ${results_df['Expected_Loss'].max():,.2f}")
        
        return results_df


def main():
    """Run complete credit risk analysis."""
    
    # Load data
    print("Loading data...")
    df = pd.read_csv('Loan_default.csv')
    print(f"Dataset shape: {df.shape}")
    
    # Initialize model
    model = CreditRiskModel(df)
    
    # Train/test split
    model.train_test_split(test_size=0.2, random_state=42)
    
    # Fit models
    model.fit_logistic_regression()
    model.fit_ridge_regression(alpha=1.0)
    model.fit_lasso_regression(alpha=0.001)
    
    # Compare models
    comparison = model.compare_models()
    
    # Plot ROC curves
    model.plot_roc_curves()
    
    # Example: Predict for new loans
    print("\n" + "="*60)
    print("PREDICTION ON NEW LOANS")
    print("="*60)
    sample_loans = df.iloc[:10].copy()
    predictions = model.predict_probability(sample_loans)
    print(predictions.head(10))
    
    # Example: Calculate expected loss
    print("\n" + "="*60)
    print("EXAMPLE: EXPECTED LOSS CALCULATION")
    print("="*60)
    # Assume average exposure = loan amount, LGD = 0.4
    exposure = df.loc[model.y_test.index, 'LoanAmount'].values
    lgd = 0.4
    el_results = model.calculate_expected_loss(exposure, lgd)
    print(f"\nTop 10 Highest Expected Losses:")
    print(el_results.nlargest(10, 'Expected_Loss')[['Probability_of_Default', 'Expected_Loss']])
    
    return model


if __name__ == "__main__":
    model = main()
    print("\n" + "="*80)
    print("Credit risk analysis complete!")
    print("="*80)
