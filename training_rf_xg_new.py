import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

class APKModelTrainer:
    def __init__(self, csv_path, test_size=0.2, random_state=42):
        """
        Initialize the trainer with dataset path.
        
        Args:
            csv_path: Path to the merged features CSV file
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
        """
        self.csv_path = csv_path
        self.test_size = test_size
        self.random_state = random_state
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.ensemble_model = None
        
    def load_and_prepare_data(self):
        """Load the dataset and prepare features and labels."""
        print("Loading dataset...")
        self.data = pd.read_csv(self.csv_path)
        print(f"Dataset shape: {self.data.shape}")
        
        # Display basic info about the dataset
        print(f"Total samples: {len(self.data)}")
        if 'label' in self.data.columns:
            print(f"Benign samples: {sum(self.data['label'] == 0)}")
            print(f"Malicious samples: {sum(self.data['label'] == 1)}")
        
        # Prepare features and labels
        # Remove non-feature columns
        feature_cols = [col for col in self.data.columns 
                       if col not in ['apk_name', 'label']]
        
        X = self.data[feature_cols]
        y = self.data['label'] if 'label' in self.data.columns else None
        
        if y is None:
            raise ValueError("No 'label' column found in the dataset!")
        
        # Handle missing values
        print("Handling missing values...")
        X = X.fillna(0)
        
        # Split the data
        print("Splitting data into train/test sets...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, 
            stratify=y
        )
        
        # Scale the features
        print("Scaling features...")
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_random_forest(self, n_estimators=100, max_depth=None, min_samples_split=2):
        """Train Random Forest model."""
        print("\n=== Training Random Forest ===")
        
        rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Train the model
        rf_model.fit(self.X_train, self.y_train)
        
        # Evaluate the model
        train_score = rf_model.score(self.X_train, self.y_train)
        test_score = rf_model.score(self.X_test, self.y_test)
        
        print(f"Random Forest - Training Accuracy: {train_score:.4f}")
        print(f"Random Forest - Testing Accuracy: {test_score:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(rf_model, self.X_train, self.y_train, 
                                   cv=5, scoring='accuracy')
        print(f"Random Forest - CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        self.models['random_forest'] = rf_model
        return rf_model
    
    def train_xgboost(self, n_estimators=100, max_depth=6, learning_rate=0.1):
        """Train XGBoost model."""
        print("\n=== Training XGBoost ===")
        
        xgb_model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=self.random_state,
            eval_metric='logloss'
        )
        
        # Train the model
        xgb_model.fit(self.X_train, self.y_train)
        
        # Evaluate the model
        train_score = xgb_model.score(self.X_train, self.y_train)
        test_score = xgb_model.score(self.X_test, self.y_test)
        
        print(f"XGBoost - Training Accuracy: {train_score:.4f}")
        print(f"XGBoost - Testing Accuracy: {test_score:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(xgb_model, self.X_train, self.y_train, 
                                   cv=5, scoring='accuracy')
        print(f"XGBoost - CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        self.models['xgboost'] = xgb_model
        return xgb_model
    
    def create_ensemble(self, voting='soft'):
        """Create ensemble model combining Random Forest and XGBoost."""
        print("\n=== Creating Ensemble Model ===")
        
        if 'random_forest' not in self.models or 'xgboost' not in self.models:
            raise ValueError("Both Random Forest and XGBoost models must be trained first!")
        
        # Create ensemble
        self.ensemble_model = VotingClassifier(
            estimators=[
                ('rf', self.models['random_forest']),
                ('xgb', self.models['xgboost'])
            ],
            voting=voting
        )
        
        # Train ensemble
        self.ensemble_model.fit(self.X_train, self.y_train)
        
        # Evaluate ensemble
        train_score = self.ensemble_model.score(self.X_train, self.y_train)
        test_score = self.ensemble_model.score(self.X_test, self.y_test)
        
        print(f"Ensemble - Training Accuracy: {train_score:.4f}")
        print(f"Ensemble - Testing Accuracy: {test_score:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.ensemble_model, self.X_train, self.y_train, 
                                   cv=5, scoring='accuracy')
        print(f"Ensemble - CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return self.ensemble_model
    
    def evaluate_models(self, save_plots=True):
        """Comprehensive evaluation of all models."""
        print("\n=== Model Evaluation ===")
        
        models_to_evaluate = self.models.copy()
        if self.ensemble_model:
            models_to_evaluate['ensemble'] = self.ensemble_model
        
        results = {}
        
        for name, model in models_to_evaluate.items():
            print(f"\n--- {name.upper()} ---")
            
            # Predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Classification report
            print("Classification Report:")
            print(classification_report(self.y_test, y_pred))
            
            # ROC AUC
            auc_score = roc_auc_score(self.y_test, y_pred_proba)
            print(f"ROC AUC Score: {auc_score:.4f}")
            
            results[name] = {
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'auc_score': auc_score
            }
            
            if save_plots:
                # Confusion Matrix
                plt.figure(figsize=(8, 6))
                cm = confusion_matrix(self.y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f'{name.title()} - Confusion Matrix')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.savefig(f'{name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
                plt.show()
                
                # ROC Curve
                plt.figure(figsize=(8, 6))
                fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
                plt.plot(fpr, tpr, label=f'{name.title()} (AUC = {auc_score:.3f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'{name.title()} - ROC Curve')
                plt.legend(loc="lower right")
                plt.savefig(f'{name}_roc_curve.png', dpi=300, bbox_inches='tight')
                plt.show()
        
        return results
    
    def get_feature_importance(self, top_n=20):
        """Get feature importance from trained models."""
        print(f"\n=== Top {top_n} Feature Importance ===")
        
        feature_names = self.X_train.columns.tolist()
        
        # Random Forest feature importance
        if 'random_forest' in self.models:
            rf_importance = self.models['random_forest'].feature_importances_
            rf_features = list(zip(feature_names, rf_importance))
            rf_features.sort(key=lambda x: x[1], reverse=True)
            
            print("\n--- Random Forest Top Features ---")
            for i, (feature, importance) in enumerate(rf_features[:top_n]):
                print(f"{i+1:2d}. {feature:<30} {importance:.6f}")
        
        # XGBoost feature importance
        if 'xgboost' in self.models:
            xgb_importance = self.models['xgboost'].feature_importances_
            xgb_features = list(zip(feature_names, xgb_importance))
            xgb_features.sort(key=lambda x: x[1], reverse=True)
            
            print("\n--- XGBoost Top Features ---")
            for i, (feature, importance) in enumerate(xgb_features[:top_n]):
                print(f"{i+1:2d}. {feature:<30} {importance:.6f}")
        
        # Plot feature importance comparison
        if len(self.models) >= 2:
            plt.figure(figsize=(12, 8))
            
            # Get top features from RF
            top_rf_features = [f[0] for f in rf_features[:top_n]]
            top_rf_importance = [f[1] for f in rf_features[:top_n]]
            
            # Get corresponding XGB importance for the same features
            xgb_importance_dict = dict(xgb_features)
            top_xgb_importance = [xgb_importance_dict.get(f, 0) for f in top_rf_features]
            
            x = np.arange(len(top_rf_features))
            width = 0.35
            
            plt.bar(x - width/2, top_rf_importance, width, label='Random Forest', alpha=0.8)
            plt.bar(x + width/2, top_xgb_importance, width, label='XGBoost', alpha=0.8)
            
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.title('Feature Importance Comparison')
            plt.xticks(x, top_rf_features, rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            plt.savefig('feature_importance_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def save_models(self, model_dir='models'):
        """Save all trained models and scaler."""
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        print(f"\n=== Saving Models to {model_dir}/ ===")
        
        # Save individual models
        for name, model in self.models.items():
            model_path = os.path.join(model_dir, f'{name}_model.joblib')
            joblib.dump(model, model_path)
            print(f"Saved {name} model to {model_path}")
        
        # Save ensemble model
        if self.ensemble_model:
            ensemble_path = os.path.join(model_dir, 'ensemble_model2.joblib')
            joblib.dump(self.ensemble_model, ensemble_path)
            print(f"Saved ensemble model to {ensemble_path}")
        
        # Save scaler
        scaler_path = os.path.join(model_dir, 'scaler2.joblib')
        joblib.dump(self.scaler, scaler_path)
        print(f"Saved scaler to {scaler_path}")
        
        # Save feature names for later use
        feature_names_path = os.path.join(model_dir, 'feature_names2.joblib')
        joblib.dump(self.X_train.columns.tolist(), feature_names_path)
        print(f"Saved feature names to {feature_names_path}")
        
        print("All models saved successfully!")
    
    def train_complete_pipeline(self):
        """Run the complete training pipeline."""
        print("Starting complete training pipeline...")
        
        # Load and prepare data
        self.load_and_prepare_data()
        
        # Train individual models
        self.train_random_forest(n_estimators=200, max_depth=20)
        self.train_xgboost(n_estimators=200, max_depth=8, learning_rate=0.1)
        
        # Create ensemble
        self.create_ensemble(voting='soft')
        
        # Evaluate all models
        results = self.evaluate_models()
        
        # Show feature importance
        self.get_feature_importance(top_n=25)
        
        # Save models
        self.save_models()
        
        print("\n🎉 Training pipeline completed successfully!")
        return results


def main():
    """Main function to run the training pipeline."""
    # Configuration
    CSV_PATH = "D:/VScode programs/Mini/final_merged_with_sensitive_flags_09.csv"  # Update this path
    
    # Initialize trainer
    trainer = APKModelTrainer(CSV_PATH, test_size=0.2, random_state=42)
    
    try:
        # Run complete training pipeline
        results = trainer.train_complete_pipeline()
        
        # Print summary
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        
        for model_name, result in results.items():
            print(f"{model_name.upper()}: AUC = {result['auc_score']:.4f}")
        
        print("\nBest model files saved in 'models/' directory")
        print("Use 'ensemble_model.joblib2' for best performance")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()