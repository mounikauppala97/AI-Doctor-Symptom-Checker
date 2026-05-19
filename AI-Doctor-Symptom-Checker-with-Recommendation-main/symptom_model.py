import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from tqdm import tqdm
from sklearn.model_selection import KFold

class SymptomModel:
    def __init__(self):
        # Initialize multiple models for ensemble
        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.gb_model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.model = VotingClassifier(
            estimators=[
                ('rf', self.rf_model),
                ('gb', self.gb_model)
            ],
            voting='soft'
        )
        self.label_encoder = LabelEncoder()
        self.symptoms = None
        self.diseases = None
        
    def load_data(self, training_file):
        """Load and preprocess the training data"""
        print("Starting to load data...")
        # Read the CSV file
        df = pd.read_csv(training_file)
        print(f"CSV file loaded. Shape: {df.shape}")
        
        # Drop any columns that are completely empty
        df = df.dropna(axis=1, how='all')
        print("Dropped empty columns")
        
        # Find the prognosis column
        columns = df.columns.tolist()
        if columns[-1].strip().lower() != 'prognosis':
            for i, col in enumerate(columns):
                if col.strip().lower() == 'prognosis':
                    prognosis_col = col
                    break
            else:
                raise ValueError("No 'prognosis' column found in the dataset.")
        else:
            prognosis_col = columns[-1]
        print(f"Found prognosis column: {prognosis_col}")
        
        self.symptoms = [col for col in columns if col != prognosis_col]
        print(f"Number of symptoms: {len(self.symptoms)}")
        
        # Separate features (X) and target (y)
        X = df[self.symptoms].astype(float)
        y = df[prognosis_col].astype(str).str.strip()
        print("Separated features and target")
        
        # Add feature engineering
        print("Starting feature engineering...")
        X = self._engineer_features(X)
        print("Feature engineering completed")
        
        # Encode the disease labels
        y = self.label_encoder.fit_transform(y)
        self.diseases = self.label_encoder.classes_
        print(f"Number of unique diseases: {len(self.diseases)}")
        
        return X, y
    
    def _engineer_features(self, X):
        """Add engineered features to improve model performance"""
        print("Creating feature dictionary...")
        # Create a new DataFrame to store all features
        features = {}
        
        # Add symptom count as a feature
        features['symptom_count'] = X.sum(axis=1)
        print("Added symptom count feature")
        
        # Add symptom combinations (interactions) more efficiently
        n_symptoms = len(self.symptoms)
        print(f"Creating {n_symptoms * (n_symptoms-1) // 2} interaction features...")
        for i in range(n_symptoms):
            for j in range(i+1, n_symptoms):
                col_name = f'interaction_{i}_{j}'
                features[col_name] = X[self.symptoms[i]] * X[self.symptoms[j]]
        print("Interaction features created")
        
        # Combine all features at once using pd.concat
        print("Combining all features...")
        result = pd.concat([X, pd.DataFrame(features, index=X.index)], axis=1)
        print("Features combined successfully")
        return result
    
    def train(self, training_file):
        """Train the model on the provided data"""
        try:
            print("\n=== Starting Model Training ===")
            # Load and preprocess the data
            print("Loading and preprocessing data...")
            X, y = self.load_data(training_file)
            print("Data loaded and preprocessed")
            
            # Split the data into training and validation sets
            print("Splitting data into train and validation sets...")
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            print("Data split completed")
            
            # Perform cross-validation with fewer folds and progress tracking
            print("\nPerforming cross-validation (this may take a few minutes)...")
            
            # Use 3 folds instead of 5 for faster training
            kf = KFold(n_splits=3, shuffle=True, random_state=42)
            cv_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(tqdm(kf.split(X_train), total=3, desc="Cross-validation")):
                # Handle both DataFrame and numpy array indexing
                if isinstance(X_train, pd.DataFrame):
                    X_fold_train = X_train.iloc[train_idx]
                    X_fold_val = X_train.iloc[val_idx]
                else:
                    X_fold_train = X_train[train_idx]
                    X_fold_val = X_train[val_idx]
                
                y_fold_train = y_train[train_idx]
                y_fold_val = y_train[val_idx]
                
                # Train and evaluate
                self.model.fit(X_fold_train, y_fold_train)
                score = self.model.score(X_fold_val, y_fold_val)
                cv_scores.append(score)
                print(f"Fold {fold + 1} score: {score:.3f}")
            
            cv_scores = np.array(cv_scores)
            print(f"\nCross-validation scores: {cv_scores}")
            print(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            # Train the final model
            print("\nTraining the final model...")
            self.model.fit(X_train, y_train)
            print("Model training completed")
            
            # Calculate and print accuracy
            train_accuracy = self.model.score(X_train, y_train)
            val_accuracy = self.model.score(X_val, y_val)
            print(f"\nTraining accuracy: {train_accuracy:.3f}")
            print(f"Validation accuracy: {val_accuracy:.3f}")
            
            # Print detailed classification report
            y_pred = self.model.predict(X_val)
            print("\nClassification Report:")
            print(classification_report(y_val, y_pred, target_names=self.diseases))
            
            # Create model directory if it doesn't exist
            os.makedirs('model', exist_ok=True)
            
            # Save the model
            print("\nSaving the model...")
            self.save_model('model/symptom_model.joblib')
            print("=== Model Training Completed ===\n")
            
            return train_accuracy, val_accuracy
        except Exception as e:
            print(f"\nError training model: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def predict(self, symptoms):
        """Predict disease based on symptoms"""
        try:
            # Convert symptoms to the same format as training data
            symptom_vector = np.zeros(len(self.symptoms))
            for symptom in symptoms:
                if symptom in self.symptoms:
                    idx = self.symptoms.index(symptom)
                    symptom_vector[idx] = 1
            
            # Create feature vector with engineered features
            X = pd.DataFrame([symptom_vector], columns=self.symptoms)
            X = self._engineer_features(X)
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
            
            # Get the predicted disease name
            predicted_disease = self.label_encoder.inverse_transform([prediction])[0]
            
            # Calculate confidence score based on multiple factors
            base_confidence = probabilities[prediction]
            symptom_coverage = sum(symptom_vector) / len(self.symptoms)
            confidence_score = (base_confidence * 0.7 + symptom_coverage * 0.3)  # Weighted combination
            
            # Get top prediction with adjusted confidence
            predictions = [{
                "disease": predicted_disease,
                "probability": float(confidence_score)
            }]
            
            return predictions
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            return None
    
    def save_model(self, model_path):
        """Save the trained model and related data"""
        try:
            model_data = {
                'model': self.model,
                'label_encoder': self.label_encoder,
                'symptoms': self.symptoms,
                'diseases': self.diseases
            }
            joblib.dump(model_data, model_path)
            print(f"Model saved successfully to {model_path}")
        except Exception as e:
            print(f"Error saving model: {str(e)}")
    
    def load_model(self, model_path):
        """Load a trained model and related data"""
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.label_encoder = model_data['label_encoder']
            self.symptoms = model_data['symptoms']
            self.diseases = model_data['diseases']
            print(f"Model loaded successfully from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

# Create a singleton instance
symptom_model = SymptomModel()

# Train the model if this file is run directly
if __name__ == "__main__":
    model_path = 'model/symptom_model.joblib'
    train_file = 'dataset/Training.csv'
    
    # Check if model already exists
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        if symptom_model.load_model(model_path):
            print("Model loaded successfully!")
        else:
            print("Failed to load existing model. Training new model...")
            if os.path.exists(train_file):
                train_accuracy, val_accuracy = symptom_model.train(train_file)
                if train_accuracy is not None:
                    print("Model training completed successfully!")
            else:
                print(f"Training file not found: {train_file}")
    else:
        print("No existing model found. Training new model...")
        if os.path.exists(train_file):
            train_accuracy, val_accuracy = symptom_model.train(train_file)
            if train_accuracy is not None:
                print("Model training completed successfully!")
        else:
            print(f"Training file not found: {train_file}")