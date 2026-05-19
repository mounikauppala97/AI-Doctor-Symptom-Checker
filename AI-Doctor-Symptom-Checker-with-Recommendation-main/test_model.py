from symptom_model import SymptomModel

def test_model():
    # Initialize the model
    model = SymptomModel()
    
    # Train the model
    print("Training model...")
    train_accuracy, val_accuracy = model.train('dataset/Training.csv')
    print(f"Training accuracy: {train_accuracy:.2f}")
    print(f"Validation accuracy: {val_accuracy:.2f}")
    
    # Save the model
    print("\nSaving model...")
    model.save_model('model/symptom_model.joblib')
    
    # Test predictions
    print("\nTesting predictions...")
    test_symptoms = ['itching', 'skin_rash', 'nodal_skin_eruptions']
    predictions = model.predict(test_symptoms)
    
    print("\nTop 3 predictions:")
    for pred in predictions:
        print(f"Disease: {pred['disease']}, Probability: {pred['probability']:.2f}")

if __name__ == "__main__":
    test_model() 