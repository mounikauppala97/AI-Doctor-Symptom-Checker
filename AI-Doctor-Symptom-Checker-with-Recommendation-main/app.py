from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
from datetime import datetime
from symptom_model import SymptomModel
import re

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ai_doctor.db'
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize the symptom model
symptom_model = SymptomModel()
model_path = 'model/symptom_model.joblib'

# Try to load existing model, if not available, train a new one
if os.path.exists(model_path):
    print(f"Loading existing model from {model_path}")
    if not symptom_model.load_model(model_path):
        print("Failed to load existing model. Training new model...")
        symptom_model.train('dataset/Training.csv')
else:
    print("No existing model found. Training new model...")
    symptom_model.train('dataset/Training.csv')

# Create a list of symptoms with their descriptions for autocomplete
symptom_suggestions = []
for symptom in symptom_model.symptoms:
    symptom_suggestions.append({
        'name': symptom.replace('_', ' '),
        'description': ''  # If you have descriptions, add them here
    })

# User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    name = db.Column(db.String(100))
    symptoms = db.relationship('Symptom', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# Symptom Model
class Symptom(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    symptoms = db.Column(db.Text, nullable=False)
    diagnosis = db.Column(db.Text)
    recommendations = db.Column(db.Text)
    confidence = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def preprocess_symptoms(symptoms_text):
    # Convert to lowercase
    symptoms_text = symptoms_text.lower()
    
    # Split by commas and clean each symptom
    symptom_parts = [part.strip() for part in symptoms_text.split(',')]
    
    # Extract matching symptoms
    found_symptoms = []
    for part in symptom_parts:
        # Remove special characters and extra spaces
        cleaned_part = re.sub(r'[^\w\s]', ' ', part)
        cleaned_part = re.sub(r'\s+', ' ', cleaned_part).strip()
        
        if not cleaned_part:
            continue
            
        # Check each symptom against the cleaned part
        best_match = None
        best_match_score = 0
        
        for symptom in symptom_model.symptoms:
            symptom_name = symptom.replace('_', ' ')
            
            # Calculate similarity score
            if cleaned_part == symptom_name:
                # Exact match
                best_match = symptom
                best_match_score = 1.0
                break
            elif cleaned_part in symptom_name or symptom_name in cleaned_part:
                # Partial match - calculate similarity
                score = len(set(cleaned_part.split()) & set(symptom_name.split())) / max(len(cleaned_part.split()), len(symptom_name.split()))
                if score > best_match_score and score > 0.5:  # Require at least 50% similarity
                    best_match = symptom
                    best_match_score = score
        
        if best_match and best_match not in found_symptoms:
            found_symptoms.append(best_match)
    
    return ' '.join(found_symptoms)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        name = request.form.get('name')
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered')
            return redirect(url_for('signup'))
        
        user = User(email=email, name=name)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('dashboard'))
        
        flash('Invalid email or password')
    return render_template('login.html')

@app.route('/get_symptom_suggestions')
def get_symptom_suggestions():
    query = request.args.get('query', '').lower()
    if not query:
        return jsonify([])
    
    suggestions = []
    for symptom in symptom_suggestions:
        if query in symptom['name'].lower() or query in symptom['description'].lower():
            suggestions.append(symptom)
    
    return jsonify(suggestions[:5])  # Return top 5 matches

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        
        if not symptoms:
            flash('Please enter your symptoms')
            return render_template('dashboard.html')
        
        # Preprocess symptoms
        processed_symptoms = preprocess_symptoms(symptoms)
        if not processed_symptoms:
            flash('No recognizable symptoms found. Please describe your symptoms in detail.')
            return render_template('dashboard.html')
        
        # Convert processed_symptoms string to list
        symptoms_list = processed_symptoms.split()
        print(symptoms_list)
        
        # Use the ML model to analyze symptoms
        predictions = symptom_model.predict(symptoms_list)
        print(predictions)
        
        if not predictions:
            flash('Unable to make a diagnosis. Please provide more detailed symptoms.')
            return render_template('dashboard.html')
        
        # Get top prediction
        top_prediction = predictions[0]
        diagnosis = top_prediction['disease']
        
        # Generate dynamic recommendations based on the disease
        recommendations = _get_recommendations(diagnosis)
        
        # Save to database
        new_symptom = Symptom(
            symptoms=symptoms,
            diagnosis=diagnosis,
            recommendations=recommendations,
            confidence=top_prediction['probability'] * 100,
            user_id=current_user.id
        )
        db.session.add(new_symptom)
        db.session.commit()
        
        # Always show results
        return render_template('dashboard.html', 
                            diagnosis=diagnosis, 
                            recommendations=recommendations,
                            processed_symptoms=processed_symptoms)
    
    return render_template('dashboard.html')

@app.route('/history')
@login_required
def history():
    symptoms = Symptom.query.filter_by(user_id=current_user.id).order_by(Symptom.timestamp.desc()).all()
    return render_template('history.html', symptoms=symptoms)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

def _get_recommendations(diagnosis):
    """Generate recommendations based on the diagnosed disease"""
    recommendations = {
        "Fungal infection": [
            "Keep the affected area clean and dry",
            "Use antifungal medications as prescribed",
            "Avoid sharing personal items",
            "Wear breathable clothing"
        ],
        "Allergy": [
            "Identify and avoid triggers",
            "Take antihistamines as prescribed",
            "Keep a symptom diary",
            "Consider allergy testing"
        ],
        "GERD": [
            "Avoid spicy and acidic foods",
            "Eat smaller, more frequent meals",
            "Don't lie down immediately after eating",
            "Elevate the head of your bed"
        ],
        "Diabetes": [
            "Monitor blood sugar regularly",
            "Follow a balanced diet",
            "Exercise regularly",
            "Take medications as prescribed"
        ],
        "Hypertension": [
            "Reduce salt intake",
            "Exercise regularly",
            "Monitor blood pressure",
            "Take medications as prescribed"
        ],
        "Malaria": [
            "Take prescribed antimalarial medications",
            "Rest and stay hydrated",
            "Monitor for severe symptoms",
            "Use mosquito prevention measures"
        ],
        "Chicken pox": [
            "Keep the affected area clean and dry",
            "Use calamine lotion for itching",
            "Take prescribed antiviral medications",
            "Stay isolated until all blisters have crusted"
        ],
        "Dengue": [
            "Rest and stay hydrated",
            "Monitor for severe symptoms",
            "Take prescribed medications",
            "Use mosquito prevention measures"
        ],
        "Typhoid": [
            "Take prescribed antibiotics",
            "Rest and stay hydrated",
            "Follow a light diet",
            "Monitor for complications"
        ],
        "Hepatitis A": [
            "Rest and stay hydrated",
            "Follow a low-fat diet",
            "Avoid alcohol",
            "Take prescribed medications"
        ],
        "Hepatitis B": [
            "Take prescribed antiviral medications",
            "Rest and stay hydrated",
            "Avoid alcohol",
            "Monitor liver function"
        ],
        "Hepatitis C": [
            "Take prescribed antiviral medications",
            "Rest and stay hydrated",
            "Avoid alcohol",
            "Monitor liver function"
        ],
        "Hepatitis D": [
            "Take prescribed antiviral medications",
            "Rest and stay hydrated",
            "Avoid alcohol",
            "Monitor liver function"
        ],
        "Hepatitis E": [
            "Rest and stay hydrated",
            "Follow a low-fat diet",
            "Avoid alcohol",
            "Take prescribed medications"
        ],
        "Alcoholic hepatitis": [
            "Stop alcohol consumption",
            "Follow a balanced diet",
            "Take prescribed medications",
            "Monitor liver function"
        ],
        "Tuberculosis": [
            "Take prescribed antibiotics",
            "Complete the full course of treatment",
            "Rest and stay hydrated",
            "Follow infection control measures"
        ],
        "Common Cold": [
            "Rest and stay hydrated",
            "Use over-the-counter cold medications",
            "Use a humidifier",
            "Get adequate sleep"
        ],
        "Pneumonia": [
            "Take prescribed antibiotics",
            "Rest and stay hydrated",
            "Use a humidifier",
            "Monitor for severe symptoms"
        ],
        "Dimorphic hemmorhoids(piles)": [
            "Use prescribed topical medications",
            "Maintain good hygiene",
            "Increase fiber intake",
            "Stay hydrated"
        ],
        "Heart attack": [
            "Seek immediate medical attention",
            "Take prescribed medications",
            "Follow cardiac rehabilitation",
            "Make lifestyle changes"
        ],
        "Varicose veins": [
            "Wear compression stockings",
            "Exercise regularly",
            "Elevate legs when resting",
            "Maintain a healthy weight"
        ],
        "Hypothyroidism": [
            "Take prescribed thyroid hormone",
            "Follow up with regular blood tests",
            "Maintain a balanced diet",
            "Exercise regularly"
        ],
        "Hyperthyroidism": [
            "Take prescribed medications",
            "Follow up with regular blood tests",
            "Manage stress",
            "Maintain a balanced diet"
        ],
        "Hypoglycemia": [
            "Eat regular meals",
            "Carry fast-acting carbohydrates",
            "Monitor blood sugar",
            "Follow prescribed treatment"
        ],
        "Osteoarthristis": [
            "Exercise regularly",
            "Maintain a healthy weight",
            "Use prescribed pain medications",
            "Consider physical therapy"
        ],
        "Arthritis": [
            "Exercise regularly",
            "Maintain a healthy weight",
            "Use prescribed medications",
            "Consider physical therapy"
        ],
        "(vertigo) Paroymsal  Positional Vertigo": [
            "Perform prescribed exercises (e.g., Epley maneuver)",
            "Avoid sudden head movements",
            "Sleep with head elevated",
            "Consult a specialist if symptoms persist"
        ],
        "Acne": [
            "Keep skin clean",
            "Use prescribed topical medications",
            "Avoid touching face",
            "Follow a healthy diet"
        ],
        "Urinary tract infection": [
            "Take prescribed antibiotics",
            "Stay hydrated",
            "Urinate frequently",
            "Maintain good hygiene"
        ],
        "Psoriasis": [
            "Use prescribed topical treatments",
            "Moisturize regularly",
            "Avoid triggers",
            "Consider phototherapy"
        ],
        "Impetigo": [
            "Keep affected area clean",
            "Take prescribed antibiotics",
            "Avoid scratching",
            "Maintain good hygiene"
        ],
        "Chronic cholestasis": [
            "Follow a low-fat diet",
            "Take prescribed medications",
            "Monitor liver function",
            "Consult a hepatologist"
        ],
        "Cervical spondylosis": [
            "Perform neck exercises as advised",
            "Maintain good posture",
            "Use a supportive pillow",
            "Consult a physiotherapist"
        ],
        "Gastroenteritis": [
            "Stay hydrated",
            "Follow a bland diet",
            "Rest",
            "Consult a doctor if symptoms persist"
        ],
        "Migraine": [
            "Rest in a quiet, dark room",
            "Take prescribed medications",
            "Avoid known triggers",
            "Stay hydrated"
        ],
        "Jaundice": [
            "Rest",
            "Stay hydrated",
            "Avoid alcohol",
            "Consult a doctor for underlying cause"
        ],
        "Paralysis (brain hemorrhage)": [
            "Seek immediate medical attention",
            "Follow prescribed rehabilitation",
            "Monitor vital signs",
            "Provide supportive care"
        ],
        "Peptic ulcer diseae": [
            "Take prescribed medications",
            "Avoid spicy and acidic foods",
            "Eat smaller, frequent meals",
            "Avoid NSAIDs and alcohol"
        ],
        "Drug Reaction": [
            "Stop the suspected medication (under medical supervision)",
            "Consult a healthcare provider immediately",
            "Monitor for severe symptoms",
            "Follow prescribed treatment"
        ],
        "AIDS": [
            "Take antiretroviral therapy (ART) as prescribed",
            "Maintain a healthy diet",
            "Practice safe hygiene and avoid infections",
            "Regularly consult your healthcare provider"
        ],
        "Bronchial Asthma": [
            "Use inhalers and medications as prescribed",
            "Avoid known triggers (allergens, smoke, etc.)",
            "Monitor your breathing and symptoms",
            "Have an asthma action plan and seek help if symptoms worsen"
        ]
    }
    
    # Get recommendations for the diagnosed disease
    disease_recommendations = recommendations.get(diagnosis, [
        "Consult with a healthcare provider for proper diagnosis and treatment",
        "Keep track of your symptoms",
        "Follow any prescribed medications",
        "Maintain a healthy lifestyle"
    ])
    
    return "\n".join([f"• {rec}" for rec in disease_recommendations])

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True) 