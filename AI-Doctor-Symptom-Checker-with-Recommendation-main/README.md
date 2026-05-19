# AI Doctor - Symptom Checker with Recommendations

An intelligent AI-powered web application that helps users analyze their symptoms and receive personalized health recommendations. The system uses machine learning to provide accurate symptom analysis and medical advice.

## 🌟 Features

- **Smart Symptom Analysis**: Real-time analysis of symptoms using advanced AI algorithms
- **Personalized Recommendations**: Tailored health advice based on symptom analysis
- **Health History Tracking**: Comprehensive record of symptoms and diagnoses
- **User Authentication**: Secure login and registration system
- **Responsive Design**: Mobile-friendly interface for all devices
- **Interactive Dashboard**: Easy-to-use interface for symptom input and results

## 🛠️ Technology Stack

- **Backend**: Python, Flask
- **Frontend**: HTML, Tailwind CSS, JavaScript
- **Database**: SQLite
- **Machine Learning**: scikit-learn, pandas, numpy
- **Authentication**: Flask-Login
- **Other Libraries**: joblib, tqdm

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/AI-Doctor-Symptom-Checker-with-Recommendation.git
   cd AI-Doctor-Symptom-Checker-with-Recommendation
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Initialize the database:
   ```bash
   flask db init
   flask db migrate
   flask db upgrade
   ```

## 💻 Usage

1. Start the application:
   ```bash
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

3. Create an account or login to access the dashboard

4. Enter your symptoms in the dashboard to receive analysis and recommendations

## 🤖 How It Works

1. **Data Collection**: The system uses a comprehensive dataset of symptoms and diseases
2. **Feature Engineering**: Advanced feature engineering techniques for better accuracy
3. **Model Training**: Ensemble learning approach combining multiple models
4. **Prediction**: Real-time symptom analysis and disease prediction
5. **Recommendations**: Personalized health advice based on the analysis

## 📊 Model Performance

The system uses an ensemble of machine learning models:
- Random Forest Classifier
- Gradient Boosting Classifier
- Voting Classifier for final predictions

The model achieves high accuracy through:
- Cross-validation
- Feature engineering
- Hyperparameter tuning
- Ensemble learning

## 🔒 Security

- Secure password hashing
- User authentication
- Protected routes
- Input validation
- SQL injection prevention

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This application is not a replacement for professional medical advice. Always consult with a healthcare provider for proper diagnosis and treatment.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

For any questions or suggestions, please open an issue in the repository.

## 🙏 Acknowledgments

- Dataset providers
- Open-source community
- Contributors and maintainers