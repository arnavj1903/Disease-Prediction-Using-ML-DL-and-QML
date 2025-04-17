# pylint: disable=R0903
# pylint: disable=E0602
# pylint: disable=W0612

"""
Medical Disease Prediction Web Application.

This Flask application provides a platform for medical professionals to predict disease
outcomes using various machine learning models. It supports heart attack, breast cancer,
diabetes, and lung cancer predictions.
"""

import logging
import pickle
from uuid import uuid4

from dotenv import load_dotenv
from flask import (Flask, redirect, render_template, request, session,
                   url_for)
from flask_sqlalchemy import SQLAlchemy

# Load environment variables
load_dotenv()

# Initialize Flask application
app = Flask(_name_)
app.config['SECRET_KEY'] = str(uuid4())  # Use a random UUID for security
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///medical.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(_name_)


class Doctor(db.Model):
    """Database model for doctor accounts."""

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)


class PatientData(db.Model):
    """Database model for patient medical records."""

    id = db.Column(db.Integer, primary_key=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey('doctor.id'), nullable=False)
    name = db.Column(db.String(80), nullable=False)
    disease = db.Column(db.String(20), nullable=False)
    age = db.Column(db.Integer, nullable=True)
    features = db.Column(db.PickleType, nullable=False)
    result = db.Column(db.Float, nullable=True)


def load_model_files():
    """Load all machine learning models and scalers from disk."""
    models_dict = {
        'heart-attack': _load_disease_models('heart_attack', 'heart'),
        'breast-cancer': _load_disease_models('breast_cancer', 'breast'),
        'diabetes': _load_disease_models('diabetes', 'diabetes'),
        'lung-cancer': _load_disease_models('lung_cancer', 'lung')
    }
    logger.info("Models and scalers loaded successfully")
    return models_dict


def _load_disease_models(folder_name, scaler_prefix):
    """Helper function to load models for a specific disease."""
    return {
        'scaler': pickle.load(open(f'{folder_name}/{scaler_prefix}_scaler', 'rb')),
        'KNN': pickle.load(open(f'{folder_name}/KNN_model', 'rb')),
        'DT': pickle.load(open(f'{folder_name}/DT_model', 'rb')),
        'RF': pickle.load(open(f'{folder_name}/RF_model', 'rb')),
        'LR': pickle.load(open(f'{folder_name}/LR_model', 'rb')),
        'SVM': pickle.load(open(f'{folder_name}/SVM_model', 'rb')),
        'NB': pickle.load(open(f'{folder_name}/NB_model', 'rb')),
        'DL': pickle.load(open(f'{folder_name}/DL_model', 'rb'))
    }


# Load models and initialize database
MODELS = load_model_files()

with app.app_context():
    db.create_all()
    logger.info("Database initialized successfully")


# Define feature lists for each disease
FEATURE_LISTS = {
    'heart-attack': [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
        'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ],
    'breast-cancer': [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
        'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean',
        'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
        'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se',
        'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
        'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
        'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
    ],
    'diabetes': [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
        'BMI', 'DiabetesPedigreeFunction', 'Age'
    ],
    'lung-cancer': [
        'GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE',
        'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOL_CONSUMING',
        'COUGHING', 'SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN'
    ]
}

# Mapping of human-readable labels to numeric values
LABEL_TO_NUMERIC = {
    'heart-attack': {
        'sex': {'female': 0, 'male': 1},
        'cp': {
            'typical_angina': 0, 'atypical_angina': 1,
            'non_anginal_pain': 2, 'asymptomatic': 3
        },
        'fbs': {'false': 0, 'true': 1},
        'restecg': {'normal': 0, 'st_t_abnormality': 1, 'lv_hypertrophy': 2},
        'exang': {'no': 0, 'yes': 1},
        'slope': {'upsloping': 0, 'flat': 1, 'downsloping': 2},
        'ca': {'0': 0, '1': 1, '2': 2, '3': 3},
        'thal': {'normal': 0, 'fixed_defect': 1, 'reversible_defect': 2, 'other': 3}
    },
    'breast-cancer': {
        'diagnosis': {'B': 0, 'M': 1}
    },
    'diabetes': {},
    'lung-cancer': {
        'GENDER': {'F': 0, 'M': 1},
        'SMOKING': {'1': 0, '2': 1},
        'YELLOW_FINGERS': {'1': 0, '2': 1},
        'ANXIETY': {'1': 0, '2': 1},
        'PEER_PRESSURE': {'1': 0, '2': 1},
        'CHRONIC_DISEASE': {'1': 0, '2': 1},
        'FATIGUE': {'1': 0, '2': 1},
        'ALLERGY': {'1': 0, '2': 1},
        'WHEEZING': {'1': 0, '2': 1},
        'ALCOHOL_CONSUMING': {'1': 0, '2': 1},
        'COUGHING': {'1': 0, '2': 1},
        'SHORTNESS_OF_BREATH': {'1': 0, '2': 1},
        'SWALLOWING_DIFFICULTY': {'1': 0, '2': 1},
        'CHEST_PAIN': {'1': 0, '2': 1}
    }
}


@app.route('/favicon.ico')
def favicon():
    """Handle favicon requests."""
    return '', 204


@app.route('/')
def home():
    """Render the home page."""
    if 'username' in session:
        return redirect(url_for('authenticated_home'))
    return render_template('home.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login and account creation."""
    if 'username' in session:
        return redirect(url_for('authenticated_home'))

    error = None
    if request.method == 'POST':
        action = request.form.get('action')
        username = request.form['username']
        password = request.form['password']

        if action == 'Login':
            return _handle_login(username, password)
        return _handle_account_creation(username, password)

    return render_template('login.html', error=error)


def _handle_login(username, password):
    """Process login attempts."""
    doctor = Doctor.query.filter_by(username=username, password=password).first()
    if doctor:
        session['username'] = doctor.username
        session['doctor_id'] = doctor.id
        logger.info("Doctor %s logged in successfully", username)
        return redirect(url_for('authenticated_home'))

    logger.warning("Failed login attempt for username: %s", username)
    return render_template('login.html', error="Invalid username or password")


def _handle_account_creation(username, password):
    """Process account creation attempts."""
    if Doctor.query.filter_by(username=username).first():
        logger.warning("Attempt to create existing username: %s", username)
        return render_template('login.html', error="Username already exists")

    new_doctor = Doctor(username=username, password=password)
    db.session.add(new_doctor)
    db.session.commit()
    session['username'] = username
    session['doctor_id'] = new_doctor.id
    logger.info("New doctor account created: %s", username)
    return redirect(url_for('authenticated_home'))


@app.route('/authenticated_home')
def authenticated_home():
    """Render the authenticated home page."""
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('authenticated_home.html')


@app.route('/logout')
def logout():
    """Handle user logout."""
    session.clear()
    logger.info("User logged out")
    return redirect(url_for('home'))


@app.route('/<disease>')
def disease_page(disease):
    """Render disease-specific prediction page."""
    if 'username' not in session:
        return redirect(url_for('login'))

    name = request.args.get('name', '')
    records = None

    if name:
        records = _get_patient_records(name, disease)

    return render_template(
        f'{disease}.html',
        prediction=None,
        name=name
    )
