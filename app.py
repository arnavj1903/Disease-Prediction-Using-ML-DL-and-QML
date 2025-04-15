from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_sqlalchemy import SQLAlchemy
import pickle
import os
from dotenv import load_dotenv
import logging
import numpy as np
from uuid import uuid4

load_dotenv()
app = Flask(__name__)
app.config['SECRET_KEY'] = str(uuid4())  # Use a random UUID for security
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///medical.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load models and scalers
models = {
    'heart-attack': {
        'scaler': pickle.load(open('heart_attack/heart_scaler', 'rb')),
        'KNN': pickle.load(open('heart_attack/KNN_model', 'rb')),
        'DT': pickle.load(open('heart_attack/DT_model', 'rb')),
        'RF': pickle.load(open('heart_attack/RF_model', 'rb')),
        'LR': pickle.load(open('heart_attack/LR_model', 'rb')),
        'SVM': pickle.load(open('heart_attack/SVM_model', 'rb')),
        'NB': pickle.load(open('heart_attack/NB_model', 'rb')),
        'DL': pickle.load(open('heart_attack/DL_model', 'rb'))
    },
    'breast-cancer': {
        'scaler': pickle.load(open('breast_cancer/breast_scaler', 'rb')),
        'KNN': pickle.load(open('breast_cancer/KNN_model', 'rb')),
        'DT': pickle.load(open('breast_cancer/DT_model', 'rb')),
        'RF': pickle.load(open('breast_cancer/RF_model', 'rb')),
        'LR': pickle.load(open('breast_cancer/LR_model', 'rb')),
        'SVM': pickle.load(open('breast_cancer/SVM_model', 'rb')),
        'NB': pickle.load(open('breast_cancer/NB_model', 'rb')),
        'DL': pickle.load(open('breast_cancer/DL_model', 'rb'))
    },
    'diabetes': {
        'scaler': pickle.load(open('diabetes/diabetes_scaler', 'rb')),
        'KNN': pickle.load(open('diabetes/KNN_model', 'rb')),
        'DT': pickle.load(open('diabetes/DT_model', 'rb')),
        'RF': pickle.load(open('diabetes/RF_model', 'rb')),
        'LR': pickle.load(open('diabetes/LR_model', 'rb')),
        'SVM': pickle.load(open('diabetes/SVM_model', 'rb')),
        'NB': pickle.load(open('diabetes/NB_model', 'rb')),
        'DL': pickle.load(open('diabetes/DL_model', 'rb'))
    },
    'lung-cancer': {
        'scaler': pickle.load(open('lung_cancer/lung_scaler', 'rb')),
        'KNN': pickle.load(open('lung_cancer/KNN_model', 'rb')),
        'DT': pickle.load(open('lung_cancer/DT_model', 'rb')),
        'RF': pickle.load(open('lung_cancer/RF_model', 'rb')),
        'LR': pickle.load(open('lung_cancer/LR_model', 'rb')),
        'SVM': pickle.load(open('lung_cancer/SVM_model', 'rb')),
        'NB': pickle.load(open('lung_cancer/NB_model', 'rb')),
        'DL': pickle.load(open('lung_cancer/DL_model', 'rb'))
    }
}
logger.info("Models and scalers loaded successfully")

# Database Models
class Doctor(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

class PatientData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey('doctor.id'), nullable=False)
    name = db.Column(db.String(80), nullable=False)
    disease = db.Column(db.String(20), nullable=False)
    age = db.Column(db.Integer, nullable=True)
    features = db.Column(db.PickleType, nullable=False)
    result = db.Column(db.Float, nullable=True)

with app.app_context():
    db.create_all()
logger.info("Database initialized successfully")

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('authenticated_home'))
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'username' in session:
        return redirect(url_for('authenticated_home'))
    
    error = None
    if request.method == 'POST':
        action = request.form.get('action')
        username = request.form['username']
        password = request.form['password']
        
        if action == 'Login':
            doctor = Doctor.query.filter_by(username=username, password=password).first()
            if doctor:
                session['username'] = doctor.username
                session['doctor_id'] = doctor.id
                logger.info(f"Doctor {username} logged in successfully")
                return redirect(url_for('authenticated_home'))
            else:
                error = "Invalid username or password"
                logger.warning(f"Failed login attempt for username: {username}")
        elif action == 'Create Account':
            if Doctor.query.filter_by(username=username).first():
                error = "Username already exists"
                logger.warning(f"Attempt to create existing username: {username}")
            else:
                new_doctor = Doctor(username=username, password=password)
                db.session.add(new_doctor)
                db.session.commit()
                session['username'] = username
                session['doctor_id'] = new_doctor.id
                logger.info(f"New doctor account created: {username}")
                return redirect(url_for('authenticated_home'))
    
    return render_template('login.html', error=error)

@app.route('/authenticated_home')
def authenticated_home():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('authenticated_home.html')

@app.route('/logout')
def logout():
    session.clear()
    logger.info("User logged out")
    return redirect(url_for('home'))

@app.route('/<disease>')
def disease_page(disease):
    if 'username' not in session:
        return redirect(url_for('login'))
    name = request.args.get('name', '')
    records = None
    no_records = False
    if name:
        doctor = Doctor.query.filter_by(username=session['username']).first()
        records = PatientData.query.filter_by(doctor_id=doctor.id, name=name, disease=disease).all()
        if records:
            records = [{
                'id': r.id,
                'age': r.age,
                'features': r.features,
                'result': r.result
            } for r in records]
        else:
            no_records = True
    return render_template(f'{disease}.html', prediction=None, name=name, error=None, records=records, no_records=no_records)

@app.route('/search/<disease>', methods=['POST'])
def search_patient(disease):
    if 'username' not in session:
        return jsonify({'records': None, 'no_records': True})
    
    name = request.form.get('name')
    if not name:
        return jsonify({'records': None, 'no_records': True})
    
    doctor = Doctor.query.filter_by(username=session['username']).first()
    if not doctor:
        return jsonify({'records': None, 'no_records': True})
    
    records = PatientData.query.filter_by(doctor_id=doctor.id, name=name, disease=disease).all()
    if records:
        return jsonify({
            'records': [{
                'id': r.id,
                'age': r.age,
                'features': r.features,
                'result': r.result
            } for r in records],
            'no_records': False
        })
    return jsonify({'records': None, 'no_records': True})

@app.route('/predict/<disease>', methods=['POST'])
def predict(disease):
    if 'username' not in session:
        return redirect(url_for('login'))
    doctor = Doctor.query.filter_by(username=session['username']).first()
    model_type = request.form['model']
    name = request.form.get('name', '')
    
    # Define feature lists for each disease
    feature_lists = {
        'heart-attack': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'],
        'breast-cancer': [
            'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
            'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
            'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
            'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst',
            'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
        ],
        'diabetes': ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
        'lung-cancer': [
            'GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY',
            'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN'
        ]
    }
    
    # Mapping of human-readable labels to numeric values
    label_to_numeric = {
        'heart-attack': {
            'sex': {'female': 0, 'male': 1},
            'cp': {'typical_angina': 0, 'atypical_angina': 1, 'non_anginal_pain': 2, 'asymptomatic': 3},
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
    
    # Convert features
    features = []
    age = None
    for feature in feature_lists[disease]:
        value = request.form.get(feature)
        if not value:
            return render_template(f'{disease}.html', prediction=None, name=name, error=f"Missing value for {feature}", records=None, no_records=False)
        
        if disease in label_to_numeric and feature in label_to_numeric[disease]:
            value = label_to_numeric[disease][feature].get(value, None)
            if value is None:
                return render_template(f'{disease}.html', prediction=None, name=name, error=f"Invalid value for {feature}", records=None, no_records=False)
        else:
            try:
                value = float(value)
            except ValueError:
                return render_template(f'{disease}.html', prediction=None, name=name, error=f"Invalid value for {feature}", records=None, no_records=False)
        
        features.append(value)
        if feature.lower() in ['age', 'AGE']:
            age = int(value)
    
    # Scale features
    try:
        scaler = models[disease]['scaler']
        scaled_features = scaler.transform([features])[0]
        logger.info(f"Features scaled for {disease} prediction")
    except Exception as e:
        logger.error(f"Scaling error for {disease}: {str(e)}")
        return render_template(f'{disease}.html', prediction=None, name=name, error="Error in scaling features", records=None, no_records=False)
    
    # Make prediction
    try:
        if model_type == 'DL':
            model_input = np.array([scaled_features])
            prediction = models[disease][model_type].predict(model_input)[0]
            prediction = 1 if prediction >= 0.5 else 0
        else:
            model_input = [scaled_features]
            prediction = models[disease][model_type].predict(model_input)[0]
        logger.info(f"Prediction for {name} with {model_type}: {prediction}")
    except Exception as e:
        logger.error(f"Prediction error for {disease} with {model_type}: {str(e)}")
        return render_template(f'{disease}.html', prediction=None, name=name, error="Error in prediction", records=None, no_records=False)
    
    # Store or update patient data
    if name:
        patient = PatientData.query.filter_by(doctor_id=doctor.id, name=name, disease=disease, age=age).first()
        if patient:
            patient.features = features
            patient.result = prediction
        else:
            new_patient = PatientData(doctor_id=doctor.id, name=name, disease=disease, age=age, features=features, result=prediction)
            db.session.add(new_patient)
        db.session.commit()
    
    # Fetch records for display
    records = None
    no_records = False
    if name:
        records = PatientData.query.filter_by(doctor_id=doctor.id, name=name, disease=disease).all()
        if records:
            records = [{
                'id': r.id,
                'age': r.age,
                'features': r.features,
                'result': r.result
            } for r in records]
        else:
            no_records = True
    
    return render_template(f'{disease}.html', prediction=prediction, name=name, error=None, records=records, no_records=no_records)

if __name__ == '__main__':
    try:
        app.run(debug=True)
    except Exception as e:
        logger.error(f"Application failed to start: {str(e)}")