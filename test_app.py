"""
Unit tests for the Medical Disease Prediction Flask Application.

This module contains comprehensive test cases for testing the functionality
of the Flask application for medical disease prediction.
"""

import unittest
import os
import numpy as np
from unittest.mock import patch, MagicMock
from flask import session
from werkzeug.security import generate_password_hash, check_password_hash

# Import the Flask application
from app import (
    app, db, Doctor, PatientData, load_model_files,
    FEATURE_LISTS, LABEL_TO_NUMERIC, _classify_risk
)


class FlaskMedicalAppTests(unittest.TestCase):
    """Unit tests for the Medical Disease Prediction Flask Application."""

    def setUp(self):
        """Set up test environment before each test."""
        # Configure the Flask app for testing
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        self.app = app.test_client()

        # Create the database and tables
        with app.app_context():
            db.create_all()

            # Add test doctor with encrypted password
            test_doctor = Doctor(username='testdoctor')
            test_doctor.set_password('testpassword')
            db.session.add(test_doctor)
            db.session.commit()

            # Add test patient data
            test_patient = PatientData(
                doctor_id=1,
                name='Test Patient',
                disease='heart-attack',
                age=45,
                features={'age': 45, 'sex': 1},
                result=0.75,
                risk_label="Medium Risk"
            )
            db.session.add(test_patient)
            db.session.commit()

    def tearDown(self):
        """Clean up after each test."""
        with app.app_context():
            db.session.remove()
            db.drop_all()

    def test_home_page(self):
        """Test that home page loads correctly."""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'MediScope AI', response.data)

    def test_login_page(self):
        """Test that login page loads correctly."""
        response = self.app.get('/login')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Login', response.data)

    def test_successful_login(self):
        """Test successful login with valid credentials."""
        response = self.app.post('/login', data={
            'username': 'testdoctor',
            'password': 'testpassword',
            'action': 'Login'
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'authenticated_home', response.data)

    def test_failed_login(self):
        """Test failed login with invalid credentials."""
        response = self.app.post('/login', data={
            'username': 'testdoctor',
            'password': 'wrongpassword',
            'action': 'Login'
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Invalid username or password', response.data)

    def test_create_account(self):
        """Test creating a new account."""
        response = self.app.post('/login', data={
            'username': 'newdoctor',
            'password': 'newpassword',
            'action': 'Create Account'
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)

        # Check if the new account was added to the database
        with app.app_context():
            doctor = Doctor.query.filter_by(username='newdoctor').first()
            self.assertIsNotNone(doctor)
            self.assertTrue(doctor.check_password('newpassword'))

    def test_create_duplicate_account(self):
        """Test attempting to create an account with an existing username."""
        response = self.app.post('/login', data={
            'username': 'testdoctor',
            'password': 'somepassword',
            'action': 'Create Account'
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Username already exists', response.data)

    def test_logout(self):
        """Test logout functionality."""
        # Login first
        with self.app as client:
            with client.session_transaction() as sess:
                sess['username'] = 'testdoctor'
                sess['doctor_id'] = 1

            # Then logout
            response = client.get('/logout', follow_redirects=True)
            self.assertEqual(response.status_code, 200)
            self.assertIn(b'MediScope AI', response.data)

            # Verify session is cleared
            with client.session_transaction() as sess:
                self.assertNotIn('username', sess)
                self.assertNotIn('doctor_id', sess)

    def test_authenticated_home_with_session(self):
        """Test authenticated home page with valid session."""
        with self.app as client:
            with client.session_transaction() as sess:
                sess['username'] = 'testdoctor'
                sess['doctor_id'] = 1

            response = client.get('/authenticated_home')
            self.assertEqual(response.status_code, 200)
            self.assertIn(b'authenticated_home', response.data)

    def test_authenticated_home_without_session(self):
        """Test that authenticated home redirects to login when no session."""
        response = self.app.get('/authenticated_home', follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Login', response.data)

    def test_disease_page_with_session(self):
        """Test disease prediction page with valid session."""
        with self.app as client:
            with client.session_transaction() as sess:
                sess['username'] = 'testdoctor'
                sess['doctor_id'] = 1

            # Test each available disease page
            for disease in ['heart-attack', 'breast-cancer', 'diabetes', 'lung-cancer']:
                with self.subTest(disease=disease):
                    response = client.get(f'/{disease}')
                    self.assertEqual(response.status_code, 200)
                    self.assertIn(disease.encode(), response.data.lower())

    def test_disease_page_without_session(self):
        """Test that disease page redirects to login when no session."""
        response = self.app.get('/heart-attack', follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Login', response.data)

    def test_disease_page_with_patient_name(self):
        """Test disease page with patient name parameter."""
        with self.app as client:
            with client.session_transaction() as sess:
                sess['username'] = 'testdoctor'
                sess['doctor_id'] = 1

            response = client.get('/heart-attack?name=Test%20Patient')
            self.assertEqual(response.status_code, 200)
            # Check if the page contains patient data section
            self.assertIn(b'patient', response.data.lower())

    def test_password_security(self):
        """Test that passwords are stored securely (TC_F1.3.1)."""
        with app.app_context():
            # Create a doctor user with password that should be hashed
            test_doctor = Doctor(username='securitytest')
            test_doctor.set_password('testpassword')
            db.session.add(test_doctor)
            db.session.commit()

            # Verify password is hashed in database
            doctor = Doctor.query.filter_by(username='securitytest').first()

            # Check password is not stored in plaintext
            self.assertNotEqual(doctor.password, 'testpassword')

            # Verify the password can be checked against the hash
            self.assertTrue(doctor.check_password('testpassword'))
            self.assertFalse(doctor.check_password('wrongpassword'))

    def test_model_loading(self):
        """Test that models are loaded correctly at application startup."""
        with patch('pickle.load') as mock_pickle:
            # Setup mock models
            mock_model = MagicMock()
            mock_pickle.return_value = mock_model

            # Call the load function
            models = load_model_files()

            # Verify all expected diseases are loaded
            self.assertIn('heart-attack', models)
            self.assertIn('breast-cancer', models)
            self.assertIn('diabetes', models)
            self.assertIn('lung-cancer', models)

            # Verify each disease has all required model types
            for disease in models:
                self.assertIn('scaler', models[disease])
                self.assertIn('KNN', models[disease])
                self.assertIn('DT', models[disease])
                self.assertIn('RF', models[disease])
                self.assertIn('LR', models[disease])
                self.assertIn('SVM', models[disease])
                self.assertIn('NB', models[disease])
                self.assertIn('DL', models[disease])

def test_model_predictions(self):
    """Test predictions for known positive and negative cases using specific models."""
    # First, ensure we have a doctor account and are logged in
    with app.app_context():
        # Make sure test doctor exists
        doctor = Doctor.query.filter_by(username='testdoctor').first()
        if not doctor:
            test_doctor = Doctor(username='testdoctor')
            test_doctor.set_password('testpassword')
            db.session.add(test_doctor)
            db.session.commit()
            doctor_id = test_doctor.id
        else:
            doctor_id = doctor.id

    # Login as the test doctor
    login_response = self.app.post('/login', data={
        'username': 'testdoctor',
        'password': 'testpassword',
        'action': 'Login'
    }, follow_redirects=True)
    
    self.assertEqual(login_response.status_code, 200)

    # Now run the test cases with the authenticated session
    with self.app as client:
        # Verify we're logged in
        with client.session_transaction() as sess:
            sess['username'] = 'testdoctor'
            sess['doctor_id'] = doctor_id

        test_cases = [
            {
                "disease": "heart-attack",
                "model": "DT",
                "positive": {
                    'name': 'HeartPos',
                    'data': {'age': 20.0, 'sex': 1, 'cp': 0, 'trestbps': 90.0, 'chol': 200.0, 'fbs': 0,
                            'restecg': 1, 'thalach': 140.0, 'exang': 0, 'oldpeak': 1.0,
                            'slope': 2, 'ca': 1, 'thal': 2}
                },
                "negative": {
                    'name': 'HeartNeg',
                    'data': {'age': 20.0, 'sex': 1, 'cp': 0, 'trestbps': 120.0, 'chol': 200.0, 'fbs': 0,
                            'restecg': 0, 'thalach': 140.0, 'exang': 0, 'oldpeak': 2.0,
                            'slope': 0, 'ca': 3, 'thal': 0}
                }
            },
            {
                "disease": "breast-cancer",
                "model": "DT",
                "positive": {
                    'name': 'BCPos',
                    'data': {
                        'radius_mean': 18.0, 'texture_mean': 10.4, 'perimeter_mean': 122.8, 'area_mean': 1001.0,
                        'smoothness_mean': 0.12, 'compactness_mean': 0.3, 'concavity_mean': 0.3,
                        'concave_points_mean': 0.15, 'symmetry_mean': 0.24, 'fractal_dimension_mean': 0.08,
                        'radius_se': 1.1, 'texture_se': 0.88, 'perimeter_se': 8.59, 'area_se': 153.4,
                        'smoothness_se': 0.006, 'compactness_se': 0.05, 'concavity_se': 0.05,
                        'concave_points_se': 0.0198, 'symmetry_se': 0.03, 'fractal_dimension_se': 0.0053,
                        'radius_worst': 25.4, 'texture_worst': 17.32, 'perimeter_worst': 184.0, 'area_worst': 2019.0,
                        'smoothness_worst': 0.16, 'compactness_worst': 0.67, 'concavity_worst': 0.72,
                        'concave_points_worst': 0.2699, 'symmetry_worst': 0.46, 'fractal_dimension_worst': 0.12
                    }
                },
                "negative": {
                    'name': 'BCNeg',
                    'data': {
                        'radius_mean': 8.0, 'texture_mean': 10.0, 'perimeter_mean': 60.0, 'area_mean': 300.0,
                        'smoothness_mean': 0.05, 'compactness_mean': 0.02, 'concavity_mean': 0.02,
                        'concave_points_mean': 0.01, 'symmetry_mean': 0.1, 'fractal_dimension_mean': 0.05,
                        'radius_se': 0.2, 'texture_se': 0.3, 'perimeter_se': 1.0, 'area_se': 10.0,
                        'smoothness_se': 0.002, 'compactness_se': 0.01, 'concavity_se': 0.01,
                        'concave_points_se': 0.001, 'symmetry_se': 0.01, 'fractal_dimension_se': 0.002,
                        'radius_worst': 9.0, 'texture_worst': 12.0, 'perimeter_worst': 70.0, 'area_worst': 400.0,
                        'smoothness_worst': 0.06, 'compactness_worst': 0.03, 'concavity_worst': 0.02,
                        'concave_points_worst': 0.02, 'symmetry_worst': 0.15, 'fractal_dimension_worst': 0.06
                    }
                }
            },
            {
                "disease": "diabetes",
                "positive": {
                    'name': 'DiabetesPos',
                    'model': 'DT',
                    'data': {
                        'Pregnancies': 0.0, 'Glucose': 150.0, 'BloodPressure': 130.0, 'SkinThickness': 35.0,
                        'Insulin': 0.0, 'BMI': 22.0, 'DiabetesPedigreeFunction': 0.8, 'Age': 20.0
                    }
                },
                "negative": {
                    'name': 'DiabetesNeg',
                    'model': 'LR',
                    'data': {
                        'Pregnancies': 0.0, 'Glucose': 200.0, 'BloodPressure': 80.0, 'SkinThickness': 5.0,
                        'Insulin': 5.0, 'BMI': 22.0, 'DiabetesPedigreeFunction': 0.5, 'Age': 20.0
                    }
                }
            },
            {
                "disease": "lung-cancer",
                "positive": {
                    'name': 'LungPos',
                    'model': 'KNN',
                    'data': {
                        'GENDER': 1, 'AGE': 67.0, 'SMOKING': 1, 'YELLOW_FINGERS': 1, 'ANXIETY': 0,
                        'PEER_PRESSURE': 0, 'CHRONIC_DISEASE': 1, 'FATIGUE': 1, 'ALLERGY': 1,
                        'WHEEZING': 1, 'ALCOHOL_CONSUMING': 1, 'COUGHING': 1,
                        'SHORTNESS_OF_BREATH': 1, 'SWALLOWING_DIFFICULTY': 1, 'CHEST_PAIN': 1
                    }
                },
                "negative": {
                    'name': 'LungNeg',
                    'model': 'DT',
                    'data': {
                        'GENDER': 1, 'AGE': 20.0, 'SMOKING': 1, 'YELLOW_FINGERS': 1, 'ANXIETY': 1,
                        'PEER_PRESSURE': 0, 'CHRONIC_DISEASE': 0, 'FATIGUE': 1, 'ALLERGY': 0,
                        'WHEEZING': 1, 'ALCOHOL_CONSUMING': 1, 'COUGHING': 1,
                        'SHORTNESS_OF_BREATH': 1, 'SWALLOWING_DIFFICULTY': 0, 'CHEST_PAIN': 0
                    }
                }
            }
        ]

        for case in test_cases:
            for label in ['positive', 'negative']:
                patient = case[label]
                model = patient.get('model', case.get('model', 'DT'))
                
                # Convert data to form format - everything needs to be strings
                form_data = {}
                for key, value in patient['data'].items():
                    form_data[key] = str(value)
                
                # Add model and name to the form data
                form_data['model'] = model
                form_data['name'] = patient['name']
                
                print(f"\nSubmitting prediction for {patient['name']} ({case['disease']})")
                response = client.post(f"/predict/{case['disease']}", data=form_data, follow_redirects=True)
                self.assertEqual(response.status_code, 200)
                
                # Force the session to save changes before checking the database
                db.session.commit()

                # Match the age field for DB lookup
                age_field = 'age' if 'age' in patient['data'] else 'AGE'
                age_value = int(float(patient['data'][age_field]))

                with app.app_context():
                    record = PatientData.query.filter_by(
                        doctor_id=doctor_id,
                        name=patient['name'],
                        disease=case['disease'],
                        age=age_value
                    ).first()

                    print(f"Checking database for {patient['name']} ({case['disease']}) with age {age_value}")
                    if record:
                        print(f"Found record: {record.id}, result: {record.result}")
                    else:
                        print(f"No record found for {patient['name']}")

                    self.assertIsNotNone(record, f"Patient record not found for {patient['name']} ({case['disease']})")

                    expected_result = 1.0 if label == 'positive' else 0.0
                    expected_risk = "High Risk" if expected_result == 1.0 else "Low Risk"

                    self.assertEqual(record.result, expected_result,
                                    f"{label.title()} prediction mismatch for {case['disease']}")
                    self.assertEqual(record.risk_label, expected_risk,
                                    f"{label.title()} risk label mismatch for {case['disease']}")




    def test_risk_classification(self):
        """Test the risk classification function."""
        # Test high risk
        self.assertEqual(_classify_risk(0.85), "High Risk")

        # Test medium risk
        self.assertEqual(_classify_risk(0.75), "Medium Risk")
        self.assertEqual(_classify_risk(0.5), "Medium Risk")

        # Test low risk
        self.assertEqual(_classify_risk(0.49), "Low Risk")
        self.assertEqual(_classify_risk(0.0), "Low Risk")

    def test_patient_record_retrieval(self):
        """Test retrieval of patient records."""
        with self.app as client:
            with client.session_transaction() as sess:
                sess['username'] = 'testdoctor'
                sess['doctor_id'] = 1

            # Test with existing patient
            response = client.get('/heart-attack?name=Test%20Patient')
            self.assertEqual(response.status_code, 200)
            self.assertIn(b'Test Patient', response.data)

            # Test with non-existent patient
            response = client.get('/heart-attack?name=Nonexistent')
            self.assertEqual(response.status_code, 200)
            self.assertIn(b'No records found', response.data)

    def test_gemini_recommendations(self):
        """Test Gemini recommendations generation (mocked)."""
        with patch('google.generativeai.GenerativeModel') as mock_model:
            # Setup mock response
            mock_response = MagicMock()
            mock_response.text = "1. Recommendation one\n2. Recommendation two"
            mock_model.return_value.generate_content.return_value = mock_response

            # Call the function
            from app import get_gemini_recommendations
            recommendations = get_gemini_recommendations('heart-attack', {'age': 50})

            # Verify the response
            self.assertEqual(len(recommendations), 2)
            self.assertIn('Recommendation one', recommendations[0])
            self.assertIn('Recommendation two', recommendations[1])

    def test_feature_lists(self):
        """Test that feature lists are properly defined."""
        self.assertIn('heart-attack', FEATURE_LISTS)
        self.assertIn('breast-cancer', FEATURE_LISTS)
        self.assertIn('diabetes', FEATURE_LISTS)
        self.assertIn('lung-cancer', FEATURE_LISTS)

        # Verify some key features for each disease
        self.assertIn('age', FEATURE_LISTS['heart-attack'])
        self.assertIn('radius_mean', FEATURE_LISTS['breast-cancer'])
        self.assertIn('Glucose', FEATURE_LISTS['diabetes'])
        self.assertIn('SMOKING', FEATURE_LISTS['lung-cancer'])

    def test_label_to_numeric_mappings(self):
        """Test that label to numeric mappings are properly defined."""
        self.assertIn('heart-attack', LABEL_TO_NUMERIC)
        self.assertIn('lung-cancer', LABEL_TO_NUMERIC)

        # Verify some key mappings
        self.assertEqual(LABEL_TO_NUMERIC['heart-attack']['sex']['male'], 1)
        self.assertEqual(LABEL_TO_NUMERIC['lung-cancer']['GENDER']['M'], 1)

if __name__ == '__main__':
    unittest.main()
