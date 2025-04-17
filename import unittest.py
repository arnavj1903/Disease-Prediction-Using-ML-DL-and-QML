import unittest
from unittest.mock import patch, MagicMock
from flask import session
import sys
import types

# test_app.py


# Absolute import for app and models (namespace package)
from Disease-Prediction-Using-ML-DL-and-QML import app

class AppTestCase(unittest.TestCase):
    def setUp(self):
        app.app.config['TESTING'] = True
        app.app.config['WTF_CSRF_ENABLED'] = False
        self.client = app.app.test_client()
        self.ctx = app.app.app_context()
        self.ctx.push()

        # Patch models and scaler for all tests
        self.patcher_models = patch('Disease-Prediction-Using-ML-DL-and-QML.app.models', {
            'heart-attack': {
                'scaler': MagicMock(transform=lambda x: x),
                'KNN': MagicMock(predict=lambda x: [1]),
                'DT': MagicMock(predict=lambda x: [1]),
                'RF': MagicMock(predict=lambda x: [1]),
                'LR': MagicMock(predict=lambda x: [1]),
                'SVM': MagicMock(predict=lambda x: [1]),
                'NB': MagicMock(predict=lambda x: [1]),
                'DL': MagicMock(predict=lambda x: [1])
            }
        })
        self.patcher_models.start()

        # Patch DB session commit to avoid real DB writes
        self.patcher_commit = patch.object(app.db.session, 'commit', return_value=None)
        self.patcher_commit.start()

    def tearDown(self):
        self.patcher_models.stop()
        self.patcher_commit.stop()
        self.ctx.pop()

    def test_home_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'home', response.data)

    def test_authenticated_home_requires_login(self):
        response = self.client.get('/authenticated_home')
        self.assertEqual(response.status_code, 302)
        self.assertIn('/login', response.headers['Location'])

    @patch('Disease-Prediction-Using-ML-DL-and-QML.app.Doctor')
    def test_login_logout_flow(self, mock_doctor):
        # Mock Doctor query
        mock_doctor.query.filter_by.return_value.first.return_value = type('User', (), {'username': 'test', 'id': 1})()
        # Login
        response = self.client.post('/login', data={'username': 'test', 'password': 'pw', 'action': 'Login'}, follow_redirects=True)
        self.assertIn(b'authenticated_home', response.data)
        # Logout
        response = self.client.get('/logout', follow_redirects=True)
        self.assertIn(b'home', response.data)

    @patch('Disease-Prediction-Using-ML-DL-and-QML.app.Doctor')
    @patch('Disease-Prediction-Using-ML-DL-and-QML.app.PatientData')
    def test_disease_page_requires_login(self, mock_patient, mock_doctor):
        response = self.client.get('/heart-attack')
        self.assertEqual(response.status_code, 302)
        self.assertIn('/login', response.headers['Location'])

    @patch('Disease-Prediction-Using-ML-DL-and-QML.app.Doctor')
    @patch('Disease-Prediction-Using-ML-DL-and-QML.app.PatientData')
    def test_search_patient_no_session(self, mock_patient, mock_doctor):
        response = self.client.post('/search/heart-attack', data={'name': 'Alice'})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'no_records', response.data)

    @patch('Disease-Prediction-Using-ML-DL-and-QML.app.Doctor')
    @patch('Disease-Prediction-Using-ML-DL-and-QML.app.PatientData')
    def test_predict_missing_feature(self, mock_patient, mock_doctor):
        with self.client.session_transaction() as sess:
            sess['username'] = 'test'
        mock_doctor.query.filter_by.return_value.first.return_value = type('User', (), {'username': 'test', 'id': 1})()
        response = self.client.post('/predict/heart-attack', data={'model': 'KNN', 'name': 'Alice'})
        self.assertIn(b'Missing value', response.data)

    @patch('Disease-Prediction-Using-ML-DL-and-QML.app.Doctor')
    @patch('Disease-Prediction-Using-ML-DL-and-QML.app.PatientData')
    @patch('Disease-Prediction-Using-ML-DL-and-QML.app.get_gemini_recommendations')
    def test_predict_valid(self, mock_gemini, mock_patient, mock_doctor):
        with self.client.session_transaction() as sess:
            sess['username'] = 'test'
        mock_doctor.query.filter_by.return_value.first.return_value = type('User', (), {'username': 'test', 'id': 1})()
        mock_patient.query.filter_by.return_value.first.return_value = None
        mock_patient.query.filter_by.return_value.all.return_value = []
        mock_gemini.return_value = ['Rec1', 'Rec2', 'Rec3', 'Rec4', 'Rec5']
        data = {
            'model': 'KNN',
            'name': 'Alice',
            'age': '60',
            'sex': 'male',
            'cp': 'typical_angina',
            'trestbps': '120',
            'chol': '200',
            'fbs': 'true',
            'restecg': 'normal',
            'thalach': '150',
            'exang': 'no',
            'oldpeak': '1.0',
            'slope': 'flat',
            'ca': '0',
            'thal': 'normal'
        }
        response = self.client.post('/predict/heart-attack', data=data)
        self.assertIn(b'prediction', response.data)

    @patch('Disease-Prediction-Using-ML-DL-and-QML.app.genai')
    def test_get_gemini_recommendations_success(self, mock_genai):
        mock_model = MagicMock()
        mock_model.generate_content.return_value.text = "1. Rec1\n2. Rec2\n3. Rec3\n4. Rec4\n5. Rec5"
        mock_genai.GenerativeModel.return_value = mock_model
        recs = app.get_gemini_recommendations('heart-attack', {'age': 60})
        self.assertEqual(len(recs), 5)
        self.assertTrue(all(isinstance(r, str) for r in recs))

    @patch('Disease-Prediction-Using-ML-DL-and-QML.app.genai')
    def test_get_gemini_recommendations_error(self, mock_genai):
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = Exception("API error")
        mock_genai.GenerativeModel.return_value = mock_model
        result = app.get_gemini_recommendations('heart-attack', {'age': 60})
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()