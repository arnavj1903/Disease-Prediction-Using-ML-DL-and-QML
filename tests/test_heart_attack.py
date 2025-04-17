import unittest
import numpy as np
#from src.models.heart_attack import HeartAttackPredictor

class TestHeartAttackModel(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment before each test method"""
        self.model = HeartAttackPredictor()
        
    def test_preprocessing(self):
        """Test feature preprocessing"""
        test_data = {
            'age': 65, 
            'sex': 'male', 
            'cp': 'typical_angina', 
            'trestbps': 145, 
            'chol': 233, 
            'fbs': 'true', 
            'restecg': 'normal', 
            'thalach': 150, 
            'exang': 'no', 
            'oldpeak': 2.1, 
            'slope': 'flat', 
            'ca': '1', 
            'thal': 'normal'
        }
        
        features = self.model.preprocess_features(test_data)
        self.assertEqual(len(features), 13, "Incorrect number of features after preprocessing")
        
    def test_prediction(self):
        """Test prediction functionality"""
        sample_features = np.array([
            [65, 1, 0, 145, 233, 1, 0, 150, 0, 2.1, 1, 1, 0]
        ])
        
        prediction = self.model.predict(sample_features, model_type='RF')
        self.assertIn(prediction, [0, 1], "Prediction should be binary (0 or 1)")
        
    def test_model_loading(self):
        """Test that all models load correctly"""
        model_types = ['KNN', 'DT', 'RF', 'LR', 'SVM', 'NB', 'DL']
        for model_type in model_types:
            self.assertIsNotNone(self.model.get_model(model_type), f"{model_type} model failed to load")

if __name__ == '__main__':
    unittest.main()