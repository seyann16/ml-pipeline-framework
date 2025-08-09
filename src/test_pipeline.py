import unittest
import os
import pandas as pd
from data_loader import load_data
from preprocessor import preprocess_data

# TestCase is a template for testing
class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls): # runs once before all tests in a class.
        # create test data
        cls.test_data = pd.DataFrame({
            'age': [25, 30, 35, 200], # outlier value
            'income': [50000, 75000, None, 10000],
            'target': [0, 1, 0, 1]
        })
        cls.test_data.to_csv('test_data.csv', index=False)
    
    # test the data loader
    def test_data_loading(self):
        data = load_data('test_data.csv')
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(data.shape, (4,3))
    
    def test_data_preprocessing(self):
        data = load_data('test_data.csv')
        clean_data = preprocess_data(data)
        # check if there are any missing values
        self.assertFalse(clean_data.isnull().any().any())
        # check outlier handling
        # check if one value is less than or equal to another value
        self.assertLessEqual(clean_data['age'].max(), 100)
        
    @classmethod
    def tearDownClass(cls):
        # clean up test file
        os.remove('test_data.csv')
        
if __name__ == "__main__":
    unittest.main()