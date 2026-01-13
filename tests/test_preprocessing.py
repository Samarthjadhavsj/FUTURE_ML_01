import unittest
import sys
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.append('../src')
from data_preprocessing import load_data, preprocess_data, create_features

class TestDataPreprocessing(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        self.df = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-02', '2023-01-04'],
            'sales': [100, 200, np.nan, 200, 400]
        })
    
    def test_preprocess_data(self):
        """Test data preprocessing"""
        df_processed = preprocess_data(self.df.copy(), 'date', 'sales')
        
        # Check if date is datetime
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df_processed['date']))
        
        # Check if sorted by date
        self.assertTrue(df_processed['date'].is_monotonic_increasing)
        
        # Check if duplicates removed
        self.assertEqual(len(df_processed), 4)
        
        # Check if NaN filled
        self.assertFalse(df_processed['sales'].isna().any())
        
        print("✓ Data preprocessing test passed")
    
    def test_create_features(self):
        """Test feature creation"""
        df = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=10, freq='D'),
            'sales': np.random.randint(100, 500, 10)
        })
        
        df_features = create_features(df, 'date')
        
        # Check if new features are created
        self.assertIn('year', df_features.columns)
        self.assertIn('month', df_features.columns)
        self.assertIn('day', df_features.columns)
        self.assertIn('dayofweek', df_features.columns)
        self.assertIn('quarter', df_features.columns)
        
        # Check if values are correct
        self.assertEqual(df_features['year'].iloc[0], 2023)
        self.assertEqual(df_features['month'].iloc[0], 1)
        self.assertEqual(df_features['day'].iloc[0], 1)
        
        print("✓ Feature creation test passed")
    
    def test_missing_values_handling(self):
        """Test missing values are handled correctly"""
        df = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=5, freq='D'),
            'sales': [100, np.nan, np.nan, 400, 500]
        })
        
        df_processed = preprocess_data(df, 'date', 'sales')
        
        # Check no missing values remain
        self.assertEqual(df_processed['sales'].isna().sum(), 0)
        
        print("✓ Missing values handling test passed")

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Running Data Preprocessing Tests")
    print("="*60 + "\n")
    
    unittest.main(verbosity=2)
