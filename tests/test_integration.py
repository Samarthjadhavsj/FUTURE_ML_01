import unittest
import sys
import pandas as pd
import numpy as np

sys.path.append('../src')
from data_preprocessing import preprocess_data, create_features
from model import SalesForecaster

class TestIntegration(unittest.TestCase):
    
    def test_end_to_end_pipeline(self):
        """Test complete pipeline from data loading to prediction"""
        print("\n" + "="*60)
        print("Running End-to-End Integration Test")
        print("="*60 + "\n")
        
        # Step 1: Create sample data
        print("Step 1: Creating sample data...")
        dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
        sales = np.random.randint(800, 1500, len(dates))
        df = pd.DataFrame({'date': dates, 'sales': sales})
        print(f"✓ Created {len(df)} records")
        
        # Step 2: Preprocess data
        print("\nStep 2: Preprocessing data...")
        df = preprocess_data(df, 'date', 'sales')
        df = create_features(df, 'date')
        print(f"✓ Data preprocessed with {len(df.columns)} features")
        
        # Step 3: Train model
        print("\nStep 3: Training Prophet model...")
        forecaster = SalesForecaster()
        model = forecaster.train_prophet(df, 'date', 'sales')
        self.assertIsNotNone(model)
        print("✓ Model trained successfully")
        
        # Step 4: Generate forecast
        print("\nStep 4: Generating 30-day forecast...")
        forecast = forecaster.predict(periods=30, freq='D')
        self.assertGreater(len(forecast), len(df))
        print(f"✓ Forecast generated with {len(forecast)} total predictions")
        
        # Step 5: Evaluate model
        print("\nStep 5: Evaluating model performance...")
        historical_forecast = forecast[forecast['ds'].isin(df['date'])]
        actual = df[df['date'].isin(historical_forecast['ds'])]['sales'].values
        predicted = historical_forecast['yhat'].values
        
        metrics = forecaster.evaluate(actual, predicted)
        print(f"✓ MAE: {metrics['MAE']:.2f}")
        print(f"✓ RMSE: {metrics['RMSE']:.2f}")
        print(f"✓ MAPE: {metrics['MAPE']:.2f}%")
        
        # Assertions
        self.assertLess(metrics['MAPE'], 50)  # MAPE should be reasonable
        
        print("\n" + "="*60)
        print("✓ End-to-End Pipeline Test PASSED")
        print("="*60 + "\n")
    
    def test_different_forecast_periods(self):
        """Test forecasting with different time periods"""
        print("\nTesting different forecast periods...")
        
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        sales = np.random.randint(800, 1500, len(dates))
        df = pd.DataFrame({'date': dates, 'sales': sales})
        df = preprocess_data(df, 'date', 'sales')
        
        forecaster = SalesForecaster()
        forecaster.train_prophet(df, 'date', 'sales')
        
        for periods in [7, 30, 60, 90]:
            forecast = forecaster.predict(periods=periods, freq='D')
            future = forecast[forecast['ds'] > df['date'].max()]
            self.assertEqual(len(future), periods)
            print(f"✓ {periods}-day forecast: PASSED")
    
    def test_data_quality_checks(self):
        """Test data quality validations"""
        print("\nTesting data quality checks...")
        
        # Test with good data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        sales = np.random.randint(800, 1500, 100)
        df = pd.DataFrame({'date': dates, 'sales': sales})
        
        df_processed = preprocess_data(df, 'date', 'sales')
        
        # Verify no missing values
        self.assertEqual(df_processed.isna().sum().sum(), 0)
        print("✓ No missing values")
        
        # Verify no duplicates
        self.assertEqual(df_processed.duplicated(subset=['date']).sum(), 0)
        print("✓ No duplicate dates")
        
        # Verify sorted
        self.assertTrue(df_processed['date'].is_monotonic_increasing)
        print("✓ Data sorted by date")

if __name__ == '__main__':
    unittest.main(verbosity=2)
