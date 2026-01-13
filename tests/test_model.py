import unittest
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.append('../src')
from model import SalesForecaster

class TestSalesForecaster(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        # Create sample data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        sales = np.random.randint(800, 1500, len(dates))
        self.df = pd.DataFrame({'date': dates, 'sales': sales})
        self.forecaster = SalesForecaster()
    
    def test_model_initialization(self):
        """Test if model initializes correctly"""
        self.assertIsNone(self.forecaster.model)
        print("✓ Model initialization test passed")
    
    def test_train_prophet(self):
        """Test if Prophet model trains successfully"""
        model = self.forecaster.train_prophet(self.df, 'date', 'sales')
        self.assertIsNotNone(model)
        self.assertIsNotNone(self.forecaster.model)
        print("✓ Prophet training test passed")
    
    def test_predict(self):
        """Test if predictions are generated"""
        self.forecaster.train_prophet(self.df, 'date', 'sales')
        forecast = self.forecaster.predict(periods=30, freq='D')
        
        self.assertIsNotNone(forecast)
        self.assertGreater(len(forecast), len(self.df))
        self.assertIn('yhat', forecast.columns)
        self.assertIn('yhat_lower', forecast.columns)
        self.assertIn('yhat_upper', forecast.columns)
        print("✓ Prediction test passed")
    
    def test_evaluate(self):
        """Test model evaluation metrics"""
        actual = np.array([100, 200, 300, 400, 500])
        predicted = np.array([110, 190, 310, 390, 510])
        
        metrics = self.forecaster.evaluate(actual, predicted)
        
        self.assertIn('MAE', metrics)
        self.assertIn('RMSE', metrics)
        self.assertIn('MAPE', metrics)
        self.assertGreater(metrics['MAE'], 0)
        self.assertGreater(metrics['RMSE'], 0)
        self.assertGreater(metrics['MAPE'], 0)
        print("✓ Evaluation metrics test passed")
    
    def test_forecast_length(self):
        """Test if forecast generates correct number of periods"""
        self.forecaster.train_prophet(self.df, 'date', 'sales')
        
        for periods in [7, 30, 90]:
            forecast = self.forecaster.predict(periods=periods, freq='D')
            future_forecast = forecast[forecast['ds'] > self.df['date'].max()]
            self.assertEqual(len(future_forecast), periods)
        
        print("✓ Forecast length test passed")
    
    def test_confidence_intervals(self):
        """Test if confidence intervals are valid"""
        self.forecaster.train_prophet(self.df, 'date', 'sales')
        forecast = self.forecaster.predict(periods=30, freq='D')
        
        # Check if lower bound <= prediction <= upper bound
        valid_intervals = (forecast['yhat_lower'] <= forecast['yhat']).all() and \
                         (forecast['yhat'] <= forecast['yhat_upper']).all()
        
        self.assertTrue(valid_intervals)
        print("✓ Confidence intervals test passed")

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Running Sales Forecasting Model Tests")
    print("="*60 + "\n")
    
    unittest.main(verbosity=2)
