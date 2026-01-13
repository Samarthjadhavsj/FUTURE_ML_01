import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

class SalesForecaster:
    def __init__(self):
        self.model = None
        
    def train_prophet(self, df, date_column='date', sales_column='sales'):
        """
        Train Prophet model for time series forecasting
        
        Args:
            df: Dataframe with date and sales columns
            date_column: Name of date column
            sales_column: Name of sales column
        """
        # Prophet requires columns named 'ds' and 'y'
        prophet_df = df[[date_column, sales_column]].copy()
        prophet_df.columns = ['ds', 'y']
        
        # Initialize and train model
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False
        )
        self.model.fit(prophet_df)
        
        return self.model
    
    def predict(self, periods=30, freq='D'):
        """
        Make future predictions
        
        Args:
            periods: Number of periods to forecast
            freq: Frequency ('D' for daily, 'W' for weekly, 'M' for monthly)
        
        Returns:
            Dataframe with predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        forecast = self.model.predict(future)
        
        return forecast
    
    def evaluate(self, actual, predicted):
        """Calculate evaluation metrics"""
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
