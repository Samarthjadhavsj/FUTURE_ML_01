import pandas as pd
import numpy as np

def load_data(filepath):
    """Load sales data from CSV file"""
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df, date_column='date', sales_column='sales'):
    """
    Preprocess sales data for forecasting
    
    Args:
        df: Input dataframe
        date_column: Name of date column
        sales_column: Name of sales column
    
    Returns:
        Preprocessed dataframe
    """
    # Convert date column to datetime
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Sort by date
    df = df.sort_values(date_column)
    
    # Handle missing values
    df[sales_column] = df[sales_column].fillna(method='ffill')
    
    # Remove duplicates
    df = df.drop_duplicates(subset=[date_column])
    
    return df

def create_features(df, date_column='date'):
    """Create time-based features for modeling"""
    df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    df['day'] = df[date_column].dt.day
    df['dayofweek'] = df[date_column].dt.dayofweek
    df['quarter'] = df[date_column].dt.quarter
    
    return df
