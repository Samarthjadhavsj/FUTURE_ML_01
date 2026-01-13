import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_style('whitegrid')

def plot_sales_trend(df, date_column='date', sales_column='sales'):
    """Plot historical sales trend"""
    plt.figure(figsize=(12, 6))
    plt.plot(df[date_column], df[sales_column], linewidth=2)
    plt.title('Historical Sales Trend', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Sales', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('sales_trend.png', dpi=300)
    plt.show()

def plot_forecast(model, forecast):
    """Plot Prophet forecast with components"""
    fig1 = model.plot(forecast, figsize=(12, 6))
    plt.title('Sales Forecast', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('forecast.png', dpi=300)
    plt.show()
    
    fig2 = model.plot_components(forecast, figsize=(12, 8))
    plt.tight_layout()
    plt.savefig('forecast_components.png', dpi=300)
    plt.show()

def plot_seasonality(df, date_column='date', sales_column='sales'):
    """Plot seasonal patterns"""
    df['month'] = pd.to_datetime(df[date_column]).dt.month
    df['dayofweek'] = pd.to_datetime(df[date_column]).dt.dayofweek
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Monthly seasonality
    monthly_avg = df.groupby('month')[sales_column].mean()
    axes[0].bar(monthly_avg.index, monthly_avg.values, color='steelblue')
    axes[0].set_title('Average Sales by Month', fontweight='bold')
    axes[0].set_xlabel('Month')
    axes[0].set_ylabel('Average Sales')
    
    # Weekly seasonality
    weekly_avg = df.groupby('dayofweek')[sales_column].mean()
    axes[1].bar(weekly_avg.index, weekly_avg.values, color='coral')
    axes[1].set_title('Average Sales by Day of Week', fontweight='bold')
    axes[1].set_xlabel('Day of Week (0=Monday)')
    axes[1].set_ylabel('Average Sales')
    
    plt.tight_layout()
    plt.savefig('seasonality.png', dpi=300)
    plt.show()
