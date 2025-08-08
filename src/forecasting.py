"""
Additional forecasting capabilities using traditional ML methods
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime, timedelta
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SalesForecaster:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.target_column = 'sales'
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for forecasting"""
        df = df.copy()
        
        # Ensure date column is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
            # Extract time-based features
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['quarter'] = df['date'].dt.quarter
            df['day_of_week'] = df['date'].dt.dayofweek
            df['day_of_year'] = df['date'].dt.dayofyear
            
            # Create lag features
            df['sales_lag_1'] = df['sales'].shift(1)
            df['sales_lag_7'] = df['sales'].shift(7)
            df['sales_lag_30'] = df['sales'].shift(30)
            
            # Rolling averages
            df['sales_ma_7'] = df['sales'].rolling(window=7).mean()
            df['sales_ma_30'] = df['sales'].rolling(window=30).mean()
            
            # Seasonal features
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
            df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)
        
        # Product and region features
        if 'product' in df.columns:
            df = pd.get_dummies(df, columns=['product'], prefix='product')
        
        if 'region' in df.columns:
            df = pd.get_dummies(df, columns=['region'], prefix='region')
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    def train_model(self, df: pd.DataFrame, model_type: str = 'linear') -> Dict[str, Any]:
        """Train a forecasting model"""
        # Prepare features
        df_processed = self.prepare_features(df)
        
        # Define feature columns (exclude date and target)
        exclude_cols = ['date', self.target_column]
        self.feature_columns = [col for col in df_processed.columns if col not in exclude_cols]
        
        # Prepare X and y
        X = df_processed[self.feature_columns]
        y = df_processed[self.target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        if model_type == 'linear':
            model = LinearRegression()
        elif model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError("model_type must be 'linear' or 'random_forest'")
        
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'mae_train': mean_absolute_error(y_train, y_pred_train),
            'mae_test': mean_absolute_error(y_test, y_pred_test),
            'mse_train': mean_squared_error(y_train, y_pred_train),
            'mse_test': mean_squared_error(y_test, y_pred_test),
            'r2_train': r2_score(y_train, y_pred_train),
            'r2_test': r2_score(y_test, y_pred_test),
            'rmse_test': np.sqrt(mean_squared_error(y_test, y_pred_test))
        }
        
        # Store model and scaler
        self.models[model_type] = model
        self.scalers[model_type] = scaler
        
        logger.info(f"Trained {model_type} model with R² score: {metrics['r2_test']:.4f}")
        
        return {
            'model': model,
            'scaler': scaler,
            'metrics': metrics,
            'feature_importance': self._get_feature_importance(model, self.feature_columns)
        }
    
    def _get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance from model"""
        if hasattr(model, 'feature_importances_'):
            # Random Forest
            importance_dict = dict(zip(feature_names, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            # Linear Regression
            importance_dict = dict(zip(feature_names, np.abs(model.coef_)))
        else:
            importance_dict = {}
        
        # Sort by importance
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    def predict_future(self, df: pd.DataFrame, periods: int = 30, 
                      model_type: str = 'linear') -> pd.DataFrame:
        """Predict future sales"""
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not trained. Call train_model first.")
        
        model = self.models[model_type]
        scaler = self.scalers[model_type]
        
        # Get the last date from the data
        last_date = pd.to_datetime(df['date'].max())
        
        # Create future dates
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=periods,
            freq='D'
        )
        
        # Create future dataframe
        future_df = pd.DataFrame({'date': future_dates})
        
        # Prepare features for future data
        future_df_processed = self.prepare_features(future_df)
        
        # Use the same feature columns as training
        X_future = future_df_processed[self.feature_columns]
        
        # Scale features
        X_future_scaled = scaler.transform(X_future)
        
        # Make predictions
        predictions = model.predict(X_future_scaled)
        
        # Create result dataframe
        result_df = pd.DataFrame({
            'date': future_dates,
            'predicted_sales': predictions
        })
        
        return result_df
    
    def generate_forecast_report(self, df: pd.DataFrame, 
                               forecast_periods: int = 90) -> Dict[str, Any]:
        """Generate a comprehensive forecast report"""
        # Train models
        linear_results = self.train_model(df, 'linear')
        rf_results = self.train_model(df, 'random_forest')
        
        # Choose the better model based on R² score
        if linear_results['metrics']['r2_test'] > rf_results['metrics']['r2_test']:
            best_model = 'linear'
            best_results = linear_results
        else:
            best_model = 'random_forest'
            best_results = rf_results
        
        # Generate predictions
        predictions = self.predict_future(df, forecast_periods, best_model)
        
        # Calculate forecast statistics
        forecast_stats = {
            'mean_prediction': predictions['predicted_sales'].mean(),
            'std_prediction': predictions['predicted_sales'].std(),
            'min_prediction': predictions['predicted_sales'].min(),
            'max_prediction': predictions['predicted_sales'].max(),
            'total_predicted_sales': predictions['predicted_sales'].sum()
        }
        
        # Create report
        report = {
            'model_used': best_model,
            'model_metrics': best_results['metrics'],
            'feature_importance': best_results['feature_importance'],
            'predictions': predictions.to_dict('records'),
            'forecast_stats': forecast_stats,
            'training_data_points': len(df),
            'forecast_periods': forecast_periods,
            'generated_at': datetime.now().isoformat()
        }
        
        return report
    
    def plot_forecast(self, df: pd.DataFrame, predictions: pd.DataFrame, 
                     title: str = "Sales Forecast"):
        """Create a plot showing historical data and predictions"""
        plt.figure(figsize=(12, 6))
        
        # Plot historical data
        plt.plot(df['date'], df['sales'], label='Historical Sales', color='blue', alpha=0.7)
        
        # Plot predictions
        plt.plot(predictions['date'], predictions['predicted_sales'], 
                label='Forecasted Sales', color='red', linestyle='--')
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return plt
    
    def save_forecast_report(self, report: Dict[str, Any], filename: str = "forecast_report.json"):
        """Save forecast report to file"""
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Forecast report saved to {filename}")

# Example usage
def create_sample_forecast():
    """Create a sample forecast using the forecaster"""
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
    
    # Generate sample sales data with some seasonality
    sales = []
    for date in dates:
        # Base sales
        base_sales = 1000
        
        # Add seasonality (higher in December, lower in January)
        seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * date.dayofyear / 365)
        
        # Add trend (slight growth)
        trend_factor = 1 + 0.0001 * (date - dates[0]).days
        
        # Add some randomness
        noise = np.random.normal(0, 0.1)
        
        daily_sales = base_sales * seasonal_factor * trend_factor * (1 + noise)
        sales.append(max(0, daily_sales))
    
    # Create dataframe
    df = pd.DataFrame({
        'date': dates,
        'sales': sales
    })
    
    # Initialize forecaster
    forecaster = SalesForecaster()
    
    # Generate forecast report
    report = forecaster.generate_forecast_report(df, forecast_periods=90)
    
    # Save report
    forecaster.save_forecast_report(report)
    
    # Print summary
    print("Forecast Report Summary:")
    print(f"Model used: {report['model_used']}")
    print(f"R² Score: {report['model_metrics']['r2_test']:.4f}")
    print(f"RMSE: {report['model_metrics']['rmse_test']:.2f}")
    print(f"Total predicted sales: ${report['forecast_stats']['total_predicted_sales']:,.2f}")
    
    return report

if __name__ == "__main__":
    # Run sample forecast
    report = create_sample_forecast()
    print("Forecast generation completed!") 