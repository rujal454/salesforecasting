import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def forecast_sales(df: pd.DataFrame, periods: int = 30):
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # Simple linear regression forecast as fallback
    y = df['sales'].values
    X = np.arange(len(y)).reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, y)

    # Predict future values
    future_X = np.arange(len(y), len(y) + periods).reshape(-1, 1)
    forecast = model.predict(future_X)

    last_date = df['date'].max()
    forecast_dates = pd.date_range(last_date, periods=periods+1, freq='D')[1:]
    return {
        "forecast": [
            {"date": d.strftime("%Y-%m-%d"), "forecast": float(f)}
            for d, f in zip(forecast_dates, forecast)
        ]
    }