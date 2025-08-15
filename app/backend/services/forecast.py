import pandas as pd
import numpy as np
import pmdarima as pm

def forecast_sales(df: pd.DataFrame, periods: int = 30):
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    y = df['sales'].values
    model = pm.auto_arima(y, seasonal=False, suppress_warnings=True)
    forecast = model.predict(n_periods=periods)
    last_date = df['date'].max()
    forecast_dates = pd.date_range(last_date, periods=periods+1, freq='D')[1:]
    return {
        "forecast": [
            {"date": d.strftime("%Y-%m-%d"), "forecast": float(f)}
            for d, f in zip(forecast_dates, forecast)
        ]
    }