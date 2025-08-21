from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from prophet import Prophet
import requests
import os
from dotenv import load_dotenv
import io
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Global variable to store uploaded data for analysis
uploaded_data_store = {}

def analyze_top_products(df, sales_col, product_col):
    """Analyze top performing products"""
    if not sales_col or not product_col:
        return {
            "answer": "I need both sales and product columns to analyze top products. Please ensure your data has these columns.",
            "model": "Local Sales Analyzer",
            "status": "error"
        }

    try:
        top_products = df.groupby(product_col)[sales_col].agg(['sum', 'count', 'mean']).round(2)
        top_products = top_products.sort_values('sum', ascending=False).head(10)

        analysis = f"📊 **TOP PERFORMING PRODUCTS ANALYSIS**\n\n"
        analysis += f"Based on your sales data with {len(df)} transactions:\n\n"

        for i, (product, data) in enumerate(top_products.iterrows(), 1):
            analysis += f"{i}. **{product}**\n"
            analysis += f"   • Total Sales: ${data['sum']:,.2f}\n"
            analysis += f"   • Number of Orders: {data['count']}\n"
            analysis += f"   • Average Order Value: ${data['mean']:,.2f}\n\n"

        total_revenue = df[sales_col].sum()
        top_10_revenue = top_products['sum'].sum()
        percentage = (top_10_revenue / total_revenue) * 100

        analysis += f"💡 **KEY INSIGHTS:**\n"
        analysis += f"• Top 10 products generate ${top_10_revenue:,.2f} ({percentage:.1f}%) of total revenue\n"
        analysis += f"• Total revenue across all products: ${total_revenue:,.2f}\n"
        analysis += f"• Best performing product: {top_products.index[0]} with ${top_products.iloc[0]['sum']:,.2f}"

        return {
            "answer": analysis,
            "model": "Local Sales Analyzer",
            "status": "success"
        }
    except Exception as e:
        return {
            "answer": f"Error analyzing products: {str(e)}",
            "model": "Local Sales Analyzer",
            "status": "error"
        }

def analyze_top_customers(df, sales_col, customer_col):
    """Analyze top customers"""
    if not sales_col or not customer_col:
        return {
            "answer": "I need both sales and customer columns to analyze top customers.",
            "model": "Local Sales Analyzer",
            "status": "error"
        }

    try:
        top_customers = df.groupby(customer_col)[sales_col].agg(['sum', 'count', 'mean']).round(2)
        top_customers = top_customers.sort_values('sum', ascending=False).head(10)

        analysis = f"👥 **TOP CUSTOMERS ANALYSIS**\n\n"

        for i, (customer, data) in enumerate(top_customers.iterrows(), 1):
            analysis += f"{i}. **{customer}**\n"
            analysis += f"   • Total Spent: ${data['sum']:,.2f}\n"
            analysis += f"   • Number of Orders: {data['count']}\n"
            analysis += f"   • Average Order: ${data['mean']:,.2f}\n\n"

        return {
            "answer": analysis,
            "model": "Local Sales Analyzer",
            "status": "success"
        }
    except Exception as e:
        return {
            "answer": f"Error analyzing customers: {str(e)}",
            "model": "Local Sales Analyzer",
            "status": "error"
        }

def analyze_top_sales(df, sales_col, date_col):
    """Analyze top sales periods"""
    if not sales_col:
        return {
            "answer": "I need a sales column to analyze top sales periods.",
            "model": "Local Sales Analyzer",
            "status": "error"
        }

    try:
        top_sales = df.nlargest(10, sales_col)
        analysis = f"💰 **TOP SALES TRANSACTIONS**\n\n"

        for i, (_, row) in enumerate(top_sales.iterrows(), 1):
            analysis += f"{i}. ${row[sales_col]:,.2f}"
            if date_col and date_col in row:
                analysis += f" on {row[date_col]}"
            analysis += "\n"

        total = df[sales_col].sum()
        avg = df[sales_col].mean()
        analysis += f"\n📊 **SUMMARY:**\n"
        analysis += f"• Total Sales: ${total:,.2f}\n"
        analysis += f"• Average Sale: ${avg:,.2f}\n"
        analysis += f"• Highest Sale: ${df[sales_col].max():,.2f}"

        return {
            "answer": analysis,
            "model": "Local Sales Analyzer",
            "status": "success"
        }
    except Exception as e:
        return {
            "answer": f"Error analyzing top sales: {str(e)}",
            "model": "Local Sales Analyzer",
            "status": "error"
        }

def analyze_trends(df, sales_col, date_col):
    """Analyze sales trends over time"""
    if not sales_col or not date_col:
        return {
            "answer": "I need both sales and date columns to analyze trends over time.",
            "model": "Local Sales Analyzer",
            "status": "error"
        }

    try:
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])

        # Monthly trends
        monthly = df_copy.groupby(df_copy[date_col].dt.to_period('M'))[sales_col].sum()

        analysis = f"📈 **SALES TRENDS ANALYSIS**\n\n"
        analysis += f"**Monthly Sales Trends:**\n"

        for period, sales in monthly.tail(12).items():
            analysis += f"• {period}: ${sales:,.2f}\n"

        # Growth analysis
        if len(monthly) > 1:
            recent_growth = ((monthly.iloc[-1] - monthly.iloc[-2]) / monthly.iloc[-2]) * 100
            analysis += f"\n📊 **Growth Analysis:**\n"
            analysis += f"• Month-over-month growth: {recent_growth:+.1f}%\n"
            analysis += f"• Best month: {monthly.idxmax()} (${monthly.max():,.2f})\n"
            analysis += f"• Lowest month: {monthly.idxmin()} (${monthly.min():,.2f})"

        return {
            "answer": analysis,
            "model": "Local Sales Analyzer",
            "status": "success"
        }
    except Exception as e:
        return {
            "answer": f"Error analyzing trends: {str(e)}",
            "model": "Local Sales Analyzer",
            "status": "error"
        }

def analyze_totals(df, sales_col):
    """Analyze total sales and revenue"""
    if not sales_col:
        return {
            "answer": "I need a sales column to calculate totals.",
            "model": "Local Sales Analyzer",
            "status": "error"
        }

    try:
        total_sales = df[sales_col].sum()
        total_transactions = len(df)

        analysis = f"💰 **TOTAL SALES ANALYSIS**\n\n"
        analysis += f"• **Total Revenue:** ${total_sales:,.2f}\n"
        analysis += f"• **Total Transactions:** {total_transactions:,}\n"
        analysis += f"• **Average Transaction:** ${total_sales/total_transactions:,.2f}\n"
        analysis += f"• **Highest Sale:** ${df[sales_col].max():,.2f}\n"
        analysis += f"• **Lowest Sale:** ${df[sales_col].min():,.2f}"

        return {
            "answer": analysis,
            "model": "Local Sales Analyzer",
            "status": "success"
        }
    except Exception as e:
        return {
            "answer": f"Error calculating totals: {str(e)}",
            "model": "Local Sales Analyzer",
            "status": "error"
        }

def analyze_averages(df, sales_col):
    """Analyze average sales metrics"""
    if not sales_col:
        return {
            "answer": "I need a sales column to calculate averages.",
            "model": "Local Sales Analyzer",
            "status": "error"
        }

    try:
        mean_sale = df[sales_col].mean()
        median_sale = df[sales_col].median()
        std_dev = df[sales_col].std()

        analysis = f"📊 **AVERAGE SALES ANALYSIS**\n\n"
        analysis += f"• **Mean (Average):** ${mean_sale:,.2f}\n"
        analysis += f"• **Median:** ${median_sale:,.2f}\n"
        analysis += f"• **Standard Deviation:** ${std_dev:,.2f}\n"

        if mean_sale > median_sale:
            analysis += f"\n💡 **Insight:** Your sales data shows a right-skewed distribution, meaning you have some high-value transactions pulling the average up."
        else:
            analysis += f"\n💡 **Insight:** Your sales data shows a fairly normal distribution."

        return {
            "answer": analysis,
            "model": "Local Sales Analyzer",
            "status": "success"
        }
    except Exception as e:
        return {
            "answer": f"Error calculating averages: {str(e)}",
            "model": "Local Sales Analyzer",
            "status": "error"
        }

def get_general_insights(df, sales_col, date_col, product_col, customer_col):
    """Provide general insights about the sales data"""
    try:
        insights = f"🔍 **GENERAL SALES DATA INSIGHTS**\n\n"
        insights += f"**Dataset Overview:**\n"
        insights += f"• Total records: {len(df):,}\n"
        insights += f"• Columns available: {len(df.columns)}\n"

        if sales_col:
            total_revenue = df[sales_col].sum()
            avg_sale = df[sales_col].mean()
            insights += f"• Total revenue: ${total_revenue:,.2f}\n"
            insights += f"• Average sale: ${avg_sale:,.2f}\n"

        if date_col:
            try:
                df_temp = df.copy()
                df_temp[date_col] = pd.to_datetime(df_temp[date_col])
                date_range = df_temp[date_col].max() - df_temp[date_col].min()
                insights += f"• Date range: {date_range.days} days\n"
            except:
                pass

        if product_col:
            unique_products = df[product_col].nunique()
            insights += f"• Unique products: {unique_products}\n"

        if customer_col:
            unique_customers = df[customer_col].nunique()
            insights += f"• Unique customers: {unique_customers}\n"

        insights += f"\n💡 **What you can ask me:**\n"
        insights += f"• 'What are the top products?' - for product analysis\n"
        insights += f"• 'Forecast sales for next month' - for sales predictions\n"
        insights += f"• 'Show me sales trends' - for time-based analysis\n"
        insights += f"• 'What's the total revenue?' - for summary statistics\n"
        insights += f"• 'Who are the best customers?' - for customer analysis\n"
        insights += f"• 'Analyze seasonal patterns' - for seasonality insights\n"
        insights += f"• 'Show performance metrics' - for KPI analysis\n"
        insights += f"• 'Which products are declining?' - for decline analysis"

        return {
            "answer": insights,
            "model": "Local Sales Analyzer",
            "status": "success"
        }
    except Exception as e:
        return {
            "answer": f"Error generating insights: {str(e)}",
            "model": "Local Sales Analyzer",
            "status": "error"
        }

def generate_forecast_analysis(df, sales_col, date_col, query_lower):
    """Generate sales forecast analysis"""
    if not sales_col or not date_col:
        return {
            "answer": "I need both sales and date columns to generate forecasts. Please ensure your data has these columns.",
            "model": "Local Sales Analyzer",
            "status": "error"
        }

    try:
        # Prepare data for Prophet
        df_clean = df[[date_col, sales_col]].copy()
        df_clean = df_clean.dropna()
        df_clean[date_col] = pd.to_datetime(df_clean[date_col])
        df_clean = df_clean.sort_values(date_col)

        # Aggregate daily sales
        daily_sales = df_clean.groupby(date_col)[sales_col].sum().reset_index()
        daily_sales.columns = ['ds', 'y']

        if len(daily_sales) < 10:
            return {
                "answer": "I need at least 10 data points to generate reliable forecasts. Your dataset is too small.",
                "model": "Local Sales Analyzer",
                "status": "error"
            }

        # Create and fit Prophet model
        model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
        model.fit(daily_sales)

        # Determine forecast period based on query
        periods = 30  # default
        if 'week' in query_lower:
            periods = 7
        elif 'month' in query_lower:
            periods = 30
        elif 'quarter' in query_lower:
            periods = 90
        elif 'year' in query_lower:
            periods = 365

        # Generate forecast
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)

        # Get future predictions only
        future_forecast = forecast.tail(periods)

        # Calculate metrics
        total_forecast = future_forecast['yhat'].sum()
        avg_daily_forecast = future_forecast['yhat'].mean()
        historical_avg = daily_sales['y'].mean()
        growth_rate = ((avg_daily_forecast - historical_avg) / historical_avg) * 100

        analysis = f"🔮 **SALES FORECAST ANALYSIS**\n\n"
        analysis += f"**Forecast Period:** Next {periods} days\n\n"
        analysis += f"**Predictions:**\n"
        analysis += f"• Total Forecasted Sales: ${total_forecast:,.2f}\n"
        analysis += f"• Average Daily Sales: ${avg_daily_forecast:,.2f}\n"
        analysis += f"• Historical Daily Average: ${historical_avg:,.2f}\n"
        analysis += f"• Projected Growth Rate: {growth_rate:+.1f}%\n\n"

        analysis += f"**Key Forecast Insights:**\n"
        if growth_rate > 5:
            analysis += f"📈 Strong growth expected ({growth_rate:+.1f}%)\n"
        elif growth_rate > 0:
            analysis += f"📊 Moderate growth expected ({growth_rate:+.1f}%)\n"
        else:
            analysis += f"📉 Decline expected ({growth_rate:+.1f}%)\n"

        # Show next few days
        analysis += f"\n**Next 7 Days Forecast:**\n"
        for i, (_, row) in enumerate(future_forecast.head(7).iterrows()):
            date_str = row['ds'].strftime('%Y-%m-%d')
            analysis += f"• {date_str}: ${row['yhat']:,.2f}\n"

        analysis += f"\n💡 **Recommendation:** "
        if growth_rate > 5:
            analysis += "Strong growth is predicted. Consider increasing inventory and marketing efforts."
        elif growth_rate > 0:
            analysis += "Steady growth expected. Maintain current strategies with minor optimizations."
        else:
            analysis += "Decline predicted. Review pricing, marketing, and product strategies."

        return {
            "answer": analysis,
            "model": "Local Sales Analyzer (Prophet)",
            "status": "success"
        }

    except Exception as e:
        return {
            "answer": f"Error generating forecast: {str(e)}. Please ensure your data has proper date and sales columns.",
            "model": "Local Sales Analyzer",
            "status": "error"
        }

def analyze_seasonality(df, sales_col, date_col):
    """Analyze seasonal patterns in sales data"""
    if not sales_col or not date_col:
        return {
            "answer": "I need both sales and date columns to analyze seasonality.",
            "model": "Local Sales Analyzer",
            "status": "error"
        }

    try:
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])

        # Monthly analysis
        df_copy['month'] = df_copy[date_col].dt.month
        df_copy['quarter'] = df_copy[date_col].dt.quarter
        df_copy['day_of_week'] = df_copy[date_col].dt.day_name()

        monthly_sales = df_copy.groupby('month')[sales_col].mean().round(2)
        quarterly_sales = df_copy.groupby('quarter')[sales_col].mean().round(2)
        weekly_sales = df_copy.groupby('day_of_week')[sales_col].mean().round(2)

        analysis = f"🌟 **SEASONALITY ANALYSIS**\n\n"

        analysis += f"**Monthly Patterns:**\n"
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for month, sales in monthly_sales.items():
            analysis += f"• {month_names[month-1]}: ${sales:,.2f} avg\n"

        analysis += f"\n**Quarterly Patterns:**\n"
        quarter_names = ['Q1', 'Q2', 'Q3', 'Q4']
        for quarter, sales in quarterly_sales.items():
            analysis += f"• {quarter_names[quarter-1]}: ${sales:,.2f} avg\n"

        analysis += f"\n**Weekly Patterns:**\n"
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for day in day_order:
            if day in weekly_sales.index:
                analysis += f"• {day}: ${weekly_sales[day]:,.2f} avg\n"

        # Find best periods
        best_month = monthly_sales.idxmax()
        best_quarter = quarterly_sales.idxmax()
        best_day = weekly_sales.idxmax()

        analysis += f"\n💡 **Key Insights:**\n"
        analysis += f"• Best month: {month_names[best_month-1]} (${monthly_sales[best_month]:,.2f})\n"
        analysis += f"• Best quarter: {quarter_names[best_quarter-1]} (${quarterly_sales[best_quarter]:,.2f})\n"
        analysis += f"• Best day: {best_day} (${weekly_sales[best_day]:,.2f})\n"

        return {
            "answer": analysis,
            "model": "Local Sales Analyzer",
            "status": "success"
        }

    except Exception as e:
        return {
            "answer": f"Error analyzing seasonality: {str(e)}",
            "model": "Local Sales Analyzer",
            "status": "error"
        }

def analyze_performance_metrics(df, sales_col, date_col, product_col, customer_col):
    """Analyze comprehensive performance metrics"""
    try:
        analysis = f"📊 **PERFORMANCE METRICS DASHBOARD**\n\n"

        if sales_col:
            total_revenue = df[sales_col].sum()
            avg_transaction = df[sales_col].mean()
            median_transaction = df[sales_col].median()

            analysis += f"**Revenue Metrics:**\n"
            analysis += f"• Total Revenue: ${total_revenue:,.2f}\n"
            analysis += f"• Average Transaction: ${avg_transaction:,.2f}\n"
            analysis += f"• Median Transaction: ${median_transaction:,.2f}\n"
            analysis += f"• Total Transactions: {len(df):,}\n\n"

        if product_col:
            unique_products = df[product_col].nunique()
            top_product = df.groupby(product_col)[sales_col].sum().idxmax()
            analysis += f"**Product Metrics:**\n"
            analysis += f"• Total Products: {unique_products}\n"
            analysis += f"• Top Product: {top_product}\n\n"

        if customer_col:
            unique_customers = df[customer_col].nunique()
            repeat_customers = df[customer_col].value_counts()
            repeat_rate = (repeat_customers > 1).sum() / len(repeat_customers) * 100
            analysis += f"**Customer Metrics:**\n"
            analysis += f"• Total Customers: {unique_customers}\n"
            analysis += f"• Repeat Customer Rate: {repeat_rate:.1f}%\n\n"

        if date_col:
            df_temp = df.copy()
            df_temp[date_col] = pd.to_datetime(df_temp[date_col])
            date_range = (df_temp[date_col].max() - df_temp[date_col].min()).days
            analysis += f"**Time Metrics:**\n"
            analysis += f"• Data Period: {date_range} days\n"
            analysis += f"• Daily Average: ${total_revenue/max(date_range, 1):,.2f}\n"

        return {
            "answer": analysis,
            "model": "Local Sales Analyzer",
            "status": "success"
        }

    except Exception as e:
        return {
            "answer": f"Error analyzing performance metrics: {str(e)}",
            "model": "Local Sales Analyzer",
            "status": "error"
        }

def analyze_declining_performance(df, sales_col, product_col, customer_col):
    """Analyze declining products and customers"""
    try:
        analysis = f"📉 **DECLINING PERFORMANCE ANALYSIS**\n\n"

        if product_col and sales_col:
            product_sales = df.groupby(product_col)[sales_col].agg(['sum', 'count', 'mean']).round(2)
            worst_products = product_sales.sort_values('sum').head(5)

            analysis += f"**Lowest Performing Products:**\n"
            for i, (product, data) in enumerate(worst_products.iterrows(), 1):
                analysis += f"{i}. {product}: ${data['sum']:,.2f} total\n"

        if customer_col and sales_col:
            customer_sales = df.groupby(customer_col)[sales_col].sum().round(2)
            worst_customers = customer_sales.sort_values().head(5)

            analysis += f"\n**Lowest Spending Customers:**\n"
            for i, (customer, sales) in enumerate(worst_customers.items(), 1):
                analysis += f"{i}. {customer}: ${sales:,.2f}\n"

        # General insights
        if sales_col:
            low_value_threshold = df[sales_col].quantile(0.25)
            low_value_count = (df[sales_col] <= low_value_threshold).sum()
            low_value_percentage = (low_value_count / len(df)) * 100

            analysis += f"\n💡 **Insights:**\n"
            analysis += f"• {low_value_count} transactions ({low_value_percentage:.1f}%) are below ${low_value_threshold:.2f}\n"
            analysis += f"• Consider strategies to improve low-performing segments\n"
            analysis += f"• Focus on upselling and customer retention"

        return {
            "answer": analysis,
            "model": "Local Sales Analyzer",
            "status": "success"
        }

    except Exception as e:
        return {
            "answer": f"Error analyzing declining performance: {str(e)}",
            "model": "Local Sales Analyzer",
            "status": "error"
        }

# Global variable to store uploaded data for analysis
uploaded_data_store = {}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def prophet_forecast(df: pd.DataFrame, periods: int = 30):
    """Advanced Prophet forecasting model"""
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Try to find date and sales columns
    date_col = None
    sales_col = None

    # Look for date columns
    for col in df.columns:
        col_lower = col.lower()
        if any(word in col_lower for word in ['date', 'time', 'day', 'month', 'year']):
            date_col = col
            break

    # Look for sales/revenue columns
    for col in df.columns:
        col_lower = col.lower()
        if any(word in col_lower for word in ['sales', 'revenue', 'amount', 'value', 'price', 'total']):
            sales_col = col
            break

    # If no specific columns found, try to use numeric columns
    if not sales_col:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            sales_col = numeric_cols[0]

    print(f"Found date column: {date_col}")
    print(f"Found sales column: {sales_col}")

    if not sales_col:
        return {"error": f"Could not find a sales/numeric column. Available columns: {df.columns.tolist()}"}

    # If no date column, create a synthetic one
    if not date_col:
        df['synthetic_date'] = pd.date_range(start='2023-01-01', periods=len(df), freq='D')
        date_col = 'synthetic_date'
        print("Created synthetic date column")
    else:
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except Exception as e:
            print(f"Error parsing dates: {e}")
            df['synthetic_date'] = pd.date_range(start='2023-01-01', periods=len(df), freq='D')
            date_col = 'synthetic_date'

    # Prepare data for Prophet
    df_clean = df[[date_col, sales_col]].copy()
    df_clean = df_clean.dropna()
    df_clean = df_clean.sort_values(date_col)

    # Rename columns for Prophet (requires 'ds' and 'y')
    df_prophet = pd.DataFrame({
        'ds': df_clean[date_col],
        'y': df_clean[sales_col]
    })

    if len(df_prophet) < 2:
        return {"error": "Not enough valid data points for forecasting"}

    print(f"Training Prophet model with {len(df_prophet)} data points")

    # Create and fit Prophet model
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        interval_width=0.95
    )

    try:
        model.fit(df_prophet)

        # Create future dataframe
        future = model.make_future_dataframe(periods=periods)

        # Make predictions
        forecast = model.predict(future)

        # Extract only the future predictions
        future_forecast = forecast.tail(periods)

        # Calculate model performance metrics on historical data
        historical_forecast = forecast.head(len(df_prophet))
        mae = np.mean(np.abs(historical_forecast['yhat'] - df_prophet['y']))
        mape = np.mean(np.abs((df_prophet['y'] - historical_forecast['yhat']) / df_prophet['y'])) * 100

        return {
            "forecast": [
                {
                    "date": row['ds'].strftime("%Y-%m-%d"),
                    "forecast": float(row['yhat']),
                    "lower_bound": float(row['yhat_lower']),
                    "upper_bound": float(row['yhat_upper'])
                }
                for _, row in future_forecast.iterrows()
            ],
            "model_performance": {
                "mae": float(mae),
                "mape": float(mape),
                "training_samples": len(df_prophet)
            },
            "info": {
                "date_column": date_col,
                "sales_column": sales_col,
                "data_points": len(df_prophet),
                "forecast_periods": periods,
                "model": "Prophet"
            }
        }

    except Exception as e:
        print(f"Prophet model error: {e}")
        return {"error": f"Prophet forecasting failed: {str(e)}"}

def read_csv_with_encoding(file_content):
    """Try to read CSV with different encodings"""
    encodings_to_try = ['utf-8', 'utf-8-sig', 'cp1252', 'latin-1', 'iso-8859-1']

    for encoding in encodings_to_try:
        try:
            # Reset file pointer
            file_content.seek(0)
            df = pd.read_csv(file_content, encoding=encoding)
            print(f"Successfully read CSV with encoding: {encoding}")
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error with encoding {encoding}: {e}")
            continue

    # If all encodings fail, try with error handling
    file_content.seek(0)
    return pd.read_csv(file_content, encoding='utf-8', errors='ignore')

@app.post("/forecast")
async def forecast(csv: UploadFile = File(...)):
    try:
        df = read_csv_with_encoding(csv.file)
        print("Uploaded DF:", df.head())  # debug print
        print("Columns:", df.columns.tolist())

        # Store the data for chatbot analysis
        import time
        timestamp = str(int(time.time()))
        uploaded_data_store[timestamp] = df

        # Keep only the last 5 uploads to manage memory
        if len(uploaded_data_store) > 5:
            oldest_key = min(uploaded_data_store.keys())
            del uploaded_data_store[oldest_key]

        result = prophet_forecast(df)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

@app.post("/embed-index")
async def embed_index(csv: UploadFile = File(...), text_column: str = Form(...)):
    try:
        df = read_csv_with_encoding(csv.file)
        print(f"Indexing {len(df)} rows with text column: {text_column}")

        # Store the data for chatbot analysis
        import time
        timestamp = str(int(time.time()))
        uploaded_data_store[timestamp] = df

        # Keep only the last 5 uploads to manage memory
        if len(uploaded_data_store) > 5:
            oldest_key = min(uploaded_data_store.keys())
            del uploaded_data_store[oldest_key]

        return {"indexed": len(df)}
    except Exception as e:
        return {"error": str(e)}

def analyze_sales_data_locally(query: str, df: pd.DataFrame = None):
    """Smart local analysis of sales data"""
    query_lower = query.lower()

    if df is None or df.empty:
        return {
            "answer": "I don't have access to your sales data yet. Please upload a CSV file first to get detailed analysis.",
            "model": "Local Sales Analyzer",
            "status": "success"
        }

    try:
        # Identify key columns
        sales_col = None
        date_col = None
        product_col = None
        customer_col = None

        for col in df.columns:
            col_lower = col.lower()
            if any(word in col_lower for word in ['sales', 'revenue', 'amount', 'total']):
                sales_col = col
            elif any(word in col_lower for word in ['date', 'time']):
                date_col = col
            elif any(word in col_lower for word in ['product', 'item']):
                product_col = col
            elif any(word in col_lower for word in ['customer', 'client']):
                customer_col = col

        # Analyze based on query type
        if any(word in query_lower for word in ['forecast', 'predict', 'future', 'next month', 'next quarter', 'projection']):
            return generate_forecast_analysis(df, sales_col, date_col, query_lower)

        elif any(word in query_lower for word in ['top', 'best', 'highest', 'most']):
            if any(word in query_lower for word in ['product', 'item']):
                return analyze_top_products(df, sales_col, product_col)
            elif any(word in query_lower for word in ['customer', 'client']):
                return analyze_top_customers(df, sales_col, customer_col)
            else:
                return analyze_top_sales(df, sales_col, date_col)

        elif any(word in query_lower for word in ['trend', 'pattern', 'over time', 'growth']):
            return analyze_trends(df, sales_col, date_col)

        elif any(word in query_lower for word in ['total', 'sum', 'revenue']):
            return analyze_totals(df, sales_col)

        elif any(word in query_lower for word in ['average', 'mean']):
            return analyze_averages(df, sales_col)

        elif any(word in query_lower for word in ['seasonal', 'season', 'monthly', 'quarterly']):
            return analyze_seasonality(df, sales_col, date_col)

        elif any(word in query_lower for word in ['performance', 'kpi', 'metrics']):
            return analyze_performance_metrics(df, sales_col, date_col, product_col, customer_col)

        elif any(word in query_lower for word in ['decline', 'drop', 'worst', 'lowest']):
            return analyze_declining_performance(df, sales_col, product_col, customer_col)

        else:
            return get_general_insights(df, sales_col, date_col, product_col, customer_col)

    except Exception as e:
        return {
            "answer": f"I encountered an error analyzing your data: {str(e)}. Please make sure your CSV has sales/revenue data.",
            "model": "Local Sales Analyzer",
            "status": "error"
        }

def chat_with_openrouter(query: str, context: str = ""):
    """Chat with OpenRouter API ONLY - No local analyzer"""
    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("MISTRAL_MODEL", "mistralai/mistral-7b-instruct")

    if not api_key or api_key == "your_actual_openrouter_api_key_here":
        return {
            "error": "❌ OpenRouter API key required! Please update your API key in the .env file.",
            "status": "error",
            "model": "None - API key missing"
        }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Sales Forecasting AI"
    }

    print(f"Using OpenRouter API with key: {api_key[:20]}...")  # Only show first 20 chars for security

    # Get sales data context if available
    data_context = ""
    if uploaded_data_store:
        latest_key = max(uploaded_data_store.keys())
        df = uploaded_data_store[latest_key]

        # Create detailed data context
        data_context = f"""
SALES DATA OVERVIEW:
- Total Records: {len(df):,}
- Columns: {', '.join(df.columns.tolist())}
- Date Range: {df.shape[0]} transactions
"""

        # Add sales summary if sales column exists
        sales_cols = [col for col in df.columns if any(word in col.lower() for word in ['sales', 'revenue', 'amount', 'total', 'price'])]
        if sales_cols:
            sales_col = sales_cols[0]
            total_sales = df[sales_col].sum()
            avg_sales = df[sales_col].mean()
            data_context += f"""
SALES METRICS:
- Total Revenue: ${total_sales:,.2f}
- Average Transaction: ${avg_sales:,.2f}
- Number of Transactions: {len(df):,}
"""

        # Add product info if available
        product_cols = [col for col in df.columns if any(word in col.lower() for word in ['product', 'item'])]
        if product_cols:
            product_col = product_cols[0]
            unique_products = df[product_col].nunique()
            top_products = df.groupby(product_col)[sales_cols[0]].sum().nlargest(5).to_dict() if sales_cols else {}
            data_context += f"""
PRODUCT INFO:
- Total Products: {unique_products}
- Top 5 Products by Sales: {dict(list(top_products.items())[:5])}
"""

    # Create a comprehensive prompt
    system_prompt = """You are an expert sales analyst and business intelligence assistant with access to real sales data. You provide detailed, data-driven insights and recommendations.

Your expertise includes:
- Sales performance analysis and KPI interpretation
- Revenue forecasting and trend analysis
- Product performance and portfolio optimization
- Customer segmentation and behavior analysis
- Seasonal pattern recognition
- Business strategy recommendations
- Market opportunity identification

Always provide specific, actionable insights with concrete numbers when possible. Format responses professionally with clear sections and bullet points."""

    user_prompt = f"""
SALES DATA CONTEXT:
{data_context}

BUSINESS QUESTION: {query}

Please analyze this question in the context of the sales data provided above. Give specific insights, numbers, and actionable recommendations based on the actual data metrics shown.
"""

    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 1000
    }

    try:
        print(f"Sending request to OpenRouter with model: {model}")
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=60
        )

        print(f"OpenRouter response status: {response.status_code}")
        print(f"OpenRouter response: {response.text[:500]}...")

        response.raise_for_status()

        result = response.json()

        # More robust response parsing
        if "choices" in result and len(result["choices"]) > 0:
            answer = result["choices"][0]["message"]["content"]
            return {
                "answer": answer,
                "model": model,
                "tokens_used": result.get("usage", {}).get("total_tokens", 0),
                "status": "success"
            }
        else:
            return {
                "error": "No response from AI model",
                "raw_response": result,
                "status": "error"
            }

    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return {"error": f"OpenRouter API error: {str(e)}", "status": "error"}
    except Exception as e:
        print(f"Processing error: {e}")
        return {"error": f"Chat processing error: {str(e)}", "status": "error"}

@app.post("/chat")
async def chat(query: str = Form(...)):
    try:
        # For now, we'll use basic context. In a full implementation,
        # this would include actual data from uploaded files
        context = "Sales data analysis context - user has uploaded sales data for analysis."
        result = chat_with_openrouter(query, context)
        return result
    except Exception as e:
        return {"error": str(e)}

@app.post("/vector-search")
async def vector_search(query: str = Form(...), n_results: int = Form(5)):
    # Simple mock response
    return {"results": [{"document": f"Mock result for query: {query}", "distance": 0.5}]}

@app.get("/")
async def root():
    return {"message": "Sales Forecasting API is running!"}

@app.get("/test-openrouter")
async def test_openrouter():
    """Test OpenRouter API connection"""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return {"error": "OpenRouter API key not found"}

    result = chat_with_openrouter("Hello, can you respond with a simple greeting?", "Test context")
    return {"test_result": result}