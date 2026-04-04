# Sales Demand Forecasting System

## Project Overview
This project is a Machine Learning-powered web dashboard built with Python and Streamlit to track long-term business performance and forecast proactive retail demand. By leveraging a Random Forest regression model on historical sales data, the dashboard computes average monthly sales, annual trend growth, the next month's forecast, and key seasonal indicators, helping businesses transition from reactive reporting to proactive operational planning.

## Dataset Description
The system relies on historical sales tracking data. It supports loading either `sales_data.csv` or `superstore.csv`. 
At a minimum, the dataset must contain:
- **`Date`**: The specific date and time of the sales records.
- **`Sales`**: Numerical values representing the total sales on each given date.

## Libraries Used
This project relies on the following major Python libraries:
- **Streamlit** (`streamlit==1.55.0`): For building the interactive front-end web dashboard.
- **Pandas** (`pandas==2.3.3`): For data manipulation, feature engineering, and time-series resampling.
- **NumPy** (`numpy==2.4.3`): For numerical operations and calculating error metrics like RMSE.
- **Matplotlib** (`matplotlib==3.10.8`): For rendering the historical sales and seasonal analytics charts.
- **Scikit-Learn** (`scikit-learn==1.8.0`): Specifically the `RandomForestRegressor` for training the predictive forecasting model and extracting evaluation metrics.

## How to Run Project

### 1. Prerequisites
Ensure you have Python 3.8+ installed on your system.

### 2. Setup the Environment
Open a terminal in the project directory and create a virtual environment:
```bash
python -m venv venv
```
Activate it:
- **Windows:** `.\venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

### 3. Install Dependencies
Install the required libraries listed in `requirements.txt` (or via `package.json` if using an automated node environment):
```bash
pip install -r requirements.txt
```

### 4. Launch the Dashboard
Because the dashboard is built with Streamlit, it must be run as a module to launch the web server. Run the following command:
```bash
python -m streamlit run dashboard.py
```
*(Note: Attempting to run it with standard `python dashboard.py` will result in a `Missing ScriptRunContext` warning).*

Once executed, the dashboard will automatically open in your default web browser at `http://localhost:8501`.

## Business Impact
Deploying this forecasting system drives immediate, measurable business impact:
1. **Optimized Procurement:** By predicting the next 14 to 120 days of sales demand with high accuracy, procurement managers can adjust bulk ordering to strictly match expected velocity, directly reducing warehousing costs and food/product spoilage.
2. **Staffing Efficiency:** Using the interactive forecast horizon, management can guarantee adequate shift coverage during projected busy weekends and confidently dial back labor during predicted lulls.
3. **Strategic Marketing Alignment:** The visual "Average Seasonal Strength" indicator isolates precisely when the business historically hits peak seasons, allowing the marketing team to allocate ad spend for maximum momentum.
