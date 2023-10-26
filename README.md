# BTC-Forecast-Aesthetics-Analyzer

This project aims to predict Bitcoin prices using three different machine learning models: Prophet, ARIMA, and LSTM and evaluate the aesthetics of the generated prediction charts.

Usage:

>python main.py [predicted_days] [past_days]
>predicted_days: The number of days to predict Bitcoin prices. (Default: 25)
>past_days: The number of past days of Bitcoin data to consider. (Default: 100)

Output:

The predicted Bitcoin prices for the next predicted_days days.
The aesthetics scores of the generated prediction charts.
The generated prediction charts saved as PNG files in the project directory.

Example:

>python main.py 50 200
This will predict Bitcoin prices for the next 50 days using the past 200 days of Bitcoin data. The predicted prices and the aesthetics scores of the generated prediction charts will be printed to the console. The generated prediction charts will be saved as PNG files in the project directory.
