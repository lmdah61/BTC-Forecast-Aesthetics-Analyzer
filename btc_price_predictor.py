import yfinance as yf
import numpy as np
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates


class BitcoinPricePredictor:
    def __init__(self):
        self.prophet_model = None
        self.arima_model = None
        self.lstm_model = None

    # Fetch historical Bitcoin price data using the yfinance library.
    def fetch_bitcoin_price_data(self, past_days):
        ticker = yf.Ticker("BTC-USD")
        bitcoin_data = ticker.history(period=f"{past_days}d")
        return bitcoin_data

    # Prepare data for Prophet model by renaming and formatting columns.
    def prepare_prophet_data(self, data):
        data = data[['Close']]
        data = data.rename(columns={'Close': 'y'})
        data.reset_index(inplace=True)
        data['ds'] = data['Date'].dt.strftime('%Y-%m-%d')
        return data

    def train_prophet_model(self, data, prediction_days):
        self.prophet_model = Prophet()
        self.prophet_model.fit(data)
        future = self.prophet_model.make_future_dataframe(periods=prediction_days)
        prophet_forecast = self.prophet_model.predict(future)
        prophet_predictions = prophet_forecast.tail(prediction_days)['yhat'].values
        return prophet_predictions

    # Prepare data for ARIMA model by setting the frequency.
    def prepare_arima_data(self, data):
        data.index.freq = 'D'
        return data['Close']

    def train_arima_model(self, data, prediction_days):
        self.arima_model = ARIMA(data, order=(5, 1, 0))
        arima_model_fit = self.arima_model.fit()
        arima_predictions_table = arima_model_fit.forecast(steps=prediction_days)
        arima_predictions = list(arima_predictions_table.values)
        return arima_predictions

    # Prepare data for LSTM model by scaling values.
    def prepare_lstm_data(self, data):
        data_values = data.values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data_values)
        return data_scaled, scaler

    def train_lstm_model(self, data, prediction_days, scaler):
        look_back = 1
        X, Y = self.create_dataset(data, look_back)

        self.lstm_model = Sequential()
        self.lstm_model.add(LSTM(4, input_shape=(look_back, 1)))
        self.lstm_model.add(Dense(1))
        self.lstm_model.compile(loss='mean_squared_error', optimizer='adam')
        self.lstm_model.fit(X, Y, epochs=100, batch_size=1, verbose=0)

        lstm_predictions = []
        test_input = data[-look_back:]
        test_input = test_input.reshape(1, 1, 1)

        for i in range(prediction_days):
            prediction = self.lstm_model.predict(test_input)
            lstm_predictions.append(scaler.inverse_transform(prediction)[0, 0])
            if i < prediction_days - 1:
                test_input = prediction.reshape(1, 1, 1)

        return lstm_predictions

    def create_dataset(self, dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)

    def train_and_predict_bitcoin_prices(self, prediction_days, past_days):
        bitcoin_data = self.fetch_bitcoin_price_data(past_days)

        # Prophet
        prophet_data = self.prepare_prophet_data(bitcoin_data)
        prophet_predictions = self.train_prophet_model(prophet_data, prediction_days)

        # ARIMA
        arima_data = self.prepare_arima_data(bitcoin_data)
        arima_predictions_table = self.train_arima_model(arima_data, prediction_days)
        arima_predictions = list(arima_predictions_table)

        # LSTM
        lstm_data, scaler = self.prepare_lstm_data(bitcoin_data['Close'])
        lstm_predictions = self.train_lstm_model(lstm_data, prediction_days, scaler)

        return {
            "Prophet Predictions": prophet_predictions,
            "ARIMA Predictions": arima_predictions,
            "LSTM Predictions": lstm_predictions
        }

    # Generate and save a chart to visualize actual and predicted Bitcoin prices.
    def generate_and_save_prediction_chart(self, bitcoin_data, predictions, model_name, filename):
        today = datetime.today()
        forecast_dates = [today + timedelta(days=i) for i in range(len(predictions))]
        forecast_dates_num = mdates.date2num(forecast_dates)

        plt.figure(figsize=(12, 6))
        plt.plot(bitcoin_data.index, bitcoin_data['Close'], label='Actual Bitcoin Price', marker='o', linestyle='-',
                 color='b')
        plt.plot(forecast_dates_num, predictions, label=f'{model_name} Forecast', marker='o', linestyle='--', color='r')

        plt.title(f'{model_name} Bitcoin Price Forecast')
        plt.xlabel('Date')
        plt.ylabel('Bitcoin Price (USD)')
        plt.legend()

        plt.savefig(filename)
        plt.close()
