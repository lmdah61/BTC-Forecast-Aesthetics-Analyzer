import sys
from btc_price_predictor import BitcoinPricePredictor
from chart_aesthethics_evaluator import AestheticModel

if __name__ == "__main__":
    # The first command-line argument, if provided, determines the number of days to predict Bitcoin prices.
    # The second command-line argument, if provided, specifies the number of past days of Bitcoin data to consider.
    # Check if command-line arguments are provided, otherwise, use default values
    if len(sys.argv) > 1:
        predicted_days = int(sys.argv[1])
    else:
        predicted_days = 25

    if len(sys.argv) > 2:
        past_days = int(sys.argv[2])
    else:
        past_days = 100

    predictor = BitcoinPricePredictor()
    predictions = predictor.train_and_predict_bitcoin_prices(predicted_days, past_days)

    # Visualize the price predictions
    print("Prophet Predictions:", predictions["Prophet Predictions"])
    print("ARIMA Predictions:", predictions["ARIMA Predictions"])
    print("LSTM Predictions:", predictions["LSTM Predictions"])

    bitcoin_data = predictor.fetch_bitcoin_price_data(past_days)

    # Save charts
    predictor.generate_and_save_prediction_chart(bitcoin_data, predictions["Prophet Predictions"], 'Prophet',
                                                 'prophet_forecast.png')
    predictor.generate_and_save_prediction_chart(bitcoin_data, predictions["ARIMA Predictions"], 'ARIMA',
                                                 'arima_forecast.png')
    predictor.generate_and_save_prediction_chart(bitcoin_data, predictions["LSTM Predictions"], 'LSTM',
                                                 'lstm_forecast.png')

    # Evaluate the aesthetics of the saved forecast charts.
    aesthetic_model = AestheticModel()

    print("Prophet Score:", aesthetic_model.evaluate_aesthetics('prophet_forecast.png'))
    print("ARIMA Score:", aesthetic_model.evaluate_aesthetics('arima_forecast.png'))
    print("LSTM Score:", aesthetic_model.evaluate_aesthetics('lstm_forecast.png'))
