from all_indicator import *

def detect_crossover(series1, series2):
    """Detects crossovers between two series. Returns a series where 1 indicates a bullish market, -1 a bearish market."""
    crossover = np.where(series1 > series2, 1, -1)
    # Detect when crossover changes
    crossover_change = np.diff(crossover, prepend=crossover[0])
    # Only change the trend on actual crossover events
    market_condition = np.where(crossover_change != 0, crossover, 0)
    return np.cumsum(market_condition)  # Cumulative sum to maintain market state

def main():
    path = r"D:\Sorted_data_stock_data\20MICRONS.csv"

    # Load the data
    df = pd.read_csv(path, usecols=['High', 'Low', 'Close'])
    close = df['Close'].values

    # Initialize indicators
    rsi_indicator = RSIIndicator(period=14, lower_band=40, upper_band=60)
    ema_indicator = EMAIndicator(period=10)
    sma_indicator = SMAIndicator(period=10)

    # Calculate indicators
    df['RSI'] = rsi_indicator.calculate(close)
    df['EMA'] = ema_indicator.calculate(close)
    df['SMA'] = sma_indicator.calculate(close)

    # Detect crossovers between SMA and EMA and maintain market condition
    df['Market_Condition'] = detect_crossover(df['SMA'], df['EMA'])

    print(df[['Close', 'SMA', 'EMA', 'RSI', 'Market_Condition']])

if __name__ == "__main__":
    main()
