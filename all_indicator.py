import talib
import numpy as np
import pandas as pd

class RSIIndicator:
    def __init__(self, period, lower_band, upper_band):
        self.period = period
        self.lower_band = lower_band
        self.upper_band = upper_band

    def calculate(self, prices):
        try:
            prices = np.asarray(prices, dtype=float)
            if prices.size < self.period:
                raise ValueError("Not enough data to calculate RSI")

            rsi = talib.RSI(prices, timeperiod=self.period)

            rsi_band = np.full(rsi.shape, 'neutral', dtype=object)
            rsi_band[rsi < self.lower_band] = 'oversold'
            rsi_band[rsi > self.upper_band] = 'overbought'

            return rsi_band
        except Exception as e:
            print(f"Error calculating RSI: {e}")
            return None

class EMAIndicator:
    def __init__(self, period):
        self.period = period

    def calculate(self, prices):
        try:
            prices = np.asarray(prices, dtype=float)
            if prices.size < self.period:
                raise ValueError("Not enough data to calculate EMA")

            ema = talib.EMA(prices, timeperiod=self.period)
            return ema
        except Exception as e:
            print(f"Error calculating EMA: {e}")
            return None

class SMAIndicator:
    def __init__(self, period):
        self.period = period

    def calculate(self, prices):
        try:
            prices = np.asarray(prices, dtype=float)
            if prices.size < self.period:
                raise ValueError("Not enough data to calculate SMA")

            sma = talib.SMA(prices, timeperiod=self.period)
            return sma
        except Exception as e:
            print(f"Error calculating SMA: {e}")
            return None

class BollingerBands:
    def __init__(self, period, nbdevup=2, nbdevdn=2, matype=0):
        self.period = period
        self.nbdevup = nbdevup
        self.nbdevdn = nbdevdn
        self.matype = matype

    def calculate(self, prices):
        try:
            prices = np.asarray(prices, dtype=float)
            if prices.size < self.period:
                raise ValueError("Not enough data to calculate Bollinger Bands")

            upperband, middleband, lowerband = talib.BBANDS(
                prices, timeperiod=self.period,
                nbdevup=self.nbdevup, nbdevdn=self.nbdevdn, matype=self.matype
            )
            return upperband, middleband, lowerband
        except Exception as e:
            print(f"Error calculating Bollinger Bands: {e}")
            return None, None, None

class MACDIndicator:
    def __init__(self, fastperiod=12, slowperiod=26, signalperiod=9):
        self.fastperiod = fastperiod
        self.slowperiod = slowperiod
        self.signalperiod = signalperiod

    def calculate(self, prices):
        try:
            prices = np.asarray(prices, dtype=float)
            if prices.size < self.slowperiod:
                raise ValueError("Not enough data to calculate MACD")

            macd, macdsignal, macdhist = talib.MACD(
                prices, fastperiod=self.fastperiod,
                slowperiod=self.slowperiod, signalperiod=self.signalperiod
            )
            return macd, macdsignal, macdhist
        except Exception as e:
            print(f"Error calculating MACD: {e}")
            return None, None, None

class SupertrendIndicator:
    def __init__(self, period, multiplier):
        self.period = period
        self.multiplier = multiplier

    def calculate(self, high, low, close):
        try:
            high = np.asarray(high, dtype=float)
            low = np.asarray(low, dtype=float)
            close = np.asarray(close, dtype=float)
            if high.size < self.period or low.size < self.period or close.size < self.period:
                raise ValueError("Not enough data to calculate Supertrend")

            atr = talib.ATR(high, low, close, timeperiod=self.period)
            hl2 = (high + low) / 2
            basic_upperband = hl2 + (self.multiplier * atr)
            basic_lowerband = hl2 - (self.multiplier * atr)

            final_upperband = np.copy(basic_upperband)
            final_lowerband = np.copy(basic_lowerband)

            for i in range(1, len(close)):
                if close[i - 1] <= final_upperband[i - 1]:
                    final_upperband[i] = min(basic_upperband[i], final_upperband[i - 1])
                else:
                    final_upperband[i] = basic_upperband[i]

                if close[i - 1] >= final_lowerband[i - 1]:
                    final_lowerband[i] = max(basic_lowerband[i], final_lowerband[i - 1])
                else:
                    final_lowerband[i] = basic_lowerband[i]

            supertrend = np.where(close > final_upperband, final_lowerband, final_upperband)
            return supertrend
        except Exception as e:
            print(f"Error calculating Supertrend: {e}")
            return None
