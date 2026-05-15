# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports! ---
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Union

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)

import talib.abstract as ta
from scipy.signal import argrelextrema


class DivergenceStrategy(IStrategy):
    INTERFACE_VERSION = 3

    # Optymalizowalne parametry
    swing_order = IntParameter(2, 8, default=5, space="buy", optimize=True)
    min_swing_gap = IntParameter(4, 24, default=12, space="buy", optimize=True)
    confluence_window = IntParameter(1, 8, default=3, space="buy", optimize=True)
    rsi_period = IntParameter(10, 21, default=14, space="buy", optimize=True)
    stoch_k = IntParameter(5, 21, default=14, space="buy", optimize=True)
    stoch_d = IntParameter(3, 9, default=3, space="buy", optimize=True)
    stoch_smooth = IntParameter(3, 9, default=3, space="buy", optimize=True)

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI designed for the strategy.
    minimal_roi = {
        "0": 0.624,
        "60": 0.183,
        "240": 0.062,
        "480": 0
    }

    # Optimal stoploss designed for the strategy.
    stoploss = -0.088

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.045
    trailing_stop_positive_offset = 0.048
    trailing_only_offset_is_reached = False

    # Optimal timeframe for the strategy.
    timeframe = '1h'

    # Run "populate" only once per new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 100

    # Optional order type mapping.
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC'
    }

    plot_config = {
        'main_plot': {
            'bb_lowerband': {'color': 'grey'},
            'bb_upperband': {'color': 'grey'},
        },
        'subplots': {
            "RSI": {
                'rsi': {'color': 'blue'},
            },
            "Stoch": {
                'stoch_k': {'color': 'orange'},
                'stoch_d': {'color': 'green'},
            },
            "MACD": {
                'macd': {'color': 'blue'},
                'macdsignal': {'color': 'orange'},
            },
        }
    }

    def _find_swings(self, series: pd.Series, order: int, min_gap: int):
        s = series.dropna()
        if len(s) < 2 * order + 1:
            return [], []
        max_idx = argrelextrema(s.values, np.greater_equal, order=order)[0]
        min_idx = argrelextrema(s.values, np.less_equal, order=order)[0]
        max_pos = s.index[max_idx]
        min_pos = s.index[min_idx]

        def _filter_close(idx_list):
            if len(idx_list) == 0:
                return idx_list
            filtered = [idx_list[0]]
            for pos in idx_list[1:]:
                if (pos - filtered[-1]) >= min_gap:
                    filtered.append(pos)
            return filtered

        return _filter_close(min_pos), _filter_close(max_pos)

    def _detect_hidden_div(self, df: DataFrame, price_col: str, ind_col: str,
                           swing_order: int, min_gap: int) -> pd.Series:
        price = df[price_col]
        indicator = df[ind_col]
        pmin, pmax = self._find_swings(price, swing_order, min_gap)
        imin, imax = self._find_swings(indicator, swing_order, min_gap)

        bullish = pd.Series(0, index=df.index)
        bearish = pd.Series(0, index=df.index)

        pm = [(t, float(price.loc[t])) for t in pmin if t in price.index]
        im = [(t, float(indicator.loc[t])) for t in imin if t in indicator.index]
        for i in range(1, len(pm)):
            t1, p1 = pm[i-1]
            t2, p2 = pm[i]
            if p2 <= p1:
                continue
            m = [(t, v) for t, v in im if t1 <= t <= t2]
            if len(m) < 2:
                continue
            if m[-1][1] < m[0][1]:
                if t2 in bullish.index:
                    bullish.loc[t2] = 1

        px = [(t, float(price.loc[t])) for t in pmax if t in price.index]
        ix = [(t, float(indicator.loc[t])) for t in imax if t in indicator.index]
        for i in range(1, len(px)):
            t1, p1 = px[i-1]
            t2, p2 = px[i]
            if p2 >= p1:
                continue
            m = [(t, v) for t, v in ix if t1 <= t <= t2]
            if len(m) < 2:
                continue
            if m[-1][1] > m[0][1]:
                if t2 in bearish.index:
                    bearish.loc[t2] = 1

        return bullish, bearish

    def _confluence(self, signals: pd.DataFrame, window: int) -> pd.Series:
        bullish = signals.filter(like='bullish_').sum(axis=1)
        bearish = signals.filter(like='bearish_').sum(axis=1)
        conf_bull = bullish.rolling(window=window, min_periods=1).max().shift(0)
        conf_bear = bearish.rolling(window=window, min_periods=1).max().shift(0)
        return conf_bull >= 2, conf_bear >= 2

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        order = self.swing_order.value
        gap = self.min_swing_gap.value
        window = self.confluence_window.value
        rsi_p = self.rsi_period.value
        k = self.stoch_k.value
        d = self.stoch_d.value
        smooth = self.stoch_smooth.value

        # Wskaźniki
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=rsi_p)
        dataframe['macd'] = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)['macd']
        dataframe['macdsignal'] = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)['macdsignal']
        stoch = ta.STOCH(dataframe, fastk_period=k, slowk_period=d, slowd_period=smooth)
        dataframe['stoch_k'] = stoch['slowk']
        dataframe['stoch_d'] = stoch['slowd']

        # Divergence detection per indicator
        signals = pd.DataFrame(index=dataframe.index)
        for col in ['rsi', 'macd', 'stoch_k']:
            bull, bear = self._detect_hidden_div(dataframe, 'close', col, order, gap)
            signals[f'bullish_{col}'] = bull
            signals[f'bearish_{col}'] = bear

        # Confluence filter
        dataframe['bullish_confluence'], dataframe['bearish_confluence'] = self._confluence(signals, window)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['bullish_confluence'] == 1) &
            (dataframe['volume'] > 0),
            'enter_long'
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['bearish_confluence'] == 1) &
            (dataframe['volume'] > 0),
            'exit_long'
        ] = 1

        return dataframe
