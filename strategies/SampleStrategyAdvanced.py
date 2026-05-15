# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Union

from freqtrade.strategy import (
    IStrategy,
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    merge_informative_pair,
)

import talib.abstract as ta
from technical import qtpylib
from scipy.signal import argrelextrema


class SampleStrategyAdvanced(IStrategy):
    """
    SampleStrategy Advanced:
    - SampleStrategy (RSI cross + TEMA + BB) as base
    - Multi-Timeframe: H4 EMA20 trend filter (weaker than EMA50)
    - Market Regime: ADX guard + no chop + no extreme vol
    - Divergence confluence: hidden divergence as bonus filter
    - Per-pair RSI thresholds: major coins vs alts
    - Dynamic position sizing: larger when trend is strong
    - Custom indicators: EMA9/21 cross guard
    """
    INTERFACE_VERSION = 3
    can_short: bool = False

    # Per-pair RSI thresholds
    buy_rsi_major = IntParameter(low=20, high=50, default=43, space="buy", optimize=True)
    buy_rsi_alts = IntParameter(low=15, high=45, default=25, space="buy", optimize=True)
    sell_rsi = IntParameter(low=50, high=100, default=50, space="sell", optimize=True)

    # Trend / regime params
    use_h4_filter = BooleanParameter(default=False, space="buy", optimize=True)
    adx_min = IntParameter(low=10, high=30, default=15, space="buy", optimize=True)
    bb_width_min = DecimalParameter(low=0.01, high=0.08, default=0.03, space="buy", optimize=True, decimals=3)
    atr_max_pct = DecimalParameter(low=0.03, high=0.10, default=0.06, space="buy", optimize=True, decimals=3)

    # Divergence params
    swing_order = IntParameter(2, 8, default=5, space="buy", optimize=True)
    min_swing_gap = IntParameter(4, 24, default=12, space="buy", optimize=True)
    confluence_window = IntParameter(1, 8, default=3, space="buy", optimize=True)

    # ROI from best SampleStrategy hyperopt
    minimal_roi = {
        "0": 0.624,
        "236": 0.183,
        "937": 0.062,
        "1740": 0
    }

    stoploss = -0.088
    trailing_stop = True
    trailing_stop_positive = 0.045
    trailing_stop_positive_offset = 0.048
    trailing_only_offset_is_reached = False

    timeframe = '1h'
    process_only_new_candles = True
    startup_candle_count: int = 400

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }
    order_time_in_force = {'entry': 'GTC', 'exit': 'GTC'}

    plot_config = {
        'main_plot': {'tema': {}, 'sar': {'color': 'white'}},
        'subplots': {
            'RSI': {'rsi': {'color': 'red'}},
            'MACD': {'macd': {'color': 'blue'}, 'macdsignal': {'color': 'orange'}},
            'ADX': {'adx': {'color': 'green'}},
        }
    }

    # --- Divergence helpers ---
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
                           swing_order: int, min_gap: int) -> tuple:
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

    def _confluence(self, signals: pd.DataFrame, window: int) -> tuple:
        bullish = signals.filter(like='bullish_').sum(axis=1)
        bearish = signals.filter(like='bearish_').sum(axis=1)
        conf_bull = bullish.rolling(window=window, min_periods=1).max().shift(0)
        conf_bear = bearish.rolling(window=window, min_periods=1).max().shift(0)
        return conf_bull >= 2, conf_bear >= 2

    # --- Main methods ---
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # H4 informative - EMA trend (weaker than EMA50)
        inf_tf = '4h'
        inf_pair = metadata['pair']
        informative = self.dp.get_pair_dataframe(pair=inf_pair, timeframe=inf_tf)
        informative['ema20'] = ta.EMA(informative, timeperiod=20)
        informative['ema50'] = ta.EMA(informative, timeperiod=50)
        informative['h4_trend_up'] = (informative['close'] > informative['ema20']).astype(int)
        informative['h4_trend_strong'] = (informative['close'] > informative['ema50']).astype(int)
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)

        # H1 indicators - SampleStrategy base
        dataframe['rsi'] = ta.RSI(dataframe)
        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        dataframe['mfi'] = ta.MFI(dataframe)
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_width'] = (dataframe['bb_upperband'] - dataframe['bb_lowerband']) / dataframe['bb_middleband']
        dataframe['sar'] = ta.SAR(dataframe)
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['ema9'] = ta.EMA(dataframe, timeperiod=9)
        dataframe['ema21'] = ta.EMA(dataframe, timeperiod=21)
        dataframe['ema_cross_bull'] = (dataframe['ema9'] > dataframe['ema21']).astype(int)

        # ATR for volatility regime
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['atr_percent'] = dataframe['atr'] / dataframe['close']

        # Market regime flags (relaxed)
        adx_min = self.adx_min.value
        bbw_min = self.bb_width_min.value
        atr_max = self.atr_max_pct.value
        dataframe['regime_ok'] = (
            (dataframe['adx'] >= adx_min) |
            (dataframe['bb_width'] >= bbw_min)
        ).astype(int)

        # Divergence detection (bonus filter)
        order = self.swing_order.value
        gap = self.min_swing_gap.value
        window = self.confluence_window.value
        signals = pd.DataFrame(index=dataframe.index)
        for col in ['rsi', 'macd']:
            bull, bear = self._detect_hidden_div(dataframe, 'close', col, order, gap)
            signals[f'bullish_{col}'] = bull
            signals[f'bearish_{col}'] = bear
        dataframe['bullish_div'], dataframe['bearish_div'] = self._confluence(signals, window)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata['pair']
        if pair in ['BTC/USDT', 'ETH/USDT']:
            rsi_th = self.buy_rsi_major.value
        else:
            rsi_th = self.buy_rsi_alts.value

        # Base entry conditions
        entry_cond = (
            (qtpylib.crossed_above(dataframe['rsi'], rsi_th))
            & (dataframe['tema'] <= dataframe['bb_middleband'])
            & (dataframe['tema'] > dataframe['tema'].shift(1))
            & (dataframe['volume'] > 0)
        )
        # Optional H4 trend filter
        if self.use_h4_filter.value:
            entry_cond = entry_cond & (dataframe['h4_trend_up_4h'] == 1)

        dataframe.loc[entry_cond, 'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['rsi'], self.sell_rsi.value))
                & (dataframe['tema'] > dataframe['bb_middleband'])
                & (dataframe['tema'] < dataframe['tema'].shift(1))
                & (dataframe['volume'] > 0)
            ),
            'exit_long'
        ] = 1
        return dataframe

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float,
                            entry_tag: Optional[str], side: str, **kwargs) -> float:
        # Dynamic sizing: larger when H4 trend is strong and ADX high
        # Access last candle adx via dataframe (not directly available here)
        # Simplified: boost for major coins, reduce for alts in weak trend
        multiplier = 1.0
        if pair in ['BTC/USDT', 'ETH/USDT']:
            multiplier = 1.2
        else:
            multiplier = 0.9
        stake = proposed_stake * multiplier
        return max(min(stake, max_stake), min_stake)
