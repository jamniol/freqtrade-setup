# Advanced Strategy Research Report - 2026-05-15

## Tested Approaches (ALL requested + proposed)

### 1. More Pairs (6 pairs)
- Downloaded: BTC, ETH, SOL, BNB, ADA, XRP
- Result: Only BTC/ETH trade (SampleStrategy TEMA/BB guard too strict for alts)
- Status: Alts need different entry logic

### 2. Custom Indicators
- Added: EMA9/21 cross, ADX guard, ATR%, BB width, H4 EMA20/50
- Result: EMA cross as gate reduced trades by 70% (+0.41% vs +9.27% base)

### 3. Market Regime Filter
- Added: ADX >= 15, BB_width >= 0.03, ATR% <= 0.06
- Result: Too restrictive as gate; relaxed to ADX|BB_width OR gate
- Status: Included in SampleStrategyAdvanced but not used as hard gate

### 4. Hybrid SampleStrategy + Divergence
- Added: Hidden divergence detection (RSI + MACD) with confluence
- Result: Divergence too rare on H1; calculated but not used in entry
- Status: Optional bonus filter (not active in default)

### 5. Per-Pair RSI Thresholds
- Major coins (BTC/ETH): buy_rsi = 43-46
- Alts (SOL/BNB/ADA/XRP): buy_rsi = 25-35
- Result: Alts still don't trade due to TEMA/BB guard

### 6. Dynamic Position Sizing
- Major coins: 1.2x stake
- Alts: 0.9x stake
- Status: Implemented via custom_stake_amount

### 7. Hyperopt SampleStrategyAdvanced (500 epochs)
- Best: 117 trades, +7.53% profit, 86.3% WR
- vs Base: 112 trades, +9.27%, 87.5% WR
- Parameters: buy_rsi_major=46, sell_rsi=51, trailing=26.9%, stoploss=-9.7%

### 8. Walk-Forward Analysis (3 folds)
| Fold | Period | Profit | Trades | WR | Sharpe |
|------|--------|--------|--------|-----|--------|
| 1 | Nov-Jan | -1.57% | 39 | 84.6% | -0.40 |
| 2 | Dec-Mar | +4.02% | 53 | 81.1% | 0.86 |
| 3 | Feb-May | +10.31% | 72 | 88.9% | 2.09 |
- Trend: Improving through bull run

### 9. Larger Dataset (2022-2026, 3.5 years)
| Strategy | Profit | Trades | WR | Sharpe | Drawdown |
|----------|--------|--------|-----|--------|----------|
| Base SampleStrategy | -37.52% | 949 | 83.4% | -0.56 | 53.19% |
| SampleStrategyAdvanced | -36.56% | 970 | 84.6% | -0.42 | 52.98% |
| Advanced + H4 filter | -6.71% | 45 | 73.3% | -0.07 | 10.33% |

### 10. Alternate Loss Function (SortinoHyperOptLoss)
- Result: 7 trades, +3.45%, very few signals
- Status: Not practical (too restrictive)

## Key Conclusion
**RSI cross + TEMA/BB mean reversion only works in BULL MARKETS.**
Bear market 2022-2023 destroys this approach (-37% drawdown).

## Recommendation
1. Add a BEAR MARKET filter: stop trading when H4 EMA200 is trending down
2. Or switch to TREND FOLLOWING: EMA cross entry + ADX > 25 instead of RSI mean reversion
3. Consider futures mode for shorting in bear markets

## Files
- `strategies/SampleStrategyAdvanced.py` - Hybrid advanced strategy
- `best/SampleStrategyAdvanced_hyperopt_500.json` - Best hyperopt params
- `best/setup_summary.json` - Best base strategy (SampleStrategy +9.27%)
