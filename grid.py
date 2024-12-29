import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta  # Technical analysis library for indicators
import itertools
from tqdm import tqdm  # Progress bar library
import yfinance as yf


def calculate_indicators(df):
    """
    Calculate indicators once. Drop NaNs at the end to avoid
    repeated dropna calls in apply_signals.
    """
    # Calculate RSI
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()

    # Calculate MACD and associated signals
    macd_indicator = ta.trend.MACD(
        close=df['Close'], window_slow=26, window_fast=12, window_sign=9
    )
    df['MACD'] = macd_indicator.macd()
    df['MACD_Signal'] = macd_indicator.macd_signal()
    df['MACD_Hist'] = macd_indicator.macd_diff()

    # Calculate EMAs
    df['EMA10'] = ta.trend.EMAIndicator(close=df['Close'], window=10).ema_indicator()
    df['EMA20'] = ta.trend.EMAIndicator(close=df['Close'], window=20).ema_indicator()

    # Calculate 20-day volume moving average
    df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()

    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_Middle'] = bollinger.bollinger_mavg()
    df['BB_Upper'] = bollinger.bollinger_hband()
    df['BB_Lower'] = bollinger.bollinger_lband()

    # Stochastic Oscillator
    stochastic = ta.momentum.StochasticOscillator(
        high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3
    )
    df['Stochastic_%K'] = stochastic.stoch()
    df['Stochastic_%D'] = stochastic.stoch_signal()

    # Average Directional Index (ADX)
    adx = ta.trend.ADXIndicator(
        high=df['High'], low=df['Low'], close=df['Close'], window=14
    )
    df['ADX'] = adx.adx()

    # On-Balance Volume (OBV)
    obv = ta.volume.OnBalanceVolumeIndicator(
        close=df['Close'], volume=df['Volume']
    )
    df['OBV'] = obv.on_balance_volume()

    # Drop rows with NaNs that result from rolling calculations
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def apply_signals(
    df,
    ema_diff,
    rsi_buy_threshold,
    rsi_sell_threshold,
    use_macd_hist_cross,
    vol_mult_buy,
    vol_mult_sell,
    bb_multiplier,
    stochastic_buy_threshold,
    stochastic_sell_threshold,
    adx_threshold,
    obv_trend,
):
    """
    Vectorized signal assignment based on various conditions.
    We'll store conditions in boolean arrays (NumPy arrays)
    and then combine them via logical operations for 'buy'/'sell'.
    """
    # Calculate EMA difference
    df['EMA_Diff'] = df['EMA10'] - df['EMA20']
    df['OBV_Change'] = df['OBV'].diff()

    # Basic RSI conditions
    buy_cond = (df['RSI'] < rsi_buy_threshold)
    sell_cond = (df['RSI'] > rsi_sell_threshold)

    # EMA difference condition
    buy_cond &= (df['EMA_Diff'] > ema_diff)
    sell_cond &= (df['EMA_Diff'] < -ema_diff)

    # Optionally add MACD Histogram crossovers
    if use_macd_hist_cross:
        # We need shift to detect a zero-cross from negative to positive (buy) or vice versa (sell).
        macd_hist_prev = df['MACD_Hist'].shift(1)
        buy_cond &= (df['MACD_Hist'] > 0) & (macd_hist_prev <= 0)
        sell_cond &= (df['MACD_Hist'] < 0) & (macd_hist_prev >= 0)

    # Volume conditions
    buy_cond &= (df['Volume'] > df['Volume_MA20'] * vol_mult_buy)
    sell_cond &= (df['Volume'] > df['Volume_MA20'] * vol_mult_sell)

    # Bollinger Bands conditions
    buy_cond &= (df['Close'] < df['BB_Lower'] * bb_multiplier)
    sell_cond &= (df['Close'] > df['BB_Upper'] * bb_multiplier)

    # Stochastic Oscillator conditions
    buy_cond &= (df['Stochastic_%K'] < stochastic_buy_threshold)
    sell_cond &= (df['Stochastic_%K'] > stochastic_sell_threshold)

    # ADX condition
    buy_cond &= (df['ADX'] > adx_threshold)
    sell_cond &= (df['ADX'] > adx_threshold)

    # OBV trend condition
    if obv_trend == 'up':
        buy_cond &= (df['OBV_Change'] > 0)
        sell_cond &= (df['OBV_Change'] < 0)
    elif obv_trend == 'down':
        buy_cond &= (df['OBV_Change'] < 0)
        sell_cond &= (df['OBV_Change'] > 0)
    # If obv_trend == 'flat', no additional condition

    # Vectorized signal assignment
    # Start with an array of 'None'
    signals = np.array([None] * len(df), dtype=object)

    signals[buy_cond] = 'buy'
    signals[sell_cond] = 'sell'

    df['Signal'] = signals

    # Shift signals by 1 day to avoid lookahead bias (optional but common practice)
    df['Signal'] = df['Signal'].shift(1)

    # Drop the first row (or any row that became NaN after shift)
    df.dropna(subset=['Signal'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def simulate_trading_v4(df, initial_investment, commission=0.001, stop_loss=None, take_profit=None):
    """
    Row-by-row simulation of trades. Uses integer index iteration for speed.
    """
    position = 0  # 0: No position, 1: Long position
    cash = initial_investment
    stock_quantity = 0
    trade_log = []
    entry_price = 0.0

    # For returns calculation at the end
    start_date = df['Date'].iloc[0]
    end_date = df['Date'].iloc[-1]

    # Use numeric indexing for speed
    for i in range(len(df) - 1):  # up to second-to-last row
        # current row info
        signal = df['Signal'].iloc[i]
        date = df['Date'].iloc[i]

        # next day info for entry/exit price
        next_date = df['Date'].iloc[i + 1]
        # Use next day's open price to avoid lookahead bias
        close_price = df['Open'].iloc[i + 1]

        if close_price <= 0:
            continue

        if position == 1:
            # Update potential sell price for stop-loss and take-profit
            change = (close_price - entry_price) / entry_price

            # Check stop-loss
            if stop_loss is not None and change <= -stop_loss:
                # Sell due to stop-loss
                total_proceeds = stock_quantity * close_price * (1 - commission)
                cash += total_proceeds
                trade_log.append({
                    'action': 'sell',
                    'price': close_price,
                    'stocks_sold': stock_quantity,
                    'cash_left': cash,
                    'total_value': cash,
                    'date': next_date,
                    'reason': 'stop-loss',
                })
                stock_quantity = 0
                position = 0
                continue

            # Check take-profit
            if take_profit is not None and change >= take_profit:
                # Sell due to take-profit
                total_proceeds = stock_quantity * close_price * (1 - commission)
                cash += total_proceeds
                trade_log.append({
                    'action': 'sell',
                    'price': close_price,
                    'stocks_sold': stock_quantity,
                    'cash_left': cash,
                    'total_value': cash,
                    'date': next_date,
                    'reason': 'take-profit',
                })
                stock_quantity = 0
                position = 0
                continue

        # Execute signals
        if signal == 'buy' and position == 0:
            # Basic liquidity check
            if df['Volume'].iloc[i] <= 0:
                continue
            # How many shares we can buy
            num_shares = cash // (close_price * (1 + commission))
            if num_shares > 0:
                cost = num_shares * close_price * (1 + commission)
                cash -= cost
                position = 1
                stock_quantity = num_shares
                entry_price = close_price
                trade_log.append({
                    'action': 'buy',
                    'price': close_price,
                    'stocks_bought': num_shares,
                    'cash_left': cash,
                    'total_value': stock_quantity * close_price + cash,
                    'date': next_date,
                    'reason': 'buy signal',
                })

        elif signal == 'sell' and position == 1:
            total_proceeds = stock_quantity * close_price * (1 - commission)
            cash += total_proceeds
            trade_log.append({
                'action': 'sell',
                'price': close_price,
                'stocks_sold': stock_quantity,
                'cash_left': cash,
                'total_value': cash,
                'date': next_date,
                'reason': 'sell signal',
            })
            stock_quantity = 0
            position = 0

    # If still holding a position at the end, sell at the last closing price
    if position == 1 and df['Close'].iloc[-1] > 0:
        final_close_price = df['Close'].iloc[-1]
        total_proceeds = stock_quantity * final_close_price * (1 - commission)
        cash += total_proceeds
        trade_log.append({
            'action': 'sell',
            'price': final_close_price,
            'stocks_sold': stock_quantity,
            'cash_left': cash,
            'total_value': cash,
            'date': end_date,
            'reason': 'end of data',
        })

    final_amount = cash
    time_period_days = (end_date - start_date).days or 1
    # Calculate annualized return if we have at least 1 year
    if time_period_days < 365:
        annualized_return = 0
    else:
        years = time_period_days / 365.25
        annualized_return = ((final_amount / initial_investment)**(1 / years) - 1) * 100
    overall_return = ((final_amount - initial_investment) / initial_investment) * 100

    trade_df = pd.DataFrame(trade_log)
    results = {
        'initial_amount': initial_investment,
        'final_amount': final_amount,
        'time_period_days': time_period_days,
        'annualized_return_rate': annualized_return,
        'overall_returns': overall_return,
    }

    return trade_df, results


if __name__ == '__main__':
    # Load your data
    # df = pd.read_feather('intermediary_files/Hist_Data/UDAICEMENT.feather')
    df = yf.download("UDAICEMENT.NS", period="max").reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Calculate indicators ONCE
    df = calculate_indicators(df)

    # Initial investment
    initial_investment = 100000

    # Define parameter ranges for grid search
    param_grid = {
        'ema_diff': [0.0, 0.5, 1.0, 1.5, 2.0],
        'rsi_buy_threshold': [40, 45, 55, 60],
        'rsi_sell_threshold': [65, 70, 75, 80],
        'use_macd_hist_cross': [True, False],
        'vol_mult_buy': [1.0, 1.1, 1.2],
        'vol_mult_sell': [0.95, 1.0, 1.1, 1.2, 1.3],
        'bb_multiplier': [1.0, 1.05],
        'stochastic_buy_threshold': [20, 25, 30],
        'stochastic_sell_threshold': [60, 70, 75, 80],
        'adx_threshold': [20, 25],
        'obv_trend': ['up', 'down', 'flat'],
        'stop_loss': [None, 0.05, 0.1],
        'take_profit': [None, 0.1, 0.2],
    }

    # Generate all combinations
    keys = list(param_grid.keys())
    combinations = list(itertools.product(*(param_grid[key] for key in keys)))

    best_params = None
    best_return = -np.inf
    best_results = None

    # Grid Search
    for combo in tqdm(combinations, desc="Optimizing parameters"):
        params = dict(zip(keys, combo))

        # Split out the stop_loss / take_profit separately
        stop_loss = params.pop('stop_loss')
        take_profit = params.pop('take_profit')

        # Apply signals
        df_signals = apply_signals(df.copy(), **params)
        if df_signals['Signal'].isnull().all():
            # No signals generated, skip
            continue

        # Simulate trading
        _, results = simulate_trading_v4(
            df_signals,
            initial_investment,
            commission=0.001,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

        # Skip if data period < 1 year
        if results['time_period_days'] < 365:
            continue

        # Check improvement in annualized return
        if results['annualized_return_rate'] > best_return:
            best_return = results['annualized_return_rate']
            best_params = {**params, 'stop_loss': stop_loss, 'take_profit': take_profit}
            best_results = results

    # Print the best parameters and the corresponding return
    print("\nBest parameters found:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    print(f"\nBest annualized return: {best_return:.2f}%")
    print(f"Best Results: {best_results}")
