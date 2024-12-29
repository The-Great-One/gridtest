import pandas as pd
import numpy as np
from datetime import datetime
import ta
import itertools
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

def calculate_indicators(df):
    """Vectorized indicator calculation."""
    # Pre-allocate a dictionary for indicators
    indicators = {}
    
    # Calculate all indicators in one pass
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    volume = df['Volume'].values
    
    # RSI
    indicators['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    
    # MACD
    macd = ta.trend.MACD(close=df['Close'])
    indicators['MACD'] = macd.macd()
    indicators['MACD_Signal'] = macd.macd_signal()
    indicators['MACD_Hist'] = macd.macd_diff()
    
    # EMAs
    indicators['EMA10'] = ta.trend.EMAIndicator(close=df['Close'], window=10).ema_indicator()
    indicators['EMA20'] = ta.trend.EMAIndicator(close=df['Close'], window=20).ema_indicator()
    
    # Volume MA using numpy for speed
    indicators['Volume_MA20'] = pd.Series(volume).rolling(window=20).mean()
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close=df['Close'])
    indicators['BB_Middle'] = bb.bollinger_mavg()
    indicators['BB_Upper'] = bb.bollinger_hband()
    indicators['BB_Lower'] = bb.bollinger_lband()
    
    # Stochastic
    stoch = ta.momentum.StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'])
    indicators['Stochastic_%K'] = stoch.stoch()
    indicators['Stochastic_%D'] = stoch.stoch_signal()
    
    # ADX
    indicators['ADX'] = ta.trend.ADXIndicator(high=df['High'], low=df['Low'], close=df['Close']).adx()
    
    # OBV using numpy cumsum for speed
    price_direction = np.sign(np.diff(close, prepend=close[0]))
    indicators['OBV'] = pd.Series(np.where(price_direction >= 0, volume, -volume).cumsum())
    
    # Convert to DataFrame and handle NaN values
    result = pd.DataFrame(indicators)
    result = result.fillna(method='ffill').fillna(0)
    
    return pd.concat([df, result], axis=1)

def apply_signals_vectorized(df, params):
    """Vectorized signal generation."""
    # Pre-calculate common values
    ema_diff = df['EMA10'] - df['EMA20']
    obv_change = df['OBV'].diff()
    
    # Buy conditions using numpy operations
    buy_conditions = (
        (df['RSI'] < params['rsi_buy_threshold']) &
        (ema_diff > params['ema_diff']) &
        (df['Volume'] > df['Volume_MA20'] * params['vol_mult_buy']) &
        (df['Close'] < df['BB_Lower'] * params['bb_multiplier']) &
        (df['Stochastic_%K'] < params['stochastic_buy_threshold']) &
        (df['ADX'] > params['adx_threshold'])
    )
    
    # Sell conditions using numpy operations
    sell_conditions = (
        (df['RSI'] > params['rsi_sell_threshold']) &
        (ema_diff < -params['ema_diff']) &
        (df['Volume'] > df['Volume_MA20'] * params['vol_mult_sell']) &
        (df['Close'] > df['BB_Upper'] * params['bb_multiplier']) &
        (df['Stochastic_%K'] > params['stochastic_sell_threshold']) &
        (df['ADX'] > params['adx_threshold'])
    )
    
    if params['use_macd_hist_cross']:
        macd_cross_up = (df['MACD_Hist'] > 0) & (df['MACD_Hist'].shift(1) <= 0)
        macd_cross_down = (df['MACD_Hist'] < 0) & (df['MACD_Hist'].shift(1) >= 0)
        buy_conditions &= macd_cross_up
        sell_conditions &= macd_cross_down
    
    if params['obv_trend'] == 'up':
        buy_conditions &= (obv_change > 0)
        sell_conditions &= (obv_change < 0)
    elif params['obv_trend'] == 'down':
        buy_conditions &= (obv_change < 0)
        sell_conditions &= (obv_change > 0)
    
    signals = pd.Series(index=df.index, dtype=str)
    signals[buy_conditions] = 'buy'
    signals[sell_conditions] = 'sell'
    
    return signals.shift(1).fillna('')

def simulate_trading_vectorized(df, signals, params, initial_investment=100000, commission=0.001):
    """Vectorized trading simulation."""
    n = len(df)
    position = np.zeros(n)
    cash = np.full(n, initial_investment)
    stock_qty = np.zeros(n)
    portfolio_value = np.zeros(n)
    entry_prices = np.zeros(n)
    
    for i in range(1, n):
        position[i] = position[i-1]
        cash[i] = cash[i-1]
        stock_qty[i] = stock_qty[i-1]
        
        price = df['Open'].iloc[i]
        if price <= 0:
            continue
            
        if position[i-1] == 1:
            change = (price - entry_prices[i-1]) / entry_prices[i-1]
            if (params['stop_loss'] is not None and change <= -params['stop_loss']) or \
               (params['take_profit'] is not None and change >= params['take_profit']):
                cash[i] += stock_qty[i-1] * price * (1 - commission)
                stock_qty[i] = 0
                position[i] = 0
                continue
        
        if signals.iloc[i-1] == 'buy' and position[i-1] == 0:
            shares = cash[i] // (price * (1 + commission))
            if shares > 0:
                cost = shares * price * (1 + commission)
                cash[i] -= cost
                stock_qty[i] = shares
                position[i] = 1
                entry_prices[i] = price
                
        elif signals.iloc[i-1] == 'sell' and position[i-1] == 1:
            cash[i] += stock_qty[i-1] * price * (1 - commission)
            stock_qty[i] = 0
            position[i] = 0
    
    portfolio_value = cash + (stock_qty * df['Close'].values)
    final_amount = portfolio_value[-1]
    time_period_days = (df['Date'].iloc[-1] - df['Date'].iloc[0]).days
    
    if time_period_days >= 365:
        years = time_period_days / 365.25
        annualized_return = ((final_amount / initial_investment) ** (1/years) - 1) * 100
    else:
        annualized_return = 0
        
    overall_return = ((final_amount - initial_investment) / initial_investment) * 100
    
    return {
        'initial_amount': initial_investment,
        'final_amount': final_amount,
        'time_period_days': time_period_days,
        'annualized_return_rate': annualized_return,
        'overall_returns': overall_return,
        'params': params
    }

def evaluate_params(params, df, initial_investment):
    """Evaluate a single parameter combination."""
    signals = apply_signals_vectorized(df, params)
    
    if signals.str.len().sum() == 0:
        return None
        
    results = simulate_trading_vectorized(df, signals, params, initial_investment)
    
    if results['time_period_days'] < 365:
        return None
        
    return results

def optimize_strategy_parallel(df, param_grid, initial_investment=100000, n_processes=None):
    """Parallelized parameter optimization."""
    if n_processes is None:
        n_processes = mp.cpu_count() - 1  # Leave one core free for system processes
    
    # Calculate indicators once before parallelization
    df = calculate_indicators(df)
    
    # Generate parameter combinations
    param_combinations = [dict(zip(param_grid.keys(), v)) 
                        for v in itertools.product(*param_grid.values())]
    
    # Create partial function with fixed arguments
    evaluate_partial = partial(evaluate_params, df=df, initial_investment=initial_investment)
    
    # Initialize multiprocessing pool
    with mp.Pool(processes=n_processes) as pool:
        # Use tqdm to show progress
        results = list(tqdm(
            pool.imap(evaluate_partial, param_combinations),
            total=len(param_combinations),
            desc=f"Optimizing with {n_processes} processes"
        ))
    
    # Filter out None results and find best parameters
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        return None, None, None
    
    best_result = max(valid_results, key=lambda x: x['annualized_return_rate'])
    best_params = best_result['params']
    best_return = best_result['annualized_return_rate']
    
    return best_params, best_return, best_result

if __name__ == '__main__':
    # Load and prepare data
    df = pd.read_feather('UDAICEMENT.feather')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Parameter grid
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
    
    # Run parallel optimization
    best_params, best_return, best_results = optimize_strategy_parallel(df, param_grid)
    
    # Print results
    if best_params:
        print("\nBest parameters found:")
        for param, value in best_params.items():
            print(f"{param}: {value}")
        print(f"\nBest annualized return: {best_return:.2f}%")
        print(f"\nBest Results: {best_results}")
    else:
        print("No valid parameter combinations found.")
