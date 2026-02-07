#!/usr/bin/env python3
"""Debug script to trace capital calculation."""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from strategies.ma_alignment.strategy import MAAlignmentStrategy
from config.schema import load_and_validate_strategy_config
from engine.backtest_engine import BacktestEngine

def main():
    # Load config
    config_path = Path(__file__).parent.parent.parent / 'strategies' / 'ma_alignment' / 'config.yml'
    strategy_config = load_and_validate_strategy_config(config_path)
    
    # Create strategy
    strategy = MAAlignmentStrategy(strategy_config)
    
    # Create engine
    engine = BacktestEngine(strategy=strategy, initial_capital=10000.0)
    
    # Load and run backtest on small sample
    data_path = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'EURUSD_M15_2021_2025.csv'
    df_base = engine.load_data(data_path)
    
    # Use first month only for debugging
    df_base = df_base[df_base.index < '2020-02-01']
    
    # Prepare data
    df_by_tf = engine.prepare_data(df_base)
    
    # Generate signals
    signals_df = strategy.generate_signals(df_by_tf)
    
    print(f"Initial capital: ${engine.initial_capital:,.2f}")
    print(f"Signals generated: {len(signals_df)}")
    print(f"\nTracing first 5 trades...\n")
    
    # Get entry timeframe
    entry_tf = strategy.config.timeframes.entry_tf
    df_entry = df_by_tf[entry_tf]
    
    # Process first few trades manually to trace
    signals_sorted = signals_df.sort_index()
    executed_signal_times = set()
    trade_count = 0
    
    for idx in range(len(df_entry)):
        if trade_count >= 5:
            break
            
        current_time = df_entry.index[idx]
        bar = df_entry.iloc[idx]
        
        execution_config = strategy.config.execution
        can_enter = len(engine.open_positions) < execution_config.max_positions
        
        if can_enter:
            unexecuted_signals = signals_sorted[
                (signals_sorted.index <= current_time) & 
                (~signals_sorted.index.isin(executed_signal_times))
            ]
            
            if not unexecuted_signals.empty:
                signal_time = unexecuted_signals.index[-1]
                signal = unexecuted_signals.iloc[-1]
                
                capital_before = engine.current_capital
                position = engine.enter_position(signal, current_time)
                
                if position:
                    capital_after_entry = engine.current_capital
                    print(f"Trade {trade_count + 1}:")
                    print(f"  Time: {current_time}")
                    print(f"  Direction: {signal['direction']}")
                    print(f"  Entry price: ${signal['entry_price']:,.2f}")
                    print(f"  Quantity: {position.quantity:.6f}")
                    print(f"  Capital before entry: ${capital_before:,.2f}")
                    print(f"  Capital after entry: ${capital_after_entry:,.2f}")
                    print(f"  Capital change: ${capital_after_entry - capital_before:,.2f}")
                    
                    executed_signal_times.add(signal_time)
                    trade_count += 1
        
        # Check for exits
        positions_to_close = []
        for pos_idx, position in enumerate(engine.open_positions):
            exit_time, exit_price, exit_reason = engine._check_position_exit(
                position, bar, current_time, df_entry, idx
            )
            
            if exit_time is not None:
                capital_before_exit = engine.current_capital
                trade = engine._close_position(position, exit_time, exit_price, exit_reason, [])
                capital_after_exit = engine.current_capital
                
                print(f"\n  Exit:")
                print(f"    Exit price: ${exit_price:,.2f}")
                print(f"    Exit reason: {exit_reason}")
                print(f"    Capital before exit: ${capital_before_exit:,.2f}")
                print(f"    Capital after exit: ${capital_after_exit:,.2f}")
                print(f"    Capital change: ${capital_after_exit - capital_before_exit:,.2f}")
                print(f"    Trade P&L: ${trade.pnl_after_costs:,.2f}")
                print(f"    Final capital: ${engine.current_capital:,.2f}\n")
                
                positions_to_close.append(pos_idx)
        
        for pos_idx in reversed(positions_to_close):
            engine.open_positions.pop(pos_idx)
    
    print(f"\nFinal capital: ${engine.current_capital:,.2f}")
    print(f"Total trades: {len(engine.trades)}")

if __name__ == '__main__':
    main()

