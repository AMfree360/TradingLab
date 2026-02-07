#!/usr/bin/env python3
"""Debug script to compare Trading Lab backtest with MT5 expectations."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from strategies.ma_alignment.strategy import MAAlignmentStrategy
from config.schema import load_and_validate_strategy_config
from engine.backtest_engine import BacktestEngine
import pandas as pd

def main():
    config_path = project_root / "strategies" / "ma_alignment" / "config.yml"
    config = load_and_validate_strategy_config(config_path)
    strategy = MAAlignmentStrategy(config)
    
    # Load data
    data_path = project_root / "data" / "raw" / "EURUSD_M15_2021_2025.csv"
    engine = BacktestEngine(strategy, initial_capital=10000.0, commission_rate=0.0004)
    
    # Run backtest
    result = engine.run(data_path)
    
    print("\n" + "="*60)
    print("BACKTEST DIAGNOSTICS")
    print("="*60)
    print(f"Total Trades: {result.total_trades}")
    print(f"Winning Trades: {result.winning_trades}")
    print(f"Losing Trades: {result.losing_trades}")
    print(f"Win Rate: {result.win_rate:.2%}")
    print(f"Profit Factor: {result.profit_factor:.2f}")
    print(f"Total P&L: ${result.total_pnl:.2f}")
    print(f"Final Capital: ${result.final_capital:.2f}")
    print(f"Expected Capital: ${result.initial_capital + result.total_pnl:.2f}")
    print(f"Capital Difference: ${result.final_capital - (result.initial_capital + result.total_pnl):.2f}")
    
    # Analyze first 10 trades
    print("\n" + "="*60)
    print("FIRST 10 TRADES ANALYSIS")
    print("="*60)
    for i, trade in enumerate(result.trades[:10], 1):
        print(f"\nTrade {i}:")
        print(f"  Entry: {trade.entry_time} @ ${trade.entry_price:.5f}")
        print(f"  Exit:  {trade.exit_time} @ ${trade.exit_price:.5f}")
        print(f"  Direction: {trade.direction}")
        print(f"  Quantity: {trade.quantity:.2f}")
        print(f"  P&L: ${trade.pnl_after_costs:.2f}")
        print(f"  Exit Reason: {trade.exit_reason}")
        if trade.partial_exits:
            print(f"  Partial Exits: {len(trade.partial_exits)}")
            for pe in trade.partial_exits:
                print(f"    - {pe.exit_time} @ ${pe.exit_price:.5f}, Qty: {pe.quantity:.2f}, P&L: ${pe.pnl_after_costs:.2f}")
    
    # Check for issues
    print("\n" + "="*60)
    print("POTENTIAL ISSUES")
    print("="*60)
    
    if result.profit_factor < 1.0:
        print("⚠️  Profit Factor < 1.0 - strategy is losing money")
    
    if abs(result.final_capital - (result.initial_capital + result.total_pnl)) > 1.0:
        print(f"⚠️  Capital mismatch: {abs(result.final_capital - (result.initial_capital + result.total_pnl)):.2f}")
        print("   This suggests a capital tracking bug")
    
    losing_trades = [t for t in result.trades if t.pnl_after_costs < 0]
    if losing_trades:
        avg_loss = sum(t.pnl_after_costs for t in losing_trades) / len(losing_trades)
        print(f"⚠️  Average loss: ${avg_loss:.2f}")
        print(f"   Check if stop losses are being hit correctly")
    
    winning_trades = [t for t in result.trades if t.pnl_after_costs > 0]
    if winning_trades:
        avg_win = sum(t.pnl_after_costs for t in winning_trades) / len(winning_trades)
        print(f"✓ Average win: ${avg_win:.2f}")
        print(f"  Win/Loss ratio: {abs(avg_win/avg_loss) if losing_trades else 0:.2f}")

if __name__ == "__main__":
    main()

