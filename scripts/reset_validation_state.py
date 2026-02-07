#!/usr/bin/env python3
"""Manually reset validation state for development and testing purposes."""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from validation.state import ValidationStateManager


def main():
    parser = argparse.ArgumentParser(
        description='Reset validation state for development and testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
WARNING: Resetting validation state should only be used for:
- Development and testing of the validation system
- Debugging validation issues
- Testing after fixing bugs in validation code
- Complete strategy redesign

DO NOT use this to data mine or cherry-pick results!

Examples:
  # Reset Phase 1 (Training Validation)
    python3 scripts/reset_validation_state.py \\
        --strategy ema_crossover \\
    --data data/raw/EURUSD_M15_2021_2025.csv \\
    --phase 1

  # Reset Phase 2 (OOS Validation) - for testing
    python3 scripts/reset_validation_state.py \\
        --strategy ema_crossover \\
    --data data/raw/EURUSD_M15_2021_2025.csv \\
    --phase 2

  # Reset Phase 3 (Stationarity)
    python3 scripts/reset_validation_state.py \\
        --strategy ema_crossover \\
    --data data/raw/EURUSD_M15_2021_2025.csv \\
    --phase 3

  # Reset ALL phases (use with extreme caution!)
    python3 scripts/reset_validation_state.py \\
        --strategy ema_crossover \\
    --data data/raw/EURUSD_M15_2021_2025.csv \\
    --phase all \\
    --confirm

  # Reset Phase 1 but keep failure count (for tracking)
    python3 scripts/reset_validation_state.py \\
        --strategy ema_crossover \\
    --data data/raw/EURUSD_M15_2021_2025.csv \\
    --phase 1 \\
    --keep-failure-count
        """
    )
    
    parser.add_argument(
        '--strategy',
        type=str,
        required=True,
        help='Strategy name (must match folder in strategies/)'
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to data file (CSV or Parquet)'
    )
    parser.add_argument(
        '--phase',
        type=str,
        choices=['1', '2', '3', 'all'],
        required=True,
        help='Which phase to reset: 1 (Training), 2 (OOS), 3 (Stationarity), or all'
    )
    parser.add_argument(
        '--keep-failure-count',
        action='store_true',
        help='Keep failure count when resetting Phase 1 (default: reset failure count)'
    )
    parser.add_argument(
        '--confirm',
        action='store_true',
        help='Skip confirmation prompt (required for --phase all)'
    )
    
    args = parser.parse_args()
    
    # Confirm action
    print("="*60)
    if args.phase == '1':
        print("RESET PHASE 1: TRAINING VALIDATION")
        phase_desc = "Phase 1 (Training Validation)"
    elif args.phase == '2':
        print("RESET PHASE 2: OOS VALIDATION")
        phase_desc = "Phase 2 (OOS Validation)"
        print("⚠️  WARNING: Resetting OOS validation allows re-using OOS data!")
        print("   This should ONLY be used for development/testing purposes.")
    elif args.phase == '3':
        print("RESET PHASE 3: STATIONARITY")
        phase_desc = "Phase 3 (Stationarity)"
    else:  # all
        print("RESET ALL PHASES")
        phase_desc = "ALL validation phases"
        print("⚠️  WARNING: This will reset ALL validation phases!")
        print("   Use with extreme caution!")
        if not args.confirm:
            print("\nERROR: --phase all requires --confirm flag for safety.")
            return 1
    
    print("="*60)
    print(f"Strategy: {args.strategy}")
    print(f"Data: {args.data}")
    print(f"Phase: {phase_desc}")
    if args.phase == '1' and args.keep_failure_count:
        print("⚠️  Will reset Phase 1 state but KEEP failure count")
    print()
    
    if not args.confirm:
        response = input("Are you sure you want to reset? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Reset cancelled.")
            return 0
    
    # Reset state
    state_manager = ValidationStateManager()
    
    if args.phase == '1':
        state_manager.reset_phase1(
            strategy_name=args.strategy,
            data_file=args.data,
            reset_failure_count=not args.keep_failure_count
        )
        print("\n✓ Phase 1 reset complete. You can now re-run Phase 1 validation.")
    elif args.phase == '2':
        state_manager.reset_phase2(
            strategy_name=args.strategy,
            data_file=args.data
        )
        print("\n✓ Phase 2 reset complete. You can now re-run Phase 2 validation.")
    elif args.phase == '3':
        state_manager.reset_phase3(
            strategy_name=args.strategy,
            data_file=args.data
        )
        print("\n✓ Phase 3 reset complete. You can now re-run Phase 3 analysis.")
    else:  # all
        state_manager.reset_all(
            strategy_name=args.strategy,
            data_file=args.data,
            confirm=True
        )
        print("\n✓ All phases reset complete. You can now re-run the complete validation workflow.")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

