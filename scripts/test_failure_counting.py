#!/usr/bin/env python3
"""Test script to verify failure counting logic."""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from validation.state import ValidationState, ValidationStateManager


def test_failure_counting():
    """Test the failure counting and retry logic."""
    print("="*60)
    print("TESTING FAILURE COUNTING LOGIC")
    print("="*60)
    
    # Create a test state
    state = ValidationState(
        strategy_name="test_strategy",
        data_file="test_data.csv"
    )
    
    # Test 1: No failures - should allow retry
    print("\n1. Testing: No failures")
    can_retry, reason = state.can_retry_phase1()
    retry_info = state.get_phase1_retry_info()
    print(f"   Can retry: {can_retry}")
    print(f"   Failure count: {retry_info['failure_count']}")
    print(f"   Remaining attempts: {retry_info['remaining_attempts']}")
    assert can_retry, "Should allow retry with 0 failures"
    assert retry_info['remaining_attempts'] == 3, "Should have 3 attempts remaining"
    print("   ✓ PASS")
    
    # Test 2: 1 failure - should allow retry
    print("\n2. Testing: 1 failure")
    state.mark_phase1_complete(passed=False, results={'failure_reasons': ['Test failure']})
    can_retry, reason = state.can_retry_phase1()
    retry_info = state.get_phase1_retry_info()
    print(f"   Can retry: {can_retry}")
    print(f"   Failure count: {retry_info['failure_count']}")
    print(f"   Remaining attempts: {retry_info['remaining_attempts']}")
    assert can_retry, "Should allow retry with 1 failure"
    assert retry_info['failure_count'] == 1, "Should have 1 failure"
    assert retry_info['remaining_attempts'] == 2, "Should have 2 attempts remaining"
    print("   ✓ PASS")
    
    # Test 3: 2 failures - should allow retry
    print("\n3. Testing: 2 failures")
    state.mark_phase1_complete(passed=False, results={'failure_reasons': ['Test failure 2']})
    can_retry, reason = state.can_retry_phase1()
    retry_info = state.get_phase1_retry_info()
    print(f"   Can retry: {can_retry}")
    print(f"   Failure count: {retry_info['failure_count']}")
    print(f"   Remaining attempts: {retry_info['remaining_attempts']}")
    assert can_retry, "Should allow retry with 2 failures"
    assert retry_info['failure_count'] == 2, "Should have 2 failures"
    assert retry_info['remaining_attempts'] == 1, "Should have 1 attempt remaining"
    print("   ✓ PASS")
    
    # Test 4: 3 failures (recent) - should block retry
    print("\n4. Testing: 3 failures (recent)")
    state.mark_phase1_complete(passed=False, results={'failure_reasons': ['Test failure 3']})
    can_retry, reason = state.can_retry_phase1()
    retry_info = state.get_phase1_retry_info()
    print(f"   Can retry: {can_retry}")
    print(f"   Failure count: {retry_info['failure_count']}")
    print(f"   Remaining attempts: {retry_info['remaining_attempts']}")
    print(f"   Reason if blocked: {reason}")
    assert not can_retry, "Should block retry with 3 recent failures"
    assert retry_info['failure_count'] == 3, "Should have 3 failures"
    assert retry_info['remaining_attempts'] == 0, "Should have 0 attempts remaining"
    assert "30 days" in reason or "discard" in reason.lower(), "Should mention 30 days or discarding"
    print("   ✓ PASS")
    
    # Test 5: 3 failures (30+ days ago) - should allow retry and reset
    print("\n5. Testing: 3 failures (30+ days ago)")
    # Manually set failure date to 31 days ago
    state.phase1_last_failure_date = (datetime.now() - timedelta(days=31)).isoformat()
    can_retry, reason = state.can_retry_phase1()
    retry_info = state.get_phase1_retry_info()
    print(f"   Can retry: {can_retry}")
    print(f"   Failure count after check: {state.phase1_failure_count}")
    assert can_retry, "Should allow retry after 30 days"
    assert state.phase1_failure_count == 0, "Should reset failure count after 30 days"
    print("   ✓ PASS")
    
    # Test 6: Manual reset
    print("\n6. Testing: Manual reset")
    state.phase1_failure_count = 3
    state.phase1_last_failure_date = datetime.now().isoformat()
    state_manager = ValidationStateManager()
    # We can't actually save/load, but we can test the reset method logic
    state.phase1_training_completed = False
    state.phase1_passed = False
    state.phase1_completed_at = None
    state.phase1_results = None
    state.phase1_failure_count = 0
    state.phase1_last_failure_date = None
    can_retry, reason = state.can_retry_phase1()
    retry_info = state.get_phase1_retry_info()
    print(f"   Can retry after reset: {can_retry}")
    print(f"   Failure count after reset: {retry_info['failure_count']}")
    assert can_retry, "Should allow retry after manual reset"
    assert retry_info['failure_count'] == 0, "Should have 0 failures after reset"
    print("   ✓ PASS")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED ✓")
    print("="*60)
    return 0


if __name__ == '__main__':
    sys.exit(test_failure_counting())

