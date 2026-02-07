"""Validation state tracking to prevent OOS/holdout reuse and enforce dataset locks."""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from pathlib import Path
import json
from datetime import datetime


def _normalize_lock(lock: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a lock payload for stable comparisons."""
    # Keep this intentionally conservative: only coerce types that frequently differ.
    normalized = json.loads(json.dumps(lock, default=str))
    return normalized


def _lock_hash(lock: Optional[Dict[str, Any]]) -> Optional[str]:
    if not lock:
        return None
    try:
        return str(lock.get('manifest_hash'))
    except Exception:
        return None


@dataclass
class ValidationState:
    """Tracks validation progress for a strategy."""
    strategy_name: str
    data_file: str
    training_start_date: Optional[str] = None
    training_end_date: Optional[str] = None
    oos_start_date: Optional[str] = None
    oos_end_date: Optional[str] = None
    
    # Phase completion status
    phase1_training_completed: bool = False
    phase1_passed: bool = False
    phase1_completed_at: Optional[str] = None
    phase1_failure_count: int = 0  # Track number of Phase 1 failures
    phase1_last_failure_date: Optional[str] = None  # Date of last failure
    
    phase2_oos_completed: bool = False
    phase2_passed: bool = False
    phase2_completed_at: Optional[str] = None

    # Phase 2 consumption semantics: OOS should be consumed on attempt (not on success).
    phase2_oos_consumed: bool = False
    phase2_oos_consumed_at: Optional[str] = None
    
    phase3_stationarity_completed: bool = False
    phase3_passed: bool = False
    phase3_completed_at: Optional[str] = None
    
    # Best parameters from optimization
    optimized_params: Optional[Dict[str, Any]] = None
    
    # Validation results summary
    phase1_results: Optional[Dict[str, Any]] = None
    phase2_results: Optional[Dict[str, Any]] = None
    phase3_results: Optional[Dict[str, Any]] = None
    
    # Holdout test results (final OOS test on reserved period)
    holdout_tests: Optional[Dict[str, Dict[str, Any]]] = None

    # Dataset phase locks: refuse to continue a phase if the dataset slice identity changes.
    phase1_dataset_lock: Optional[Dict[str, Any]] = None
    phase2_dataset_lock: Optional[Dict[str, Any]] = None
    holdout_dataset_locks: Optional[Dict[str, Dict[str, Any]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidationState':
        """Create from dictionary."""
        # Handle migration: if phase1_failure_count doesn't exist but phase1 failed, initialize it
        if 'phase1_failure_count' not in data:
            data['phase1_failure_count'] = 0
            # If Phase 1 was previously completed and failed, we should have a count
            # But we can't know how many times it failed before, so we start at 0
            # The next failure will increment it to 1
        if 'phase1_last_failure_date' not in data:
            data['phase1_last_failure_date'] = None
        if 'holdout_tests' not in data:
            data['holdout_tests'] = None

        # Migration for new fields
        if 'phase2_oos_consumed' not in data:
            # If Phase 2 was already completed, treat it as consumed.
            data['phase2_oos_consumed'] = bool(data.get('phase2_oos_completed', False))
        if 'phase2_oos_consumed_at' not in data:
            data['phase2_oos_consumed_at'] = data.get('phase2_completed_at')

        if 'phase1_dataset_lock' not in data:
            data['phase1_dataset_lock'] = None
        if 'phase2_dataset_lock' not in data:
            data['phase2_dataset_lock'] = None
        if 'holdout_dataset_locks' not in data:
            data['holdout_dataset_locks'] = None
        
        return cls(**data)
    
    def is_qualified_for_live(self) -> bool:
        """Check if strategy has passed all required phases for live trading."""
        return self.phase1_passed and self.phase2_passed
    
    def can_use_oos_data(self) -> bool:
        """Check if OOS data can still be used (not yet used)."""
        return not self.phase2_oos_consumed

    def mark_phase2_consumed(self):
        """Mark Phase 2 OOS as consumed (on attempt)."""
        if not self.phase2_oos_consumed:
            self.phase2_oos_consumed = True
            self.phase2_oos_consumed_at = datetime.now().isoformat()
    
    def mark_phase1_complete(self, passed: bool, results: Optional[Dict[str, Any]] = None):
        """Mark Phase 1 (Training Validation) as complete."""
        self.phase1_training_completed = True
        self.phase1_passed = passed
        self.phase1_completed_at = datetime.now().isoformat()
        if not passed:
            # Increment failure count and record failure date
            self.phase1_failure_count += 1
            self.phase1_last_failure_date = datetime.now().isoformat()
        if results:
            self.phase1_results = results
    
    def can_retry_phase1(self) -> tuple[bool, Optional[str]]:
        """
        Check if Phase 1 can be retried after failure.
        
        Returns:
            (can_retry, reason_if_not)
        """
        if self.phase1_failure_count == 0:
            return True, None
        
        if self.phase1_failure_count >= 3:
            # After 3 failures, check if enough time has passed (30 days)
            if self.phase1_last_failure_date:
                from datetime import datetime, timedelta
                last_failure = datetime.fromisoformat(self.phase1_last_failure_date)
                days_since_failure = (datetime.now() - last_failure).days
                
                if days_since_failure < 30:
                    days_remaining = 30 - days_since_failure
                    return False, f"Strategy has failed Phase 1 validation 3 times. " \
                                  f"Please wait {days_remaining} more days before retrying, " \
                                  f"or consider discarding this strategy entirely."
                else:
                    # 30 days have passed, allow retry but reset count
                    self.phase1_failure_count = 0
                    self.phase1_last_failure_date = None
                    return True, None
            else:
                return False, "Strategy has failed Phase 1 validation 3 times. " \
                              "Consider discarding this strategy entirely."
        
        # 1 or 2 failures - allow retry
        remaining_attempts = 3 - self.phase1_failure_count
        return True, None
    
    def get_phase1_retry_info(self) -> Dict[str, Any]:
        """Get information about Phase 1 retry status."""
        can_retry, reason = self.can_retry_phase1()
        remaining_attempts = max(0, 3 - self.phase1_failure_count)
        
        return {
            'failure_count': self.phase1_failure_count,
            'remaining_attempts': remaining_attempts,
            'can_retry': can_retry,
            'reason_if_blocked': reason,
            'last_failure_date': self.phase1_last_failure_date,
            'previous_failure_reasons': self.phase1_results.get('failure_reasons', []) if self.phase1_results else []
        }
    
    def mark_phase2_complete(self, passed: bool, results: Optional[Dict[str, Any]] = None):
        """Mark Phase 2 (OOS Validation) as complete."""
        # Ensure it's treated as consumed even if older callers only mark completion.
        self.mark_phase2_consumed()
        self.phase2_oos_completed = True
        self.phase2_passed = passed
        self.phase2_completed_at = datetime.now().isoformat()
        if results:
            self.phase2_results = results

    def verify_or_set_phase_lock(self, phase: str, lock: Dict[str, Any]):
        """Verify lock matches the existing lock, or set it if absent.

        Raises:
            ValueError if an existing lock is present and differs.
        """
        normalized = _normalize_lock(lock)

        if phase == 'phase1':
            existing = self.phase1_dataset_lock
            if existing is None:
                self.phase1_dataset_lock = normalized
                return
            if _lock_hash(existing) != _lock_hash(normalized):
                raise ValueError(
                    "Dataset manifest changed during Phase 1. "
                    "Refusing to proceed to prevent invalid validation."
                )
            return

        if phase == 'phase2':
            existing = self.phase2_dataset_lock
            if existing is None:
                self.phase2_dataset_lock = normalized
                return
            if _lock_hash(existing) != _lock_hash(normalized):
                raise ValueError(
                    "Dataset manifest changed during Phase 2. "
                    "Refusing to proceed to prevent invalid OOS validation."
                )
            return

        if phase.startswith('holdout:'):
            holdout_key = phase.split(':', 1)[1]
            if self.holdout_dataset_locks is None:
                self.holdout_dataset_locks = {}
            existing = self.holdout_dataset_locks.get(holdout_key)
            if existing is None:
                self.holdout_dataset_locks[holdout_key] = normalized
                return
            if _lock_hash(existing) != _lock_hash(normalized):
                raise ValueError(
                    "Dataset manifest changed for this holdout period. "
                    "Refusing to proceed to protect one-shot OOS integrity."
                )
            return

        raise ValueError(f"Unknown phase for lock verification: {phase}")
    
    def mark_phase3_complete(self, passed: bool, results: Optional[Dict[str, Any]] = None):
        """Mark Phase 3 (Stationarity) as complete."""
        self.phase3_stationarity_completed = True
        self.phase3_passed = passed
        self.phase3_completed_at = datetime.now().isoformat()
        if results:
            self.phase3_results = results


class ValidationStateManager:
    """Manages validation state persistence."""
    
    def __init__(self, state_dir: Optional[Path] = None):
        """
        Initialize state manager.
        
        Args:
            state_dir: Directory to store state files. Defaults to .validation_state/
        """
        if state_dir is None:
            state_dir = Path.cwd() / ".validation_state"
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(exist_ok=True)
    
    def get_state_file(self, strategy_name: str, data_file: str) -> Path:
        """Get state file path for a strategy/data combination."""
        # Create a safe filename from strategy and data file
        safe_strategy = strategy_name.replace("/", "_").replace("\\", "_")
        safe_data = Path(data_file).stem.replace("/", "_").replace("\\", "_")
        filename = f"{safe_strategy}_{safe_data}.json"
        return self.state_dir / filename
    
    def load_state(
        self,
        strategy_name: str,
        data_file: str
    ) -> Optional[ValidationState]:
        """Load validation state for a strategy."""
        state_file = self.get_state_file(strategy_name, data_file)
        
        if not state_file.exists():
            return None
        
        try:
            with open(state_file, 'r') as f:
                data = json.load(f)
            return ValidationState.from_dict(data)
        except Exception as e:
            print(f"Warning: Could not load validation state: {e}")
            return None
    
    def save_state(self, state: ValidationState, strategy_name: Optional[str] = None, data_file: Optional[str] = None):
        """Save validation state."""
        # Support both old signature (state only) and new signature (with explicit params)
        if strategy_name and data_file:
            state_file = self.get_state_file(strategy_name, data_file)
        else:
            state_file = self.get_state_file(state.strategy_name, state.data_file)
        
        try:
            # Convert to dict and handle non-serializable types
            state_dict = state.to_dict()
            
            # Convert numpy types and other non-serializable types
            def convert_value(v):
                if isinstance(v, (bool, int, float, str, type(None))):
                    return v
                elif isinstance(v, dict):
                    return {k: convert_value(val) for k, val in v.items()}
                elif isinstance(v, (list, tuple)):
                    return [convert_value(item) for item in v]
                else:
                    # Convert to string for other types
                    return str(v)
            
            state_dict = convert_value(state_dict)
            
            with open(state_file, 'w') as f:
                json.dump(state_dict, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Could not save validation state: {e}")
    
    def reset_phase1(self, strategy_name: str, data_file: str, reset_failure_count: bool = True):
        """
        Reset Phase 1 validation state (allows re-running after reoptimization).
        
        WARNING: Only use this if you've made significant changes to the strategy
        (e.g., reoptimized parameters, changed logic). Do not use to data mine.
        
        Args:
            strategy_name: Name of strategy
            data_file: Path to data file
            reset_failure_count: If True, also resets failure count (default: True).
                                Set to False to keep failure count for tracking purposes.
        """
        state = self.load_state(strategy_name, data_file)
        if state:
            state.phase1_training_completed = False
            state.phase1_passed = False
            state.phase1_completed_at = None
            state.phase1_results = None
            state.phase1_dataset_lock = None
            if reset_failure_count:
                state.phase1_failure_count = 0
                state.phase1_last_failure_date = None
                print("Phase 1 state and failure count reset. You can now re-run Phase 1 validation.")
            else:
                print("Phase 1 state reset (failure count preserved). You can now re-run Phase 1 validation.")
            self.save_state(state)
        else:
            print("No validation state found. Nothing to reset.")
    
    def reset_phase2(self, strategy_name: str, data_file: str):
        """
        Reset Phase 2 (OOS Validation) state (allows re-running OOS validation).
        
        WARNING: Only use this for development/testing purposes. In production,
        OOS data should only be used ONCE. Resetting Phase 2 allows you to:
        - Test the validation system end-to-end
        - Debug OOS validation issues
        - Re-run after fixing bugs in the validation code
        
        DO NOT use this to data mine or cherry-pick OOS results.
        
        Args:
            strategy_name: Name of strategy
            data_file: Path to data file
        """
        state = self.load_state(strategy_name, data_file)
        if state:
            state.phase2_oos_completed = False
            state.phase2_passed = False
            state.phase2_completed_at = None
            state.phase2_results = None
            state.phase2_dataset_lock = None
            state.phase2_oos_consumed = False
            state.phase2_oos_consumed_at = None
            print("⚠️  Phase 2 (OOS Validation) state reset.")
            print("   You can now re-run Phase 2 validation.")
            print("   WARNING: This should only be used for development/testing!")
            self.save_state(state)
        else:
            print("No validation state found. Nothing to reset.")
    
    def reset_phase3(self, strategy_name: str, data_file: str):
        """
        Reset Phase 3 (Stationarity) state (allows re-running stationarity analysis).
        
        Args:
            strategy_name: Name of strategy
            data_file: Path to data file
        """
        state = self.load_state(strategy_name, data_file)
        if state:
            state.phase3_stationarity_completed = False
            state.phase3_passed = False
            state.phase3_completed_at = None
            state.phase3_results = None
            print("Phase 3 (Stationarity) state reset. You can now re-run Phase 3 analysis.")
            self.save_state(state)
        else:
            print("No validation state found. Nothing to reset.")
    
    def reset_all(self, strategy_name: str, data_file: str, confirm: bool = False):
        """
        Reset ALL validation phases (use with extreme caution!).
        
        WARNING: This completely resets all validation state. Only use for:
        - Complete strategy redesign
        - Testing the entire validation system
        - Starting fresh after major changes
        
        Args:
            strategy_name: Name of strategy
            data_file: Path to data file
            confirm: Must be True to actually reset (safety check)
        """
        if not confirm:
            print("ERROR: reset_all requires confirm=True for safety.")
            print("This will reset ALL validation phases. Use with extreme caution!")
            return
        
        state = self.load_state(strategy_name, data_file)
        if state:
            # Reset all phases
            self.reset_phase1(strategy_name, data_file, reset_failure_count=True)
            self.reset_phase2(strategy_name, data_file)
            self.reset_phase3(strategy_name, data_file)
            print("\n✅ All validation phases reset.")
            print("   You can now re-run the complete validation workflow.")
        else:
            print("No validation state found. Nothing to reset.")
    
    def create_or_load_state(
        self,
        strategy_name: str,
        data_file: str,
        training_start: Optional[str] = None,
        training_end: Optional[str] = None,
        oos_start: Optional[str] = None,
        oos_end: Optional[str] = None
    ) -> ValidationState:
        """Create new state or load existing one."""
        state = self.load_state(strategy_name, data_file)
        
        if state is None:
            state = ValidationState(
                strategy_name=strategy_name,
                data_file=data_file,
                training_start_date=training_start,
                training_end_date=training_end,
                oos_start_date=oos_start,
                oos_end_date=oos_end
            )
        else:
            # Update date ranges if provided
            if training_start:
                state.training_start_date = training_start
            if training_end:
                state.training_end_date = training_end
            if oos_start:
                state.oos_start_date = oos_start
            if oos_end:
                state.oos_end_date = oos_end
        
        return state

