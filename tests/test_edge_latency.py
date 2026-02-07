"""Unit tests for edge latency metric."""

import unittest
import numpy as np
from metrics.edge_latency import compute_edge_latency


class TestEdgeLatency(unittest.TestCase):
    """Test edge latency calculations."""
    
    def test_strong_edge(self):
        """Test with strong edge (high signal ratio)."""
        # Strong edge: mean=0.5, std=1.0 → signal_ratio=0.5
        r_multiples = np.array([0.5, 1.0, -0.5, 1.5, 0.0, 0.5, 1.0, -0.5, 0.5, 1.0])
        result = compute_edge_latency(r_multiples)
        
        self.assertIsNotNone(result['mean_r'])
        self.assertIsNotNone(result['std_r'])
        self.assertIsNotNone(result['signal_ratio'])
        self.assertIsNotNone(result['required_trades'])
        self.assertGreater(result['mean_r'], 0)
        self.assertGreater(result['signal_ratio'], 0)
        self.assertLess(result['required_trades'], 100)  # Strong edge should require < 100 trades
    
    def test_weak_edge(self):
        """Test with weak edge (low signal ratio)."""
        # Weak edge: mean=0.1, std=1.0 → signal_ratio=0.1
        # Construct deterministically: values 1.1 and -0.9 have mean 0.1 and std 1.0.
        r_multiples = np.array([1.1, -0.9] * 50)
        result = compute_edge_latency(r_multiples)
        
        self.assertIsNotNone(result['required_trades'])
        self.assertGreater(result['required_trades'], 300)  # Weak edge requires many trades
    
    def test_no_edge(self):
        """Test with no positive edge (mean <= 0)."""
        # No edge: mean <= 0
        r_multiples = np.array([-0.5, -1.0, -0.5, -1.5, 0.0, -0.5, -1.0, -0.5])
        result = compute_edge_latency(r_multiples)
        
        self.assertLessEqual(result['mean_r'], 0)
        self.assertIsNone(result['signal_ratio'])
        self.assertIsNone(result['required_trades'])
        self.assertIsNone(result['years_to_significance'])
    
    def test_zero_variance(self):
        """Test with zero variance (all trades same R-multiple)."""
        # All trades have same R-multiple
        r_multiples = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        result = compute_edge_latency(r_multiples)
        
        self.assertEqual(result['std_r'], 0.0)
        self.assertIsNone(result['signal_ratio'])
        self.assertIsNone(result['required_trades'])
    
    def test_all_wins(self):
        """Test with all winning trades."""
        r_multiples = np.array([1.0, 2.0, 1.5, 0.5, 1.0])
        result = compute_edge_latency(r_multiples)
        
        self.assertGreater(result['mean_r'], 0)
        self.assertIsNotNone(result['required_trades'])
    
    def test_all_losses(self):
        """Test with all losing trades."""
        r_multiples = np.array([-1.0, -2.0, -1.5, -0.5, -1.0])
        result = compute_edge_latency(r_multiples)
        
        self.assertLessEqual(result['mean_r'], 0)
        self.assertIsNone(result['required_trades'])
    
    def test_empty_array(self):
        """Test with empty array."""
        r_multiples = np.array([])
        result = compute_edge_latency(r_multiples)
        
        self.assertIsNone(result['mean_r'])
        self.assertIsNone(result['std_r'])
        self.assertIsNone(result['signal_ratio'])
        self.assertIsNone(result['required_trades'])
    
    def test_single_trade(self):
        """Test with single trade."""
        r_multiples = np.array([1.0])
        result = compute_edge_latency(r_multiples)
        
        self.assertEqual(result['mean_r'], 1.0)
        self.assertEqual(result['std_r'], 0.0)  # Single value has zero std
        self.assertIsNone(result['signal_ratio'])  # Can't compute with zero variance
    
    def test_time_conversion(self):
        """Test time conversion with trades_per_year."""
        r_multiples = np.array([0.5, 1.0, -0.5, 1.5, 0.0, 0.5, 1.0, -0.5])
        trades_per_year = 100.0
        result = compute_edge_latency(r_multiples, trades_per_year=trades_per_year)
        
        self.assertIsNotNone(result['years_to_significance'])
        self.assertGreater(result['years_to_significance'], 0)
        # Verify calculation: years = required_trades / trades_per_year
        expected_years = result['required_trades'] / trades_per_year
        self.assertAlmostEqual(result['years_to_significance'], expected_years, places=2)
    
    def test_custom_confidence_power(self):
        """Test with custom confidence and power levels."""
        r_multiples = np.array([0.5, 1.0, -0.5, 1.5, 0.0, 0.5, 1.0, -0.5])
        result = compute_edge_latency(r_multiples, confidence=0.99, power=0.90)
        
        self.assertEqual(result['confidence'], 0.99)
        self.assertEqual(result['power'], 0.90)
        # Higher confidence/power should require more trades
        result_default = compute_edge_latency(r_multiples, confidence=0.95, power=0.80)
        self.assertGreater(result['required_trades'], result_default['required_trades'])
    
    # Note: edge latency warnings were intentionally removed in the implementation
    # (values are surfaced in reports instead). Tests should validate numeric outputs.
    
    def test_realistic_strategy(self):
        """Test with realistic strategy R-multiples."""
        # Realistic strategy: 60% win rate, avg win=1.5R, avg loss=-1.0R
        # Expected R = 0.6 * 1.5 + 0.4 * (-1.0) = 0.5
        wins = np.random.normal(1.5, 0.5, 60)
        losses = np.random.normal(-1.0, 0.5, 40)
        r_multiples = np.concatenate([wins, losses])
        np.random.shuffle(r_multiples)
        
        result = compute_edge_latency(r_multiples, trades_per_year=200.0)
        
        self.assertIsNotNone(result['required_trades'])
        self.assertIsNotNone(result['years_to_significance'])
        self.assertGreater(result['mean_r'], 0)
        # Realistic strategy should have reasonable latency (50-200 trades)
        self.assertGreater(result['required_trades'], 20)
        self.assertLess(result['required_trades'], 500)


if __name__ == '__main__':
    unittest.main()

