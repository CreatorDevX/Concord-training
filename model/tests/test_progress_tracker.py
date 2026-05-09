import pytest
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from model.train import ProgressTracker

def test_progress_tracker():
    tracker = ProgressTracker(total_target=70_000_000_000)
    
    assert tracker.tokens_seen == 0
    assert tracker.get_percentage() == 0.0
    
    tracker.add_tokens(35_000_000_000)
    assert abs(tracker.get_percentage() - 50.0) < 1e-4
    
    tracker.add_tokens(35_000_000_000)
    assert abs(tracker.get_percentage() - 100.0) < 1e-4

def test_progress_tracker_state_dict():
    tracker = ProgressTracker(total_target=1000)
    tracker.add_tokens(250)
    
    state = tracker.state_dict()
    assert state['tokens_seen'] == 250
    assert state['percentage'] == 25.0
    
    tracker2 = ProgressTracker()
    tracker2.load_state_dict(state)
    assert tracker2.tokens_seen == 250
    assert tracker2.total_target == 1000
