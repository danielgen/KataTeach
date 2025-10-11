#!/usr/bin/env python3
"""
Test script for Snorkel integration.

This script creates sample data and tests the Snorkel board position processor
to ensure everything works correctly.
"""

import json
import sys
from pathlib import Path
import tempfile
import shutil

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from snorkel_board_positions import SnorkelBoardPositionProcessor, BoardPositionData


def create_sample_policy_data():
    """Create sample policy data for testing."""
    sample_data = {
        "sgf": "(;FF[4]CA[UTF-8]GM[1]SZ[19];B[dd];W[pd];B[dp];W[pp])",
        "policy": {
            "0": {
                "suggestions": [
                    {"move": "C16", "winrate": 0.75, "policy_prob": 0.15},
                    {"move": "C4", "winrate": 0.72, "policy_prob": 0.12},
                    {"move": "D17", "winrate": 0.70, "policy_prob": 0.10},
                    {"move": "Q3", "winrate": 0.68, "policy_prob": 0.08},
                    {"move": "R4", "winrate": 0.65, "policy_prob": 0.06}
                ],
                "actual_move": {
                    "move": "D16",
                    "winrate": 0.60,
                    "policy_prob": 0.05,
                    "player": "b"
                }
            },
            "1": {
                "suggestions": [
                    {"move": "P16", "winrate": 0.55, "policy_prob": 0.20},
                    {"move": "P4", "winrate": 0.52, "policy_prob": 0.18},
                    {"move": "Q17", "winrate": 0.50, "policy_prob": 0.15},
                    {"move": "Q3", "winrate": 0.48, "policy_prob": 0.12},
                    {"move": "R4", "winrate": 0.45, "policy_prob": 0.10}
                ],
                "actual_move": {
                    "move": "P16",
                    "winrate": 0.55,
                    "policy_prob": 0.20,
                    "player": "w"
                }
            },
            "2": {
                "suggestions": [
                    {"move": "D3", "winrate": 0.80, "policy_prob": 0.30},
                    {"move": "D17", "winrate": 0.78, "policy_prob": 0.25},
                    {"move": "C4", "winrate": 0.75, "policy_prob": 0.20},
                    {"move": "Q17", "winrate": 0.72, "policy_prob": 0.15},
                    {"move": "R4", "winrate": 0.70, "policy_prob": 0.10}
                ],
                "actual_move": {
                    "move": "D3",
                    "winrate": 0.80,
                    "policy_prob": 0.30,
                    "player": "b"
                }
            }
        }
    }
    return sample_data


def test_board_position_data():
    """Test the BoardPositionData class."""
    print("Testing BoardPositionData class...")
    
    sample_policy = create_sample_policy_data()
    position_data = sample_policy["policy"]["0"]
    
    board_pos = BoardPositionData(position_data)
    
    # Test feature extraction
    assert len(board_pos.Q) == 5, f"Expected 5 Q values, got {len(board_pos.Q)}"
    assert board_pos.Q[0] == 0.75, f"Expected first Q value 0.75, got {board_pos.Q[0]}"
    
    # Test dictionary conversion
    pos_dict = board_pos.to_dict()
    assert "Q" in pos_dict, "Q values not in dictionary"
    assert "suggestions" in pos_dict, "Suggestions not in dictionary"
    
    print("✓ BoardPositionData class working correctly")


def test_snorkel_processor():
    """Test the SnorkelBoardPositionProcessor."""
    print("Testing SnorkelBoardPositionProcessor...")
    
    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create sample policy file
        sample_data = create_sample_policy_data()
        test_file = temp_path / "test_game.json"
        
        with open(test_file, 'w') as f:
            json.dump(sample_data, f)
        
        # Create output directory
        output_dir = temp_path / "output"
        
        try:
            # Test processor
            processor = SnorkelBoardPositionProcessor(output_dir)
            
            # Test data loading
            positions = processor.load_policy_data(test_file)
            assert len(positions) == 3, f"Expected 3 positions, got {len(positions)}"
            
            # Test dataframe creation
            df = processor.create_dataframe(positions)
            assert len(df) == 3, f"Expected 3 rows in dataframe, got {len(df)}"
            assert "position_id" in df.columns, "position_id column missing"
            assert "Q" in df.columns, "Q column missing"
            
            # Test labeling function application
            L_train = processor.apply_labeling_functions(df)
            assert L_train.shape[0] == 3, f"Expected 3 rows in L_train, got {L_train.shape[0]}"
            assert L_train.shape[1] == len(processor.labeling_functions), \
                f"Expected {len(processor.labeling_functions)} columns in L_train, got {L_train.shape[1]}"
            
            print("✓ SnorkelBoardPositionProcessor working correctly")
            
        except Exception as e:
            print(f"✗ Error in SnorkelBoardPositionProcessor: {e}")
            raise


def test_full_pipeline():
    """Test the complete Snorkel pipeline."""
    print("Testing complete Snorkel pipeline...")
    
    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create sample policy file
        sample_data = create_sample_policy_data()
        test_file = temp_path / "test_game.json"
        
        with open(test_file, 'w') as f:
            json.dump(sample_data, f)
        
        # Create output directory
        output_dir = temp_path / "output"
        
        try:
            # Run complete pipeline
            processor = SnorkelBoardPositionProcessor(output_dir)
            df_result, label_model = processor.process(test_file)
            
            # Check results
            assert len(df_result) == 3, f"Expected 3 positions in results, got {len(df_result)}"
            assert "snorkel_prob_positive" in df_result.columns, "snorkel_prob_positive column missing"
            assert "snorkel_prediction" in df_result.columns, "snorkel_prediction column missing"
            
            # Check output files
            assert (output_dir / "snorkel_results.csv").exists(), "Results CSV not created"
            assert (output_dir / "label_model.pkl").exists(), "Label model not saved"
            assert (output_dir / "lf_statistics.json").exists(), "LF statistics not saved"
            assert (output_dir / "summary_report.txt").exists(), "Summary report not saved"
            
            print("✓ Complete Snorkel pipeline working correctly")
            
        except Exception as e:
            print(f"✗ Error in complete pipeline: {e}")
            raise


def main():
    """Run all tests."""
    print("Running Snorkel integration tests...")
    print("=" * 50)
    
    try:
        test_board_position_data()
        test_snorkel_processor()
        test_full_pipeline()
        
        print("\n" + "=" * 50)
        print("✓ All tests passed! Snorkel integration is working correctly.")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
