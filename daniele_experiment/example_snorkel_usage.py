#!/usr/bin/env python3
"""
Example usage of the Snorkel board position processor.

This script demonstrates how to use the snorkel_board_positions.py module
to apply weak supervision to Go board positions.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from snorkel_board_positions import SnorkelBoardPositionProcessor


def main():
    """Example usage of the Snorkel processor."""
    
    # Example 1: Process a single policy file
    print("Example 1: Processing single policy file")
    print("-" * 40)
    
    # Check if we have any policy files to work with
    games_dir = Path("games/policy")
    if games_dir.exists():
        policy_files = list(games_dir.glob("*.json"))
        if policy_files:
            # Use the first available policy file
            input_file = policy_files[0]
            output_dir = Path("snorkel_output_single")
            
            print(f"Processing: {input_file}")
            print(f"Output directory: {output_dir}")
            
            try:
                processor = SnorkelBoardPositionProcessor(output_dir)
                df_result, label_model = processor.process(input_file)
                
                print(f"Successfully processed {len(df_result)} positions")
                print(f"Results saved to: {output_dir}")
                
                # Show some sample results
                print("\nSample results:")
                print(df_result[['position_id', 'snorkel_prob_positive', 'snorkel_prediction']].head())
                
            except Exception as e:
                print(f"Error processing single file: {e}")
        else:
            print("No policy files found in games/policy/")
    else:
        print("games/policy/ directory not found")
    
    print("\n" + "=" * 60 + "\n")
    
    # Example 2: Process all policy files in a directory
    print("Example 2: Processing all policy files in directory")
    print("-" * 40)
    
    if games_dir.exists():
        output_dir = Path("snorkel_output_batch")
        
        print(f"Processing directory: {games_dir}")
        print(f"Output directory: {output_dir}")
        
        try:
            processor = SnorkelBoardPositionProcessor(output_dir)
            df_result, label_model = processor.process(games_dir)
            
            print(f"Successfully processed {len(df_result)} positions from all games")
            print(f"Results saved to: {output_dir}")
            
            # Show summary statistics
            print("\nSummary statistics:")
            print(f"Total positions: {len(df_result)}")
            print(f"Average positive probability: {df_result['snorkel_prob_positive'].mean():.3f}")
            print(f"Positive predictions: {(df_result['snorkel_prediction'] == 1).sum()}")
            print(f"Negative predictions: {(df_result['snorkel_prediction'] == 0).sum()}")
            
        except Exception as e:
            print(f"Error processing directory: {e}")
    else:
        print("games/policy/ directory not found")
    
    print("\n" + "=" * 60 + "\n")
    
    # Example 3: Show how to analyze results
    print("Example 3: Analyzing Snorkel results")
    print("-" * 40)
    
    # Look for existing results
    result_dirs = [Path("snorkel_output_single"), Path("snorkel_output_batch")]
    
    for result_dir in result_dirs:
        if result_dir.exists():
            results_file = result_dir / "snorkel_results.csv"
            if results_file.exists():
                print(f"Found results in: {result_dir}")
                
                # Load and analyze results
                import pandas as pd
                df = pd.read_csv(results_file)
                
                print(f"Loaded {len(df)} positions")
                print("\nLabeling function coverage:")
                
                # Show coverage for each labeling function
                lf_columns = [col for col in df.columns if col.startswith('lf_')]
                for col in lf_columns:
                    lf_name = col.replace('lf_', '')
                    coverage = (df[col] != -1).mean()
                    positive_rate = (df[col] == 1).mean()
                    print(f"  {lf_name}: {coverage:.1%} coverage, {positive_rate:.1%} positive")
                
                print(f"\nSnorkel predictions:")
                print(f"  Average positive probability: {df['snorkel_prob_positive'].mean():.3f}")
                print(f"  Prediction distribution: {df['snorkel_prediction'].value_counts().to_dict()}")
                
                break
    else:
        print("No existing results found. Run the processor first!")


if __name__ == "__main__":
    main()
