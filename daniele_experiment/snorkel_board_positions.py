#!/usr/bin/env python3
"""
Snorkel-based weak supervision for Go board positions.

This script applies Snorkel's weak supervision framework to Go board positions
using the existing labeling functions and data structures in the daniele_experiment
package. It provides a complete pipeline for:

1. Loading board position data from policy files
2. Applying weak labeling functions using Snorkel
3. Training a label model to combine weak labels
4. Generating probabilistic labels for downstream tasks

Usage:
    python snorkel_board_positions.py --input-dir games/policy/ --output-dir snorkel_output/
    python snorkel_board_positions.py --input-file games/policy/game.json --output-dir snorkel_output/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from snorkel.labeling import LabelingFunction, labeling_function
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LabelModel
from snorkel.labeling import filter_unlabeled_dataframe

# Add the current directory to path for imports
sys.path.append(str(Path(__file__).parent))


class BoardPositionData:
    """Container for board position data with KataGo analysis."""
    
    def __init__(self, position_data: Dict[str, Any]):
        self.position_data = position_data
        self.suggestions = position_data.get("suggestions", [])
        self.actual_move = position_data.get("actual_move", {})
        
        # Extract features for labeling functions
        self._extract_features()
    
    def _extract_features(self):
        """Extract features needed by labeling functions."""
        # Extract Q-values (winrates) from suggestions
        self.Q = [s.get("winrate", 0.0) for s in self.suggestions]
        
        # Extract policy probabilities
        self.policy_probs = [s.get("policy_prob", 0.0) for s in self.suggestions]
        
        # Extract visits (approximate from policy probabilities)
        # In a real implementation, you'd want actual visit counts
        self.visits = [max(1, int(p * 1000)) for p in self.policy_probs]
        
        # For now, we'll use placeholder values for features not available
        # In a full implementation, you'd extract these from KataGo analysis
        self.O_T = np.array([0.0] * 361)  # Ownership tensor (placeholder)
        self.ladder_works = 0.0  # Ladder flag (placeholder)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format expected by labeling functions."""
        return {
            "Q": self.Q,
            "visits": self.visits,
            "O_T": self.O_T,
            "ladder_works": self.ladder_works,
            "suggestions": self.suggestions,
            "actual_move": self.actual_move
        }


# Snorkel Labeling Functions
@labeling_function()
def lf_tenuki_ok(x):
    """Labeling function for tenuki_ok concept."""
    try:
        return 1 if tenuki_ok(x) > 0.5 else 0
    except:
        return -1  # Abstain

@labeling_function()
def lf_invasion_viable(x):
    """Labeling function for invasion_viable concept."""
    try:
        return 1 if invasion_viable(x) > 0.5 else 0
    except:
        return -1  # Abstain

@labeling_function()
def lf_cut_available(x):
    """Labeling function for cut_available concept."""
    try:
        return 1 if cut_available(x) > 0.5 else 0
    except:
        return -1  # Abstain

@labeling_function()
def lf_ladder_works(x):
    """Labeling function for ladder_works concept."""
    try:
        return 1 if ladder_works(x) > 0.5 else 0
    except:
        return -1  # Abstain

@labeling_function()
def lf_sente_line(x):
    """Labeling function for sente_line concept."""
    try:
        return 1 if sente_line(x) > 0.5 else 0
    except:
        return -1  # Abstain

# Additional heuristic labeling functions
@labeling_function()
def lf_high_winrate_move(x):
    """Label if the best move has very high winrate (>0.7)."""
    try:
        q_values = x.get("Q", [])
        if not q_values:
            return -1
        max_q = max(q_values)
        return 1 if max_q > 0.7 else 0
    except:
        return -1

@labeling_function()
def lf_close_competition(x):
    """Label if multiple moves have similar winrates (close competition)."""
    try:
        q_values = x.get("Q", [])
        if len(q_values) < 2:
            return -1
        sorted_q = sorted(q_values, reverse=True)
        diff = sorted_q[0] - sorted_q[1]
        return 1 if diff < 0.05 else 0  # Within 5% winrate
    except:
        return -1

@labeling_function()
def lf_policy_concentration(x):
    """Label if policy is highly concentrated on few moves."""
    try:
        policy_probs = x.get("policy_probs", [])
        if not policy_probs:
            return -1
        # Check if top 3 moves have >80% of policy mass
        sorted_probs = sorted(policy_probs, reverse=True)
        top3_mass = sum(sorted_probs[:3])
        return 1 if top3_mass > 0.8 else 0
    except:
        return -1

@labeling_function()
def lf_many_candidates(x):
    """Label if there are many candidate moves (>5 with >1% policy)."""
    try:
        policy_probs = x.get("policy_probs", [])
        if not policy_probs:
            return -1
        candidates = sum(1 for p in policy_probs if p > 0.01)
        return 1 if candidates > 5 else 0
    except:
        return -1


class SnorkelBoardPositionProcessor:
    """Main processor for applying Snorkel to board positions."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define labeling functions
        self.labeling_functions = [
            lf_tenuki_ok,
            lf_invasion_viable,
            lf_cut_available,
            lf_ladder_works,
            lf_sente_line,
            lf_high_winrate_move,
            lf_close_competition,
            lf_policy_concentration,
            lf_many_candidates,
        ]
        
        self.lf_names = [lf.name for lf in self.labeling_functions]
    
    def load_policy_data(self, input_path: Path) -> List[Dict[str, Any]]:
        """Load board position data from policy files."""
        positions = []
        
        if input_path.is_file():
            # Single file
            with open(input_path, 'r') as f:
                data = json.load(f)
                policy_data = data.get("policy", {})
                
                for pos_idx, pos_data in policy_data.items():
                    position = BoardPositionData(pos_data)
                    positions.append({
                        "position_id": f"{input_path.stem}_{pos_idx}",
                        "game_id": input_path.stem,
                        "position_idx": int(pos_idx),
                        **position.to_dict()
                    })
        
        elif input_path.is_dir():
            # Directory of policy files
            for policy_file in input_path.glob("*.json"):
                with open(policy_file, 'r') as f:
                    data = json.load(f)
                    policy_data = data.get("policy", {})
                    
                    for pos_idx, pos_data in policy_data.items():
                        position = BoardPositionData(pos_data)
                        positions.append({
                            "position_id": f"{policy_file.stem}_{pos_idx}",
                            "game_id": policy_file.stem,
                            "position_idx": int(pos_idx),
                            **position.to_dict()
                        })
        
        return positions
    
    def create_dataframe(self, positions: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert positions to pandas DataFrame for Snorkel."""
        df = pd.DataFrame(positions)
        
        # Add some derived features
        df['num_suggestions'] = df['suggestions'].apply(len)
        df['best_winrate'] = df['Q'].apply(lambda q: max(q) if q else 0.0)
        df['winrate_spread'] = df['Q'].apply(
            lambda q: max(q) - min(q) if len(q) > 1 else 0.0
        )
        
        return df
    
    def apply_labeling_functions(self, df: pd.DataFrame) -> np.ndarray:
        """Apply all labeling functions to the dataframe."""
        applier = PandasLFApplier(lfs=self.labeling_functions)
        L_train = applier.apply(df=df)
        return L_train
    
    def train_label_model(self, L_train: np.ndarray) -> LabelModel:
        """Train Snorkel's label model to combine weak labels."""
        label_model = LabelModel(cardinality=2, verbose=True)
        
        # Filter out positions where all LFs abstained
        L_train_filtered = L_train[~np.all(L_train == -1, axis=1)]
        
        if len(L_train_filtered) == 0:
            raise ValueError("No positions with non-abstaining labels found!")
        
        print(f"Training label model on {len(L_train_filtered)} positions...")
        label_model.fit(L_train=L_train_filtered, n_epochs=500, log_freq=100, seed=123)
        
        return label_model
    
    def generate_probabilistic_labels(self, df: pd.DataFrame, label_model: LabelModel) -> pd.DataFrame:
        """Generate probabilistic labels for all positions."""
        L_train = self.apply_labeling_functions(df)
        
        # Get probabilistic predictions
        probs_train = label_model.predict_proba(L=L_train)
        
        # Add predictions to dataframe
        df_result = df.copy()
        df_result['snorkel_prob_positive'] = probs_train[:, 1]
        df_result['snorkel_prob_negative'] = probs_train[:, 0]
        df_result['snorkel_prediction'] = label_model.predict(L=L_train)
        
        # Add individual LF outputs for analysis
        for i, lf_name in enumerate(self.lf_names):
            df_result[f'lf_{lf_name}'] = L_train[:, i]
        
        return df_result
    
    def save_results(self, df_result: pd.DataFrame, label_model: LabelModel):
        """Save results and model to output directory."""
        # Save dataframe with results
        df_result.to_csv(self.output_dir / "snorkel_results.csv", index=False)
        
        # Save label model
        label_model.save(self.output_dir / "label_model.pkl")
        
        # Save labeling function statistics
        L_train = self.apply_labeling_functions(df_result)
        lf_stats = self._compute_lf_statistics(L_train)
        
        with open(self.output_dir / "lf_statistics.json", 'w') as f:
            json.dump(lf_stats, f, indent=2)
        
        # Save summary report
        self._save_summary_report(df_result, lf_stats)
    
    def _compute_lf_statistics(self, L_train: np.ndarray) -> Dict[str, Any]:
        """Compute statistics for each labeling function."""
        stats = {}
        
        for i, lf_name in enumerate(self.lf_names):
            lf_outputs = L_train[:, i]
            
            stats[lf_name] = {
                "total_positions": len(lf_outputs),
                "positive_labels": int(np.sum(lf_outputs == 1)),
                "negative_labels": int(np.sum(lf_outputs == 0)),
                "abstentions": int(np.sum(lf_outputs == -1)),
                "coverage": float(np.sum(lf_outputs != -1) / len(lf_outputs)),
                "positive_rate": float(np.sum(lf_outputs == 1) / np.sum(lf_outputs != -1)) if np.sum(lf_outputs != -1) > 0 else 0.0
            }
        
        return stats
    
    def _save_summary_report(self, df_result: pd.DataFrame, lf_stats: Dict[str, Any]):
        """Save a human-readable summary report."""
        report_path = self.output_dir / "summary_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("SNORKEL BOARD POSITION ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total positions analyzed: {len(df_result)}\n")
            f.write(f"Number of labeling functions: {len(self.lf_names)}\n\n")
            
            f.write("LABELING FUNCTION STATISTICS:\n")
            f.write("-" * 30 + "\n")
            for lf_name, stats in lf_stats.items():
                f.write(f"{lf_name}:\n")
                f.write(f"  Coverage: {stats['coverage']:.2%}\n")
                f.write(f"  Positive rate: {stats['positive_rate']:.2%}\n")
                f.write(f"  Abstentions: {stats['abstentions']}\n\n")
            
            f.write("SNORKEL PREDICTIONS SUMMARY:\n")
            f.write("-" * 30 + "\n")
            pred_counts = df_result['snorkel_prediction'].value_counts()
            f.write(f"Positive predictions: {pred_counts.get(1, 0)}\n")
            f.write(f"Negative predictions: {pred_counts.get(0, 0)}\n")
            f.write(f"Average positive probability: {df_result['snorkel_prob_positive'].mean():.3f}\n")
    
    def process(self, input_path: Path):
        """Main processing pipeline."""
        print(f"Loading data from {input_path}...")
        positions = self.load_policy_data(input_path)
        
        if not positions:
            raise ValueError(f"No positions found in {input_path}")
        
        print(f"Loaded {len(positions)} positions")
        
        print("Creating dataframe...")
        df = self.create_dataframe(positions)
        
        print("Applying labeling functions...")
        L_train = self.apply_labeling_functions(df)
        
        print("Training label model...")
        label_model = self.train_label_model(L_train)
        
        print("Generating probabilistic labels...")
        df_result = self.generate_probabilistic_labels(df, label_model)
        
        print("Saving results...")
        self.save_results(df_result, label_model)
        
        print(f"Results saved to {self.output_dir}")
        return df_result, label_model


def main():
    parser = argparse.ArgumentParser(
        description="Apply Snorkel weak supervision to Go board positions"
    )
    parser.add_argument(
        "--input-dir", 
        type=Path, 
        help="Directory containing policy JSON files"
    )
    parser.add_argument(
        "--input-file", 
        type=Path, 
        help="Single policy JSON file to process"
    )
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        required=True,
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    if not args.input_dir and not args.input_file:
        parser.error("Must specify either --input-dir or --input-file")
    
    if args.input_dir and args.input_file:
        parser.error("Cannot specify both --input-dir and --input-file")
    
    input_path = args.input_dir or args.input_file
    
    if not input_path.exists():
        parser.error(f"Input path does not exist: {input_path}")
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    
    try:
        processor = SnorkelBoardPositionProcessor(args.output_dir)
        df_result, label_model = processor.process(input_path)
        
        print("\nProcessing completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        print(f"Summary report: {args.output_dir / 'summary_report.txt'}")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
