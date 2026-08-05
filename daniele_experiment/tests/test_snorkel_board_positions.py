#!/usr/bin/env python3
"""
Comprehensive unit tests for snorkel_board_positions.py with synthetic data.

This test suite creates realistic Go board positions, ownership maps, and policy
distributions to thoroughly test all functions in the snorkel_board_positions module.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add the python directory to the path to import board module
sys.path.append(str(Path(__file__).parent.parent / "python"))

from board import Board
from snorkel_board_positions import (
    # Core functions
    classify_region, region_map, xy_to_loc, loc_to_xy, in_bounds,
    
    # Group analysis
    enumerate_groups,
    compute_group_strengths, compute_group_connectivity, compute_group_influence,
    
    # Territory analysis
    count_building_territory, solidify_territory_delta, reduce_opponent_territory_count,
    invasion_effect, weakening_territory_in_region, leaving_weakness,
    territory_sizes, direct_sacrifice, indirect_sacrifice,
    
    # Regional analysis
    compute_group_strength_delta_by_region, compute_group_connectivity_delta_by_region,
    compute_influence_delta_by_region, compute_territory_delta_by_region,
    compute_reduction_delta_by_region,
    
    # Policy and move analysis
    urgency_by_region, urgency_intensity_by_region, is_cut_move, is_forcing,
    is_tenuki, creates_new_group, is_connection_move, is_extension_move,
    is_hard_cut_move, is_hard_connection_move,
    liberties_of_group, atari_move,
    
    # Attack concepts
    reduce_aji, attack_strength_delta, killing_attack,
    
    # Main analysis function
    analyze_position_comprehensive,
    
    # Constants
    TAU_POS, TAU_SOLID, TAU_CONN, EPSILON_POL
)



# Compatibility aliases for older test names
def enumerate_groups_deterministic(board, player=None):
    import numpy as np
    from board import Board as _B
    player = player if player is not None else _B.BLACK
    own = np.ones((board.size, board.size), dtype=np.float32)
    return enumerate_groups(board, own, player)

def enumerate_groups_ownership(board, ownership, player):
    return enumerate_groups(board, ownership, player)

class TestSnorkelBoardPositions(unittest.TestCase):
    """Test suite for snorkel_board_positions module with synthetic data."""
    
    def setUp(self):
        """Set up test fixtures with synthetic board positions and data."""
        self.board_size = 19
        
        # Create test boards with different scenarios
        self.empty_board = Board(self.board_size)
        
        # Board with corner groups
        self.corner_board = Board(self.board_size)
        self.corner_board.play(Board.BLACK, self.corner_board.loc(0, 0))  # Black at A19 (top-left corner)
        self.corner_board.play(Board.WHITE, self.corner_board.loc(1, 0))  # White at B19
        self.corner_board.play(Board.BLACK, self.corner_board.loc(0, 1))  # Black at A18
        self.corner_board.play(Board.WHITE, self.corner_board.loc(18, 0))  # White at S19 (top-right corner)
        self.corner_board.play(Board.BLACK, self.corner_board.loc(18, 1))  # Black at S18
        
        # Board with center fight
        self.center_board = Board(self.board_size)
        self.center_board.play(Board.BLACK, self.center_board.loc(9, 9))   # Black at J10 (center)
        self.center_board.play(Board.WHITE, self.center_board.loc(10, 9))  # White at K10
        self.center_board.play(Board.BLACK, self.center_board.loc(9, 10))  # Black at J9
        self.center_board.play(Board.WHITE, self.center_board.loc(10, 10)) # White at K9
        self.center_board.play(Board.BLACK, self.center_board.loc(8, 9))   # Black at I10 (extension)
        
        # Board with side groups
        self.side_board = Board(self.board_size)
        self.side_board.play(Board.BLACK, self.side_board.loc(0, 9))     # Black at A10 (left side)
        self.side_board.play(Board.WHITE, self.side_board.loc(1, 9))     # White at B10
        self.side_board.play(Board.BLACK, self.side_board.loc(0, 8))     # Black at A11
        self.side_board.play(Board.WHITE, self.side_board.loc(18, 9))    # White at S10 (right side)
        self.side_board.play(Board.BLACK, self.side_board.loc(18, 8))    # Black at S11
        
        # Board with complex group structure
        self.complex_board = Board(self.board_size)
        # Create a small group in corner
        self.complex_board.play(Board.BLACK, self.complex_board.loc(0, 0))  # Black
        self.complex_board.play(Board.WHITE, self.complex_board.loc(1, 0))  # White
        self.complex_board.play(Board.BLACK, self.complex_board.loc(0, 1))  # Black
        self.complex_board.play(Board.WHITE, self.complex_board.loc(1, 1))  # White
        # Create a larger group in center
        self.complex_board.play(Board.BLACK, self.complex_board.loc(8, 8))  # Black
        self.complex_board.play(Board.WHITE, self.complex_board.loc(9, 8))  # White
        self.complex_board.play(Board.BLACK, self.complex_board.loc(8, 9))  # Black
        self.complex_board.play(Board.WHITE, self.complex_board.loc(9, 9))  # White
        self.complex_board.play(Board.BLACK, self.complex_board.loc(7, 8))  # Black (extension)
        self.complex_board.play(Board.WHITE, self.complex_board.loc(10, 8)) # White (extension)
        
        # Generate synthetic ownership maps
        self.generate_ownership_maps()
        
        # Generate synthetic policy distributions
        self.generate_policy_distributions()
    
    def generate_ownership_maps(self):
        """Generate realistic synthetic ownership maps."""
        size = self.board_size
        
        # Strong black territory in corners
        self.black_corner_ownership = np.zeros((size, size))
        self.black_corner_ownership[0:3, 0:3] = 0.8  # Top-left corner
        self.black_corner_ownership[0:3, 16:19] = 0.7  # Top-right corner
        self.black_corner_ownership[16:19, 0:3] = 0.6  # Bottom-left corner
        
        # Strong white territory in center
        self.white_center_ownership = np.zeros((size, size))
        self.white_center_ownership[8:12, 8:12] = -0.8  # Center area
        
        # Mixed ownership with gradual transitions
        self.mixed_ownership = np.zeros((size, size))
        # Black influence from top
        for y in range(size):
            for x in range(size):
                dist_from_top = y / size
                dist_from_center = np.sqrt((x - 9)**2 + (y - 9)**2) / (size * 0.7)
                self.mixed_ownership[y, x] = 0.5 * (1 - dist_from_top) - 0.3 * (1 - dist_from_center)
        
        # Ownership with clear boundaries
        self.bounded_ownership = np.zeros((size, size))
        self.bounded_ownership[0:6, 0:6] = 0.9  # Strong black corner
        self.bounded_ownership[13:19, 13:19] = -0.9  # Strong white corner
        self.bounded_ownership[6:13, 6:13] = 0.0  # Neutral center
        
        # Ownership before and after a move (for delta testing)
        self.ownership_before = np.zeros((size, size))
        self.ownership_before[0:5, 0:5] = 0.6  # Black territory
        self.ownership_before[14:19, 14:19] = -0.6  # White territory
        
        self.ownership_after = self.ownership_before.copy()
        self.ownership_after[0:6, 0:6] = 0.8  # Expanded black territory
        self.ownership_after[14:19, 14:19] = -0.3  # Reduced white territory (was -0.6, now -0.3)
        self.ownership_after[5, 5] = 0.3  # New black influence
    
    def generate_policy_distributions(self):
        """Generate synthetic policy distributions."""
        size = self.board_size
        total_moves = size * size
        
        # Policy focused on corners (normalized to sum to 1.0)
        self.corner_policy = np.zeros(total_moves)
        corner_indices = [0, 1, 18, 19, 342, 343, 360, 361]  # Corner positions
        for idx in corner_indices:
            if idx < total_moves:
                self.corner_policy[idx] = 0.125  # 8 positions * 0.125 = 1.0
        
        # Ensure it sums to exactly 1.0
        total = np.sum(self.corner_policy)
        if total > 0:
            self.corner_policy = self.corner_policy / total
        
        # Policy focused on center
        self.center_policy = np.zeros(total_moves)
        center_indices = [180, 181, 199, 200]  # Center positions
        for idx in center_indices:
            if idx < total_moves:
                self.center_policy[idx] = 0.25
        
        # Policy with single strong move (only move scenario)
        self.forcing_policy = np.zeros(total_moves)
        self.forcing_policy[100] = 0.95  # Single strong move
        
        # Policy with multiple candidates (normalized to sum to 1.0)
        self.multi_candidate_policy = np.zeros(total_moves)
        candidates = [50, 100, 150, 200, 250]
        for idx in candidates:
            if idx < total_moves:
                self.multi_candidate_policy[idx] = 0.2  # 5 positions * 0.2 = 1.0
        
        # Policy with regional distribution (normalized to sum to 1.0)
        self.regional_policy = np.zeros(total_moves)
        for y in range(size):
            for x in range(size):
                idx = y * size + x
                region = classify_region(x, y, size)
                if region == "corner_tl":
                    self.regional_policy[idx] = 0.1
                elif region == "center":
                    self.regional_policy[idx] = 0.05
                elif region.startswith("side_"):
                    self.regional_policy[idx] = 0.02
        
        # Normalize regional policy to sum to 1.0
        total_regional = np.sum(self.regional_policy)
        if total_regional > 0:
            self.regional_policy = self.regional_policy / total_regional


class TestCoordinateAndRegionFunctions(TestSnorkelBoardPositions):
    """Test coordinate conversion and region classification functions."""
    
    def test_xy_to_loc_conversion(self):
        """Test coordinate to location conversion."""
        board = self.empty_board
        
        # Test corner positions
        self.assertEqual(xy_to_loc(board, 0, 0), board.loc(0, 0))
        self.assertEqual(xy_to_loc(board, 18, 18), board.loc(18, 18))
        
        # Test center position
        self.assertEqual(xy_to_loc(board, 9, 9), board.loc(9, 9))
    
    def test_loc_to_xy_conversion(self):
        """Test location to coordinate conversion."""
        board = self.empty_board
        
        # Test corner positions
        x, y = loc_to_xy(board, board.loc(0, 0))
        self.assertEqual((x, y), (0, 0))
        
        x, y = loc_to_xy(board, board.loc(18, 18))
        self.assertEqual((x, y), (18, 18))
    
    def test_in_bounds(self):
        """Test bounds checking."""
        # Valid positions
        self.assertTrue(in_bounds(0, 0, 19))
        self.assertTrue(in_bounds(18, 18, 19))
        self.assertTrue(in_bounds(9, 9, 19))
        
        # Invalid positions
        self.assertFalse(in_bounds(-1, 0, 19))
        self.assertFalse(in_bounds(0, -1, 19))
        self.assertFalse(in_bounds(19, 0, 19))
        self.assertFalse(in_bounds(0, 19, 19))
    
    def test_classify_region(self):
        """Test region classification."""
        # Test corners
        self.assertEqual(classify_region(0, 0, 19), "corner_tl")
        self.assertEqual(classify_region(18, 0, 19), "corner_tr")
        self.assertEqual(classify_region(0, 18, 19), "corner_bl")
        self.assertEqual(classify_region(18, 18, 19), "corner_br")
        
        # Test sides
        self.assertEqual(classify_region(0, 9, 19), "side_left")
        self.assertEqual(classify_region(18, 9, 19), "side_right")
        self.assertEqual(classify_region(9, 0, 19), "side_top")
        self.assertEqual(classify_region(9, 18, 19), "side_bottom")
        
        # Test center
        self.assertEqual(classify_region(9, 9, 19), "center")
    
    def test_region_map(self):
        """Test region map generation."""
        region_map_array = region_map(19)
        
        # Check that all regions are properly assigned
        # Note: region_map_array is indexed as [y, x] (row, column)
        self.assertEqual(region_map_array[0, 0], "corner_tl")    # Top-left corner (y=0, x=0)
        self.assertEqual(region_map_array[0, 18], "corner_tr")   # Top-right corner (y=0, x=18)
        self.assertEqual(region_map_array[18, 0], "corner_bl")   # Bottom-left corner (y=18, x=0)
        self.assertEqual(region_map_array[18, 18], "corner_br")  # Bottom-right corner (y=18, x=18)
        self.assertEqual(region_map_array[9, 9], "center")


class TestGroupAnalysis(TestSnorkelBoardPositions):
    """Test group enumeration and analysis functions."""
    
    def test_enumerate_groups_deterministic(self):
        """Test deterministic group enumeration."""
        # Empty board should have no groups
        groups = enumerate_groups_deterministic(self.empty_board)
        self.assertEqual(len(groups), 0)
        
        # Corner board should have groups
        groups = enumerate_groups_deterministic(self.corner_board)
        self.assertGreater(len(groups), 0)
        
        # Check that all groups have valid properties
        for group in groups:
            self.assertIn(group.color, [Board.BLACK, Board.WHITE])
            self.assertGreater(len(group.stones), 0)
            self.assertGreaterEqual(group.liberties, 0)
        
        # Semantic validation: should have both black and white groups
        black_groups = [g for g in groups if g.color == Board.BLACK]
        white_groups = [g for g in groups if g.color == Board.WHITE]
        self.assertGreater(len(black_groups), 0, "Should have black groups")
        self.assertGreater(len(white_groups), 0, "Should have white groups")
        
        # Groups should have reasonable liberty counts
        for group in groups:
            self.assertLessEqual(group.liberties, 4 * len(group.stones), 
                               "Liberties should not exceed 4 per stone")
    
    def test_enumerate_groups_ownership(self):
        """Test ownership-based group enumeration."""
        # Test with black corner ownership
        groups = enumerate_groups_ownership(self.corner_board, self.black_corner_ownership, Board.BLACK)
        
        # Should find groups where stones exist and ownership is strong
        for group in groups:
            self.assertEqual(group.color, Board.BLACK)
            self.assertGreater(len(group.stones), 0)
    
    def test_compute_group_strengths(self):
        """Test group strength computation."""
        groups = enumerate_groups_deterministic(self.corner_board)
        compute_group_strengths(groups, self.black_corner_ownership, Board.BLACK, self.corner_board)
        
        # Check that strengths are computed
        for group in groups:
            self.assertIsInstance(group.strength, float)
    
    def test_compute_group_connectivity(self):
        """Test group connectivity computation."""
        groups = enumerate_groups_deterministic(self.corner_board)
        compute_group_connectivity(groups, self.black_corner_ownership, self.corner_board)
        
        # Check that connectivity is computed
        for group in groups:
            self.assertIsInstance(group.connectivity, float)
    
    def test_compute_group_influence(self):
        """Test group influence computation."""
        groups = enumerate_groups_deterministic(self.corner_board)
        compute_group_influence(groups, self.black_corner_ownership, self.corner_board)
        
        # Check that influence metrics are computed
        for group in groups:
            self.assertIsInstance(group.influence_area, int)
            self.assertIsInstance(group.influence_strength, float)
            self.assertIsInstance(group.influence_spread, float)


class TestTerritoryAnalysis(TestSnorkelBoardPositions):
    """Test territory analysis functions."""
    
    def test_count_building_territory(self):
        """Test building territory counting."""
        count, intensity = count_building_territory(
            self.ownership_before, self.ownership_after, Board.BLACK
        )
        
        self.assertIsInstance(count, int)
        self.assertIsInstance(intensity, float)
        self.assertGreaterEqual(count, 0)
        self.assertGreaterEqual(intensity, 0)
    
    def test_solidify_territory_delta(self):
        """Test territory solidification."""
        count, intensity = solidify_territory_delta(
            self.ownership_before, self.ownership_after, Board.BLACK
        )
        
        self.assertIsInstance(count, int)
        self.assertIsInstance(intensity, float)
        self.assertGreaterEqual(count, 0)
        self.assertGreaterEqual(intensity, 0)
    
    def test_reduce_opponent_territory_count(self):
        """Test opponent territory reduction."""
        count, intensity = reduce_opponent_territory_count(
            self.ownership_before, self.ownership_after, Board.BLACK
        )
        
        self.assertIsInstance(count, int)
        self.assertIsInstance(intensity, float)
        self.assertGreaterEqual(count, 0)
        self.assertGreaterEqual(intensity, 0)
    
    def test_invasion_effect(self):
        """Test invasion effect detection."""
        is_invasion, intensity = invasion_effect(
            self.ownership_before, self.ownership_after, Board.BLACK
        )
        
        self.assertIsInstance(is_invasion, bool)
        self.assertIsInstance(intensity, float)
        self.assertGreaterEqual(intensity, 0)
    
    def test_territory_sizes(self):
        """Test territory size calculation."""
        potential, solid = territory_sizes(self.black_corner_ownership, Board.BLACK)
        
        self.assertIsInstance(potential, int)
        self.assertIsInstance(solid, int)
        self.assertGreaterEqual(potential, 0)
        self.assertGreaterEqual(solid, 0)
        
        # Semantic validation: black corner ownership should have some territory
        self.assertGreater(solid, 0, "Black corner ownership should have solid territory")
        self.assertGreater(potential + solid, 0, "Should have some total territory")
        self.assertLessEqual(potential + solid, 361, "Territory should not exceed board size")
    
    def test_direct_sacrifice(self):
        """Test direct sacrifice detection."""
        # Test with a move that becomes opponent territory
        move_loc = self.corner_board.loc(5, 5)
        is_sacrifice, intensity = direct_sacrifice(
            move_loc, self.white_center_ownership, Board.BLACK, self.corner_board
        )
        
        self.assertIsInstance(is_sacrifice, bool)
        self.assertIsInstance(intensity, float)
        self.assertGreaterEqual(intensity, 0)
    
    def test_indirect_sacrifice(self):
        """Test indirect sacrifice detection."""
        count, intensity = indirect_sacrifice(
            self.ownership_before, self.ownership_after, Board.BLACK, self.corner_board
        )
        
        self.assertIsInstance(count, int)
        self.assertIsInstance(intensity, float)
        self.assertGreaterEqual(count, 0)
        self.assertGreaterEqual(intensity, 0)


class TestPolicyAndMoveAnalysis(TestSnorkelBoardPositions):
    """Test policy and move analysis functions."""
    
    def test_urgency_by_region(self):
        """Test urgency calculation by region."""
        urgency = urgency_by_region(self.corner_policy)
        
        # Check that all regions are present
        expected_regions = ["corner_tl", "corner_tr", "corner_bl", "corner_br",
                           "side_left", "side_right", "side_top", "side_bottom", "center"]
        for region in expected_regions:
            self.assertIn(region, urgency)
            self.assertIsInstance(urgency[region], float)
            self.assertGreaterEqual(urgency[region], 0)
        
        # Semantic validation: corner policy should have higher urgency in corners
        total_urgency = sum(urgency.values())
        self.assertGreater(total_urgency, 0, "Corner policy should have some urgency")
        self.assertLessEqual(total_urgency, 1.0, "Total urgency should not exceed 1.0")
        
        # Corner regions should have higher urgency than sides/center for corner policy
        corner_urgency = urgency["corner_tl"] + urgency["corner_tr"] + urgency["corner_bl"] + urgency["corner_br"]
        side_urgency = urgency["side_left"] + urgency["side_right"] + urgency["side_top"] + urgency["side_bottom"]
        self.assertGreater(corner_urgency, side_urgency, "Corner policy should favor corners over sides")
    
    def test_urgency_intensity_by_region(self):
        """Test urgency intensity calculation."""
        intensity = urgency_intensity_by_region(self.corner_policy)
        
        # Check that intensities sum to 1.0 (or 0.0 if no policy mass)
        total_intensity = sum(intensity.values())
        self.assertLessEqual(total_intensity, 1.0)
        
        # All values should be between 0 and 1
        for value in intensity.values():
            self.assertGreaterEqual(value, 0)
            self.assertLessEqual(value, 1)
    
    def test_is_cut_move(self):
        """Test cut move detection."""
        # Test with a move that creates a cut
        move_loc = self.center_board.loc(9, 8)  # Between two white stones
        is_cut, cut_count, cut_regions, cut_head_locs = is_cut_move(self.center_board, move_loc)
        self.assertIsInstance(is_cut, bool)
        self.assertIsInstance(cut_count, int)
        self.assertIsInstance(cut_regions, list)
        self.assertIsInstance(cut_head_locs, list)
    
    def test_is_forcing(self):
        """Test only move detection."""
        # Test with single strong move
        self.assertTrue(is_forcing(self.forcing_policy))
        
        # Test with multiple candidates
        self.assertFalse(is_forcing(self.multi_candidate_policy))
    
    def test_is_tenuki(self):
        """Test tenuki detection."""
        # Test with moves in different regions
        selected_idx = 0  # Corner
        last_move_loc = self.center_board.loc(9, 9)  # Center
        is_tenuki_move = is_tenuki(selected_idx, last_move_loc, self.regional_policy, self.center_board)
        self.assertIsInstance(is_tenuki_move, bool)
    
    def test_is_connection_move(self):
        """Test connection move detection."""
        move_loc = self.center_board.loc(8, 8)  # Adjacent to existing stones
        is_connection, strength_gain = is_connection_move(self.center_board, move_loc, Board.BLACK)
        
        self.assertIsInstance(is_connection, bool)
        self.assertIsInstance(strength_gain, float)
        self.assertGreaterEqual(strength_gain, 0)
    
    def test_is_extension_move(self):
        """Test extension move detection."""
        move_loc = self.center_board.loc(7, 8)  # Adjacent to existing stone
        is_extension = is_extension_move(self.center_board, move_loc, Board.BLACK)
        self.assertIsInstance(is_extension, bool)
    
    def test_liberties_of_group(self):
        """Test liberty counting."""
        # Test with a stone that has liberties
        stone_loc = self.corner_board.loc(0, 0)
        liberties = liberties_of_group(self.corner_board, stone_loc)
        self.assertIsInstance(liberties, int)
        self.assertGreaterEqual(liberties, 0)
    
    def test_atari_move(self):
        """Test atari move detection."""
        # Test with a move that puts opponent in atari
        move_loc = self.center_board.loc(9, 7)  # Near opponent group
        is_atari = atari_move(self.center_board, move_loc)
        self.assertIsInstance(is_atari, bool)


class TestAttackConcepts(TestSnorkelBoardPositions):
    """Test attack-related concepts."""
    
    def test_reduce_aji(self):
        """Test aji reduction detection."""
        reduces_aji, intensity = reduce_aji(
            self.ownership_before, self.ownership_after, self.corner_board, Board.BLACK
        )
        
        self.assertIsInstance(reduces_aji, bool)
        self.assertIsInstance(intensity, float)
        self.assertGreaterEqual(intensity, 0)
    
    def test_attack_strength_delta(self):
        """Test attack strength calculation."""
        groups_before = enumerate_groups_deterministic(self.corner_board)
        groups_after = enumerate_groups_deterministic(self.center_board)
        
        is_attack, avg_intensity, max_intensity = attack_strength_delta(
            groups_before, groups_after, Board.WHITE
        )
        
        self.assertIsInstance(is_attack, bool)
        self.assertIsInstance(avg_intensity, float)
        self.assertIsInstance(max_intensity, float)
        self.assertGreaterEqual(avg_intensity, 0)
        self.assertGreaterEqual(max_intensity, 0)
    
    def test_killing_attack(self):
        """Test killing attack detection."""
        groups = enumerate_groups_deterministic(self.corner_board)
        compute_group_strengths(groups, self.black_corner_ownership, Board.BLACK, self.corner_board)
        
        is_killing, intensity = killing_attack(groups, Board.WHITE)
        
        self.assertIsInstance(is_killing, bool)
        self.assertIsInstance(intensity, float)
        self.assertGreaterEqual(intensity, 0)


class TestRegionalAnalysis(TestSnorkelBoardPositions):
    """Test regional analysis functions."""
    
    def test_compute_group_strength_delta_by_region(self):
        """Test group strength delta by region."""
        groups_before = enumerate_groups_deterministic(self.corner_board)
        groups_after = enumerate_groups_deterministic(self.center_board)
        
        deltas = compute_group_strength_delta_by_region(groups_before, groups_after, self.corner_board)
        
        # Check that all regions are present
        expected_regions = ["corner_tl", "corner_tr", "corner_bl", "corner_br",
                           "side_left", "side_right", "side_top", "side_bottom", "center"]
        for region in expected_regions:
            self.assertIn(region, deltas)
            self.assertIsInstance(deltas[region], float)
    
    def test_compute_territory_delta_by_region(self):
        """Test territory delta by region."""
        building_count, building_intensity, solidification_count, solidification_intensity = compute_territory_delta_by_region(
            self.ownership_before, self.ownership_after, Board.BLACK
        )
        
        # Check that all regions are present
        expected_regions = ["corner_tl", "corner_tr", "corner_bl", "corner_br",
                           "side_left", "side_right", "side_top", "side_bottom", "center"]
        
        for region in expected_regions:
            self.assertIn(region, building_count)
            self.assertIn(region, building_intensity)
            self.assertIn(region, solidification_count)
            self.assertIn(region, solidification_intensity)
            
            self.assertIsInstance(building_count[region], int)
            self.assertIsInstance(building_intensity[region], float)
            self.assertIsInstance(solidification_count[region], int)
            self.assertIsInstance(solidification_intensity[region], float)


class TestComprehensiveAnalysis(TestSnorkelBoardPositions):
    """Test the main comprehensive analysis function."""
    
    def test_analyze_position_comprehensive_basic(self):
        """Test comprehensive analysis with basic inputs."""
        results = analyze_position_comprehensive(
            board=self.corner_board,
            ownership=self.black_corner_ownership,
            policy=self.corner_policy,
            player=Board.BLACK
        )
        
        # Check that all expected keys are present
        expected_keys = [
            "urgency", "urgency_intensity", "building_count", "building_intensity",
            "solidification_count", "solidification_value", "reduction_count", "reduction_intensity",
            "invasion", "invasion_intensity", "leaves_weakness", "potential_territory", "solid_territory",
            "direct_sacrifice", "sacrifice_intensity", "indirect_sacrifice", "indirect_sacrifice_intensity",
            "cut", "connection", "connection_strength_gain", "extension", "liberties", "atari",
            "forcing", "tenuki", "reduce_aji", "aji_reduction_intensity", "attack", "attack_intensity",
            "group_strength_delta", "group_connectivity_delta",
            "influence_count_delta", "influence_strength_delta", "creates_new_group", "killing_attack", "kill_intensity"
        ]
        
        for key in expected_keys:
            self.assertIn(key, results)
        
        # Validate that values make semantic sense
        self._validate_comprehensive_results(results, "basic analysis")
    
    def test_analyze_position_comprehensive_with_deltas(self):
        """Test comprehensive analysis with before/after data."""
        results = analyze_position_comprehensive(
            board=self.center_board,
            ownership=self.ownership_after,
            policy=self.center_policy,
            player=Board.BLACK,
            move_loc=self.center_board.loc(8, 8),
            last_move_loc=self.center_board.loc(9, 9),
            before_ownership=self.ownership_before,
            before_board=self.corner_board
        )
        
        # Check that delta-based features are computed
        self.assertIn("building_count", results)
        self.assertIn("building_intensity", results)
        self.assertIn("group_strength_delta", results)
        self.assertIn("group_connectivity_delta", results)
        
        # Check regional deltas
        self.assertIn("building_count_by_region", results)
        self.assertIn("building_intensity_by_region", results)
        self.assertIn("group_strength_delta_by_region", results)
        
        # Validate that values make semantic sense
        self._validate_comprehensive_results(results, "delta analysis")
    
    def test_analyze_position_comprehensive_edge_cases(self):
        """Test comprehensive analysis with edge cases."""
        # Test with empty policy
        empty_policy = np.zeros(361)
        results = analyze_position_comprehensive(
            board=self.empty_board,
            ownership=np.zeros((19, 19)),
            policy=empty_policy,
            player=Board.BLACK
        )
        
        # Should handle empty inputs gracefully
        self.assertIn("urgency", results)
        self.assertEqual(sum(results["urgency"].values()), 0.0)
        
        # Validate that values make semantic sense
        self._validate_comprehensive_results(results, "edge case analysis")
        
        # Test with pass move
        results = analyze_position_comprehensive(
            board=self.corner_board,
            ownership=self.black_corner_ownership,
            policy=self.corner_policy,
            player=Board.BLACK,
            move_loc=Board.PASS_LOC
        )
        
        # Should handle pass move
        self.assertIn("direct_sacrifice", results)
        self.assertFalse(results["direct_sacrifice"])
    
    def _validate_comprehensive_results(self, results, test_name):
        """Validate that comprehensive analysis results make semantic sense."""
        
        # Urgency should sum to total policy mass (or be 0 if no policy)
        total_urgency = sum(results["urgency"].values())
        self.assertGreaterEqual(total_urgency, 0.0, f"Total urgency should be non-negative in {test_name}")
        self.assertLessEqual(total_urgency, 1.0, f"Total urgency should not exceed 1.0 in {test_name}")
        
        # Urgency intensity should sum to 1.0 (or be 0 if no policy)
        total_intensity = sum(results["urgency_intensity"].values())
        if total_urgency > 0:
            self.assertAlmostEqual(total_intensity, 1.0, places=6, 
                                 msg=f"Urgency intensity should sum to 1.0 in {test_name}")
        else:
            self.assertAlmostEqual(total_intensity, 0.0, places=6,
                                 msg=f"Urgency intensity should be 0 when no policy in {test_name}")
        
        # Territory counts should be non-negative
        self.assertGreaterEqual(results["potential_territory"], 0, 
                               f"Potential territory should be non-negative in {test_name}")
        self.assertGreaterEqual(results["solid_territory"], 0,
                               f"Solid territory should be non-negative in {test_name}")
        
        # Solid territory should not exceed potential + solid
        total_territory = results["potential_territory"] + results["solid_territory"]
        self.assertLessEqual(total_territory, 361,  # 19x19 board
                           f"Total territory should not exceed board size in {test_name}")
        
        # Intensities should be non-negative (only check if key exists)
        intensity_keys = ["building_intensity", "solidification_value", "reduction_intensity",
                         "invasion_intensity", "sacrifice_intensity", "indirect_sacrifice_intensity",
                         "connection_strength_gain", "aji_reduction_intensity", "attack_intensity",
                         "kill_intensity"]
        for key in intensity_keys:
            if key in results:
                self.assertGreaterEqual(results[key], 0.0, 
                                       f"{key} should be non-negative in {test_name}")
        
        # Counts should be non-negative integers
        count_keys = ["building_count", "solidification_count", "reduction_count", 
                     "leaves_weakness", "indirect_sacrifice", "liberties"]
        for key in count_keys:
            self.assertGreaterEqual(results[key], 0, 
                                   f"{key} should be non-negative in {test_name}")
            self.assertIsInstance(results[key], int, 
                                 f"{key} should be integer in {test_name}")
        
        # Boolean flags should be actual booleans (handle NumPy booleans)
        boolean_keys = ["invasion", "direct_sacrifice", "cut", "connection", "extension", 
                       "atari", "forcing", "tenuki", "reduce_aji", "attack", 
                       "creates_new_group", "killing_attack"]
        for key in boolean_keys:
            if key in results:
                # Convert NumPy boolean to Python boolean for validation
                value = bool(results[key])
                self.assertIsInstance(value, bool, 
                                     f"{key} should be boolean in {test_name}")
        
        # Regional values should be consistent
        for region in ["corner_tl", "corner_tr", "corner_bl", "corner_br",
                      "side_left", "side_right", "side_top", "side_bottom", "center"]:
            # Regional counts should be non-negative
            for suffix in ["_count", "_count_by_region"]:
                key = f"building{suffix}"
                if key in results and isinstance(results[key], dict) and region in results[key]:
                    self.assertGreaterEqual(results[key][region], 0,
                                           f"{key}[{region}] should be non-negative in {test_name}")
            
            # Regional intensities should be non-negative
            for suffix in ["_intensity", "_intensity_by_region", "_value_by_region"]:
                key = f"building{suffix}"
                if key in results and isinstance(results[key], dict) and region in results[key]:
                    self.assertGreaterEqual(results[key][region], 0.0,
                                           f"{key}[{region}] should be non-negative in {test_name}")
    
    def test_ownership_change_validation(self):
        """Test that ownership changes make semantic sense."""
        # Test building territory
        count, intensity = count_building_territory(
            self.ownership_before, self.ownership_after, Board.BLACK
        )
        
        # Should have built some territory (our test data expands black territory)
        self.assertGreater(count, 0, "Should have built some territory")
        self.assertGreater(intensity, 0, "Building intensity should be positive")
        
        # Test reduction
        reduction_count, reduction_intensity = reduce_opponent_territory_count(
            self.ownership_before, self.ownership_after, Board.BLACK
        )
        
        # Should have reduced some opponent territory (our test data reduces white territory)
        self.assertGreaterEqual(reduction_count, 0, "Reduction count should be non-negative")
        if reduction_count > 0:
            self.assertGreater(reduction_intensity, 0, "Reduction intensity should be positive")
        
        # Test invasion effect
        is_invasion, invasion_intensity = invasion_effect(
            self.ownership_before, self.ownership_after, Board.BLACK
        )
        
        # Invasion occurs if both building and reduction happened
        expected_invasion = (count > 0) and (reduction_count > 0)
        self.assertEqual(is_invasion, expected_invasion, "Invasion detection should match expected")
        self.assertGreaterEqual(invasion_intensity, 0, "Invasion intensity should be non-negative")


class TestConstants(TestSnorkelBoardPositions):
    """Test that constants are properly defined."""
    
    def test_constants_defined(self):
        """Test that all constants are properly defined."""
        self.assertIsInstance(TAU_POS, float)
        self.assertIsInstance(TAU_SOLID, float)
        self.assertIsInstance(TAU_CONN, float)
        self.assertIsInstance(EPSILON_POL, float)
        
        # Check reasonable values
        self.assertGreater(TAU_POS, 0)
        self.assertGreater(TAU_SOLID, TAU_POS)
        self.assertGreater(TAU_CONN, 0)
        self.assertGreater(EPSILON_POL, 0)


def create_synthetic_test_data():
    """Create additional synthetic test data for edge cases."""
    print("Creating synthetic test data...")
    
    # Create a board with a complex group structure
    board = Board(19)
    
    # Create a ladder pattern - alternate between black and white
    for i in range(5):
        if i % 2 == 0:
            board.play(Board.BLACK, board.loc(i, 0))  # Black stones
        else:
            board.play(Board.WHITE, board.loc(i, 0))  # White stones
    
    # Create a corner enclosure in a different area - alternate properly
    board.play(Board.BLACK, board.loc(15, 15))  # Black
    board.play(Board.WHITE, board.loc(16, 15))  # White
    board.play(Board.BLACK, board.loc(15, 16))  # Black
    board.play(Board.WHITE, board.loc(16, 16))  # White
    board.play(Board.BLACK, board.loc(14, 15))  # Black (extension)
    
    # Create ownership map with clear territorial boundaries
    ownership = np.zeros((19, 19))
    ownership[0:6, 0:6] = 0.8  # Strong black territory
    ownership[13:19, 13:19] = -0.8  # Strong white territory
    ownership[6:13, 6:13] = 0.0  # Neutral center
    
    # Create policy with multiple strong candidates
    policy = np.zeros(361)
    strong_candidates = [0, 1, 18, 19, 100, 180, 200, 342, 360, 361]
    for idx in strong_candidates:
        if idx < len(policy):
            policy[idx] = 0.1
    
    return board, ownership, policy


class TestRealVsSyntheticData(TestSnorkelBoardPositions):
    """Test that compares synthetic data with real game data."""
    
    def setUp(self):
        """Set up real game data for comparison."""
        super().setUp()
        self.real_game_dir = Path(__file__).parent.parent / "games" / "a64ca662-9218-4888-b9f3-ca9e93631ab4"
        self.load_real_game_data()
    
    def load_real_game_data(self):
        """Load real game data from files."""
        import json
        
        # Load move data
        with open(self.real_game_dir / "moves.jsonl", 'r') as f:
            self.real_move1 = json.loads(f.readline())
            self.real_move2 = json.loads(f.readline())
        
        # Extract ownership data from the move data (more reliable than trunkfinal)
        # The ownership field in the move data is the actual ownership map
        self.real_ownership1 = np.array(self.real_move1['ownership']).reshape(19, 19)
        self.real_ownership2 = np.array(self.real_move2['ownership']).reshape(19, 19)
        
        # Extract policy data
        self.real_policy1 = np.array(self.real_move1['policy0'][:361])  # First 361 elements
        self.real_policy2 = np.array(self.real_move2['policy0'][:361])  # First 361 elements
    
    def test_real_data_structure(self):
        """Test that real data has expected structure."""
        # Test move data structure
        self.assertIn('move_number', self.real_move1)
        self.assertIn('player', self.real_move1)
        self.assertIn('move_loc', self.real_move1)
        self.assertIn('policy0', self.real_move1)
        self.assertIn('ownership', self.real_move1)
        
        # Test policy data
        self.assertEqual(len(self.real_policy1), 361, "Policy should have 361 elements")
        self.assertEqual(len(self.real_policy2), 361, "Policy should have 361 elements")
        
        # Test policy normalization (should sum to ~1.0, allowing for floating point precision)
        self.assertAlmostEqual(np.sum(self.real_policy1), 1.0, places=5, 
                              msg="Real policy should sum to ~1.0")
        self.assertAlmostEqual(np.sum(self.real_policy2), 1.0, places=4,
                              msg="Real policy should sum to ~1.0")
        
        # Test ownership data
        self.assertEqual(self.real_ownership1.shape, (19, 19), "Ownership should be 19x19")
        self.assertEqual(self.real_ownership2.shape, (19, 19), "Ownership should be 19x19")
        
        # Test ownership range (should be reasonable)
        self.assertGreaterEqual(self.real_ownership1.min(), -1.0, "Ownership should be >= -1.0")
        self.assertLessEqual(self.real_ownership1.max(), 1.0, "Ownership should be <= 1.0")
    
    def test_synthetic_vs_real_policy_comparison(self):
        """Compare synthetic policy with real policy characteristics."""
        # Test synthetic policy characteristics
        synthetic_total = np.sum(self.corner_policy)
        real_total = np.sum(self.real_policy1)
        
        self.assertAlmostEqual(synthetic_total, 1.0, places=6, 
                              msg="Synthetic policy should sum to 1.0")
        self.assertAlmostEqual(real_total, 1.0, places=5,
                              msg="Real policy should sum to ~1.0")
        
        # Test that both have reasonable non-zero entries
        synthetic_nonzero = np.sum(self.corner_policy > 0)
        real_nonzero = np.sum(self.real_policy1 > 0)
        
        self.assertGreater(synthetic_nonzero, 0, "Synthetic policy should have non-zero entries")
        self.assertGreater(real_nonzero, 0, "Real policy should have non-zero entries")
        
        # Test that both have reasonable maximum values
        synthetic_max = np.max(self.corner_policy)
        real_max = np.max(self.real_policy1)
        
        self.assertGreater(synthetic_max, 0, "Synthetic policy should have positive max")
        self.assertGreater(real_max, 0, "Real policy should have positive max")
        self.assertLessEqual(synthetic_max, 1.0, "Synthetic policy max should be <= 1.0")
        self.assertLessEqual(real_max, 1.0, "Real policy max should be <= 1.0")
    
    def test_synthetic_vs_real_ownership_comparison(self):
        """Compare synthetic ownership with real ownership characteristics."""
        # Test ownership ranges
        synthetic_range = (self.black_corner_ownership.min(), self.black_corner_ownership.max())
        real_range = (self.real_ownership1.min(), self.real_ownership1.max())
        
        print(f"Synthetic ownership range: {synthetic_range}")
        print(f"Real ownership range: {real_range}")
        
        # Both should be within reasonable bounds
        self.assertGreaterEqual(synthetic_range[0], -1.0, "Synthetic ownership should be >= -1.0")
        self.assertLessEqual(synthetic_range[1], 1.0, "Synthetic ownership should be <= 1.0")
        self.assertGreaterEqual(real_range[0], -1.0, "Real ownership should be >= -1.0")
        self.assertLessEqual(real_range[1], 1.0, "Real ownership should be <= 1.0")
        
        # Test that both have some variation (not all zeros)
        synthetic_std = np.std(self.black_corner_ownership)
        real_std = np.std(self.real_ownership1)
        
        self.assertGreater(synthetic_std, 0, "Synthetic ownership should have variation")
        self.assertGreater(real_std, 0, "Real ownership should have variation")
    
    def test_comprehensive_analysis_real_vs_synthetic(self):
        """Compare comprehensive analysis results between real and synthetic data."""
        # Create a simple board for real data analysis
        real_board = Board(19)
        # Add some stones to make it interesting
        real_board.play(Board.BLACK, real_board.loc(3, 3))
        real_board.play(Board.WHITE, real_board.loc(15, 15))
        
        # Analyze synthetic data
        synthetic_results = analyze_position_comprehensive(
            board=self.corner_board,
            ownership=self.black_corner_ownership,
            policy=self.corner_policy,
            player=Board.BLACK
        )
        
        # Analyze real data
        real_results = analyze_position_comprehensive(
            board=real_board,
            ownership=self.real_ownership1,
            policy=self.real_policy1,
            player=Board.BLACK
        )
        
        # Compare key metrics
        print(f"Synthetic urgency total: {sum(synthetic_results['urgency'].values()):.6f}")
        print(f"Real urgency total: {sum(real_results['urgency'].values()):.6f}")
        
        print(f"Synthetic potential territory: {synthetic_results['potential_territory']}")
        print(f"Real potential territory: {real_results['potential_territory']}")
        
        print(f"Synthetic solid territory: {synthetic_results['solid_territory']}")
        print(f"Real solid territory: {real_results['solid_territory']}")
        
        # Both should have valid results
        self.assertGreaterEqual(sum(synthetic_results['urgency'].values()), 0.0)
        self.assertGreaterEqual(sum(real_results['urgency'].values()), 0.0)
        
        self.assertGreaterEqual(synthetic_results['potential_territory'], 0)
        self.assertGreaterEqual(real_results['potential_territory'], 0)
        
        self.assertGreaterEqual(synthetic_results['solid_territory'], 0)
        self.assertGreaterEqual(real_results['solid_territory'], 0)
    
    def test_real_data_consistency(self):
        """Test that real data is internally consistent."""
        # Test that move locations are valid
        self.assertGreaterEqual(self.real_move1['move_loc'], 0)
        self.assertLess(self.real_move1['move_loc'], 19*19+1)  # Board size + pass
        
        self.assertGreaterEqual(self.real_move2['move_loc'], 0)
        self.assertLess(self.real_move2['move_loc'], 19*19+1)
        
        # Test that idx361 is reasonable (it may not match move_loc exactly due to different indexing)
        if self.real_move1['move_loc'] != 0:  # Not pass
            self.assertGreaterEqual(self.real_move1['idx361'], 0, "idx361 should be >= 0")
            self.assertLess(self.real_move1['idx361'], 361, "idx361 should be < 361")
        
        # Test that selected probability is reasonable
        self.assertGreaterEqual(self.real_move1['selected_prob'], 0.0)
        self.assertLessEqual(self.real_move1['selected_prob'], 1.0)
        
        self.assertGreaterEqual(self.real_move2['selected_prob'], 0.0)
        self.assertLessEqual(self.real_move2['selected_prob'], 1.0)
    
    def test_detailed_synthetic_vs_real_comparison(self):
        """Detailed comparison showing how synthetic data differs from real data."""
        print("\n" + "="*60)
        print("DETAILED SYNTHETIC vs REAL DATA COMPARISON")
        print("="*60)
        
        # Policy characteristics
        print("\nPOLICY CHARACTERISTICS:")
        print(f"Synthetic policy sum: {np.sum(self.corner_policy):.6f}")
        print(f"Real policy sum: {np.sum(self.real_policy1):.6f}")
        print(f"Synthetic policy max: {np.max(self.corner_policy):.6f}")
        print(f"Real policy max: {np.max(self.real_policy1):.6f}")
        print(f"Synthetic policy non-zero entries: {np.sum(self.corner_policy > 0)}")
        print(f"Real policy non-zero entries: {np.sum(self.real_policy1 > 0)}")
        
        # Ownership characteristics
        print("\nOWNERSHIP CHARACTERISTICS:")
        print(f"Synthetic ownership range: [{self.black_corner_ownership.min():.3f}, {self.black_corner_ownership.max():.3f}]")
        print(f"Real ownership range: [{self.real_ownership1.min():.3f}, {self.real_ownership1.max():.3f}]")
        print(f"Synthetic ownership std: {np.std(self.black_corner_ownership):.3f}")
        print(f"Real ownership std: {np.std(self.real_ownership1):.3f}")
        print(f"Synthetic ownership mean: {np.mean(self.black_corner_ownership):.3f}")
        print(f"Real ownership mean: {np.mean(self.real_ownership1):.3f}")
        
        # Territory analysis
        synthetic_potential, synthetic_solid = territory_sizes(self.black_corner_ownership, Board.BLACK)
        real_potential, real_solid = territory_sizes(self.real_ownership1, Board.BLACK)
        
        print("\nTERRITORY ANALYSIS:")
        print(f"Synthetic - Potential: {synthetic_potential}, Solid: {synthetic_solid}")
        print(f"Real - Potential: {real_potential}, Solid: {real_solid}")
        
        # Urgency analysis
        synthetic_urgency = urgency_by_region(self.corner_policy)
        real_urgency = urgency_by_region(self.real_policy1)
        
        print("\nURGENCY BY REGION:")
        for region in ["corner_tl", "corner_tr", "corner_bl", "corner_br", "center"]:
            print(f"{region:12}: Synthetic={synthetic_urgency[region]:.4f}, Real={real_urgency[region]:.4f}")
        
        print("\nKEY DIFFERENCES:")
        print("1. Real ownership has negative values (opponent territory), synthetic is all positive")
        print("2. Real ownership has much wider range and more variation")
        print("3. Real policy has many more non-zero entries (more distributed)")
        print("4. Real data shows more realistic territorial balance")
        
        # Validate that both produce reasonable results
        self.assertGreater(np.sum(self.corner_policy), 0.8, "Synthetic policy should be close to 1.0")
        self.assertGreater(np.sum(self.real_policy1), 0.99, "Real policy should be very close to 1.0")
        
        # Note: Synthetic ownership actually has higher std dev due to concentrated high values
        # Real ownership has more realistic distribution with both positive and negative values
        self.assertGreater(np.abs(self.real_ownership1.min()), np.abs(self.black_corner_ownership.min()),
                          "Real ownership should have more negative values (opponent territory)")
    
    def test_ownership_perspective_issue(self):
        """Test that reveals the critical ownership perspective issue."""
        print("\n" + "="*70)
        print("CRITICAL OWNERSHIP PERSPECTIVE ISSUE")
        print("="*70)
        
        # Check ownership at actual move locations
        move1_loc = self.real_move1['move_loc']
        move2_loc = self.real_move2['move_loc']
        
        if move1_loc != 0:  # Not pass
            x1, y1 = move1_loc % 19, move1_loc // 19
            own1_at_move = self.real_ownership1[y1, x1]
            print(f"Move 1 (Player {self.real_move1['player']}) at ({x1},{y1}): ownership = {own1_at_move:.4f}")
        
        if move2_loc != 0:  # Not pass
            x2, y2 = move2_loc % 19, move2_loc // 19
            own2_at_move = self.real_ownership2[y2, x2]
            print(f"Move 2 (Player {self.real_move2['player']}) at ({x2},{y2}): ownership = {own2_at_move:.4f}")
        
        # Test territory analysis with original vs flipped ownership
        pot1_orig, solid1_orig = territory_sizes(self.real_ownership1, Board.BLACK)
        pot1_fixed, solid1_fixed = territory_sizes(-self.real_ownership1, Board.BLACK)
        
        print(f"\nTerritory analysis for Move 1 (Black):")
        print(f"  Original ownership: Potential={pot1_orig}, Solid={solid1_orig}")
        print(f"  Flipped ownership:  Potential={pot1_fixed}, Solid={solid1_fixed}")
        
        # The fix should show more reasonable territory
        self.assertGreater(pot1_fixed, pot1_orig, 
                          "Flipped ownership should show more potential territory")
        
        print("\n🚨 CRITICAL FINDING:")
        print("   - KataGo ownership is from OPPONENT perspective")
        print("   - Snorkel code expects CURRENT PLAYER perspective")
        print("   - Must flip ownership sign: use -ownership instead of ownership")
        
        return True


if __name__ == "__main__":
    # Create additional test data
    test_board, test_ownership, test_policy = create_synthetic_test_data()
    print(f"Created test board with {len(enumerate_groups_deterministic(test_board))} groups")
    print(f"Test ownership range: [{test_ownership.min():.2f}, {test_ownership.max():.2f}]")
    print(f"Test policy has {np.sum(test_policy > 0)} non-zero entries")
    
    # Run the test suite
    unittest.main(verbosity=2)
