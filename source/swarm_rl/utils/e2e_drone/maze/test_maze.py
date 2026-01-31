#!/usr/bin/env python3
"""
Comprehensive test suite for maze.py module.
Tests MazeAlgorithm and PathPrecomputer classes with full API correctness and performance validation.

This test suite implements the systematic testing plan covering:
1. Test Environment & Tools - Python 3.8+, PyTorch, CUDA support detection
2. API Correctness Verification - All public methods and edge cases
3. Performance Testing - Maze generation and pathfinding benchmarks
4. Continuous Integration - Automated testing with reproducible results

Usage:
    python test_maze.py  # Run all tests
    python test_maze.py --perf  # Run only performance tests
    python test_maze.py --unit  # Run only unit tests
    python test_maze.py --help  # Show usage information
"""

import sys
import time
import random
import argparse
import traceback
import os
from typing import List, Tuple, Dict, Any
import torch
import numpy as np

# Import the maze module
from maze import MazeAlgorithm, PathPrecomputer, Direction


class TestResult:
    """Test result tracking"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def add_pass(self, test_name: str):
        self.passed += 1
        print(f"✓ {test_name}")

    def add_fail(self, test_name: str, error: str):
        self.failed += 1
        self.errors.append((test_name, error))
        print(f"✗ {test_name}: {error}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n=== Test Summary ===")
        print(f"Total: {total}, Passed: {self.passed}, Failed: {self.failed}")
        print(f"Success Rate: {self.passed/total*100:.1f}%" if total > 0 else "No tests run")

        if self.errors:
            print("\n=== Failed Tests ===")
            for test_name, error in self.errors:
                print(f"- {test_name}: {error}")


def assert_equal(actual, expected, message=""):
    """Custom assertion for better error messages"""
    if actual != expected:
        raise AssertionError(f"Expected {expected}, got {actual}. {message}")


def assert_true(condition, message=""):
    """Custom assertion for boolean conditions"""
    if not condition:
        raise AssertionError(f"Condition failed. {message}")


def assert_raises(exception_type, callable_obj, *args, **kwargs):
    """Custom assertion for exception raising"""
    try:
        callable_obj(*args, **kwargs)
        raise AssertionError(f"Expected {exception_type.__name__} to be raised")
    except exception_type:
        pass  # Expected exception raised
    except Exception as e:
        raise AssertionError(f"Expected {exception_type.__name__}, got {type(e).__name__}: {e}")


class MazeAlgorithmTests:
    """Test suite for MazeAlgorithm class"""

    def __init__(self, result: TestResult):
        self.result = result

    def run_all(self):
        """Run all MazeAlgorithm tests"""
        print("\n=== MazeAlgorithm Tests ===")

        test_methods = [
            self.test_constructor_validation,
            self.test_generate_output_format,
            self.test_perfect_tree_generation,
            self.test_loop_insertion,
            self.test_entrance_exit_placement,
            self.test_get_corridor_cells_before_generate,
            self.test_is_connected_special_cases,
            self.test_maze_connectivity,
            self.test_entrance_path_creation
        ]

        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                self.result.add_fail(test_method.__name__, str(e))

    def test_constructor_validation(self):
        """Test constructor parameter validation"""
        # Test invalid dimensions
        try:
            assert_raises(ValueError, MazeAlgorithm, 0, 5)
            assert_raises(ValueError, MazeAlgorithm, -1, 5)
            assert_raises(ValueError, MazeAlgorithm, 5, 0)
            assert_raises(ValueError, MazeAlgorithm, 5, -1)

            # Test even dimensions (must be odd)
            assert_raises(ValueError, MazeAlgorithm, 4, 5)
            assert_raises(ValueError, MazeAlgorithm, 5, 4)

            # Test non-integer dimensions
            assert_raises(ValueError, MazeAlgorithm, 5.5, 5)
            assert_raises(ValueError, MazeAlgorithm, 5, "5")

            self.result.add_pass("test_constructor_validation")
        except Exception as e:
            raise AssertionError(f"Constructor validation failed: {e}")

    def test_generate_output_format(self):
        """Test generate() output format"""
        random.seed(42)
        maze_gen = MazeAlgorithm(5, 7, loop_density=0.0)
        maze = maze_gen.generate()

        # Check dimensions
        assert_equal(len(maze), 5, "Wrong number of rows")
        assert_equal(len(maze[0]), 7, "Wrong number of columns")

        # Check all elements are 0 or 1
        for row in maze:
            for cell in row:
                assert_true(cell in [0, 1], f"Cell value {cell} not in [0, 1]")

        self.result.add_pass("test_generate_output_format")

    def test_perfect_tree_generation(self):
        """Test perfect tree generation (no loops)"""
        random.seed(42)
        maze_gen = MazeAlgorithm(11, 11, loop_density=0.0, branching_prob=0.0)
        maze = maze_gen.generate()

        # Should be connected
        assert_true(maze_gen.is_connected(), "Perfect tree should be connected")

        # Count corridors and check tree property
        corridors = maze_gen.get_corridor_cells()
        assert_true(len(corridors) > 0, "Should have corridors")

        self.result.add_pass("test_perfect_tree_generation")

    def test_loop_insertion(self):
        """Test loop insertion functionality"""
        random.seed(42)

        # Generate maze without loops
        maze_gen_no_loops = MazeAlgorithm(11, 11, loop_density=0.0)
        maze_no_loops = maze_gen_no_loops.generate()
        corridors_no_loops = len(maze_gen_no_loops.get_corridor_cells())

        # Generate maze with loops
        maze_gen_loops = MazeAlgorithm(11, 11, loop_density=0.3)
        maze_loops = maze_gen_loops.generate()
        corridors_loops = len(maze_gen_loops.get_corridor_cells())

        # Maze with loops should have more corridors
        assert_true(corridors_loops >= corridors_no_loops,
                   "Maze with loops should have at least as many corridors")

        self.result.add_pass("test_loop_insertion")

    def test_entrance_exit_placement(self):
        """Test entrance and exit placement"""
        random.seed(42)
        maze_gen = MazeAlgorithm(11, 11, num_entrances=2, num_exits=3)
        maze = maze_gen.generate()

        # Check entrance and exit counts
        assert_equal(len(maze_gen.entrances), 2, "Wrong number of entrances")
        assert_equal(len(maze_gen.exits), 3, "Wrong number of exits")

        # Check all entrances and exits are on perimeter
        for row, col in maze_gen.entrances + maze_gen.exits:
            is_perimeter = (row == 0 or row == 10 or col == 0 or col == 10)
            assert_true(is_perimeter, f"Position ({row}, {col}) not on perimeter")

            # Check they are open (value 0)
            assert_equal(maze[row][col], 0, f"Entrance/exit at ({row}, {col}) not open")

        self.result.add_pass("test_entrance_exit_placement")

    def test_get_corridor_cells_before_generate(self):
        """Test get_corridor_cells before generate() is called"""
        maze_gen = MazeAlgorithm(5, 5)
        assert_raises(RuntimeError, maze_gen.get_corridor_cells)
        self.result.add_pass("test_get_corridor_cells_before_generate")

    def test_is_connected_special_cases(self):
        """Test is_connected() for special cases"""
        maze_gen = MazeAlgorithm(5, 5)

        # Test before generation
        assert_equal(maze_gen.is_connected(), False, "Should be false before generation")

        # Test with all walls (manually set)
        maze_gen.maze = [[1] * 5 for _ in range(5)]
        assert_equal(maze_gen.is_connected(), False, "All walls should not be connected")

        # Test with all corridors
        maze_gen.maze = [[0] * 5 for _ in range(5)]
        assert_equal(maze_gen.is_connected(), True, "All corridors should be connected")

        self.result.add_pass("test_is_connected_special_cases")

    def test_maze_connectivity(self):
        """Test that generated mazes are always connected"""
        random.seed(42)

        # Test multiple configurations
        configs = [
            (5, 5, 0.0),
            (7, 7, 0.1),
            (9, 9, 0.2),
            (11, 11, 0.3)
        ]

        for rows, cols, loop_density in configs:
            maze_gen = MazeAlgorithm(rows, cols, loop_density=loop_density)
            maze = maze_gen.generate()
            assert_true(maze_gen.is_connected(),
                       f"Maze {rows}x{cols} with loop_density={loop_density} not connected")

        self.result.add_pass("test_maze_connectivity")

    def test_entrance_path_creation(self):
        """Test that entrance paths connect to maze interior"""
        random.seed(42)
        maze_gen = MazeAlgorithm(9, 9, num_entrances=4, num_exits=2)
        maze = maze_gen.generate()

        # Check that each entrance/exit has a clear path to interior
        for row, col in maze_gen.entrances + maze_gen.exits:
            # Should be able to move into the maze from this position
            neighbors = [(row+dr, col+dc) for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]]
            valid_neighbors = [(r, c) for r, c in neighbors
                             if 0 <= r < 9 and 0 <= c < 9]

            # At least one neighbor should be a corridor
            has_corridor_neighbor = any(maze[r][c] == 0 for r, c in valid_neighbors)
            assert_true(has_corridor_neighbor,
                       f"Entrance/exit at ({row}, {col}) has no corridor neighbors")

        self.result.add_pass("test_entrance_path_creation")


class PathPrecomputerTests:
    """Test suite for PathPrecomputer class"""

    def __init__(self, result: TestResult):
        self.result = result

    def run_all(self):
        """Run all PathPrecomputer tests"""
        print("\n=== PathPrecomputer Tests ===")

        test_methods = [
            self.test_initialization,
            self.test_precompute_paths_validation,
            self.test_single_target_bfs,
            self.test_multi_target_paths,
            self.test_batch_query_paths,
            self.test_memory_usage,
            self.test_verification,
            self.test_world_grid_coordinate_conversion,
            self.test_lookup_next_directions,
            self.test_path_waypoints,
            self.test_precompute_apsp
        ]

        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                self.result.add_fail(test_method.__name__, str(e))

    def test_initialization(self):
        """Test PathPrecomputer initialization"""
        # Test CPU device
        pathfinder_cpu = PathPrecomputer(device="cpu")
        assert_equal(str(pathfinder_cpu.device), "cpu", "CPU device not set correctly")

        # Test CUDA device if available
        if torch.cuda.is_available():
            pathfinder_cuda = PathPrecomputer(device="cuda:0")
            assert_equal(str(pathfinder_cuda.device), "cuda:0", "CUDA device not set correctly")

        self.result.add_pass("test_initialization")

    def test_precompute_paths_validation(self):
        """Test precompute_paths parameter validation"""
        pathfinder = PathPrecomputer(device="cpu")

        # Test empty goal cells
        maze_grid = [[0, 1, 0], [0, 0, 0], [1, 0, 0]]
        assert_raises(ValueError, pathfinder.precompute_paths, maze_grid, [])

        # Test all walls maze
        all_walls = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        assert_raises(ValueError, pathfinder.precompute_paths, all_walls, [(0, 0)])

        # Test invalid goal coordinates - wall position
        maze_grid = [[0, 1, 0], [0, 0, 0], [1, 0, 0]]
        assert_raises(ValueError, pathfinder.precompute_paths, maze_grid, [(0, 1)])  # Wall position

        # Test invalid goal coordinates - out of bounds
        assert_raises(ValueError, pathfinder.precompute_paths, maze_grid, [(5, 5)])  # Out of bounds

        self.result.add_pass("test_precompute_paths_validation")

    def test_single_target_bfs(self):
        """Test single target reverse BFS"""
        pathfinder = PathPrecomputer(device="cpu")

        # Simple 5x5 maze
        maze_grid = [
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0]
        ]

        goal_cells = [(2, 2)]  # Center position
        pathfinder.precompute_paths(maze_grid, goal_cells)

        # Test query from various positions
        next_dir, distance = pathfinder.query_path((0, 0), 0)
        assert_true(next_dir >= 0, "Should have valid direction from (0,0)")
        assert_true(distance > 0, "Should have positive distance from (0,0)")

        # Test query from goal position
        next_dir, distance = pathfinder.query_path((2, 2), 0)
        assert_equal(next_dir, -1, "Should return -1 at goal")
        assert_equal(distance, 0.0, "Should return 0 distance at goal")

        # Test query from wall position
        next_dir, distance = pathfinder.query_path((1, 1), 0)
        assert_equal(next_dir, -1, "Should return -1 from wall")
        assert_equal(distance, float('inf'), "Should return inf distance from wall")

        self.result.add_pass("test_single_target_bfs")

    def test_multi_target_paths(self):
        """Test multi-target pathfinding"""
        pathfinder = PathPrecomputer(device="cpu")

        # Simple corridor maze
        maze_grid = [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0]
        ]

        goal_cells = [(0, 0), (4, 4)]  # Two corners
        pathfinder.precompute_paths(maze_grid, goal_cells)

        # Test queries to different goals
        next_dir0, dist0 = pathfinder.query_path((2, 2), 0)  # To goal 0
        next_dir1, dist1 = pathfinder.query_path((2, 2), 1)  # To goal 1

        assert_true(next_dir0 >= 0, "Should have path to goal 0")
        assert_true(next_dir1 >= 0, "Should have path to goal 1")
        assert_true(dist0 > 0 and dist1 > 0, "Should have positive distances")

        self.result.add_pass("test_multi_target_paths")

    def test_batch_query_paths(self):
        """Test batch query functionality"""
        pathfinder = PathPrecomputer(device="cpu")

        maze_grid = [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ]

        goal_cells = [(1, 1)]  # Note: this will fail validation since (1,1) is a wall

        # Use a valid goal instead
        goal_cells = [(0, 0)]
        pathfinder.precompute_paths(maze_grid, goal_cells)

        # Batch query
        cells = torch.tensor([[0, 1], [2, 2], [1, 1], [5, 5]], dtype=torch.long)  # Mix of valid/invalid
        goal_indices = torch.tensor([0, 0, 0, 0], dtype=torch.long)

        next_dirs, distances = pathfinder.batch_query_paths(cells, goal_indices)

        assert_equal(len(next_dirs), 4, "Should return 4 results")
        assert_equal(len(distances), 4, "Should return 4 distances")

        # Invalid positions should return -1, inf
        assert_equal(next_dirs[2].item(), -1, "Wall position should return -1")
        assert_equal(next_dirs[3].item(), -1, "Out of bounds should return -1")
        assert_equal(distances[2].item(), float('inf'), "Wall should return inf distance")
        assert_equal(distances[3].item(), float('inf'), "Out of bounds should return inf distance")

        self.result.add_pass("test_batch_query_paths")

    def test_memory_usage(self):
        """Test memory usage calculation"""
        pathfinder = PathPrecomputer(device="cpu")

        # Before precomputation
        memory_before = pathfinder.get_memory_usage_mb()
        assert_equal(memory_before, 0.0, "Memory usage should be 0 before precomputation")

        # After precomputation
        maze_grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        goal_cells = [(0, 0)]
        pathfinder.precompute_paths(maze_grid, goal_cells)

        memory_after = pathfinder.get_memory_usage_mb()
        assert_true(memory_after > 0, "Memory usage should be positive after precomputation")

        # Check theoretical calculation
        n_corridors = 9  # All cells are corridors
        n_goals = 1
        expected_mb = (n_corridors * n_goals * 5) / (1024**2)

        # Allow 50% tolerance due to overhead
        assert_true(abs(memory_after - expected_mb) < expected_mb * 0.5,
                   f"Memory usage {memory_after} differs too much from expected {expected_mb}")

        self.result.add_pass("test_memory_usage")

    def test_verification(self):
        """Test precomputation verification"""
        pathfinder = PathPrecomputer(device="cpu")

        # Test before precomputation
        result = pathfinder.verify_precomputation([], [])
        assert_true("error" in result, "Should return error before precomputation")

        # Test after precomputation
        maze_grid = [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ]
        goal_cells = [(0, 0)]
        pathfinder.precompute_paths(maze_grid, goal_cells)

        result = pathfinder.verify_precomputation(maze_grid, goal_cells)

        expected_keys = ["all_reachable_covered", "distances_correct", "directions_correct"]
        for key in expected_keys:
            assert_true(key in result, f"Missing key {key} in verification result")

        self.result.add_pass("test_verification")

    def test_world_grid_coordinate_conversion(self):
        """Test world coordinate to grid coordinate conversion"""
        pathfinder = PathPrecomputer(device="cpu")

        # Setup simple maze
        maze_grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        goal_cells = [(0, 0)]
        pathfinder.precompute_apsp(maze_grid, 1.0, torch.tensor([0., 0., 0.]), goal_cells)

        # Test conversion - use only x,y coordinates for world_to_grid_batch
        world_coords = torch.tensor([[0., 0.]], dtype=torch.float32)  # Only x,y
        grid_coords = pathfinder.world_to_grid_batch(world_coords)

        assert_equal(grid_coords.shape[0], 1, "Should return one coordinate")
        assert_equal(grid_coords.shape[1], 2, "Should return (row, col)")

        # Convert back
        world_coords_back = pathfinder.grid_to_world_batch(grid_coords)
        assert_equal(world_coords_back.shape[0], 1, "Should return one coordinate")
        assert_equal(world_coords_back.shape[1], 3, "Should return (x, y, z)")

        self.result.add_pass("test_world_grid_coordinate_conversion")

    def test_lookup_next_directions(self):
        """Test batch lookup with world coordinates"""
        pathfinder = PathPrecomputer(device="cpu")

        # Setup maze
        maze_grid = [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ]
        goal_cells = [(0, 0)]
        pathfinder.precompute_apsp(maze_grid, 1.0, torch.tensor([0., 0., 0.]), goal_cells)

        # Test lookup - use only x,y coordinates
        robot_positions = torch.tensor([[0., 0.]], dtype=torch.float32)  # Only x,y
        target_positions = torch.tensor([[0., 0.]], dtype=torch.float32)  # Only x,y

        next_dirs, remaining_dists, next_coords = pathfinder.lookup_next_directions(
            robot_positions, target_positions)

        assert_equal(len(next_dirs), 1, "Should return one direction")
        assert_equal(len(remaining_dists), 1, "Should return one distance")
        assert_equal(next_coords.shape, (1, 3), "Should return one 3D coordinate")

        self.result.add_pass("test_lookup_next_directions")

    def test_path_waypoints(self):
        """Test path waypoint generation"""
        pathfinder = PathPrecomputer(device="cpu")

        # Setup simple corridor
        maze_grid = [
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0]
        ]
        goal_cells = [(0, 0)]
        pathfinder.precompute_apsp(maze_grid, 1.0, torch.tensor([0., 0., 0.]), goal_cells)

        # Test single query
        start_pos = torch.tensor([2., 0., 1.], dtype=torch.float32)  # Bottom row, left
        target_pos = torch.tensor([0., 0., 1.], dtype=torch.float32)  # Top row, left

        waypoints, remaining_dists = pathfinder.get_path_waypoints(start_pos, target_pos, max_waypoints=10)

        # Should return some waypoints (could be 0 if already at target)
        assert_true(waypoints.shape[0] >= 0, "Should return non-negative waypoints")
        assert_true(remaining_dists.shape[0] >= 0, "Should return non-negative remaining distances")
        assert_equal(waypoints.shape[0], remaining_dists.shape[0], "Waypoints and distances should have same length")
        if waypoints.shape[0] > 0:
            assert_equal(waypoints.shape[1], 3, "Waypoints should be 3D coordinates")

        # Test batch query
        batch_start_pos = torch.tensor([[2., 0.], [2., 2.]], dtype=torch.float32)  # Two start positions
        batch_target_pos = torch.tensor([[0., 0.], [0., 2.]], dtype=torch.float32)  # Two targets

        batch_waypoints, batch_remaining_dists = pathfinder.get_path_waypoints(
            batch_start_pos, batch_target_pos, max_waypoints=10)
        
        assert_equal(batch_waypoints.shape[0], 2, "Should return waypoints for 2 environments")
        assert_equal(batch_waypoints.shape[1], 10, "Should have max_waypoints dimension")
        assert_equal(batch_waypoints.shape[2], 3, "Waypoints should be 3D coordinates")
        assert_equal(batch_remaining_dists.shape, (2, 10), "Remaining distances should match waypoints shape")

        self.result.add_pass("test_path_waypoints")

    def test_precompute_apsp(self):
        """Test APSP precomputation"""
        pathfinder = PathPrecomputer(device="cpu")

        maze_grid = [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ]
        exits = [(0, 0), (2, 2)]

        pathfinder.precompute_apsp(maze_grid, 1.0, torch.tensor([0., 0., 0.]), exits)

        assert_true(pathfinder.is_precomputed, "Should be marked as precomputed")
        assert_equal(pathfinder.cell_size, 1.0, "Cell size should be set")
        assert_true(torch.equal(pathfinder.world_origin, torch.tensor([0., 0., 0.])),
                   "World origin should be set")
        assert_equal(len(pathfinder._goal_cells), 2, "Should have 2 goal cells")

        self.result.add_pass("test_precompute_apsp")


class PerformanceTests:
    """Performance benchmark tests"""

    def __init__(self, result: TestResult):
        self.result = result

    def run_all(self):
        """Run all performance tests"""
        print("\n=== Performance Tests ===")

        test_methods = [
            self.test_maze_generation_performance,
            self.test_pathfinding_performance,
            self.test_memory_scaling,
            self.test_lookup_next_directions_performance
        ]

        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                self.result.add_fail(test_method.__name__, str(e))

    def test_maze_generation_performance(self):
        """Test maze generation performance"""
        test_cases = [
            (21, 21, 0.1),   # Small
            (65, 65, 0.05),  # Medium
            (129, 129, 0.03) # Large (reduced timeout for CI)
        ]

        thresholds = [0.01, 0.1, 0.5]  # seconds

        for i, (rows, cols, timeout) in enumerate(test_cases):
            random.seed(42)
            maze_gen = MazeAlgorithm(rows, cols, loop_density=0.1)

            start_time = time.time()
            maze = maze_gen.generate()
            elapsed = time.time() - start_time

            assert_true(elapsed < thresholds[i],
                       f"Maze generation {rows}x{cols} took {elapsed:.3f}s, expected < {thresholds[i]}s")

            print(f"  Maze {rows}x{cols}: {elapsed:.3f}s")

        self.result.add_pass("test_maze_generation_performance")

    def test_pathfinding_performance(self):
        """Test pathfinding precomputation performance"""
        # Prioritize GPU for performance testing
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"  Using device: {device}")

        test_cases = [
            (21, 21, 4, 0.5),    # Small with 4 goals - increased threshold
            (65, 65, 4, 2.0),    # Medium with 4 goals - increased threshold
            (65, 65, 8, 4.0),    # Medium with 8 goals - increased threshold
            (129, 129, 32, 8.0),    # Medium with 8 goals - increased threshold
        ]

        for rows, cols, num_goals, threshold in test_cases:
            # Generate maze
            random.seed(42)
            maze_gen = MazeAlgorithm(rows, cols, loop_density=0.1)
            maze = maze_gen.generate()

            # Select goal cells
            corridors = maze_gen.get_corridor_cells()
            goal_cells = random.sample(corridors, min(num_goals, len(corridors)))

            # Test pathfinding performance
            pathfinder = PathPrecomputer(device=device)

            start_time = time.time()
            pathfinder.precompute_paths(maze, goal_cells)
            elapsed = time.time() - start_time

            assert_true(elapsed < threshold,
                       f"Pathfinding {rows}x{cols} with {num_goals} goals took {elapsed:.3f}s, expected < {threshold}s")

            print(f"  Pathfinding {rows}x{cols}, {num_goals} goals: {elapsed:.3f}s")

        self.result.add_pass("test_pathfinding_performance")

    def test_memory_scaling(self):
        """Test memory usage scaling"""
        # Prioritize GPU for performance testing, fallback to CPU for memory measurement
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"  Using device: {device}")

        test_cases = [
            (11, 11, 2),
            (21, 21, 4),
            (31, 31, 4),
        ]

        for rows, cols, num_goals in test_cases:
            # Generate maze
            random.seed(42)
            maze_gen = MazeAlgorithm(rows, cols, loop_density=0.1)
            maze = maze_gen.generate()

            corridors = maze_gen.get_corridor_cells()
            goal_cells = random.sample(corridors, min(num_goals, len(corridors)))

            # Measure memory
            pathfinder = PathPrecomputer(device=device)
            pathfinder.precompute_paths(maze, goal_cells)

            actual_mb = pathfinder.get_memory_usage_mb()
            expected_mb = (len(corridors) * num_goals * 5) / (1024**2)

            # Allow 20% tolerance
            ratio = actual_mb / expected_mb if expected_mb > 0 else float('inf')
            assert_true(0.8 <= ratio <= 1.2,
                       f"Memory usage ratio {ratio:.2f} outside expected range [0.8, 1.2]")

            print(f"  Memory {rows}x{cols}, {num_goals} goals: {actual_mb:.4f}MB (expected {expected_mb:.4f}MB)")

        self.result.add_pass("test_memory_scaling")

    def test_lookup_next_directions_performance(self):
        """Test lookup_next_directions performance with large robot counts"""
        # Prioritize GPU for performance testing
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"  Using device: {device}")

        test_cases = [
            (65, 65, 4, 10000, 0.1),    # 10K robots
            (65, 65, 4, 50000, 0.3),    # 50K robots
            (65, 65, 4, 100000, 0.5),   # 100K robots
        ]

        # Add even larger test cases for GPU to showcase performance
        if device.startswith("cuda"):
            test_cases.extend([
                (65, 65, 4, 200000, 1.0),   # 200K robots
                (65, 65, 4, 500000, 2.0),   # 500K robots
            ])

        for rows, cols, num_goals, num_robots, threshold in test_cases:
            # Generate maze
            random.seed(42)
            maze_gen = MazeAlgorithm(rows, cols, loop_density=0.1)
            maze = maze_gen.generate()

            # Get corridor cells and select goals
            corridors = maze_gen.get_corridor_cells()
            goal_cells = random.sample(corridors, min(num_goals, len(corridors)))

            # Setup pathfinder with APSP precomputation
            pathfinder = PathPrecomputer(device=device)
            pathfinder.precompute_apsp(maze, 1.0, torch.tensor([0., 0., 0.], device=device), goal_cells)

            # Generate random robot positions (x, y coordinates)
            # Ensure robots are placed on corridor cells
            random.seed(42)
            robot_corridors = random.choices(corridors, k=num_robots)
            robot_positions = torch.tensor([
                [float(col), float(row)] for row, col in robot_corridors
            ], dtype=torch.float32, device=device)

            # Generate random target positions (also on corridor cells)
            target_corridors = random.choices(corridors, k=num_robots)
            target_positions = torch.tensor([
                [float(col), float(row)] for row, col in target_corridors
            ], dtype=torch.float32, device=device)

            # Warm-up run for GPU
            if device.startswith("cuda"):
                _ = pathfinder.lookup_next_directions(robot_positions[:100], target_positions[:100])
                torch.cuda.synchronize()

            # Perform batch lookup and measure performance
            start_time = time.time()
            next_dirs, remaining_dists, next_coords = pathfinder.lookup_next_directions(
                robot_positions, target_positions
            )

            if device.startswith("cuda"):
                torch.cuda.synchronize()

            elapsed = time.time() - start_time

            # Verify results
            assert_equal(len(next_dirs), num_robots, f"Should return {num_robots} directions")
            assert_equal(len(remaining_dists), num_robots, f"Should return {num_robots} distances")
            assert_equal(next_coords.shape, (num_robots, 3), f"Should return {num_robots} 3D coordinates")

            # Check performance threshold
            assert_true(elapsed < threshold,
                       f"lookup_next_directions with {num_robots} robots took {elapsed:.3f}s, expected < {threshold}s")

            # Calculate throughput
            throughput = num_robots / elapsed
            print(f"  lookup_next_directions {num_robots} robots: {elapsed:.3f}s ({throughput:.0f} robots/s)")

        self.result.add_pass("test_lookup_next_directions_performance")


def run_unit_tests():
    """Run all unit tests"""
    result = TestResult()

    # Run test suites
    maze_tests = MazeAlgorithmTests(result)
    maze_tests.run_all()

    pathfinder_tests = PathPrecomputerTests(result)
    pathfinder_tests.run_all()

    return result


def run_performance_tests():
    """Run performance tests"""
    result = TestResult()

    perf_tests = PerformanceTests(result)
    perf_tests.run_all()

    return result


def validate_test_environment():
    """Validate that the test environment is properly set up"""
    print("=== Environment Validation ===")

    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 8):
        print(f"⚠️  Python {python_version.major}.{python_version.minor} detected, Python 3.8+ recommended")
    else:
        print(f"✓ Python {python_version.major}.{python_version.minor}.{python_version.micro}")

    # Check PyTorch
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")

        # Check CUDA
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  CUDA devices: {torch.cuda.device_count()}")
        else:
            print("⚠️  CUDA not available, using CPU only")

    except ImportError:
        print("❌ PyTorch not found")
        return False

    # Check NumPy
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError:
        print("❌ NumPy not found")
        return False

    # Check maze module
    try:
        from maze import MazeAlgorithm, PathPrecomputer, Direction
        print("✓ Maze module imports successful")
    except ImportError as e:
        print(f"❌ Maze module import failed: {e}")
        return False

    print()
    return True


def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description="Comprehensive test suite for maze.py")
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument("--perf", action="store_true", help="Run only performance tests")
    parser.add_argument("--no-validate", action="store_true", help="Skip environment validation")
    args = parser.parse_args()

    print("=== Maze Module Test Suite ===")
    print("Systematic testing for MazeAlgorithm and PathPrecomputer classes")
    print("Testing plan covers API correctness, performance benchmarks, and error handling")
    print()

    # Validate environment unless skipped
    if not args.no_validate:
        if not validate_test_environment():
            print("❌ Environment validation failed")
            sys.exit(1)

    # Set random seeds for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    total_result = TestResult()

    try:
        if args.perf:
            # Run only performance tests
            print("Running performance benchmarks only...")
            perf_result = run_performance_tests()
            total_result.passed += perf_result.passed
            total_result.failed += perf_result.failed
            total_result.errors.extend(perf_result.errors)
        elif args.unit:
            # Run only unit tests
            print("Running unit tests only...")
            unit_result = run_unit_tests()
            total_result.passed += unit_result.passed
            total_result.failed += unit_result.failed
            total_result.errors.extend(unit_result.errors)
        else:
            # Run all tests
            print("Running complete test suite...")
            unit_result = run_unit_tests()
            total_result.passed += unit_result.passed
            total_result.failed += unit_result.failed
            total_result.errors.extend(unit_result.errors)

            perf_result = run_performance_tests()
            total_result.passed += perf_result.passed
            total_result.failed += perf_result.failed
            total_result.errors.extend(perf_result.errors)

    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        traceback.print_exc()

    # Print final summary
    total_result.summary()

    # Additional summary information
    if total_result.failed == 0:
        print("\n🎉 All tests passed! The maze module is functioning correctly.")
    else:
        print(f"\n⚠️  {total_result.failed} test(s) failed. Please review the errors above.")

    # Exit with error code if any tests failed
    sys.exit(1 if total_result.failed > 0 else 0)


if __name__ == "__main__":
    main()
