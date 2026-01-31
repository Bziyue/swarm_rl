#!/usr/bin/env python3
"""Independent maze solver for verification using standard BFS."""

from collections import deque
from typing import List, Tuple, Optional, Dict
import time

class IndependentMazeSolver:
    """Independent maze solver using standard BFS for verification."""

    def __init__(self):
        self.directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # N, E, S, W
        self.direction_names = ["North", "East", "South", "West"]

    def bfs_shortest_path(self, maze_grid: List[List[int]],
                         start: Tuple[int, int],
                         goal: Tuple[int, int]) -> Tuple[List[Tuple[int, int]], int]:
        """
        Find shortest path using standard BFS.

        Args:
            maze_grid: 2D maze where 0=corridor, 1=wall
            start: Starting position (row, col)
            goal: Goal position (row, col)

        Returns:
            (path, distance) where path is list of coordinates including start and goal
        """
        rows, cols = len(maze_grid), len(maze_grid[0])

        if maze_grid[start[0]][start[1]] == 1 or maze_grid[goal[0]][goal[1]] == 1:
            return [], float('inf')

        if start == goal:
            return [start], 0

        # BFS
        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            (current_row, current_col), path = queue.popleft()

            for dr, dc in self.directions:
                new_row, new_col = current_row + dr, current_col + dc

                if (0 <= new_row < rows and
                    0 <= new_col < cols and
                    maze_grid[new_row][new_col] == 0 and
                    (new_row, new_col) not in visited):

                    new_path = path + [(new_row, new_col)]

                    if (new_row, new_col) == goal:
                        return new_path, len(new_path) - 1  # Distance = steps

                    queue.append(((new_row, new_col), new_path))
                    visited.add((new_row, new_col))

        return [], float('inf')  # No path found

    def get_next_direction(self, maze_grid: List[List[int]],
                          current: Tuple[int, int],
                          goal: Tuple[int, int]) -> Tuple[int, float]:
        """
        Get next direction from current position to goal.

        Returns:
            (direction_index, remaining_distance) where direction_index maps to:
            0=North, 1=East, 2=South, 3=West, -1=invalid/at_goal
        """
        path, distance = self.bfs_shortest_path(maze_grid, current, goal)

        if not path or len(path) <= 1:
            return -1, 0.0 if current == goal else float('inf')

        # Get next step
        next_pos = path[1]
        current_pos = path[0]

        # Calculate direction
        dr = next_pos[0] - current_pos[0]
        dc = next_pos[1] - current_pos[1]

        for i, (direction_dr, direction_dc) in enumerate(self.directions):
            if dr == direction_dr and dc == direction_dc:
                return i, distance

        return -1, float('inf')


def compare_solvers():
    """Compare our PathPrecomputer with independent BFS solver."""
    from maze import MazeAlgorithm, PathPrecomputer

    print("=== Comparing Maze Solvers ===")

    # Generate test maze
    maze_algo = MazeAlgorithm(rows=11, cols=11, num_exits=2, loop_density=0.1)
    maze_grid = maze_algo.generate()

    print(f"Generated maze: {len(maze_grid)}x{len(maze_grid[0])}")
    print(f"Exits: {maze_algo.exits}")

    # Get corridors for testing
    corridors = maze_algo.get_corridor_cells()
    print(f"Total corridors: {len(corridors)}")

    # Initialize both solvers
    precomputer = PathPrecomputer(device="cpu")
    independent = IndependentMazeSolver()

    # Precompute paths
    goals = maze_algo.exits
    start_time = time.time()
    precomputer.precompute_paths(maze_grid, goals)
    precompute_time = time.time() - start_time
    print(f"PathPrecomputer precomputation time: {precompute_time:.4f}s")

    # Test multiple start points
    test_points = corridors[:min(10, len(corridors))]  # Test first 10 corridors

    print("\n=== Comparison Results ===")
    print("Start -> Goal: [Precomputer] vs [Independent BFS]")
    print("-" * 60)

    discrepancies = 0
    total_tests = 0

    for start_pos in test_points:
        for goal_idx, goal_pos in enumerate(goals):
            if start_pos == goal_pos:
                continue

            total_tests += 1

            # Test PathPrecomputer
            precomp_dir, precomp_dist = precomputer.query_path(start_pos, goal_idx)

            # Test Independent BFS
            indep_dir, indep_dist = independent.get_next_direction(maze_grid, start_pos, goal_pos)

            # Compare results
            direction_match = precomp_dir == indep_dir
            distance_match = abs(precomp_dist - indep_dist) < 0.01

            status = "✓" if (direction_match and distance_match) else "✗"

            if not (direction_match and distance_match):
                discrepancies += 1

            print(f"{start_pos} -> {goal_pos}: "
                  f"[dir:{precomp_dir}, dist:{precomp_dist:.1f}] vs "
                  f"[dir:{indep_dir}, dist:{indep_dist:.1f}] {status}")

            # If there's a discrepancy, show the actual path
            if not distance_match and indep_dist != float('inf'):
                path, _ = independent.bfs_shortest_path(maze_grid, start_pos, goal_pos)
                if path:
                    print(f"  Actual BFS path ({len(path)-1} steps): {' -> '.join(map(str, path[:5]))}{'...' if len(path) > 5 else ''}")

    print("-" * 60)
    print(f"Total tests: {total_tests}")
    print(f"Discrepancies: {discrepancies}")
    print(f"Accuracy: {((total_tests - discrepancies) / total_tests * 100):.1f}%" if total_tests > 0 else "N/A")


def detailed_path_analysis():
    """Detailed analysis of a specific path to debug issues."""
    from maze import MazeAlgorithm, PathPrecomputer

    print("\n=== Detailed Path Analysis ===")

    # Generate small maze for detailed analysis
    maze_algo = MazeAlgorithm(rows=7, cols=7, num_exits=1, loop_density=0.0)
    maze_grid = maze_algo.generate()

    # Print maze
    print("Maze layout (S=start, G=goal, █=wall, ·=corridor):")
    corridors = maze_algo.get_corridor_cells()
    goals = maze_algo.exits

    if not corridors or not goals:
        print("No suitable maze generated")
        return

    start_pos = corridors[0]
    goal_pos = goals[0]

    # Ensure start != goal
    if start_pos == goal_pos and len(corridors) > 1:
        start_pos = corridors[1]

    print(f"Start: {start_pos}, Goal: {goal_pos}")

    for r in range(len(maze_grid)):
        line = ""
        for c in range(len(maze_grid[0])):
            if (r, c) == start_pos:
                line += "S"
            elif (r, c) == goal_pos:
                line += "G"
            elif maze_grid[r][c] == 1:
                line += "█"
            else:
                line += "·"
        print(line)

    # Compare both methods step by step
    precomputer = PathPrecomputer(device="cpu")
    independent = IndependentMazeSolver()

    precomputer.precompute_paths(maze_grid, goals)

    # Get BFS path
    bfs_path, bfs_distance = independent.bfs_shortest_path(maze_grid, start_pos, goal_pos)
    print(f"\nBFS path length: {bfs_distance}")
    print(f"BFS path: {' -> '.join(map(str, bfs_path))}")

    # Trace precomputer path
    print(f"\nTracing PathPrecomputer path:")
    current = start_pos
    precomp_path = [current]
    steps = 0

    while steps < 50:  # Safety limit
        next_dir, dist = precomputer.query_path(current, 0)
        print(f"Step {steps}: pos={current}, next_dir={next_dir}, remaining_dist={dist}")

        if next_dir == -1:
            break

        # Move to next position
        dr, dc = [(-1, 0), (0, 1), (1, 0), (0, -1)][next_dir]
        next_pos = (current[0] + dr, current[1] + dc)

        if (0 <= next_pos[0] < len(maze_grid) and
            0 <= next_pos[1] < len(maze_grid[0]) and
            maze_grid[next_pos[0]][next_pos[1]] == 0):

            precomp_path.append(next_pos)
            current = next_pos
            steps += 1
        else:
            print(f"Invalid move to {next_pos}")
            break

    print(f"PathPrecomputer path length: {len(precomp_path) - 1}")
    print(f"PathPrecomputer path: {' -> '.join(map(str, precomp_path))}")

    # Compare paths
    if len(precomp_path) - 1 == bfs_distance:
        print("✓ Path lengths match!")
    else:
        print(f"✗ Path length mismatch: BFS={bfs_distance}, Precomputer={len(precomp_path) - 1}")


if __name__ == "__main__":
    compare_solvers()
    detailed_path_analysis()
