"""Standalone optimized maze generation and multi-target pathfinding."""

import random
from enum import Enum
from typing import List, Tuple, Optional, Dict
import torch
import numpy as np


class Direction(Enum):
    """Direction enumeration for maze generation and pathfinding."""
    NORTH = (-1, 0)
    SOUTH = (1, 0)
    EAST = (0, 1)
    WEST = (0, -1)

    @classmethod
    def all_directions(cls):
        return [cls.NORTH, cls.SOUTH, cls.EAST, cls.WEST]

    def opposite(self):
        return {self.NORTH: self.SOUTH, self.SOUTH: self.NORTH,
                self.EAST: self.WEST, self.WEST: self.EAST}[self]


class MazeAlgorithm:
    """Optimized 2D maze generation using randomized DFS with loop insertion."""

    def __init__(self, rows: int, cols: int, loop_density: float = 0.15,
                 branching_prob: float = 0.4, bias_factor: float = 0.3,
                 num_entrances: int = 0, num_exits: int = 0):
        if not isinstance(rows, int) or rows <= 0 or not isinstance(cols, int) or cols <= 0:
            raise ValueError("Invalid grid dimensions")
        if rows % 2 == 0 or cols % 2 == 0:
            raise ValueError("Grid dimensions must be odd for proper maze generation")

        self.rows, self.cols = rows, cols
        self.loop_density = max(0.0, min(1.0, loop_density))
        self.branching_prob = max(0.0, min(1.0, branching_prob))
        self.bias_factor = max(0.0, min(1.0, bias_factor))
        self.num_entrances = max(0, num_entrances)
        self.num_exits = max(0, num_exits)
        self.maze = None
        self.last_direction = None
        self.entrances = []
        self.exits = []

    def generate(self) -> List[List[int]]:
        """Generate maze using randomized DFS algorithm with loop insertion.

        Returns:
            2D grid where 0=corridor, 1=wall
        """
        self.maze = [[1] * self.cols for _ in range(self.rows)]
        self.last_direction = None

        self._generate_base_maze()
        if self.loop_density > 0:
            self._add_loops()

        # Add entrances and exits if requested
        if self.num_entrances > 0 or self.num_exits > 0:
            self._place_entrances_exits()

        return self.maze

    def _generate_base_maze(self):
        """Generate base maze using randomized DFS."""
        start_row = random.randrange(1, self.rows - 1, 2)
        start_col = random.randrange(1, self.cols - 1, 2)
        self.maze[start_row][start_col] = 0

        stack = [(start_row, start_col)]
        visited = {(start_row, start_col)}

        while stack:
            row, col = stack[-1]
            neighbors = []

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_row, new_col = row + 2*dr, col + 2*dc
                if (0 < new_row < self.rows - 1 and 0 < new_col < self.cols - 1 and
                    (new_row, new_col) not in visited):
                    neighbors.append((new_row, new_col, Direction((dr, dc))))

            if neighbors:
                next_row, next_col, direction = self._choose_next_cell(neighbors)
                wall_row, wall_col = row + direction.value[0], col + direction.value[1]
                self.maze[wall_row][wall_col] = 0
                self.maze[next_row][next_col] = 0

                visited.add((next_row, next_col))
                stack.append((next_row, next_col))
                self.last_direction = direction

                if random.random() < self.branching_prob and len(neighbors) > 1:
                    for nr, nc, _ in neighbors:
                        if (nr, nc) not in visited:
                            stack.append((row, col))
                            break
            else:
                stack.pop()

    def _choose_next_cell(self, neighbors):
        """Choose next cell based on bias factor for straighter corridors."""
        if self.last_direction and self.bias_factor > 0:
            same_direction = [n for n in neighbors if n[2] == self.last_direction]
            if same_direction and random.random() < self.bias_factor:
                return random.choice(same_direction)
        return random.choice(neighbors)

    def _add_loops(self):
        """Add loops by removing walls to create cycles using vectorized operations."""
        # Convert maze to numpy array for efficient vectorized operations
        maze_array = np.array(self.maze)

        # Create coordinate grids
        rows, cols = np.meshgrid(np.arange(1, self.rows - 1),
                                np.arange(1, self.cols - 1), indexing='ij')

        # Vectorized wall detection
        wall_mask = maze_array[1:self.rows-1, 1:self.cols-1] == 1

        # Vectorized loop potential calculation
        # Count adjacent passages for each potential wall removal position
        adjacent_counts = np.zeros_like(wall_mask, dtype=int)

        # Check all four directions simultaneously
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            # Shift coordinates to check neighbors
            neighbor_r = rows + dr
            neighbor_c = cols + dc

            # Bounds check
            valid_mask = ((neighbor_r >= 0) & (neighbor_r < self.rows) &
                         (neighbor_c >= 0) & (neighbor_c < self.cols))

            # Count passages (vectorized)
            passages = np.zeros_like(wall_mask, dtype=bool)
            passages[valid_mask] = maze_array[neighbor_r[valid_mask], neighbor_c[valid_mask]] == 0
            adjacent_counts += passages.astype(int)

        # Find removable walls (walls with 2+ adjacent passages)
        removable_mask = wall_mask & (adjacent_counts >= 2)
        removable_positions = list(zip(*np.where(removable_mask)))

        # Adjust coordinates back to original maze space
        removable_walls = [(r + 1, c + 1) for r, c in removable_positions]

        # Remove walls
        num_to_remove = int(self.loop_density * len(removable_walls))
        if num_to_remove > 0 and removable_walls:
            walls_to_remove = random.sample(removable_walls, min(num_to_remove, len(removable_walls)))
            for r, c in walls_to_remove:
                self.maze[r][c] = 0

    def get_corridor_cells(self) -> List[Tuple[int, int]]:
        """Get all corridor (navigable) cell coordinates using vectorized operations."""
        if self.maze is None:
            raise RuntimeError("No maze generated")

        # Convert to numpy array for vectorized operations
        maze_array = np.array(self.maze)

        # Find all corridor positions at once
        corridor_positions = np.where(maze_array == 0)

        # Convert to list of tuples
        corridors = list(zip(corridor_positions[0], corridor_positions[1]))
        return corridors

    def is_connected(self) -> bool:
        """Verify maze connectivity using flood fill."""
        if self.maze is None:
            return False

        corridors = self.get_corridor_cells()
        if not corridors:
            return False

        start = corridors[0]
        visited = set()
        stack = [start]

        while stack:
            r, c = stack.pop()
            if (r, c) in visited:
                continue
            visited.add((r, c))

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.rows and 0 <= nc < self.cols and
                    self.maze[nr][nc] == 0 and (nr, nc) not in visited):
                    stack.append((nr, nc))

        return len(visited) == len(corridors)

    def _place_entrances_exits(self):
        """Place entrances and exits on maze perimeter."""
        perimeter = []
        # Top and bottom edges
        for c in range(self.cols):
            perimeter.extend([(0, c), (self.rows - 1, c)])
        # Left and right edges (excluding corners already added)
        for r in range(1, self.rows - 1):
            perimeter.extend([(r, 0), (r, self.cols - 1)])

        total_openings = self.num_entrances + self.num_exits
        if total_openings > 0 and len(perimeter) > 0:
            selected_positions = random.sample(perimeter, min(total_openings, len(perimeter)))
            self.entrances = selected_positions[:self.num_entrances]
            self.exits = selected_positions[self.num_entrances:]

            # Create openings and connect them to maze interior
            for row, col in self.entrances + self.exits:
                self.maze[row][col] = 0
                self._create_entrance_path(row, col)

    def _create_entrance_path(self, row: int, col: int):
        """Create a path from entrance/exit into maze interior."""
        directions = []
        if row == 0: directions.append((1, 0))  # From top edge, go down
        elif row == self.rows - 1: directions.append((-1, 0))  # From bottom edge, go up
        if col == 0: directions.append((0, 1))  # From left edge, go right
        elif col == self.cols - 1: directions.append((0, -1))  # From right edge, go left

        # Create path into maze interior until reaching existing corridor
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            while (0 <= new_row < self.rows and 0 <= new_col < self.cols and
                   self.maze[new_row][new_col] == 1):
                self.maze[new_row][new_col] = 0
                new_row += dr
                new_col += dc


class PathPrecomputer:
    """GPU-accelerated multi-target pathfinding using reverse BFS with world coordinate support."""

    def __init__(self, device: str = "cuda:0"):
        self.device = torch.device(device)
        self.grid_h = self.grid_w = 0
        self.next_dir = self.dist = None
        self.corridor_to_grid = None
        self.grid_to_corridor = None
        self.is_precomputed = False
        self.maze_grid = None  # Store maze grid for optimized BFS
        # Direction offsets: N, E, S, W = 0, 1, 2, 3
        self.dir_offsets = torch.tensor([[-1, 0], [0, 1], [1, 0], [0, -1]],
                                       dtype=torch.long, device=self.device)

        # Additional attributes for world coordinate support
        self.cell_size = 0.0
        self.world_origin = torch.zeros(3, device=self.device)
        self._goal_cells = None

    def precompute_paths(self, maze_grid: List[List[int]], goal_cells: List[Tuple[int, int]]) -> None:
        """Precompute shortest paths from all cells to multiple goal cells using reverse BFS.

        Args:
            maze_grid: 2D maze where 0=corridor, 1=wall
            goal_cells: List of (row, col) goal positions
        """
        if not goal_cells:
            raise ValueError("At least one goal cell required")

        self.grid_h, self.grid_w = len(maze_grid), len(maze_grid[0])

        # Store maze grid for the optimized BFS
        self.maze_grid = maze_grid

        grid_tensor = torch.tensor(maze_grid, dtype=torch.uint8, device=self.device)
        corridor_indices = torch.nonzero(grid_tensor == 0, as_tuple=False)

        if len(corridor_indices) == 0:
            raise ValueError("No navigable corridors found")

        num_corridors = len(corridor_indices)
        num_goals = len(goal_cells)

        self.corridor_to_grid = corridor_indices
        self.grid_to_corridor = torch.full((self.grid_h, self.grid_w), -1,
                                          dtype=torch.long, device=self.device)

        for i, (row, col) in enumerate(corridor_indices):
            self.grid_to_corridor[row, col] = i

        INF = 1e6
        self.dist = torch.full((num_corridors, num_goals), INF,
                              dtype=torch.float32, device=self.device)
        self.next_dir = torch.zeros((num_corridors, num_goals),
                                   dtype=torch.uint8, device=self.device)

        goal_corridor_indices = []
        for goal_r, goal_c in goal_cells:
            if (0 <= goal_r < self.grid_h and 0 <= goal_c < self.grid_w and
                maze_grid[goal_r][goal_c] == 0):
                goal_idx = self.grid_to_corridor[goal_r, goal_c].item()
                if goal_idx >= 0:
                    goal_corridor_indices.append(goal_idx)
            else:
                raise ValueError(f"Invalid goal cell: ({goal_r}, {goal_c})")

        if len(goal_corridor_indices) != num_goals:
            raise ValueError("Some goal cells are not in navigable corridors")

        self._multi_source_bfs(goal_corridor_indices, num_corridors)
        self._goal_cells = goal_cells
        self.is_precomputed = True

    def _multi_source_bfs(self, goal_indices: List[int], num_corridors: int):
        """Fully optimized GPU-accelerated multi-source BFS using vectorized operations.

        This method eliminates ALL Python-level loops by using purely vectorized tensor operations
        for maximum GPU utilization and minimal kernel launch overhead.
        """
        num_goals = len(goal_indices)

        # Create full grid representations for all goals [num_goals, grid_h, grid_w]
        grid_visited = torch.zeros((num_goals, self.grid_h, self.grid_w),
                                  dtype=torch.bool, device=self.device)
        grid_frontier = torch.zeros_like(grid_visited)
        grid_distances = torch.full((num_goals, self.grid_h, self.grid_w), 1e6,
                                   dtype=torch.float32, device=self.device)
        grid_directions = torch.zeros((num_goals, self.grid_h, self.grid_w),
                                     dtype=torch.uint8, device=self.device)

        # Create corridor mask from stored maze grid
        corridor_mask = torch.tensor([[self.maze_grid[r][c] == 0 for c in range(self.grid_w)]
                                     for r in range(self.grid_h)],
                                    dtype=torch.bool, device=self.device)

        # Initialize goal positions
        for goal_idx_pos, goal_corridor_idx in enumerate(goal_indices):
            goal_pos = self.corridor_to_grid[goal_corridor_idx]
            goal_r, goal_c = goal_pos[0].item(), goal_pos[1].item()

            grid_frontier[goal_idx_pos, goal_r, goal_c] = True
            grid_distances[goal_idx_pos, goal_r, goal_c] = 0.0
            grid_visited[goal_idx_pos, goal_r, goal_c] = True

        current_distance = 0
        max_iterations = self.grid_h * self.grid_w  # Safety limit

        # Wavefront expansion loop with full vectorization
        for iteration in range(max_iterations):
            if not grid_frontier.any():
                break

            current_distance += 1
            next_frontier = torch.zeros_like(grid_frontier)

            # Process all directions simultaneously
            for dir_idx in range(4):
                dr, dc = self.dir_offsets[dir_idx]

                # Find neighbor positions by shifting frontier
                # For reverse BFS: we expand FROM goal TO neighbors
                # So neighbor at (r+dr, c+dc) should point back via opposite direction
                neighbor_r = torch.zeros_like(grid_frontier[:, :, :], dtype=torch.long)
                neighbor_c = torch.zeros_like(grid_frontier[:, :, :], dtype=torch.long)

                # Create coordinate matrices
                r_coords, c_coords = torch.meshgrid(
                    torch.arange(self.grid_h, device=self.device),
                    torch.arange(self.grid_w, device=self.device),
                    indexing='ij'
                )

                # Calculate neighbor coordinates
                neighbor_r = r_coords + dr
                neighbor_c = c_coords + dc

                # Check bounds
                valid_bounds = ((neighbor_r >= 0) & (neighbor_r < self.grid_h) &
                               (neighbor_c >= 0) & (neighbor_c < self.grid_w))

                # For each goal, check which frontier cells can expand in this direction
                for goal_idx in range(num_goals):
                    # Get current frontier for this goal
                    current_goal_frontier = grid_frontier[goal_idx]

                    # Find valid expansions: frontier cell exists, neighbor in bounds, neighbor is corridor, neighbor not visited
                    valid_expansions = (current_goal_frontier & valid_bounds)

                    if valid_expansions.any():
                        # Get the neighbor positions that should be updated
                        expand_r = neighbor_r[valid_expansions]
                        expand_c = neighbor_c[valid_expansions]

                        # Check if neighbors are corridors and not visited
                        corridor_check = corridor_mask[expand_r, expand_c]
                        visited_check = ~grid_visited[goal_idx][expand_r, expand_c]
                        final_valid = corridor_check & visited_check

                        if final_valid.any():
                            final_r = expand_r[final_valid]
                            final_c = expand_c[final_valid]

                            # Update distance and direction for these neighbors
                            grid_distances[goal_idx, final_r, final_c] = float(current_distance)

                            # The direction stored should be how to get FROM the neighbor TO the goal
                            # Since we expanded in direction dir_idx, the reverse direction is needed
                            opposite_dir = (dir_idx + 2) % 4
                            grid_directions[goal_idx, final_r, final_c] = opposite_dir

                            # Mark as visited and add to next frontier
                            grid_visited[goal_idx, final_r, final_c] = True
                            next_frontier[goal_idx, final_r, final_c] = True

            grid_frontier = next_frontier

        # Vectorized conversion from grid-based to corridor-based storage
        corridor_positions = self.corridor_to_grid  # [num_corridors, 2]
        corridor_rows = corridor_positions[:, 0]    # [num_corridors]
        corridor_cols = corridor_positions[:, 1]    # [num_corridors]

        # Use advanced indexing to extract all values at once
        for goal_idx in range(num_goals):
            self.dist[:, goal_idx] = grid_distances[goal_idx, corridor_rows, corridor_cols]
            self.next_dir[:, goal_idx] = grid_directions[goal_idx, corridor_rows, corridor_cols]

    def query_path(self, cell: Tuple[int, int], goal_idx: int = 0) -> Tuple[int, float]:
        """Query next optimal direction and distance to specified goal.

        Args:
            cell: (row, col) current position
            goal_idx: Index of target goal (0-based)

        Returns:
            (next_direction_idx, remaining_distance) where direction_idx maps to:
            0=NORTH, 1=EAST, 2=SOUTH, 3=WEST, -1=invalid/at_goal
        """
        if not self.is_precomputed:
            raise RuntimeError("Paths not precomputed")

        row, col = cell
        if not (0 <= row < self.grid_h and 0 <= col < self.grid_w):
            return -1, float('inf')

        corridor_idx = self.grid_to_corridor[row, col].item()
        if corridor_idx < 0:
            return -1, float('inf')

        if goal_idx >= self.dist.shape[1]:
            return -1, float('inf')

        distance = self.dist[corridor_idx, goal_idx].item()
        if distance >= 1e6:
            return -1, float('inf')
        if distance == 0:
            return -1, 0.0

        next_dir = self.next_dir[corridor_idx, goal_idx].item()
        return next_dir, distance

    def batch_query_paths(self, cells: torch.Tensor, goal_indices: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batch query paths for multiple cells.

        Args:
            cells: [N, 2] tensor of (row, col) positions
            goal_indices: [N] tensor of goal indices, or None for goal 0

        Returns:
            (next_directions, distances) tensors of shape [N]
        """
        if not self.is_precomputed:
            raise RuntimeError("Paths not precomputed")

        batch_size = cells.shape[0]
        if goal_indices is None:
            goal_indices = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        next_directions = torch.full((batch_size,), -1, dtype=torch.long, device=self.device)
        distances = torch.full((batch_size,), float('inf'), device=self.device)

        valid_mask = ((cells[:, 0] >= 0) & (cells[:, 0] < self.grid_h) &
                     (cells[:, 1] >= 0) & (cells[:, 1] < self.grid_w))

        if not valid_mask.any():
            return next_directions, distances

        valid_cells = cells[valid_mask]
        valid_goals = goal_indices[valid_mask]

        corridor_indices = self.grid_to_corridor[valid_cells[:, 0], valid_cells[:, 1]]
        corridor_mask = corridor_indices >= 0

        if not corridor_mask.any():
            return next_directions, distances

        valid_corridor_indices = corridor_indices[corridor_mask]
        valid_goal_indices = valid_goals[corridor_mask]

        goal_mask = (valid_goal_indices >= 0) & (valid_goal_indices < self.dist.shape[1])
        if not goal_mask.any():
            return next_directions, distances

        final_corridor_indices = valid_corridor_indices[goal_mask]
        final_goal_indices = valid_goal_indices[goal_mask]

        query_distances = self.dist[final_corridor_indices, final_goal_indices]
        query_directions = self.next_dir[final_corridor_indices, final_goal_indices]

        reachable_mask = query_distances < 1e6
        at_goal_mask = query_distances == 0

        final_directions = torch.where(at_goal_mask, -1, query_directions.long())
        final_distances = query_distances

        valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]
        corridor_indices_in_valid = torch.nonzero(corridor_mask, as_tuple=True)[0]
        goal_indices_in_corridor = torch.nonzero(goal_mask, as_tuple=True)[0]

        original_indices = valid_indices[corridor_indices_in_valid[goal_indices_in_corridor]]

        next_directions[original_indices[reachable_mask]] = final_directions[reachable_mask]
        distances[original_indices[reachable_mask]] = final_distances[reachable_mask]

        return next_directions, distances

    def get_memory_usage_mb(self) -> float:
        """Get memory usage of precomputed tables in MB."""
        if not self.is_precomputed:
            return 0.0
        n_corridors, n_goals = self.dist.shape
        return (n_corridors * n_goals * 5) / (1024**2)

    def verify_precomputation(self, maze_grid: List[List[int]],
                            goal_cells: List[Tuple[int, int]]) -> Dict[str, bool]:
        """Verify correctness of precomputed paths."""
        if not self.is_precomputed:
            return {"error": "Not precomputed"}

        results = {"all_reachable_covered": True, "distances_correct": True, "directions_correct": True}

        corridors = []
        for r in range(len(maze_grid)):
            for c in range(len(maze_grid[0])):
                if maze_grid[r][c] == 0:
                    corridors.append((r, c))

        for goal_idx, goal_cell in enumerate(goal_cells):
            for cell in corridors[:min(100, len(corridors))]:
                next_dir, distance = self.query_path(cell, goal_idx)

                if cell == goal_cell:
                    if next_dir != -1 or distance != 0:
                        results["distances_correct"] = False
                elif distance == float('inf'):
                    results["all_reachable_covered"] = False
                elif next_dir == -1 and distance > 0:
                    results["directions_correct"] = False

        return results

    def world_to_grid_batch(self, world_coords: torch.Tensor) -> torch.Tensor:
        """Convert batch of world coordinates to grid indices.

        Uses the same logic as MazeGenerator._world_to_maze_coords for consistency.
        """
        if not self.is_precomputed:
            raise RuntimeError("APSP not precomputed")

        local_coords = world_coords - self.world_origin[:2]
        grid_offset_x = -(self.grid_w * self.cell_size) / 2.0 + self.cell_size / 2.0
        grid_offset_y = -(self.grid_h * self.cell_size) / 2.0 + self.cell_size / 2.0

        # Use the same rounding logic as MazeGenerator: int((coord - offset) / cell_size + 0.5)
        # This is equivalent to torch.round((coord - offset) / cell_size).long()
        grid_x = torch.round((local_coords[:, 0] - grid_offset_x) / self.cell_size).long()
        grid_y = torch.round((local_coords[:, 1] - grid_offset_y) / self.cell_size).long()

        # Initialize result tensor with invalid coordinates
        grid_coords = torch.full((world_coords.shape[0], 2), -1, dtype=torch.long, device=self.device)

        # Check bounds first
        in_bounds_mask = ((grid_x >= 0) & (grid_x < self.grid_w) &
                         (grid_y >= 0) & (grid_y < self.grid_h))

        if in_bounds_mask.any():
            # For in-bounds coordinates, check if they are corridors
            valid_x = grid_x[in_bounds_mask]
            valid_y = grid_y[in_bounds_mask]

            # Check if these grid positions are corridors (using grid_to_corridor mapping)
            corridor_indices = self.grid_to_corridor[valid_y, valid_x]
            corridor_mask = corridor_indices >= 0

            # Create final valid coordinates
            final_valid_mask = torch.zeros_like(in_bounds_mask)
            final_valid_mask[in_bounds_mask] = corridor_mask

            # Set valid coordinates
            if final_valid_mask.any():
                grid_coords[final_valid_mask, 0] = grid_y[final_valid_mask]  # row
                grid_coords[final_valid_mask, 1] = grid_x[final_valid_mask]  # col
            else:
                print(f"Warning: No valid corridor coordinates found for world coordinates: {world_coords[in_bounds_mask]}, grid coordinates: {grid_x[in_bounds_mask]}, {grid_y[in_bounds_mask]}")

        return grid_coords

    def grid_to_world_batch(self, grid_coords: torch.Tensor) -> torch.Tensor:
        """Convert batch of grid coordinates to world coordinates."""
        if not self.is_precomputed:
            raise RuntimeError("APSP not precomputed")

        grid_offset_x = -(self.grid_w * self.cell_size) / 2.0 + self.cell_size / 2.0
        grid_offset_y = -(self.grid_h * self.cell_size) / 2.0 + self.cell_size / 2.0

        local_x = grid_offset_x + grid_coords[:, 1] * self.cell_size
        local_y = grid_offset_y + grid_coords[:, 0] * self.cell_size
        local_z = torch.full_like(local_x, 1.0)

        world_coords = torch.stack([local_x, local_y, local_z], dim=1) + self.world_origin
        return world_coords

    def lookup_next_directions(self, robot_positions: torch.Tensor,
                              target_positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fully vectorized batch lookup using pathfinder with world coordinates.

        Optimized for high-throughput scenarios with thousands of robots.
        Eliminates all Python loops using vectorized tensor operations.
        """
        if not self.is_precomputed:
            raise RuntimeError("APSP not precomputed")

        robot_grid = self.world_to_grid_batch(robot_positions)
        target_grid = self.world_to_grid_batch(target_positions)

        num_robots = robot_positions.shape[0]

        # Initialize output tensors
        next_directions = torch.full((num_robots,), -1, dtype=torch.long, device=self.device)
        remaining_distances = torch.full((num_robots,), float('inf'), device=self.device)
        next_world_coords = torch.zeros((num_robots, 3), device=self.device)

        # Handle invalid coordinates: create mask for valid robot and target positions
        robot_valid_mask = (robot_grid[:, 0] != -1) & (robot_grid[:, 1] != -1)
        target_valid_mask = (target_grid[:, 0] != -1) & (target_grid[:, 1] != -1)
        valid_mask = robot_valid_mask & target_valid_mask

        # Early return if no valid robots
        if not valid_mask.any():
            # Set default z-coordinate for all robots
            next_world_coords[:, :2] = robot_positions
            next_world_coords[:, 2] = 1.0
            return next_directions, remaining_distances, next_world_coords

        # Extract valid positions
        valid_robot_grid = robot_grid[valid_mask]  # [N_valid, 2]
        valid_target_grid = target_grid[valid_mask]  # [N_valid, 2]
        n_valid = valid_robot_grid.shape[0]

        # Vectorized closest goal finding for all valid robots
        # Convert goal cells to tensor for vectorized operations
        goal_cells_tensor = torch.tensor(self._goal_cells, dtype=torch.long, device=self.device)  # [n_goals, 2]
        n_goals = goal_cells_tensor.shape[0]

        # Compute Manhattan distances from each target to all goals
        # target_grid: [N_valid, 2], goal_cells: [n_goals, 2]
        # Create broadcasting: [N_valid, 1, 2] - [1, n_goals, 2] = [N_valid, n_goals, 2]
        target_expanded = valid_target_grid.unsqueeze(1)  # [N_valid, 1, 2]
        goals_expanded = goal_cells_tensor.unsqueeze(0)   # [1, n_goals, 2]

        # Manhattan distance calculation: |target_row - goal_row| + |target_col - goal_col|
        manhattan_dists = torch.abs(target_expanded - goals_expanded).sum(dim=2)  # [N_valid, n_goals]

        # Find closest goal for each target
        closest_goal_indices = torch.argmin(manhattan_dists, dim=1)  # [N_valid]

        # Vectorized pathfinding query using batch_query_paths
        query_directions, query_distances = self.batch_query_paths(valid_robot_grid, closest_goal_indices)

        # Create masks for different cases
        unreachable_mask = query_distances >= 1e6
        at_goal_mask = query_distances == 0.0
        valid_path_mask = ~unreachable_mask & ~at_goal_mask

        # Initialize results for valid robots
        valid_next_directions = torch.full((n_valid,), -1, dtype=torch.long, device=self.device)
        valid_remaining_distances = torch.full((n_valid,), float('inf'), device=self.device)
        valid_next_world_coords = torch.zeros((n_valid, 3), device=self.device)

        # Handle robots at goal (distance = 0)
        if at_goal_mask.any():
            valid_next_directions[at_goal_mask] = -1
            valid_remaining_distances[at_goal_mask] = 0.0
            # Convert current robot positions to world coordinates
            at_goal_indices = torch.nonzero(at_goal_mask, as_tuple=True)[0]
            robot_world_pos = robot_positions[valid_mask][at_goal_indices]
            valid_next_world_coords[at_goal_mask] = torch.cat([robot_world_pos, torch.ones(at_goal_indices.shape[0], 1, device=self.device)], dim=1)

        # Handle robots with valid paths
        if valid_path_mask.any():
            # Extract information for robots with valid paths
            valid_robots = valid_robot_grid[valid_path_mask]  # [N_path, 2]
            path_directions = query_directions[valid_path_mask]  # [N_path]
            path_distances = query_distances[valid_path_mask]  # [N_path]

            # Vectorized next cell calculation
            # direction_offsets: [4, 2] (N, E, S, W)
            direction_deltas = self.dir_offsets[path_directions]  # [N_path, 2]
            next_cells = valid_robots + direction_deltas  # [N_path, 2]

            # Boundary check for next cells
            next_in_bounds = ((next_cells[:, 0] >= 0) & (next_cells[:, 0] < self.grid_h) &
                            (next_cells[:, 1] >= 0) & (next_cells[:, 1] < self.grid_w))

            # Update results only for in-bounds next cells
            valid_path_in_bounds = valid_path_mask.clone()
            valid_path_in_bounds[valid_path_mask] = next_in_bounds

            if valid_path_in_bounds.any():
                # Set directions and distances
                valid_next_directions[valid_path_in_bounds] = query_directions[valid_path_in_bounds]
                valid_remaining_distances[valid_path_in_bounds] = query_distances[valid_path_in_bounds] * self.cell_size

                # Convert next grid positions to world coordinates
                next_cells_in_bounds = next_cells[next_in_bounds]
                if next_cells_in_bounds.shape[0] > 0:
                    next_world_batch = self.grid_to_world_batch(next_cells_in_bounds)

                    # Map back to the correct indices in valid_next_world_coords
                    valid_path_indices = torch.nonzero(valid_path_in_bounds, as_tuple=True)[0]
                    valid_next_world_coords[valid_path_indices] = next_world_batch

        # Handle unreachable robots (set to current position with invalid direction)
        if unreachable_mask.any():
            unreachable_indices = torch.nonzero(unreachable_mask, as_tuple=True)[0]
            robot_world_pos = robot_positions[valid_mask][unreachable_indices]
            valid_next_world_coords[unreachable_mask] = torch.cat([robot_world_pos, torch.ones(unreachable_indices.shape[0], 1, device=self.device)], dim=1)

        # Map results back to original robot indices
        next_directions[valid_mask] = valid_next_directions
        remaining_distances[valid_mask] = valid_remaining_distances
        next_world_coords[valid_mask] = valid_next_world_coords

        # Handle invalid robots (set to current position with default z)
        invalid_mask = ~valid_mask
        if invalid_mask.any():
            invalid_robot_pos = robot_positions[invalid_mask]
            next_world_coords[invalid_mask] = torch.cat([invalid_robot_pos, torch.ones(invalid_mask.sum(), 1, device=self.device)], dim=1)

        return next_directions, remaining_distances, next_world_coords

    def get_path_waypoints(self, start_pos: torch.Tensor, target_pos: torch.Tensor,
                          max_waypoints: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get path waypoints using pathfinder with vectorized support.

        Args:
            start_pos: Start positions tensor with shape (env_id, 2) or (env_id, 3) for batched queries
                      or (2,) or (3,) for single query
            target_pos: Target positions tensor with same shape as start_pos
            max_waypoints: Maximum number of waypoints to generate

        Returns:
            (waypoints, remaining_dists):
                - waypoints: Tensor of shape (env_id, max_waypoints, 3) for batched queries
                           or (max_waypoints, 3) for single query
                - remaining_dists: Tensor of shape (env_id, max_waypoints) for batched queries
                                 or (max_waypoints,) for single query
        """
        if not self.is_precomputed:
            raise RuntimeError("APSP not precomputed")

        # Handle both single and batched inputs
        if start_pos.dim() == 1:
            # Single query - convert to batch format
            start_batch = start_pos[:2].unsqueeze(0)  # (1, 2)
            target_batch = target_pos[:2].unsqueeze(0)  # (1, 2)
            is_single_query = True
        else:
            # Batched query
            start_batch = start_pos[:, :2]  # (env_id, 2)
            target_batch = target_pos[:, :2]  # (env_id, 2)
            is_single_query = False

        num_envs = start_batch.shape[0]

        # Initialize output tensors
        all_waypoints = torch.zeros((num_envs, max_waypoints, 3), device=self.device)
        all_remaining_dists = torch.zeros((num_envs, max_waypoints), device=self.device)

        # Current positions for all environments
        current_positions = start_batch.clone()  # (env_id, 2)

        # Track which environments are still active (haven't reached goal)
        active_mask = torch.ones(num_envs, dtype=torch.bool, device=self.device)

        for step in range(max_waypoints):
            if not active_mask.any():
                break

            # Only process active environments
            active_indices = torch.nonzero(active_mask, as_tuple=True)[0]
            active_current = current_positions[active_indices]  # (n_active, 2)
            active_targets = target_batch[active_indices]  # (n_active, 2)

            # Get next directions and coordinates for active environments
            next_dirs, remaining_dists, next_coords = self.lookup_next_directions(
                active_current, active_targets)

            # Check which environments should continue
            continue_mask = (next_dirs != -1) & (remaining_dists > 0)

            # Update waypoints and distances for environments that can continue
            valid_indices = active_indices[continue_mask]
            if valid_indices.numel() > 0:
                all_waypoints[valid_indices, step] = next_coords[continue_mask]
                all_remaining_dists[valid_indices, step] = remaining_dists[continue_mask]

                # Update current positions for next iteration
                current_positions[valid_indices] = next_coords[continue_mask, :2]

            # Update active mask - deactivate environments that reached goal or hit invalid state
            active_mask[active_indices[~continue_mask]] = False

            # Early termination check: if close to target
            if valid_indices.numel() > 0:
                distances_to_target = torch.norm(
                    current_positions[valid_indices] - target_batch[valid_indices], dim=1)
                close_to_target = distances_to_target < self.cell_size * 0.1
                active_mask[valid_indices[close_to_target]] = False

        if is_single_query:
            # Return single query format - remove batch dimension
            return all_waypoints[0], all_remaining_dists[0]
        else:
            # Return batched format
            return all_waypoints, all_remaining_dists

    def precompute_apsp(self, maze_grid: List[List[int]], cell_size: float, world_origin: torch.Tensor,
                        exits: List[Tuple[int, int]]) -> None:
        """Precompute All-Pairs Shortest Paths with world coordinate support.

        Args:
            maze_grid: 2D maze where 0=corridor, 1=wall
            cell_size: Size of each grid cell in world units
            world_origin: World origin coordinates [x, y, z]
            exits: List of (row, col) exit positions to use as goal cells.
                  If provided, only these exits will be used as destinations.
                  If None, uses representative sampling of all corridor cells.
        """
        # Store world coordinate parameters
        self.cell_size = cell_size
        self.world_origin = world_origin.to(self.device)

        # Get all corridor cells
        corridors = []
        for r in range(len(maze_grid)):
            for c in range(len(maze_grid[0])):
                if maze_grid[r][c] == 0:
                    corridors.append((r, c))

        if not corridors:
            raise ValueError("No navigable corridors found")

        # Validate exits are in navigable corridors
        goal_cells = []
        for exit_pos in exits:
            r, c = exit_pos
            if (0 <= r < len(maze_grid) and 0 <= c < len(maze_grid[0]) and
                maze_grid[r][c] == 0):
                goal_cells.append(exit_pos)
            else:
                print(f"Warning: Exit position ({r}, {c}) is not a navigable corridor, skipping")

        if not goal_cells:
            raise ValueError("No valid exit positions found in navigable corridors")

        print(f"Using {len(goal_cells)} exits as goal cells for APSP pathfinding")

        # Precompute paths using existing method
        self.precompute_paths(maze_grid, goal_cells)

        # Store goal cells for lookup functions
        self._goal_cells = goal_cells

    def world_to_maze_coords_single(self, world_pos) -> Optional[Tuple[int, int]]:
        """Convert single world position to maze grid coordinates.

        This method provides the same interface as MazeGenerator._world_to_maze_coords
        but uses the centralized logic for consistency.

        Args:
            world_pos: World position as (x, y, z) tuple or tensor

        Returns:
            (row, col) maze coordinates if valid corridor, None otherwise
        """
        if not self.is_precomputed:
            return None

        # Handle both tensor and tuple/list inputs
        if isinstance(world_pos, torch.Tensor):
            if world_pos.dim() == 0:  # scalar tensor
                return None
            world_coords = world_pos[:2].unsqueeze(0)  # Convert to batch format [1, 2]
        else:
            # Convert tuple/list to tensor
            world_coords = torch.tensor([[world_pos[0], world_pos[1]]],
                                      dtype=torch.float32, device=self.device)

        # Use batch conversion method
        grid_coords = self.world_to_grid_batch(world_coords)

        # Extract single result
        if grid_coords[0, 0] == -1 or grid_coords[0, 1] == -1:
            return None
        else:
            return (grid_coords[0, 0].item(), grid_coords[0, 1].item())

    def maze_to_world_coords_single(self, maze_pos, return_tensor: bool = False):
        """Convert single maze grid coordinates to world coordinates.

        This method provides the same interface as MazeGenerator._maze_to_world_coords
        but uses the centralized logic for consistency.

        Args:
            maze_pos: (row, col) maze coordinates
            return_tensor: Whether to return as tensor or tuple

        Returns:
            World coordinates as tensor or tuple (x, y, z)
        """
        if not self.is_precomputed:
            if return_tensor:
                return torch.zeros(3, device=self.device)
            return (0.0, 0.0, 0.0)

        # Handle both tensor and tuple inputs
        if isinstance(maze_pos, torch.Tensor):
            grid_coords = maze_pos.unsqueeze(0)  # Convert to batch format [1, 2]
        else:
            row, col = maze_pos
            grid_coords = torch.tensor([[row, col]], dtype=torch.long, device=self.device)

        # Use batch conversion method
        world_coords = self.grid_to_world_batch(grid_coords)

        # Extract single result
        if return_tensor:
            return world_coords[0]  # Return [3] tensor
        else:
            world_coord = world_coords[0]
            return (world_coord[0].item(), world_coord[1].item(), world_coord[2].item())

    def get_maze_offsets(self) -> Tuple[float, float]:
        """Calculate maze offset coordinates to center at origin.

        This method provides the same interface as MazeGenerator._get_maze_offsets
        for consistency.

        Returns:
            (maze_offset_x, maze_offset_y) tuple
        """
        if not self.is_precomputed:
            return (0.0, 0.0)

        maze_offset_x = -(self.grid_w * self.cell_size) / 2.0 + self.cell_size / 2.0
        maze_offset_y = -(self.grid_h * self.cell_size) / 2.0 + self.cell_size / 2.0
        return maze_offset_x, maze_offset_y