"""Interactive maze visualization GUI using Tkinter."""

import tkinter as tk
from tkinter import ttk, messagebox
import math
import time
from typing import List, Tuple, Optional, Dict, Set
import torch

from maze import MazeAlgorithm, PathPrecomputer


class MazeVisualizerGUI:
    """Interactive GUI for visualizing maze generation and pathfinding."""

    def __init__(self, master: tk.Tk):
        self.master = master
        self.master.title("Maze Algorithm Visualizer")
        self.master.geometry("1200x800")

        # Core components
        self.maze_algo = None
        self.path_precomputer = PathPrecomputer(device="cpu")  # Use CPU for GUI
        self.maze_grid = None
        self.precompute_goals = []  # Fixed goals for precomputation (exits)
        self.selected_goal = None   # User-selected goal for visualization
        self.start_cell = None
        self.path_waypoints = []

        # UI state
        self.cell_size = 20  # Pixel size for each cell
        self.canvas_offset_x = 10
        self.canvas_offset_y = 10
        self.is_paths_precomputed = False
        self.selected_mode = "start"  # "start" or "goal"

        # Color scheme
        self.colors = {
            'wall': '#2c2c2c',
            'corridor': '#f0f0f0',
            'precompute_goal': '#ff4444',    # Red: Fixed goals for precomputation
            'selected_goal': '#ff9900',      # Orange: User-selected goal for visualization
            'start': '#44ff44',              # Green: Start point
            'path': '#0099ff',               # Blue: Path line
            'entrance': '#4444ff',           # Blue: Entrance points
            'exit': '#ff44ff'                # Purple: Exit points
        }

        self._setup_ui()
        self._bind_events()

    def _setup_ui(self):
        """Setup the user interface."""
        # Main container
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel for controls
        control_frame = ttk.Frame(main_frame, width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)

        # Right panel for canvas
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._setup_control_panel(control_frame)
        self._setup_canvas(canvas_frame)
        self._setup_status_bar()

    def _setup_control_panel(self, parent):
        """Setup the control panel with parameters and buttons."""
        # Parameters section
        params_frame = ttk.LabelFrame(parent, text="Maze Parameters", padding=10)
        params_frame.pack(fill=tk.X, pady=(0, 10))

        # Grid dimensions
        ttk.Label(params_frame, text="Rows (odd):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.rows_var = tk.StringVar(value="21")
        ttk.Entry(params_frame, textvariable=self.rows_var, width=10).grid(row=0, column=1, sticky=tk.W, pady=2)

        ttk.Label(params_frame, text="Columns (odd):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.cols_var = tk.StringVar(value="21")
        ttk.Entry(params_frame, textvariable=self.cols_var, width=10).grid(row=1, column=1, sticky=tk.W, pady=2)

        # Algorithm parameters
        ttk.Label(params_frame, text="Loop Density:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.loop_density_var = tk.DoubleVar(value=0.15)
        ttk.Scale(params_frame, from_=0.0, to=1.0, variable=self.loop_density_var,
                 orient=tk.HORIZONTAL, length=150).grid(row=2, column=1, sticky=tk.W, pady=2)

        ttk.Label(params_frame, text="Branching Prob:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.branching_var = tk.DoubleVar(value=0.4)
        ttk.Scale(params_frame, from_=0.0, to=1.0, variable=self.branching_var,
                 orient=tk.HORIZONTAL, length=150).grid(row=3, column=1, sticky=tk.W, pady=2)

        ttk.Label(params_frame, text="Bias Factor:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.bias_var = tk.DoubleVar(value=0.3)
        ttk.Scale(params_frame, from_=0.0, to=1.0, variable=self.bias_var,
                 orient=tk.HORIZONTAL, length=150).grid(row=4, column=1, sticky=tk.W, pady=2)

        # Entrances and exits
        ttk.Label(params_frame, text="Entrances:").grid(row=5, column=0, sticky=tk.W, pady=2)
        self.entrances_var = tk.StringVar(value="2")
        ttk.Entry(params_frame, textvariable=self.entrances_var, width=10).grid(row=5, column=1, sticky=tk.W, pady=2)

        ttk.Label(params_frame, text="Exits:").grid(row=6, column=0, sticky=tk.W, pady=2)
        self.exits_var = tk.StringVar(value="2")
        ttk.Entry(params_frame, textvariable=self.exits_var, width=10).grid(row=6, column=1, sticky=tk.W, pady=2)

        # Operation buttons
        buttons_frame = ttk.LabelFrame(parent, text="Operations", padding=10)
        buttons_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(buttons_frame, text="Generate Maze",
                  command=self._generate_maze).pack(fill=tk.X, pady=2)
        ttk.Button(buttons_frame, text="Precompute Paths",
                  command=self._precompute_paths).pack(fill=tk.X, pady=2)
        ttk.Button(buttons_frame, text="Clear Paths",
                  command=self._clear_paths).pack(fill=tk.X, pady=2)
        ttk.Button(buttons_frame, text="Reset All",
                  command=self._reset_all).pack(fill=tk.X, pady=2)

        # Selection mode
        mode_frame = ttk.LabelFrame(parent, text="Selection Mode", padding=10)
        mode_frame.pack(fill=tk.X, pady=(0, 10))

        self.mode_var = tk.StringVar(value="start")
        ttk.Radiobutton(mode_frame, text="Select Start Point", variable=self.mode_var,
                       value="start").pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="Select Target Goal", variable=self.mode_var,
                       value="goal").pack(anchor=tk.W)

        # Goal selection dropdown
        goal_frame = ttk.LabelFrame(parent, text="Target Goal Selection", padding=10)
        goal_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(goal_frame, text="Select Goal:").pack(anchor=tk.W)
        self.goal_var = tk.StringVar()
        self.goal_combobox = ttk.Combobox(goal_frame, textvariable=self.goal_var,
                                         state="readonly", width=25)
        self.goal_combobox.pack(fill=tk.X, pady=2)
        self.goal_combobox.bind("<<ComboboxSelected>>", self._on_goal_selected)

        # Path visualization
        path_frame = ttk.LabelFrame(parent, text="Path Visualization", padding=10)
        path_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(path_frame, text="Show Path from Start",
                  command=self._show_path).pack(fill=tk.X, pady=2)
        ttk.Button(path_frame, text="Clear Path Visualization",
                  command=self._clear_path_vis).pack(fill=tk.X, pady=2)

        # Statistics
        stats_frame = ttk.LabelFrame(parent, text="Statistics", padding=10)
        stats_frame.pack(fill=tk.X, pady=(0, 10))

        self.stats_text = tk.Text(stats_frame, height=8, width=30, font=("Courier", 9))
        self.stats_text.pack(fill=tk.BOTH, expand=True)

    def _setup_canvas(self, parent):
        """Setup the maze visualization canvas."""
        # Canvas with scrollbars
        canvas_container = ttk.Frame(parent)
        canvas_container.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_container, bg='white', scrollregion=(0, 0, 1000, 1000))

        # Scrollbars
        v_scrollbar = ttk.Scrollbar(canvas_container, orient=tk.VERTICAL, command=self.canvas.yview)
        h_scrollbar = ttk.Scrollbar(canvas_container, orient=tk.HORIZONTAL, command=self.canvas.xview)

        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        # Pack scrollbars and canvas
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def _setup_status_bar(self):
        """Setup status bar at the bottom."""
        self.status_var = tk.StringVar(value="Click 'Generate Maze' to start")
        status_bar = ttk.Label(self.master, textvariable=self.status_var,
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _bind_events(self):
        """Bind mouse events to canvas."""
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Button-3>", self._on_canvas_right_click)
        self.canvas.bind("<Motion>", self._on_canvas_motion)

    def _generate_maze(self):
        """Generate a new maze with current parameters."""
        try:
            rows = int(self.rows_var.get())
            cols = int(self.cols_var.get())

            if rows % 2 == 0 or cols % 2 == 0:
                messagebox.showerror("Error", "Rows and columns must be odd numbers")
                return

            loop_density = self.loop_density_var.get()
            branching_prob = self.branching_var.get()
            bias_factor = self.bias_var.get()
            num_entrances = int(self.entrances_var.get())
            num_exits = int(self.exits_var.get())

            # Generate maze
            self.maze_algo = MazeAlgorithm(
                rows=rows, cols=cols,
                loop_density=loop_density,
                branching_prob=branching_prob,
                bias_factor=bias_factor,
                num_entrances=num_entrances,
                num_exits=num_exits
            )

            start_time = time.time()
            self.maze_grid = self.maze_algo.generate()
            generation_time = time.time() - start_time

            # Reset state
            self.precompute_goals = []
            self.selected_goal = None
            self.start_cell = None
            self.path_waypoints = []
            self.is_paths_precomputed = False

            # Add maze exits as precompute goals
            if hasattr(self.maze_algo, 'exits') and self.maze_algo.exits:
                self.precompute_goals = self.maze_algo.exits.copy()

            # Update goal selection dropdown
            self._update_goal_dropdown()

            # Update canvas
            self._draw_maze()
            self._update_statistics(generation_time=generation_time)
            self.status_var.set(f"Maze generated: {rows}x{cols}, {len(self.precompute_goals)} goals available")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate maze: {str(e)}")

    def _precompute_paths(self):
        """Precompute shortest paths to all goal cells."""
        if self.maze_grid is None:
            messagebox.showerror("Error", "Generate a maze first")
            return

        if not self.precompute_goals:
            messagebox.showerror("Error", "No goals available for precomputation")
            return

        try:
            start_time = time.time()
            self.path_precomputer.precompute_paths(self.maze_grid, self.precompute_goals)
            precompute_time = time.time() - start_time

            self.is_paths_precomputed = True

            # Redraw with path visualization
            self._draw_maze()
            if self.selected_goal is not None:
                self._draw_path_arrows()
                self._draw_distance_heatmap()

            self._update_statistics(precompute_time=precompute_time)
            self.status_var.set(f"Paths precomputed to {len(self.precompute_goals)} goals in {precompute_time:.3f}s")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to precompute paths: {str(e)}")

    def _clear_paths(self):
        """Clear path precomputation and visualization."""
        self.is_paths_precomputed = False
        self.path_waypoints = []
        if self.maze_grid is not None:
            self._draw_maze()
        self.status_var.set("Paths cleared")

    def _reset_all(self):
        """Reset everything to initial state."""
        self.maze_algo = None
        self.maze_grid = None
        self.precompute_goals = []
        self.selected_goal = None
        self.start_cell = None
        self.path_waypoints = []
        self.is_paths_precomputed = False

        self.canvas.delete("all")
        self.stats_text.delete(1.0, tk.END)
        self._update_goal_dropdown()
        self.status_var.set("Reset complete. Click 'Generate Maze' to start")

    def _draw_maze(self):
        """Draw the maze grid on canvas."""
        if self.maze_grid is None:
            return

        self.canvas.delete("all")

        rows, cols = len(self.maze_grid), len(self.maze_grid[0])
        canvas_width = cols * self.cell_size + 2 * self.canvas_offset_x
        canvas_height = rows * self.cell_size + 2 * self.canvas_offset_y

        # Update scroll region
        self.canvas.configure(scrollregion=(0, 0, canvas_width, canvas_height))

        # Draw grid
        for r in range(rows):
            for c in range(cols):
                x1 = self.canvas_offset_x + c * self.cell_size
                y1 = self.canvas_offset_y + r * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size

                if self.maze_grid[r][c] == 1:  # Wall
                    color = self.colors['wall']
                else:  # Corridor
                    color = self.colors['corridor']

                # Special cell types (order matters for color priority)
                if (r, c) == self.start_cell:
                    color = self.colors['start']
                elif (r, c) == self.selected_goal:
                    color = self.colors['selected_goal']
                elif (r, c) in self.precompute_goals:
                    color = self.colors['precompute_goal']
                elif hasattr(self.maze_algo, 'entrances') and (r, c) in self.maze_algo.entrances:
                    color = self.colors['entrance']

                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline='gray',
                                           tags=f"cell_{r}_{c}")

        # Draw path if exists
        if self.path_waypoints:
            self._draw_path_line()

    def _draw_distance_heatmap(self):
        """Draw distance heatmap overlay for selected goal."""
        if not self.is_paths_precomputed or self.maze_grid is None or self.selected_goal is None:
            return

        # Find goal index
        try:
            goal_idx = self.precompute_goals.index(self.selected_goal)
        except ValueError:
            return

        rows, cols = len(self.maze_grid), len(self.maze_grid[0])

        # Get max distance for normalization
        max_dist = 0
        for r in range(rows):
            for c in range(cols):
                if self.maze_grid[r][c] == 0:  # Corridor
                    _, dist = self.path_precomputer.query_path((r, c), goal_idx)
                    if dist != float('inf') and dist > max_dist:
                        max_dist = dist

        if max_dist == 0:
            return

        # Draw heatmap
        for r in range(rows):
            for c in range(cols):
                if self.maze_grid[r][c] == 0:  # Corridor
                    _, dist = self.path_precomputer.query_path((r, c), goal_idx)
                    if dist != float('inf'):
                        # Calculate heat intensity (0-255)
                        intensity = int(255 * (1 - dist / max_dist))
                        alpha_color = f"#{intensity:02x}{intensity:02x}ff"

                        x1 = self.canvas_offset_x + c * self.cell_size + 2
                        y1 = self.canvas_offset_y + r * self.cell_size + 2
                        x2 = x1 + self.cell_size - 4
                        y2 = y1 + self.cell_size - 4

                        self.canvas.create_rectangle(x1, y1, x2, y2, fill=alpha_color,
                                                   outline="", stipple="gray25",
                                                   tags="heatmap")

    def _draw_path_arrows(self):
        """Draw direction arrows for each corridor cell to selected goal."""
        if not self.is_paths_precomputed or self.maze_grid is None or self.selected_goal is None:
            return

        # Find goal index
        try:
            goal_idx = self.precompute_goals.index(self.selected_goal)
        except ValueError:
            return

        rows, cols = len(self.maze_grid), len(self.maze_grid[0])
        arrow_directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # N, E, S, W

        for r in range(rows):
            for c in range(cols):
                if self.maze_grid[r][c] == 0:  # Corridor
                    next_dir, dist = self.path_precomputer.query_path((r, c), goal_idx)

                    if next_dir >= 0 and dist > 0:  # Valid direction
                        center_x = self.canvas_offset_x + c * self.cell_size + self.cell_size // 2
                        center_y = self.canvas_offset_y + r * self.cell_size + self.cell_size // 2

                        # Calculate arrow endpoints
                        dr, dc = arrow_directions[next_dir]
                        arrow_length = self.cell_size // 3

                        end_x = center_x + dc * arrow_length
                        end_y = center_y + dr * arrow_length

                        # Draw arrow
                        self.canvas.create_line(center_x, center_y, end_x, end_y,
                                              arrow=tk.LAST, fill='red', width=2,
                                              arrowshape=(8, 10, 3), tags="arrows")

    def _show_path(self):
        """Show path from start cell to selected goal."""
        if not self.is_paths_precomputed:
            messagebox.showerror("Error", "Precompute paths first")
            return

        if self.start_cell is None:
            messagebox.showerror("Error", "Select a start cell first")
            return

        if self.selected_goal is None:
            messagebox.showerror("Error", "Select a target goal first")
            return

        # Find goal index
        try:
            goal_idx = self.precompute_goals.index(self.selected_goal)
        except ValueError:
            messagebox.showerror("Error", "Selected goal is not valid")
            return

        # Generate path waypoints
        self.path_waypoints = []
        current_cell = self.start_cell
        max_steps = 1000  # Prevent infinite loops

        # Check if start and goal are the same
        if current_cell == self.selected_goal:
            self.status_var.set("Start and goal are the same position!")
            return

        for step in range(max_steps):
            next_dir, dist = self.path_precomputer.query_path(current_cell, goal_idx)

            if next_dir == -1:  # Reached goal or invalid
                if current_cell == self.selected_goal:
                    break  # Successfully reached goal
                else:
                    # No valid path
                    self.status_var.set(f"No valid path found after {step} steps")
                    break

            # Move to next cell
            dr, dc = [(-1, 0), (0, 1), (1, 0), (0, -1)][next_dir]
            next_r, next_c = current_cell[0] + dr, current_cell[1] + dc

            if (0 <= next_r < len(self.maze_grid) and
                0 <= next_c < len(self.maze_grid[0]) and
                self.maze_grid[next_r][next_c] == 0):

                self.path_waypoints.append((next_r, next_c))
                current_cell = (next_r, next_c)
            else:
                self.status_var.set(f"Invalid move from {current_cell} to ({next_r}, {next_c}) at step {step}")
                break

        # Redraw maze with path
        self._draw_maze()
        if self.is_paths_precomputed and self.selected_goal is not None:
            self._draw_path_arrows()
            self._draw_distance_heatmap()

        self.status_var.set(f"Path shown: {len(self.path_waypoints)} steps to goal {self.selected_goal}")

    def _draw_path_line(self):
        """Draw the path line connecting waypoints."""
        if len(self.path_waypoints) == 0:
            return

        points = []

        # Start point
        if self.start_cell:
            start_x = self.canvas_offset_x + self.start_cell[1] * self.cell_size + self.cell_size // 2
            start_y = self.canvas_offset_y + self.start_cell[0] * self.cell_size + self.cell_size // 2
            points.extend([start_x, start_y])

        # Waypoints
        for r, c in self.path_waypoints:
            x = self.canvas_offset_x + c * self.cell_size + self.cell_size // 2
            y = self.canvas_offset_y + r * self.cell_size + self.cell_size // 2
            points.extend([x, y])

        # End at goal if we have waypoints
        if self.path_waypoints and self.selected_goal:
            goal_x = self.canvas_offset_x + self.selected_goal[1] * self.cell_size + self.cell_size // 2
            goal_y = self.canvas_offset_y + self.selected_goal[0] * self.cell_size + self.cell_size // 2
            points.extend([goal_x, goal_y])

        if len(points) >= 4:
            self.canvas.create_line(points, fill=self.colors['path'], width=3,
                                  smooth=True, tags="path_line")

    def _clear_path_vis(self):
        """Clear path visualization."""
        self.path_waypoints = []
        if self.maze_grid is not None:
            self._draw_maze()
            if self.is_paths_precomputed and self.selected_goal is not None:
                self._draw_path_arrows()
                self._draw_distance_heatmap()
        self.status_var.set("Path visualization cleared")

    def _on_canvas_click(self, event):
        """Handle left mouse click on canvas."""
        if self.maze_grid is None:
            return

        # Convert canvas coordinates to grid coordinates
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        col = int((canvas_x - self.canvas_offset_x) // self.cell_size)
        row = int((canvas_y - self.canvas_offset_y) // self.cell_size)

        rows, cols = len(self.maze_grid), len(self.maze_grid[0])

        if (0 <= row < rows and 0 <= col < cols and
            self.maze_grid[row][col] == 0):  # Valid corridor

            mode = self.mode_var.get()

            if mode == "start":
                self.start_cell = (row, col)
                self.status_var.set(f"Start set to ({row}, {col})")

            elif mode == "goal":
                # Only allow selection from precompute goals
                if (row, col) in self.precompute_goals:
                    self.selected_goal = (row, col)
                    self.goal_var.set(f"Goal {self.precompute_goals.index((row, col))}: ({row}, {col})")
                    self.status_var.set(f"Selected goal at ({row}, {col})")

                    # Update visualization if paths are precomputed
                    if self.is_paths_precomputed:
                        self._draw_maze()
                        self._draw_path_arrows()
                        self._draw_distance_heatmap()
                else:
                    self.status_var.set(f"({row}, {col}) is not a valid goal. Select from available goals.")

            self._draw_maze()
            if self.is_paths_precomputed and self.selected_goal is not None:
                self._draw_path_arrows()
                self._draw_distance_heatmap()

    def _on_canvas_right_click(self, event):
        """Handle right mouse click on canvas."""
        # Right click sets start point regardless of mode
        if self.maze_grid is None:
            return

        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        col = int((canvas_x - self.canvas_offset_x) // self.cell_size)
        row = int((canvas_y - self.canvas_offset_y) // self.cell_size)

        rows, cols = len(self.maze_grid), len(self.maze_grid[0])

        if (0 <= row < rows and 0 <= col < cols and
            self.maze_grid[row][col] == 0):  # Valid corridor

            self.start_cell = (row, col)
            self.status_var.set(f"Start set to ({row}, {col}) via right-click")

            self._draw_maze()
            if self.is_paths_precomputed and self.selected_goal is not None:
                self._draw_path_arrows()
                self._draw_distance_heatmap()

    def _on_canvas_motion(self, event):
        """Handle mouse motion over canvas."""
        if self.maze_grid is None:
            return

        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        col = int((canvas_x - self.canvas_offset_x) // self.cell_size)
        row = int((canvas_y - self.canvas_offset_y) // self.cell_size)

        rows, cols = len(self.maze_grid), len(self.maze_grid[0])

        if 0 <= row < rows and 0 <= col < cols:
            cell_type = "Wall" if self.maze_grid[row][col] == 1 else "Corridor"

            if self.is_paths_precomputed and self.maze_grid[row][col] == 0 and self.selected_goal is not None:
                try:
                    goal_idx = self.precompute_goals.index(self.selected_goal)
                    next_dir, dist = self.path_precomputer.query_path((row, col), goal_idx)
                    direction_names = ["North", "East", "South", "West"]
                    dir_name = direction_names[next_dir] if next_dir >= 0 else "None"

                    self.master.title(f"Maze Visualizer - ({row},{col}) {cell_type} | "
                                    f"Next: {dir_name} | Distance: {dist:.1f}")
                except ValueError:
                    self.master.title(f"Maze Visualizer - ({row},{col}) {cell_type}")
            else:
                self.master.title(f"Maze Visualizer - ({row},{col}) {cell_type}")

    def _update_statistics(self, generation_time=None, precompute_time=None):
        """Update statistics display."""
        if self.maze_grid is None:
            return

        stats = []

        # Basic maze info
        rows, cols = len(self.maze_grid), len(self.maze_grid[0])
        total_cells = rows * cols
        corridor_count = sum(row.count(0) for row in self.maze_grid)
        wall_count = total_cells - corridor_count

        stats.append(f"Maze Size: {rows} × {cols}")
        stats.append(f"Total Cells: {total_cells}")
        stats.append(f"Corridors: {corridor_count}")
        stats.append(f"Walls: {wall_count}")
        stats.append(f"Corridor %: {corridor_count/total_cells*100:.1f}%")

        if generation_time:
            stats.append(f"Gen Time: {generation_time:.3f}s")

        # Goal info
        stats.append(f"Precompute Goals: {len(self.precompute_goals)}")
        if self.selected_goal:
            goal_idx = self.precompute_goals.index(self.selected_goal) if self.selected_goal in self.precompute_goals else -1
            stats.append(f"Selected Goal: {self.selected_goal} (idx: {goal_idx})")
        if self.start_cell:
            stats.append(f"Start: {self.start_cell}")

        # Pathfinding info
        if self.is_paths_precomputed:
            stats.append("Paths: COMPUTED")
            if precompute_time:
                stats.append(f"Precomp Time: {precompute_time:.3f}s")

            # Memory usage
            memory_mb = self.path_precomputer.get_memory_usage_mb()
            stats.append(f"Memory: {memory_mb:.2f} MB")

            # Connectivity check for selected goal
            if self.selected_goal is not None:
                try:
                    goal_idx = self.precompute_goals.index(self.selected_goal)
                    reachable_count = 0
                    for r in range(rows):
                        for c in range(cols):
                            if self.maze_grid[r][c] == 0:
                                _, dist = self.path_precomputer.query_path((r, c), goal_idx)
                                if dist != float('inf'):
                                    reachable_count += 1
                    stats.append(f"Reachable to Goal: {reachable_count}/{corridor_count}")
                except ValueError:
                    pass
        else:
            stats.append("Paths: NOT COMPUTED")

        # Update display
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, "\n".join(stats))

    def _update_goal_dropdown(self):
        """Update the goal selection dropdown with available goals."""
        goal_options = []
        if self.precompute_goals:
            for i, goal in enumerate(self.precompute_goals):
                goal_options.append(f"Goal {i}: {goal}")

        self.goal_combobox['values'] = goal_options
        if goal_options:
            self.goal_combobox.set(goal_options[0])
            self.selected_goal = self.precompute_goals[0]
        else:
            self.goal_combobox.set("")
            self.selected_goal = None

    def _on_goal_selected(self, event):
        """Handle goal selection from dropdown."""
        selection = self.goal_var.get()
        if selection and self.precompute_goals:
            try:
                # Extract goal index from selection string
                goal_idx = int(selection.split(":")[0].split()[-1])
                if 0 <= goal_idx < len(self.precompute_goals):
                    self.selected_goal = self.precompute_goals[goal_idx]
                    self.status_var.set(f"Selected goal: {self.selected_goal}")

                    # Update visualization if paths are precomputed
                    if self.is_paths_precomputed:
                        self._draw_maze()
                        self._draw_path_arrows()
                        self._draw_distance_heatmap()
            except (ValueError, IndexError):
                pass


def main():
    """Main function to run the GUI."""
    root = tk.Tk()
    app = MazeVisualizerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
