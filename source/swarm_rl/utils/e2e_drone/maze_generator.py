"""Optimized Maze Generator for Isaac Lab environments."""

import math
import random
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union

import carb
import numpy as np
import omni
import torch
import trimesh
from scipy.spatial import KDTree
from pxr import UsdGeom, Gf, Vt

import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prims_utils
from isaaclab.sim.schemas import RigidBodyPropertiesCfg

from e2e_drone.utils.map_generator import flatten_instancer_and_mesh
from e2e_drone.utils.maze.maze import MazeAlgorithm, PathPrecomputer


@dataclass
class MazeConfig:
    """Configuration for maze generation."""
    cell_size: float = 2.0
    wall_height: float = 3.0
    spawn_height: float = 1.0
    clearance_distance: float = 0.20
    carve_prob: float = 0.0
    loop_density: float = 0.15
    branching_prob: float = 0.4
    bias_factor: float = 0.3
    num_entrances: int = 2
    num_exits: int = 2
    num_floaters: int = 20
    floaters_size_range: Tuple[float, float] = (0.1, 0.3)
    floaters_height_range: Tuple[float, float] = (1.0, 2.5)
    floater_min_distance: float = 1.0
    # Surface sampling parameters
    num_surface_samples: int = 500000
    min_samples_per_face: int = 3

class MazeGenerator:
    """Maze-based map generator for Isaac Lab environments."""

    def __init__(self, sim, device="cuda:0", map_origin=(0.0, 0.0, 0.0), base_prim="/World/ground"):
        self.device = torch.device(device)
        self.sim = sim
        self.map_origin = torch.tensor(map_origin, dtype=torch.float32, device=self.device)
        self.base_prim = base_prim
        self._generation_in_progress = False
        self._maze_grid = None
        self._maze_algorithm = None
        self.config = MazeConfig()
        self.apsp_pathfinder = PathPrecomputer(device=device)
        self._enable_viewport_grid()

    def _get_prim_path(self, relative_path):
        """Construct a primitive path relative to base primitive."""
        return f"{self.base_prim}/{relative_path}" if relative_path else self.base_prim

    def _translate_to_origin(self, position):
        """Translate position relative to map origin."""
        if isinstance(position, torch.Tensor):
            return position + self.map_origin
        pos_tensor = torch.tensor(position, dtype=torch.float32, device=self.device)
        return pos_tensor + self.map_origin

    def _get_maze_offsets(self) -> Tuple[float, float]:
        """Calculate maze offset coordinates to center at origin."""
        # Use centralized coordinate conversion from apsp_pathfinder if available
        if self.apsp_pathfinder.is_precomputed:
            return self.apsp_pathfinder.get_maze_offsets()

        # Fallback for when pathfinder is not yet available
        if self._maze_grid is None:
            raise RuntimeError("No maze grid available")
        rows, cols = len(self._maze_grid), len(self._maze_grid[0])
        maze_offset_x = -(cols * self.config.cell_size) / 2.0 + self.config.cell_size / 2.0
        maze_offset_y = -(rows * self.config.cell_size) / 2.0 + self.config.cell_size / 2.0
        return maze_offset_x, maze_offset_y

    def _world_to_maze_coords(self, world_pos) -> Optional[Tuple[int, int]]:
        """Convert world coordinates to maze grid coordinates."""
        # Use centralized coordinate conversion from apsp_pathfinder if available
        if self.apsp_pathfinder.is_precomputed:
            return self.apsp_pathfinder.world_to_maze_coords_single(world_pos)

        # Fallback for when pathfinder is not yet available
        if self._maze_grid is None:
            return None

        rows, cols = len(self._maze_grid), len(self._maze_grid[0])
        maze_offset_x, maze_offset_y = self._get_maze_offsets()

        if isinstance(world_pos, torch.Tensor):
            local_pos = world_pos - self.map_origin
            local_x, local_y = local_pos[0].item(), local_pos[1].item()
        else:
            local_x = world_pos[0] - self.map_origin[0].item()
            local_y = world_pos[1] - self.map_origin[1].item()

        col = int((local_x - maze_offset_x) / self.config.cell_size + 0.5)
        row = int((local_y - maze_offset_y) / self.config.cell_size + 0.5)

        if (0 <= row < rows and 0 <= col < cols and self._maze_grid[row][col] == 0):
            return (row, col)
        return None

    def _maze_to_world_coords(self, maze_pos, return_tensor: bool = False):
        """Convert maze grid coordinates to world coordinates."""
        # Use centralized coordinate conversion from apsp_pathfinder if available
        if self.apsp_pathfinder.is_precomputed:
            return self.apsp_pathfinder.maze_to_world_coords_single(maze_pos, return_tensor)

        # Fallback for when pathfinder is not yet available
        maze_offset_x, maze_offset_y = self._get_maze_offsets()

        if isinstance(maze_pos, torch.Tensor):
            world_positions = torch.zeros(maze_pos.shape[0], 3, device=self.device, dtype=torch.float32)
            world_positions[:, 0] = maze_offset_x + maze_pos[:, 1] * self.config.cell_size
            world_positions[:, 1] = maze_offset_y + maze_pos[:, 0] * self.config.cell_size
            world_positions[:, 2] = self.config.spawn_height
            return world_positions + self.map_origin.unsqueeze(0)
        else:
            row, col = maze_pos
            world_x = maze_offset_x + col * self.config.cell_size
            world_y = maze_offset_y + row * self.config.cell_size
            world_z = self.config.spawn_height

            if return_tensor:
                return self._translate_to_origin(torch.tensor([world_x, world_y, world_z],
                                                           dtype=torch.float32, device=self.device))
            result = self._translate_to_origin((world_x, world_y, world_z))
            return (result[0].item(), result[1].item(), result[2].item())

    def create_environment(self, scene, **kwargs):
        """Create maze environment with configurable parameters."""
        if self._generation_in_progress:
            return None

        self._generation_in_progress = True

        try:
            start_time = time.time()

            # Update config with kwargs
            for key, value in kwargs.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

            self._cleanup_environment()

            # Calculate maze dimensions
            max_cells = int(scene.env_spacing / self.config.cell_size)
            rows = max_cells if max_cells % 2 == 1 else max_cells - 1
            cols = rows
            rows, cols = max(rows, 11), max(cols, 11)

            # Generate maze
            self._maze_algorithm = MazeAlgorithm(
                rows=rows,
                cols=cols,
                loop_density=self.config.loop_density,
                branching_prob=self.config.branching_prob,
                bias_factor=self.config.bias_factor,
                num_entrances=self.config.num_entrances,
                num_exits=self.config.num_exits
            )
            self._maze_grid = self._maze_algorithm.generate()

            # Generate environment components
            ground_prim = self._generate_ground_plane(scene.env_spacing)
            perimeter_prims = self._generate_perimeter_walls_and_ceiling(scene.env_spacing)
            wall_prims = self._generate_maze_walls()
            floater_prims, floater_positions = self._generate_maze_floaters()

            # Generate flattened mesh
            stage = omni.usd.get_context().get_stage()
            merged_mesh_path = "/map_mesh"
            if prims_utils.is_prim_path_valid(merged_mesh_path):
                prims_utils.delete_prim(merged_mesh_path)

            env_space = scene.env_spacing / 2.0
            flattened_mesh_path = flatten_instancer_and_mesh(
                stage, "/World/ground", merged_mesh_path,
                min_bound=(-env_space * 2, -env_space * 2, 0.0),
                max_bound=(env_space * 2, env_space * 2, self.config.wall_height + 1.0),
                env_space=env_space)

            # Generate occupancy map
            kdtree, occupied_points, free_points = self.generate_occupancy_map_from_mesh(
                scene, merged_mesh_path,
                num_surface_samples=self.config.num_surface_samples,
                min_samples_per_face=self.config.min_samples_per_face)

            # Use maze exits as goal cells to reduce computation
            exit_cells = [(pos[0], pos[1]) for pos in self._maze_algorithm.exits] if self._maze_algorithm and self._maze_algorithm.exits else None
            self.apsp_pathfinder.precompute_apsp(self._maze_grid, self.config.cell_size, self.map_origin, exits=exit_cells)
            apsp_available = True


            return {
                "walls": wall_prims + [ground_prim] + perimeter_prims,
                "obstacles": wall_prims,
                "obstacle_positions": self._extract_wall_positions(),
                "floaters": floater_prims,
                "floater_positions": floater_positions,
                "kdtree": kdtree,
                "points": occupied_points,
                "free_points": free_points,
                "flattened_mesh": flattened_mesh_path,
                "generation_time": time.time() - start_time,
                "map_origin": self.map_origin,
                "base_prim": self.base_prim,
                "maze_grid": self._maze_grid,
                "maze_params": self.config.__dict__,
                "entrances": [self._maze_to_world_coords(pos) for pos in self._maze_algorithm.entrances],
                "exits": [self._maze_to_world_coords(pos) for pos in self._maze_algorithm.exits],
                "apsp_pathfinder": self.apsp_pathfinder if apsp_available else None,
                "apsp_available": apsp_available
            }

        finally:
            self._generation_in_progress = False

    def _generate_maze_walls(self) -> List[str]:
        """Generate 3D cuboid walls for maze."""
        wall_prims = []
        rows, cols = len(self._maze_grid), len(self._maze_grid[0])

        # Create wall prototype and instancer
        wall_prototype_path = self._get_prim_path("prototypes/wall_prototype")
        wall_instancer_path = self._get_prim_path("wall_instancer")

        prototype_dir = self._get_prim_path("prototypes")
        if not prims_utils.is_prim_path_valid(prototype_dir):
            prims_utils.create_prim(prototype_dir, "Xform")

        wall_cfg = sim_utils.MeshCuboidCfg(
            size=(self.config.cell_size, self.config.cell_size, self.config.wall_height),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.6, 0.6)),
            rigid_props=RigidBodyPropertiesCfg(rigid_body_enabled=False, kinematic_enabled=True),
            activate_contact_sensors=True)

        if not prims_utils.is_prim_path_valid(wall_prototype_path):
            wall_cfg.func(wall_prototype_path, wall_cfg)

        stage = omni.usd.get_context().get_stage()
        wall_instancer = UsdGeom.PointInstancer.Define(stage, wall_instancer_path)
        wall_instancer.GetPrototypesRel().AddTarget(wall_prototype_path)

        maze_offset_x, maze_offset_y = self._get_maze_offsets()
        wall_positions = []

        for row in range(rows):
            for col in range(cols):
                if self._maze_grid[row][col] == 1:
                    world_x = maze_offset_x + col * self.config.cell_size
                    world_y = maze_offset_y + row * self.config.cell_size
                    world_z = self.config.wall_height / 2.0
                    wall_positions.append(self._translate_to_origin((world_x, world_y, world_z)))

        if wall_positions:
            pos_array = Vt.Vec3fArray([Gf.Vec3f(*pos.cpu().tolist()) for pos in wall_positions])
            scale_array = Vt.Vec3fArray([Gf.Vec3f(1.0, 1.0, 1.0)] * len(wall_positions))
            orient_array = Vt.QuathArray([Gf.Quath(1.0, 0.0, 0.0, 0.0)] * len(wall_positions))
            proto_indices = Vt.IntArray([0] * len(wall_positions))

            wall_instancer.CreateProtoIndicesAttr().Set(proto_indices)
            wall_instancer.CreatePositionsAttr().Set(pos_array)
            wall_instancer.CreateScalesAttr().Set(scale_array)
            wall_instancer.CreateOrientationsAttr().Set(orient_array)

            for i in range(len(wall_positions)):
                wall_prims.append(f"{wall_instancer_path}/instance_{i}")

        return wall_prims

    def _generate_ground_plane(self, env_size: float) -> str:
        """Generate ground plane."""
        ground_size = env_size + 5.0
        ground_thickness = 0.2

        ground_cfg = sim_utils.MeshCuboidCfg(
            size=(ground_size, ground_size, ground_thickness),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2)),
            rigid_props=RigidBodyPropertiesCfg(rigid_body_enabled=False, kinematic_enabled=True),
            activate_contact_sensors=True)

        ground_path = self._get_prim_path("maze_ground_plane")
        ground_cfg.func(ground_path, ground_cfg,
                       translation=self._translate_to_origin((0.0, 0.0, -ground_thickness/2.0 - 0.01))) # Avoid touching the ground plane become one surface with ground
        return ground_path

    def _generate_perimeter_walls_and_ceiling(self, env_size: float) -> List[str]:
        """Generate perimeter walls and ceiling."""
        perimeter_prims = []
        rows, cols = len(self._maze_grid), len(self._maze_grid[0])
        maze_width, maze_height = cols * self.config.cell_size, rows * self.config.cell_size
        wall_thickness = max(0.2, self.config.wall_height / 40.0)

        wall_cfg = sim_utils.MeshCuboidCfg(
            size=(maze_width, wall_thickness, self.config.wall_height),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.4, 0.8)),
            rigid_props=RigidBodyPropertiesCfg(rigid_body_enabled=False, kinematic_enabled=True),
            activate_contact_sensors=True)

        # Generate walls
        wall_positions = [
            (0.0, maze_height / 2.0 + wall_thickness / 2.0),  # Left
            (0.0, -maze_height / 2.0 - wall_thickness / 2.0), # Right
        ]

        for i, (x, y) in enumerate(wall_positions):
            path = self._get_prim_path(f"perimeter_wall_{i}")
            wall_cfg.func(path, wall_cfg,
                         translation=self._translate_to_origin((x, y, self.config.wall_height / 2.0)))
            perimeter_prims.append(path)

        # Perpendicular walls
        perp_wall_cfg = sim_utils.MeshCuboidCfg(
            size=(wall_thickness, maze_height, self.config.wall_height),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.4, 0.8)),
            rigid_props=RigidBodyPropertiesCfg(rigid_body_enabled=False, kinematic_enabled=True),
            activate_contact_sensors=True)

        perp_positions = [
            (maze_width / 2.0 + wall_thickness / 2.0, 0.0),   # Forward
            (-maze_width / 2.0 - wall_thickness / 2.0, 0.0),  # Backward
        ]

        for i, (x, y) in enumerate(perp_positions):
            path = self._get_prim_path(f"perimeter_perp_wall_{i}")
            perp_wall_cfg.func(path, perp_wall_cfg,
                              translation=self._translate_to_origin((x, y, self.config.wall_height / 2.0)))
            perimeter_prims.append(path)

        # Ceiling
        ceiling_thickness = max(0.2, max(maze_width, maze_height) / 40.0)
        ceiling_cfg = sim_utils.MeshCuboidCfg(
            size=(maze_width, maze_height, ceiling_thickness),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1)),
            rigid_props=RigidBodyPropertiesCfg(rigid_body_enabled=False, kinematic_enabled=True),
            activate_contact_sensors=True)

        ceiling_path = self._get_prim_path("perimeter_ceiling")
        ceiling_cfg.func(ceiling_path, ceiling_cfg,
                        translation=self._translate_to_origin((0.0, 0.0, self.config.wall_height + ceiling_thickness / 2.0 + 0.01)))  # Slightly above walls

        # Make ceiling invisible
        stage = omni.usd.get_context().get_stage()
        imageable = UsdGeom.Imageable(stage.GetPrimAtPath(ceiling_path))
        imageable.MakeInvisible()

        perimeter_prims.append(ceiling_path)
        return perimeter_prims

    def _generate_maze_floaters(self) -> Tuple[List[str], List[Tuple]]:
        """Generate floating obstacles in maze corridors."""
        rows, cols = len(self._maze_grid), len(self._maze_grid[0])
        maze_offset_x, maze_offset_y = self._get_maze_offsets()

        corridor_cells = []
        for row in range(rows):
            for col in range(cols):
                if self._maze_grid[row][col] == 0:
                    cell_center_x = maze_offset_x + col * self.config.cell_size
                    cell_center_y = maze_offset_y + row * self.config.cell_size
                    corridor_cells.append((cell_center_x, cell_center_y))

        if not corridor_cells:
            return [], []

        num_floaters = min(self.config.num_floaters, len(corridor_cells) * 3)

        # Define prototype configurations
        prototype_configs = {
            "cuboid": sim_utils.MeshCuboidCfg(
                size=(1.0, 0.5, 0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.6, 0.1)),
                rigid_props=RigidBodyPropertiesCfg(rigid_body_enabled=False, kinematic_enabled=True),
                activate_contact_sensors=True),
            "cylinder": sim_utils.MeshCylinderCfg(
                radius=0.5, height=1.0,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.6, 1.0)),
                rigid_props=RigidBodyPropertiesCfg(rigid_body_enabled=False, kinematic_enabled=True),
                activate_contact_sensors=True),
            "cone": sim_utils.MeshConeCfg(
                radius=0.5, height=1.0,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 1.0, 0.1)),
                rigid_props=RigidBodyPropertiesCfg(rigid_body_enabled=False, kinematic_enabled=True),
                activate_contact_sensors=True)
        }

        # Create prototypes and instancers
        prototype_dir = self._get_prim_path("prototypes")
        if not prims_utils.is_prim_path_valid(prototype_dir):
            prims_utils.create_prim(prototype_dir, "Xform")

        stage = omni.usd.get_context().get_stage()
        instancers = {}

        for shape_type, config in prototype_configs.items():
            proto_path = self._get_prim_path(f"prototypes/{shape_type}_floater_prototype")
            instancer_path = self._get_prim_path(f"{shape_type}_floater_instancer")

            if not prims_utils.is_prim_path_valid(proto_path):
                config.func(proto_path, config)

            if not prims_utils.is_prim_path_valid(instancer_path):
                instancer = UsdGeom.PointInstancer.Define(stage, instancer_path)
                instancer.GetPrototypesRel().AddTarget(proto_path)
                instancers[shape_type] = instancer
            else:
                instancers[shape_type] = UsdGeom.PointInstancer(stage.GetPrimAtPath(instancer_path))

        # Generate floater positions
        transform_data = {shape: {"positions": [], "scales": [], "orientations": []}
                         for shape in prototype_configs.keys()}
        floater_positions = []

        def is_position_valid(new_x, new_y, new_z, existing_positions, min_distance):
            return all(math.sqrt((new_x - x)**2 + (new_y - y)**2 + (new_z - z)**2) >= min_distance
                      for x, y, z in existing_positions)

        attempts = 0
        max_attempts = num_floaters * 10

        while len(floater_positions) < num_floaters and attempts < max_attempts:
            attempts += 1
            cell_center_x, cell_center_y = random.choice(corridor_cells)

            margin = 0.2
            x = cell_center_x + random.uniform(-self.config.cell_size/2 + margin,
                                             self.config.cell_size/2 - margin)
            y = cell_center_y + random.uniform(-self.config.cell_size/2 + margin,
                                             self.config.cell_size/2 - margin)

            size = random.uniform(*self.config.floaters_size_range)
            height = random.uniform(*self.config.floaters_height_range)
            z = height + size/2.0

            if not is_position_valid(x, y, z, floater_positions, self.config.floater_min_distance):
                continue

            shape_type = random.choice(list(prototype_configs.keys()))

            # Random orientation
            u1, u2, u3 = random.random(), random.random(), random.random()
            qw = math.sqrt(1 - u1) * math.sin(2 * math.pi * u2)
            qx = math.sqrt(1 - u1) * math.cos(2 * math.pi * u2)
            qy = math.sqrt(u1) * math.sin(2 * math.pi * u3)
            qz = math.sqrt(u1) * math.cos(2 * math.pi * u3)

            translated_pos = self._translate_to_origin((x, y, z))
            transform_data[shape_type]["positions"].append(translated_pos)
            transform_data[shape_type]["scales"].append((size, size, size))
            transform_data[shape_type]["orientations"].append((qw, qx, qy, qz))
            floater_positions.append(translated_pos)

        # Apply transforms to instancers
        floater_prims = []
        for shape_type, data in transform_data.items():
            if not data["positions"]:
                continue

            instancer = instancers[shape_type]
            pos_array = Vt.Vec3fArray([Gf.Vec3f(*pos.cpu().tolist()) for pos in data["positions"]])
            scale_array = Vt.Vec3fArray([Gf.Vec3f(*scale) for scale in data["scales"]])
            orient_array = Vt.QuathArray([Gf.Quath(*orient) for orient in data["orientations"]])
            proto_indices = Vt.IntArray([0] * len(data["positions"]))

            instancer.CreateProtoIndicesAttr().Set(proto_indices)
            instancer.CreatePositionsAttr().Set(pos_array)
            instancer.CreateScalesAttr().Set(scale_array)
            instancer.CreateOrientationsAttr().Set(orient_array)

            instancer_path = self._get_prim_path(f"{shape_type}_floater_instancer")
            for i in range(len(data["positions"])):
                floater_prims.append(f"{instancer_path}/instance_{i}")

        return floater_prims, floater_positions

    def _optimized_surface_sampling(self, mesh, num_surface_samples, min_samples_per_face=5):
        """
        Optimized surface sampling to ensure small surfaces get adequate coverage.

        Uses a simple hybrid approach:
        1. Identify small faces (below 10% of average face area)
        2. Ensure minimum samples per face for small surfaces
        3. Use area-weighted sampling for remaining samples

        Args:
            mesh: trimesh.Trimesh object
            num_surface_samples: Total number of desired samples
            min_samples_per_face: Minimum samples per face for small surfaces

        Returns:
            np.ndarray: Combined surface sample points
        """
        # Calculate face areas and identify small faces
        face_areas = mesh.area_faces
        avg_face_area = mesh.area / len(face_areas)
        small_face_threshold = avg_face_area * 0.1
        small_face_indices = np.where(face_areas < small_face_threshold)[0]

        print(f"Surface sampling: {len(small_face_indices)}/{len(face_areas)} small faces "
              f"(< {small_face_threshold:.6f} area units)")

        occupied_points_list = []
        total_samples_used = 0

        # Strategy 1: Ensure minimum coverage for small faces
        if len(small_face_indices) > 0:
            small_face_samples = min(
                min_samples_per_face * len(small_face_indices),
                num_surface_samples // 4  # Cap at 25% of total samples
            )

            if small_face_samples > 0:
                # Create face weights: 1.0 for small faces, 0.0 for others
                face_weights = np.zeros_like(face_areas)
                face_weights[small_face_indices] = 1.0

                try:
                    small_points, _ = trimesh.sample.sample_surface(
                        mesh, count=small_face_samples, face_weight=face_weights
                    )
                    occupied_points_list.append(small_points)
                    total_samples_used += len(small_points)
                    print(f"Small face sampling: {len(small_points)} points")
                except Exception as e:
                    print(f"Warning: Small face sampling failed: {e}")

        # Strategy 2: Area-weighted sampling for remaining samples
        remaining_samples = max(0, num_surface_samples - total_samples_used)
        if remaining_samples > 0:
            try:
                area_points, _ = trimesh.sample.sample_surface(
                    mesh, count=remaining_samples, face_weight=face_areas
                )
                occupied_points_list.append(area_points)
                total_samples_used += len(area_points)
                print(f"Area-weighted sampling: {len(area_points)} points")
            except Exception as e:
                print(f"Warning: Area-weighted sampling failed: {e}")
                # Fallback to basic uniform sampling
                try:
                    uniform_points, _ = trimesh.sample.sample_surface(mesh, count=remaining_samples)
                    occupied_points_list.append(uniform_points)
                    total_samples_used += len(uniform_points)
                    print(f"Uniform fallback sampling: {len(uniform_points)} points")
                except Exception as e2:
                    print(f"Warning: All sampling methods failed: {e2}")

        # Combine all sampling strategies
        if occupied_points_list:
            occupied_points = np.vstack(occupied_points_list)
            print(f"Total surface samples: {len(occupied_points)} points "
                  f"(target: {num_surface_samples})")
        else:
            # Ultimate fallback: use mesh vertices
            occupied_points = mesh.vertices
            print(f"Using mesh vertices as fallback: {len(occupied_points)} points")

        return occupied_points.astype(np.float32)

    def generate_occupancy_map_from_mesh(self, scene, mesh_path="/map_mesh", num_surface_samples=500000, num_free_samples=10000, min_samples_per_face=3):
        """
        Generate an optimized occupancy map from a unified mesh using trimesh.

        Uses a hybrid sampling strategy to ensure small surfaces get adequate coverage:
        1. Area-weighted sampling for overall distribution
        2. Minimum samples per face for small surface coverage

        Args:
            scene: Scene configuration with env_spacing parameter
            mesh_path: Path to the unified mesh primitive
            num_surface_samples: Number of surface samples for occupied points
            num_free_samples: Number of free space samples
            min_samples_per_face: Minimum samples per face to ensure small surfaces get coverage

        Returns:
            KDTree of occupied points, occupied points array, free points array
        """
        print(f"Generating optimized occupancy map from mesh: {mesh_path}")
        start_time = time.time()

        # Get the mesh from USD stage
        stage = omni.usd.get_context().get_stage()
        mesh_prim = stage.GetPrimAtPath(mesh_path)

        if not mesh_prim.IsValid():
            print(f"Warning: Mesh prim at {mesh_path} not found, using fallback points")
            fallback_points = np.array([[0, 0, 0]], dtype=np.float32)
            return KDTree(fallback_points), fallback_points, fallback_points

        # Convert USD mesh to trimesh
        mesh_geom = UsdGeom.Mesh(mesh_prim)
        vertices = mesh_geom.GetPointsAttr().Get()
        face_vertex_counts = mesh_geom.GetFaceVertexCountsAttr().Get()
        face_vertex_indices = mesh_geom.GetFaceVertexIndicesAttr().Get()

        if not vertices or not face_vertex_indices:
            print("Warning: Empty mesh data, using fallback points")
            fallback_points = np.array([[0, 0, 0]], dtype=np.float32)
            return KDTree(fallback_points), fallback_points, fallback_points

        # Convert to numpy arrays
        vertices_np = np.array([[v[0], v[1], v[2]] for v in vertices], dtype=np.float32)

        # Convert face data to triangles (trimesh expects triangular faces)
        faces = []
        idx = 0
        for count in face_vertex_counts:
            if count == 3:
                # Triangle face
                faces.append([
                    face_vertex_indices[idx],
                    face_vertex_indices[idx + 1],
                    face_vertex_indices[idx + 2]
                ])
            elif count == 4:
                # Quad face - split into two triangles
                faces.append([
                    face_vertex_indices[idx],
                    face_vertex_indices[idx + 1],
                    face_vertex_indices[idx + 2]
                ])
                faces.append([
                    face_vertex_indices[idx],
                    face_vertex_indices[idx + 2],
                    face_vertex_indices[idx + 3]
                ])
            idx += count

        faces_np = np.array(faces, dtype=np.int32)

        # Create trimesh object
        try:
            mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np)
            print(f"Created trimesh with {len(vertices_np)} vertices and {len(faces_np)} faces")
        except Exception as e:
            print(f"Error creating trimesh: {e}, using fallback points")
            fallback_points = np.array([[0, 0, 0]], dtype=np.float32)
            return KDTree(fallback_points), fallback_points, fallback_points

        # Optimized surface sampling for occupied points
        surface_time = time.time()
        try:
            occupied_points = self._optimized_surface_sampling(mesh, num_surface_samples, min_samples_per_face)
            print(f"Surface sampling: {time.time() - surface_time:.3f} seconds, "
                  f"generated {len(occupied_points)} points")
        except Exception as e:
            print(f"Error in surface sampling: {e}, using mesh vertices as occupied points")
            occupied_points = vertices_np

        # Randomly sample twice the target number of points to limit computation
        free_time = time.time()
        min_bounds = self._translate_to_origin((
            -scene.env_spacing / 2.0 + 1.0,
            -scene.env_spacing / 2.0 + 1.0,
            0.5
        ))
        max_bounds = self._translate_to_origin((
            scene.env_spacing / 2.0 - 1.0,
            scene.env_spacing / 2.0 - 1.0,
            1.5
        ))
        bbox_min = np.array(min_bounds.cpu(), dtype=np.float32)
        bbox_max = np.array(max_bounds.cpu(), dtype=np.float32)

        # Generate a regular 3D grid of candidate points within bounds
        voxel_pitch = 0.1  # Fixed grid resolution
        x_vals = np.arange(bbox_min[0], bbox_max[0], voxel_pitch, dtype=np.float32)
        y_vals = np.arange(bbox_min[1], bbox_max[1], voxel_pitch, dtype=np.float32)
        z_vals = np.arange(bbox_min[2], bbox_max[2], voxel_pitch, dtype=np.float32)
        xx, yy, zz = np.meshgrid(x_vals, y_vals, z_vals, indexing='xy')
        grid_pts = np.stack((xx.ravel(), yy.ravel(), zz.ravel()), axis=-1)

        # Randomly sample twice the target number of points to limit computation
        num_candidates = min(len(grid_pts), num_free_samples * 2)
        candidate_indices = np.random.choice(len(grid_pts), size=num_candidates, replace=False)
        sample_pts = grid_pts[candidate_indices]

        # Efficient free space sampling via downward ray intersection with even/odd parity test
        ray_directions = np.tile([0.0, 0.0, -1.0], (len(sample_pts), 1))
        # Batch compute all intersection locations and ray indices
        locations, index_ray, _ = mesh.ray.intersects_location(
            ray_origins=sample_pts,
            ray_directions=ray_directions,
            multiple_hits=True
        )
        # Count hits per ray
        counts = np.bincount(index_ray, minlength=len(sample_pts))
        # Free points are outside mesh (even number of intersections - 0, 2, 4, ...)
        # Occupied points are inside mesh (odd number of intersections - 1, 3, 5, ...)
        free_mask = (counts % 2) == 0
        free_points = sample_pts[free_mask]
        print(f"Downward ray even/odd parity: tested {len(sample_pts)} rays, found {len(free_points)} free points")

        # Downsample free points to exact target count if necessary
        if len(free_points) > num_free_samples:
            chosen = np.random.choice(len(free_points), size=num_free_samples, replace=False)
            free_points = free_points[chosen]
            print(f"Downsampled to {len(free_points)} free points")

        # Handle empty points case
        if len(occupied_points) == 0:
            print("No occupied points found, using fallback")
            occupied_points = np.array([[0, 0, 0]], dtype=np.float32)

        if len(free_points) == 0:
            print("No free points found, using fallback")
            free_points = np.array([[0, 0, 1]], dtype=np.float32)

        # Create KDTree for efficient nearest neighbor queries
        kdtree = KDTree(occupied_points)

        # Filter free points by minimum clearance distance to obstacles
        clearance_time = time.time()
        min_clearance = 0.20  # Minimum clearance distance in meters

        if len(free_points) > 0 and len(occupied_points) > 0:
            # Query distances to nearest occupied points for all free points
            distances, _ = kdtree.query(free_points, k=1)

            # Keep only free points that have sufficient clearance
            clearance_mask = distances >= min_clearance
            filtered_free_points = free_points[clearance_mask]

            print(f"Clearance filtering: kept {len(filtered_free_points)}/{len(free_points)} free points "
                  f"with min clearance {min_clearance}m, time: {time.time() - clearance_time:.3f} seconds")
        else:
            filtered_free_points = free_points
        print(f"Filtered free points: {len(filtered_free_points)}, time: {time.time() - free_time:.3f} seconds")

        total_time = time.time() - start_time
        print(f"Total trimesh-based occupancy map generation: {total_time:.3f} seconds")

        return kdtree, occupied_points, filtered_free_points

    def _extract_wall_positions(self) -> List[Tuple]:
        """Extract wall positions for MapGenerator compatibility."""
        if self._maze_grid is None:
            return []

        wall_positions = []
        rows, cols = len(self._maze_grid), len(self._maze_grid[0])

        for row in range(rows):
            for col in range(cols):
                if self._maze_grid[row][col] == 1:
                    world_pos = self._maze_to_world_coords((row, col))
                    wall_positions.append((world_pos[0], world_pos[1], self.config.cell_size, self.config.cell_size))

        return wall_positions

    def get_spawn_points(self, num_points: int) -> np.ndarray:
        """Get safe spawn points for agents in maze corridors."""
        if self._maze_grid is None:
            raise RuntimeError("No maze generated")

        corridor_positions = []
        rows, cols = len(self._maze_grid), len(self._maze_grid[0])

        for row in range(rows):
            for col in range(cols):
                if self._maze_grid[row][col] == 0:
                    world_pos = self._maze_to_world_coords((row, col))
                    corridor_positions.append(list(world_pos))

        num_points = min(num_points, len(corridor_positions))
        selected_indices = random.sample(range(len(corridor_positions)), num_points)
        return np.array([corridor_positions[i] for i in selected_indices])

    def _cleanup_environment(self):
        """Clean up existing maze environment components."""
        components_to_remove = [
            "maze_wall_", "maze_ground_plane", "maze_floater_instancer", "prototypes",
            "cuboid_floater_instancer", "cylinder_floater_instancer", "cone_floater_instancer",
            "perimeter_", "wall_instancer"
        ]

        stage = omni.usd.get_context().get_stage()
        prims_to_delete = []

        for prim in stage.TraverseAll():
            prim_path = str(prim.GetPath())
            if (prim_path.startswith(self.base_prim) and
                any(component in prim_path for component in components_to_remove)):
                prims_to_delete.append(prim_path)

        for prim_path in prims_to_delete:
            if prims_utils.is_prim_path_valid(prim_path):
                prims_utils.delete_prim(prim_path)

    def is_generation_in_progress(self) -> bool:
        """Check if maze generation is in progress."""
        return self._generation_in_progress

    def _enable_viewport_grid(self):
        """Enable viewport grid."""
        try:
            carb.settings.get_settings().set("/app/viewport/grid/enabled", True)
        except Exception:
            pass

    # APSP pathfinding interface methods
    def get_optimal_directions(self, robot_positions: torch.Tensor, target_positions: torch.Tensor):
        """Get optimal movement directions using APSP pathfinding."""
        if not self.apsp_pathfinder.is_precomputed:
            raise RuntimeError("APSP pathfinding not available")
        return self.apsp_pathfinder.lookup_next_directions(robot_positions, target_positions)

    def get_path_waypoints(self, start_pos: torch.Tensor, target_pos: torch.Tensor, max_waypoints: int = 100):
        """Get complete path waypoints using APSP pathfinding."""
        if not self.apsp_pathfinder.is_precomputed:
            raise RuntimeError("APSP pathfinding not available")
        return self.apsp_pathfinder.get_path_waypoints(start_pos, target_pos, max_waypoints)

    def is_apsp_available(self) -> bool:
        """Check if APSP pathfinding is available."""
        return self.apsp_pathfinder.is_precomputed

    def get_apsp_memory_usage_mb(self) -> float:
        """Get APSP memory usage in MB."""
        return self.apsp_pathfinder.get_memory_usage_mb()

    def get_corridor_world_positions(self) -> List[Tuple[float, float, float]]:
        """Get world positions of all navigable corridor cells.

        Returns:
            List of (x, y, z) world coordinates for all corridor cells.
        """
        if self._maze_grid is None:
            return []

        corridor_positions = []
        rows, cols = len(self._maze_grid), len(self._maze_grid[0])

        for row in range(rows):
            for col in range(cols):
                if self._maze_grid[row][col] == 0:  # 0 = corridor, 1 = wall
                    world_pos = self._maze_to_world_coords((row, col))
                    corridor_positions.append(world_pos)

        return corridor_positions

    def validate_position_in_corridor(self, world_pos) -> bool:
        """Validate if a world position is within a navigable corridor cell.

        Args:
            world_pos: World position as (x, y, z) tuple or tensor

        Returns:
            True if position is in a navigable corridor, False otherwise
        """
        if self._maze_grid is None:
            return False

        maze_coords = self._world_to_maze_coords(world_pos)
        return maze_coords is not None

    def world_to_grid_coords(self, world_coords: torch.Tensor) -> torch.Tensor:
        """Convert world coordinates to grid coordinates."""
        if not self.apsp_pathfinder.is_precomputed:
            raise RuntimeError("APSP pathfinding not available")
        return self.apsp_pathfinder.world_to_grid_batch(world_coords)

    def grid_to_world_coords(self, grid_coords: torch.Tensor) -> torch.Tensor:
        """Convert grid coordinates to world coordinates."""
        if not self.apsp_pathfinder.is_precomputed:
            raise RuntimeError("APSP pathfinding not available")
        return self.apsp_pathfinder.grid_to_world_batch(grid_coords)