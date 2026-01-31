import numpy as np
import time
import os
import cv2
import textwrap
import torch

def save_points_to_pcd(points, filename):
    if os.path.exists(filename):
        os.remove(filename)
    num_points = len(points)
    header = textwrap.dedent(f"""\
        # .PCD v0.7 - Point Cloud Data file format
        VERSION 0.7
        FIELDS x y z
        SIZE 4 4 4
        TYPE F F F
        COUNT 1 1 1
        WIDTH {num_points}
        HEIGHT 1
        VIEWPOINT 0 0 0 1 0 0 0
        POINTS {num_points}
        DATA ascii
        """)
    with open(filename, 'w') as file:
        file.write(header)
        for point in points:
            file.write(f"{point[0]} {point[1]} {point[2]}\n")

class OccCollector:
    def __init__(self, env_spacing: float, data_dir: str, enable: bool = False, num_envs: int = 1, dt: float = 0.01):
        """
        Initialize the occupancy data collector.

        Args:
            env_spacing: Environment spacing parameter
            data_dir: Directory where data will be stored
            enable: Whether data collection is enabled
            num_envs: Number of environments to collect data from
            dt: Simulation timestep for virtual time calculation
        """
        self.data_dir = data_dir + "/" + str(int(time.time()))
        self.enable = enable
        self.env_spacing = env_spacing
        self.num_envs = num_envs
        self.dt = dt
        self.map_id = -1
        self.map_dir = None

        # Arrays to track environment data
        self.env_dirs = np.array([None] * num_envs)
        self.episode_count = np.zeros(num_envs, dtype=np.int32)
        self.episode_data = [[] for _ in range(num_envs)]
        self.virtual_time = np.zeros(num_envs, dtype=np.float32)

        if self.enable:
            os.makedirs(self.data_dir, exist_ok=True)

    def new_envrionment(self, map_3d):
        """
        Initialize a new environment for occupancy grid collection.
        """
        if not self.enable:
            return

        # Store previous environment data
        if self.map_id >= 0:
            # Use numpy to iterate over valid episode data indices
            valid_indices = np.where([len(data) > 0 for data in self.episode_data])[0]
            for env_idx in valid_indices:
                self._save_episode_data(env_idx)

        # Prepare for new environment
        self.map_id += 1
        self.map_dir = os.path.join(self.data_dir, f"map_{self.map_id}")
        os.makedirs(self.map_dir, exist_ok=True)

        # Create directories for all environments in a batch
        self.env_dirs = np.array([os.path.join(self.map_dir, f"env_{i}") for i in range(self.num_envs)])

        # Create directories in a batch
        for env_dir in self.env_dirs:
            os.makedirs(env_dir, exist_ok=True)

        # Reset counters and data arrays using NumPy operations
        self.episode_count = np.zeros(self.num_envs, dtype=np.int32)
        self.episode_data = [[] for _ in range(self.num_envs)]
        self.virtual_time = np.zeros(self.num_envs, dtype=np.float32)

        # Save the 3D map
        map_file = os.path.join(self.map_dir, "map.pcd")
        save_points_to_pcd(map_3d, map_file)

    def reset_env(self, env_indices):
        """
        Reset data collection for specific environments when episodes are done.

        Args:
            env_indices: NumPy array or list of environment indices to reset
        """
        if not self.enable:
            return

        # Convert to NumPy array if not already
        env_indices = np.asarray(env_indices)

        # Save data for environments with collected data
        valid_mask = np.array([len(self.episode_data[idx]) > 0 for idx in env_indices], dtype=bool)
        valid_indices = env_indices[valid_mask]

        for env_idx in valid_indices:
            self._save_episode_data(env_idx)

        # Reset episode data and counters using vectorized operations
        for env_idx in env_indices:
            self.episode_count[env_idx] += 1
            self.episode_data[env_idx] = []
            self.virtual_time[env_idx] = 0.0

    def _save_episode_data(self, env_idx):
        """
        Save episode data for a specific environment.

        Args:
            env_idx: Environment index
        """
        if len(self.episode_data[env_idx]) == 0:
            return

        episode_file = os.path.join(self.env_dirs[env_idx], f"episode_{self.episode_count[env_idx]}.npz")

        # Extract data components efficiently
        episode_frames = self.episode_data[env_idx]

        # Use list comprehensions instead of loops for better performance
        timestamps = np.array([frame[0] for frame in episode_frames])
        states = np.array([frame[1] for frame in episode_frames])
        depth_images = np.array([frame[2] for frame in episode_frames])
        left_depths = np.array([frame[3] for frame in episode_frames])
        right_depths = np.array([frame[4] for frame in episode_frames])
        back_depths = np.array([frame[5] for frame in episode_frames])
        down_depths = np.array([frame[6] for frame in episode_frames]) if len(episode_frames[0]) > 6 else np.zeros_like(left_depths)
        up_depths = np.array([frame[7] for frame in episode_frames]) if len(episode_frames[0]) > 7 else np.zeros_like(left_depths)

        # Use npz format for storing multiple arrays in one file (more efficient than multiple npy files)
        np.savez_compressed(
            episode_file,
            timestamps=timestamps,
            states=states,
            depth_images=depth_images,
            left_depths=left_depths,
            right_depths=right_depths,
            back_depths=back_depths,
            down_depths=down_depths,
            up_depths=up_depths
        )

    def collect_data(self, state, depth_image, left_depth=None, right_depth=None, back_depth=None,
                     down_depth=None, up_depth=None, env_indices=None):
        """
        Collect data for occupancy grid.
            Args:
                state: Current state of the environment [envs, pos, quat, lin_vel, ang_vel]
                depth_image: Depth image of the environment [envs, height, width]
                left_depth: Left TOF single point measurement [envs]
                right_depth: Right TOF single point measurement [envs]
                back_depth: Back TOF single point measurement [envs]
                env_indices: Optional array of environment indices to collect data for.
                             If None, collect for all environments.
        """
        if not self.enable:
            return

        # Default values for optional TOF data
        if left_depth is None:
            left_depth = np.zeros(len(state))
        if right_depth is None:
            right_depth = np.zeros(len(state))
        if back_depth is None:
            back_depth = np.zeros(len(state))
        if down_depth is None:
            down_depth = np.zeros(len(state))
        if up_depth is None:
            up_depth = np.zeros(len(state))

        # Determine which environments to collect data for
        if env_indices is None:
            env_indices = np.arange(min(self.num_envs, len(state)))
        else:
            env_indices = np.asarray(env_indices)

        # Update virtual time for all environments at once
        self.virtual_time[env_indices] += self.dt

        # Collect data for each environment
        for i, env_idx in enumerate(env_indices):
            if env_idx >= self.num_envs:
                continue

            # Create frame data tuple with efficient copy operations
            state_copy = state[env_idx].copy() if hasattr(state[env_idx], 'copy') else state[env_idx]
            depth_copy = depth_image[env_idx].copy() if hasattr(depth_image[env_idx], 'copy') else depth_image[env_idx]
            left_copy = left_depth[env_idx].copy() if hasattr(left_depth[env_idx], 'copy') else left_depth[env_idx]
            right_copy = right_depth[env_idx].copy() if hasattr(right_depth[env_idx], 'copy') else right_depth[env_idx]
            back_copy = back_depth[env_idx].copy() if hasattr(back_depth[env_idx], 'copy') else back_depth[env_idx]
            down_copy = down_depth[env_idx].copy() if hasattr(down_depth[env_idx], 'copy') else down_depth[env_idx]
            up_copy = up_depth[env_idx].copy() if hasattr(up_depth[env_idx], 'copy') else up_depth[env_idx]

            frame_data = (
                self.virtual_time[env_idx],  # Virtual timestamp
                state_copy,                  # Robot state
                depth_copy,                  # Depth image
                left_copy,                   # Left TOF
                right_copy,                  # Right TOF
                back_copy,                   # Back TOF
                down_copy,                   # Down TOF
                up_copy                      # Up TOF
            )

            self.episode_data[env_idx].append(frame_data)

def ego_occ_gen(map_points, pos_w, quat_w, grid_dims=(32, 32, 16), voxel_size=0.25):
    """
    Generate an ego-centric occupancy grid using PyTorch for efficient computation.
    Optimized for large point clouds (e.g., 3M points) with full batch vectorization.

    Args:
        map_points: Point cloud of the environment (N, 3) - approximately 3M points
        pos_w: Robot position in world frame (batch_size, 3)
        quat_w: Robot orientation as quaternion [w, x, y, z] in world frame (batch_size, 4)
        grid_dims: Dimensions of the occupancy grid (x, y, z)
        voxel_size: Size of each voxel in meters

    Returns:
        Occupancy grid as a binary tensor (batch_size, x, y, z)
        Robot is positioned at the center of the grid without offset.
    """
    # Convert inputs to PyTorch tensors if they aren't already
    device = pos_w.device
    batch_size = pos_w.shape[0]

    # Ensure map_points is a tensor on the correct device
    if not isinstance(map_points, torch.Tensor):
        map_points = torch.tensor(map_points, device=device, dtype=torch.float32)
    if map_points.device != device:
        map_points = map_points.to(device)

    # Import matrix_from_quat function (assuming it's available in the environment)
    from isaaclab.utils.math import matrix_from_quat

    # Get rotation matrices (world to robot frame transformation)
    rot_matrices = matrix_from_quat(quat_w)  # (batch_size, 3, 3)

    # Calculate grid physical dimensions - use half extents for proper centering
    grid_half_extents = torch.tensor([
        grid_dims[0] * voxel_size * 0.5,
        grid_dims[1] * voxel_size * 0.5,
        grid_dims[2] * voxel_size * 0.5
    ], device=device)

    # Robot is at the center of the grid, so min_corner is at -half_extents
    min_corner = -grid_half_extents

    # Add small epsilon to prevent floating point boundary issues
    eps = 1e-6

    # === Full Batch Vectorization ===
    # Expand dimensions for batch processing
    # map_points: (N, 3) -> (1, N, 3) -> (B, N, 3)
    # pos_w: (B, 3) -> (B, 1, 3)
    map_points_expanded = map_points.unsqueeze(0).expand(batch_size, -1, -1)  # (B, N, 3)
    pos_w_expanded = pos_w.unsqueeze(1)  # (B, 1, 3)

    # Transform points from world to robot frame in batch
    # 1. Translate to robot origin
    translated_points = map_points_expanded - pos_w_expanded  # (B, N, 3)

    # 2. Rotate to robot frame using batch matrix multiplication
    # rot_matrices: (B, 3, 3), translated_points: (B, N, 3)
    # Need: (B, 3, 3) x (B, 3, N) -> (B, 3, N) -> transpose to (B, N, 3)
    points_robot_frame = torch.bmm(
        rot_matrices,
        translated_points.transpose(1, 2)
    ).transpose(1, 2)  # (B, N, 3)

    # Convert to voxel coordinates with epsilon for boundary stability
    voxel_coords = torch.floor(
        (points_robot_frame - min_corner + eps) / voxel_size
    ).long()  # (B, N, 3)

    # Create validity mask for points within grid bounds
    valid_x = (voxel_coords[..., 0] >= 0) & (voxel_coords[..., 0] < grid_dims[0])
    valid_y = (voxel_coords[..., 1] >= 0) & (voxel_coords[..., 1] < grid_dims[1])
    valid_z = (voxel_coords[..., 2] >= 0) & (voxel_coords[..., 2] < grid_dims[2])
    valid_mask = valid_x & valid_y & valid_z  # (B, N)

    # Initialize occupancy grid
    occupancy_grid = torch.zeros((batch_size, *grid_dims), dtype=torch.bool, device=device)

    # Process each batch efficiently with deduplication
    for i in range(batch_size):
        if not torch.any(valid_mask[i]):
            continue

        # Extract valid coordinates for this batch
        valid_coords = voxel_coords[i][valid_mask[i]]  # (M, 3) where M <= N

        # Deduplicate coordinates to avoid redundant writes
        # Convert to linear indices for efficient unique operation
        linear_indices = (valid_coords[:, 0] * grid_dims[1] * grid_dims[2] +
                        valid_coords[:, 1] * grid_dims[2] +
                        valid_coords[:, 2])

        unique_linear_indices = torch.unique(linear_indices)

        # Convert back to 3D coordinates
        unique_z = unique_linear_indices % grid_dims[2]
        unique_y = (unique_linear_indices // grid_dims[2]) % grid_dims[1]
        unique_x = unique_linear_indices // (grid_dims[1] * grid_dims[2])

        # Set occupancy using efficient indexing
        occupancy_grid[i, unique_x, unique_y, unique_z] = True

    return occupancy_grid
