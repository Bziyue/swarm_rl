#!/usr/bin/env python3

"""
Data Visualization Tool for Occupancy Collector Data using Rerun

This script visualizes data collected by the OccCollector, showing 3D maps,
robot trajectories, depth images, and TOF sensor readings with Rerun.

Usage:
    python occ_data_viz.py --npz_file <path_to_episode.npz> [--server_mode]
"""

import os
import sys
import argparse
import numpy as np
import open3d as o3d
import rerun as rr # pip install rerun-sdk
import rerun.blueprint as rrb  # Import blueprint UI library
from typing import Dict, List, Any, Tuple
import time as sys_time
import cv2

# Camera offsets (pos: [x,y,z], quat_wxyz: [w,x,y,z]) relative to robot body
# Based on TiledCameraCfg.OffsetCfg in quad_point_ctrl_cfg.py
# Robot Body Frame: +X Forward, +Y Left, +Z Up (ROS standard convention)
# Isaac Sim World Frame: Z-up
# Camera Local Frame: +Z Forward, +X Right, +Y Down (OpenCV/Optical Frame convention often used by Rerun for images)
# For Rerun, when logging a Transform3D for a camera, it assumes the camera looks along its local -Z or +Z axis
# depending on convention. For rr.Image, Rerun expects the image data itself.
# For rr.Arrows3D, vectors are from origin. If TOF sensor points along its local +X axis after offset rotation:
CAMERA_OFFSETS = {
    "tiled_camera": {
        "pos": np.array([0.05, 0.0, 0.0]),      # Robot body: +0.05m along X (forward)
        "quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0]) # No rotation relative to robot body (points robot's +X)
    },
    "left_camera": {
        "pos": np.array([0.0, 0.05, 0.0]),      # Robot body: +0.05m along Y (left)
        "quat_wxyz": np.array([np.cos(np.pi/4), 0.0, 0.0, np.sin(np.pi/4)]) # Rotated +90 deg about Z (points robot's +Y)
    },
    "right_camera": {
        "pos": np.array([0.0, -0.05, 0.0]),     # Robot body: -0.05m along Y (right)
        "quat_wxyz": np.array([np.cos(-np.pi/4), 0.0, 0.0, np.sin(-np.pi/4)]) # Rotated -90 deg about Z (points robot's -Y)
    },
    "back_camera": {
        "pos": np.array([-0.05, 0.0, 0.0]),     # Robot body: -0.05m along X (backward)
        "quat_wxyz": np.array([np.cos(np.pi/2), 0.0, 0.0, np.sin(np.pi/2)]) # Rotated 180 deg about Z (points robot's -X)
    }
}

# Rotation from RDF (Rerun Pinhole: X-Right, Y-Down, Z-Fwd) to FLU (ROS Body: X-Fwd, Y-Left, Z-Up)
# This is used as parent_from_child transform, so FLU_from_RDF.
# Standard quaternion wxyz: [0.5, -0.5, 0.5, -0.5] (Corrected from [0.5, -0.5, -0.5, -0.5])
FLU_FROM_RDF_WXYZ = np.array([0.5, -0.5, 0.5, -0.5])
# Rerun expects xyzw for rr.Quaternion
FLU_FROM_RDF_XYZW = np.array([FLU_FROM_RDF_WXYZ[1], FLU_FROM_RDF_WXYZ[2], FLU_FROM_RDF_WXYZ[3], FLU_FROM_RDF_WXYZ[0]])

TOF_SENSOR_CONFIG = {
    "left_tof": {"offset_key": "left_camera", "data_key": "left_depths", "color": [0, 255, 0]},
    "right_tof": {"offset_key": "right_camera", "data_key": "right_depths", "color": [255, 0, 255]},
    "back_tof": {"offset_key": "back_camera", "data_key": "back_depths", "color": [255, 255, 0]},
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize occupancy data with Rerun")
    parser.add_argument("--npz_file", type=str, required=True,
                        help="Path to episode NPZ file to visualize")
    parser.add_argument("--server_mode", action="store_true",
                        help="Run in server mode without GUI (useful for headless environments)")
    return parser.parse_args()

def wxyz_to_xyzw(wxyz: np.ndarray) -> np.ndarray:
    """Convert quaternion from [w, x, y, z] to Rerun's [x, y, z, w]."""
    return np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])

def quat_multiply(q1_wxyz: np.ndarray, q2_wxyz: np.ndarray) -> np.ndarray:
    """Multiply two quaternions q1 * q2 (both wxyz)."""
    w1, x1, y1, z1 = q1_wxyz
    w2, x2, y2, z2 = q2_wxyz
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,  # w
        w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
        w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
        w1*z2 + x1*y2 - y1*x2 + z1*w2   # z
    ])

def rotate_vector_by_quat(v_xyz: np.ndarray, q_wxyz: np.ndarray) -> np.ndarray:
    """Rotate a vector v by quaternion q."""
    q_vec = q_wxyz[1:]
    q_scalar = q_wxyz[0]
    t = 2.0 * np.cross(q_vec, v_xyz)
    v_rotated = v_xyz + q_scalar * t + np.cross(q_vec, t)
    return v_rotated

def get_map_path_from_npz(npz_path: str) -> str:
    """Extract map.pcd path from episode NPZ file path.

    Example NPZ path: /path/to/logs/occ_data/1747330730/map_0/env_1/episode_2.npz
    Expected map path: /path/to/logs/occ_data/1747330730/map_0/map.pcd
    """
    # Parse directory structure from NPZ path
    npz_dir = os.path.dirname(npz_path)  # e.g., /path/to/logs/occ_data/1747330730/map_0/env_1
    env_dir = os.path.basename(npz_dir)  # e.g., env_1

    if not env_dir.startswith("env_"):
        raise ValueError(f"NPZ file not in expected directory structure. Expected .../map_X/env_Y/episode_Z.npz, got {npz_path}")

    map_dir = os.path.dirname(npz_dir)  # e.g., /path/to/logs/occ_data/1747330730/map_0
    map_dir_name = os.path.basename(map_dir)  # e.g., map_0

    if not map_dir_name.startswith("map_"):
        raise ValueError(f"NPZ file not in expected directory structure. Expected .../map_X/env_Y/episode_Z.npz, got {npz_path}")

    map_pcd_path = os.path.join(map_dir, "map.pcd")
    return map_pcd_path

def extract_episode_info_from_npz(npz_path: str) -> Dict[str, Any]:
    """Extract episode information from NPZ file path.

    Example NPZ path: /path/to/logs/occ_data/1747330730/map_0/env_1/episode_2.npz
    """
    # Parse directory structure from NPZ path
    if not os.path.exists(npz_path):
        raise ValueError(f"NPZ file not found: {npz_path}")

    npz_dir = os.path.dirname(npz_path)  # e.g., /path/to/logs/occ_data/1747330730/map_0/env_1
    env_dir = os.path.basename(npz_dir)  # e.g., env_1
    env_id = env_dir.split("_")[-1] if env_dir.startswith("env_") else "0"

    map_dir = os.path.dirname(npz_dir)  # e.g., /path/to/logs/occ_data/1747330730/map_0
    map_dir_name = os.path.basename(map_dir)  # e.g., map_0
    map_id = map_dir_name.split("_")[-1] if map_dir_name.startswith("map_") else "0"

    timestamp_dir = os.path.dirname(map_dir)  # e.g., /path/to/logs/occ_data/1747330730
    timestamp_name = os.path.basename(timestamp_dir)  # e.g., 1747330730

    npz_filename = os.path.basename(npz_path)  # e.g., episode_2.npz
    episode_id = npz_filename.split("_")[-1].split(".")[0] if npz_filename.startswith("episode_") else "0"

    map_pcd_path = os.path.join(map_dir, "map.pcd")

    return {
        "timestamp_folder_name": timestamp_name,
        "map_id": map_id,
        "env_id": env_id,
        "episode_id": episode_id,
        "map_pcd_path": map_pcd_path,
        "npz_file_path": npz_path
    }

def read_map_pcd(map_file_path: str):
    """Read a PCD file using Open3D."""
    if not os.path.exists(map_file_path):
        raise ValueError(f"Map file not found: {map_file_path}")
    pcd = o3d.io.read_point_cloud(map_file_path)
    return pcd

def read_episode_data(episode_file_path: str):
    """Read episode data from an NPZ file."""
    if not os.path.exists(episode_file_path):
        raise ValueError(f"Episode file not found: {episode_file_path}")
    return np.load(episode_file_path)

def colorize_depth_image(depth_image: np.ndarray, min_depth: float = 0.0, max_depth: float = 5.0) -> np.ndarray:
    """Convert depth image to colormap for better visualization.

    Args:
        depth_image: Raw depth image (H, W) with actual distance values
        min_depth: Minimum depth value to map (will be colored blue)
        max_depth: Maximum depth value to map (will be colored red)

    Returns:
        RGB image with color-mapped depth values
    """
    # Normalize depth to 0-1 range for color mapping
    normalized = np.clip((depth_image - min_depth) / (max_depth - min_depth), 0, 1)

    # Convert to uint8 and apply colormap
    normalized_u8 = (normalized * 255).astype(np.uint8)
    color_mapped = cv2.applyColorMap(normalized_u8, cv2.COLORMAP_TURBO)

    # Mark invalid/out-of-range values as black
    invalid_mask = (depth_image <= 0) | (depth_image > max_depth) | np.isnan(depth_image)
    color_mapped[invalid_mask] = [0, 0, 0]

    return color_mapped

def set_rerun_time(timestamp_seconds: float):
    """Set the current Rerun timeline position using the correct API."""
    try:
        # Different versions of Rerun may have different APIs
        rr.set_time_sequence("sim_time", timestamp_seconds)
    except (AttributeError, TypeError):
        try:
            # Fallback to older API if needed
            rr.set_time_seconds(timestamp_seconds)
        except (AttributeError, TypeError):
            # If both fail, just log without time
            pass

def log_data_to_rerun(npz_file_path: str, server_mode: bool = False):
    """Load data and log it to Rerun."""
    # Initialize Rerun
    if server_mode:
        rr.init("occ_data_visualizer", server=True)
        print("Running in server mode. Connect with: rerun --connect rerun+http://127.0.0.1:9876/proxy")
    else:
        rr.init("occ_data_visualizer", spawn=True)

    try:
        # Extract episode info from path
        episode_info = extract_episode_info_from_npz(npz_file_path)
        print(f"Visualizing episode: {episode_info['timestamp_folder_name']}/map_{episode_info['map_id']}/env_{episode_info['env_id']}/episode_{episode_info['episode_id']}")

        # Construct entity paths
        ts_name = episode_info['timestamp_folder_name']
        map_id = episode_info['map_id']
        env_id = episode_info['env_id']
        ep_id = episode_info['episode_id']

        episode_run_path = f"{ts_name}/map_{map_id}/env_{env_id}/episode_{ep_id}"
        map_entity_path = f"{ts_name}/map_{map_id}/world/map_points"

        # Add description about this dataset
        rr.log(
            "description",
            rr.TextDocument(
                f"# Occupancy Collector Data\n\n"
                f"* **Episode**: {ts_name}/map_{map_id}/env_{env_id}/episode_{ep_id}\n"
                f"* **Features**: 3D map, robot trajectory, depth images, TOF sensors\n"
                f"* **Colorized depth** is textured onto camera frustums\n",
                media_type=rr.MediaType.MARKDOWN
            )
        )

        # Load map PCD
        map_pcd_path = episode_info['map_pcd_path']
        try:
            print(f"Loading map from: {map_pcd_path}")
            map_pcd = read_map_pcd(map_pcd_path)

            # Filter map points to keep only those between 0.1 and 1.8 meters in height
            map_points = np.asarray(map_pcd.points)
            map_colors = np.asarray(map_pcd.colors) if map_pcd.has_colors() else np.full((len(map_points), 3), 150, dtype=np.uint8)

            # Apply height filter (z-coordinate) between 0.1 and 1.8m
            height_mask = (map_points[:, 2] >= 0.1) & (map_points[:, 2] <= 1.8)
            filtered_points = map_points[height_mask]
            filtered_colors = map_colors[height_mask] if map_pcd.has_colors() else np.array([150, 150, 150], dtype=np.uint8)

            # Log filtered map points
            print(f"Filtered map points: {len(filtered_points)} out of {len(map_points)} (keeping points between 0.1-1.8m height)")
            rr.log(
                map_entity_path,
                rr.Points3D(
                    positions=filtered_points,
                    colors=filtered_colors
                )
            )
        except Exception as e:
            print(f"Error loading map {map_pcd_path}: {e}", file=sys.stderr)
            print("Continuing without map visualization.")

        # Load episode data
        try:
            episode_data = read_episode_data(episode_info['npz_file_path'])
        except Exception as e:
            print(f"Error loading episode {episode_info['npz_file_path']}: {e}", file=sys.stderr)
            return

        timestamps = episode_data['timestamps']
        states = episode_data['states']
        depth_images = episode_data['depth_images']

        # Log robot trajectory
        trajectory_positions = states[:, :3]
        rr.log(
            f"{episode_run_path}/world/robot_trajectory",
            rr.LineStrips3D([trajectory_positions], colors=[0,255,0])
        )

        # Display stats about the data
        num_frames = len(timestamps)
        duration = timestamps[-1] - timestamps[0] if num_frames > 0 else 0
        print(f"Episode contains {num_frames} frames over {duration:.2f} seconds")

        # Log depth image stats (min, max, shape)
        if depth_images.size > 0:
            depth_shape = depth_images[0].shape
            depth_min = np.min(depth_images)
            depth_max = np.max(depth_images)
            print(f"Depth images: shape={depth_shape}, range=[{depth_min:.2f}, {depth_max:.2f}]")

        # Define the blueprint (but use recording method instead of set_ui_layout for compatibility)
        sensor_views = [
            rrb.Spatial2DView(
                name="tiled_camera",
                origin=f"{episode_run_path}/world/tiled_camera/camera",
                defaults=[rr.components.ImagePlaneDistance(5.0)],
                overrides={"world/anns": [rr.components.FillModeBatch("solid")]},
            )
        ]

        # Add other camera views for the UI but not as 3D frustums
        for sensor_name in ["left_camera", "right_camera", "back_camera"]:
            sensor_views.append(
                rrb.Spatial2DView(
                    name=sensor_name,
                    origin=f"{episode_run_path}/world/{sensor_name}/direction_indicator",
                    defaults=[rr.components.ImagePlaneDistance(5.0)],
                    overrides={"world/anns": [rr.components.FillModeBatch("solid")]},
                )
            )

        blueprint = rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(name="3D", origin="world"),
                rrb.Vertical(
                    rrb.TextDocumentView(origin="description", name="Description"),
                    rrb.MapView(
                        origin="world",
                        name="MapView",
                        zoom=rrb.archetypes.MapZoom(18.0),
                        background=rrb.archetypes.MapBackground(
                            rrb.components.MapProvider.OpenStreetMap
                        ),
                    ),
                    row_shares=[1,1],
                ),
                column_shares=[3,1],
            ),
            rrb.Grid(*sensor_views),
            row_shares=[4,2],
        )

        # Use blueprint.record() instead of rr.set_ui_layout for compatibility with older Rerun SDK
        try:
            # Newer Rerun SDK method
            rr.set_ui_layout(blueprint)
        except (AttributeError, TypeError):
            try:
                # Alternative method for older Rerun SDK
                blueprint.record()
            except Exception as e:
                print(f"Warning: Could not set UI layout: {e}. Continuing with default UI.")

        # Log frame data for the episode
        for i in range(len(timestamps)):
            current_time_seconds = float(timestamps[i])
            set_rerun_time(current_time_seconds)

            # Robot position and orientation
            robot_pos_w = states[i, 0:3]
            robot_quat_wxyz = states[i, 3:7]
            robot_quat_xyzw_rerun = wxyz_to_xyzw(robot_quat_wxyz)

            # Log robot transform
            robot_entity_path = f"{episode_run_path}/world/robot"
            rr.log(
                robot_entity_path,
                rr.Transform3D(
                    translation=robot_pos_w,
                    rotation=rr.Quaternion(xyzw=robot_quat_xyzw_rerun)
                )
            )
            # Log robot visualization
            rr.log(
                f"{robot_entity_path}/mesh",
                rr.Boxes3D(centers=[0,0,0], half_sizes=[0.025, 0.025, 0.0125], colors=[255,0,0])
            )

            # Main camera transforms and data
            main_cam_name = "tiled_camera"
            main_cam_offset_info = CAMERA_OFFSETS[main_cam_name]
            main_cam_offset_pos_robot = main_cam_offset_info["pos"]
            main_cam_offset_quat_robot_wxyz = main_cam_offset_info["quat_wxyz"]

            main_cam_world_pos = robot_pos_w + rotate_vector_by_quat(main_cam_offset_pos_robot, robot_quat_wxyz)
            main_cam_world_quat_wxyz = quat_multiply(robot_quat_wxyz, main_cam_offset_quat_robot_wxyz)
            main_cam_world_quat_xyzw_rerun = wxyz_to_xyzw(main_cam_world_quat_wxyz)

            main_cam_world_entity_path = f"{episode_run_path}/world/{main_cam_name}"
            rr.log(
                main_cam_world_entity_path,
                rr.Transform3D(
                    translation=main_cam_world_pos,
                    rotation=rr.Quaternion(xyzw=main_cam_world_quat_xyzw_rerun)
                )
            )

            # Get depth image and dimensions for the main camera
            depth_img = depth_images[i]
            H, W = depth_img.shape

            # Log the camera as a box
            rr.log(
                f"{main_cam_world_entity_path}/mesh",
                rr.Boxes3D(centers=[0,0,0], half_sizes=[0.01, 0.02, 0.01], colors=[0,0,255])
            )

            # Apply the FLU_from_RDF rotation to the '/camera' sub-entity.
            # This orients the Pinhole sensor correctly (Z-forward) within the camera body (which is X-forward).
            rr.log(
                f"{main_cam_world_entity_path}/camera",
                rr.Transform3D(rotation=rr.Quaternion(xyzw=FLU_FROM_RDF_XYZW))
            )

            # Add Pinhole camera entity for the main camera to the same '/camera' path.
            # It will inherit the local FLU_from_RDF transform and the parent's world transform.
            # Using intrinsics appropriate for 8x8 images
            fx, fy = 8.0, 8.0  # Adjusted focal length for 8x8 image
            cx, cy = W/2, H/2  # Principal point (center of image)
            rr.log(
                f"{main_cam_world_entity_path}/camera",
                rr.Pinhole(
                    resolution=[W, H],
                    focal_length=[fx, fy],
                    principal_point=[cx, cy],
                )
            )

            # Create colorized version for better visualization and log it as camera/image
            # for texturing onto the image plane
            colorized_depth = colorize_depth_image(depth_img, min_depth=0.1, max_depth=5.0)
            rr.log(
                f"{main_cam_world_entity_path}/camera/image",
                rr.Image(colorized_depth)
            )

            # Keep the raw depth image
            rr.log(
                f"{main_cam_world_entity_path}/depth_raw",
                rr.Image(depth_img)
            )

            # TOF sensor visualization
            for tof_sensor_name, config in TOF_SENSOR_CONFIG.items():
                offset_key = config["offset_key"]
                data_key = config["data_key"]
                tof_color = config["color"]

                if data_key not in episode_data:
                    continue

                tof_depth_value = float(episode_data[data_key][i])

                tof_cam_offset_info = CAMERA_OFFSETS[offset_key]
                tof_cam_offset_pos_robot = tof_cam_offset_info["pos"]
                tof_cam_offset_quat_robot_wxyz = tof_cam_offset_info["quat_wxyz"]

                tof_cam_world_pos = robot_pos_w + rotate_vector_by_quat(tof_cam_offset_pos_robot, robot_quat_wxyz)
                tof_cam_world_quat_wxyz = quat_multiply(robot_quat_wxyz, tof_cam_offset_quat_robot_wxyz)
                tof_cam_world_quat_xyzw_rerun = wxyz_to_xyzw(tof_cam_world_quat_wxyz)

                tof_cam_world_entity_path = f"{episode_run_path}/world/{offset_key}"
                rr.log(
                    tof_cam_world_entity_path,
                    rr.Transform3D(
                        translation=tof_cam_world_pos,
                        rotation=rr.Quaternion(xyzw=tof_cam_world_quat_xyzw_rerun)
                    )
                )
                rr.log(
                    f"{tof_cam_world_entity_path}/marker",
                    rr.Points3D(positions=[[0,0,0]], radii=[0.01], colors=tof_color)
                )

                # Add direction indicator arrow to show camera orientation
                forward_vector = np.array([0.2, 0, 0])  # Local +X is forward in our setup
                rr.log(
                    f"{tof_cam_world_entity_path}/direction_indicator",
                    rr.Arrows3D(
                        origins=[[0,0,0]],
                        vectors=[forward_vector],
                        colors=tof_color,
                        radii=0.005
                    )
                )

                # Only add Pinhole camera for the main front camera (tiled_camera)
                if offset_key == "tiled_camera":
                    # Add Pinhole camera entity for the main front camera
                    rr.log(
                        f"{tof_cam_world_entity_path}/camera",
                        rr.Pinhole(
                            resolution=[W, H],
                            focal_length=[fx, fy],  # Using the same fx, fy as main camera (8.0)
                            principal_point=[cx, cy],
                        )
                    )

                    # Create colorized version for better visualization and log it as camera/image
                    # for texturing onto the image plane
                    colorized_depth = colorize_depth_image(depth_img, min_depth=0.1, max_depth=5.0)
                    rr.log(
                        f"{tof_cam_world_entity_path}/camera/image",
                        rr.Image(colorized_depth)
                    )
                else:
                    # For other cameras, create a simple view without a 3D frustum
                    # Create a simple depth visualization for secondary cameras
                    if 0.01 < tof_depth_value < 5.0:
                        # Create a simple colorized depth image based on the TOF value
                        norm_depth = np.clip((tof_depth_value - 0.1) / 4.9, 0, 1)
                        r, g, b = tof_color
                        brightness = 1.0 - norm_depth  # Closer is brighter
                        simple_img = np.ones((H, W, 3), dtype=np.uint8)
                        simple_img[:, :, 0] = int(r * brightness)
                        simple_img[:, :, 1] = int(g * brightness)
                        simple_img[:, :, 2] = int(b * brightness)

                        # Log the image to the direction_indicator path for the 2D view
                        rr.log(
                            f"{tof_cam_world_entity_path}/direction_indicator/image",
                            rr.Image(simple_img)
                        )

                # TOF sensor visualization (arrows showing distance)
                tof_max_viz_range = 4.0
                if 0.01 < tof_depth_value < tof_max_viz_range:
                    # Assuming TOF sensor points along its local +X axis after offset rotation
                    local_ray_vector = np.array([tof_depth_value, 0.0, 0.0])
                    rr.log(
                        f"{tof_cam_world_entity_path}/ray",
                        rr.Arrows3D(
                            origins=[[0,0,0]],
                            vectors=[local_ray_vector],
                            colors=tof_color,
                            radii=0.005
                        )
                    )

                # Display TOF depth value as text
                rr.log(
                    f"{tof_cam_world_entity_path}/depth_value",
                    rr.TextDocument(f"{tof_depth_value:.2f}m", media_type=rr.MediaType.MARKDOWN)
                )

        print("Data logging to Rerun complete. The Rerun viewer should be open or accessible.")
        if server_mode:
            print("Running in server mode. Connect with: rerun --connect rerun+http://127.0.0.1:9876/proxy")
            # Keep server running until interrupted
            try:
                while True:
                    sys_time.sleep(1)
            except KeyboardInterrupt:
                print("Keyboard interrupt received, shutting down server...")
        else:
            print("If the viewer did not spawn, ensure your DISPLAY environment is correctly set (e.g., 'export DISPLAY=:0'),")
            print("or consider running this script with the --server_mode flag for headless operation.")

    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()

def main():
    """Main execution function."""
    args = parse_args()
    log_data_to_rerun(args.npz_file, args.server_mode)

if __name__ == "__main__":
    main()