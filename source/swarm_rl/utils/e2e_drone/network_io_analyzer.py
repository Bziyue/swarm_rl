#!/usr/bin/env python3

import numpy as np
import argparse
from pathlib import Path

class NetworkIOAnalyzer:
    """Analyze network input/output distributions from collected data."""

    def __init__(self):
        self.obs_components = {}
        self.action_components = {}
        self._setup_observation_layout()

    def _setup_observation_layout(self):
        """Define observation structure from quad_baseline_env.py lines 757-769, 813-817"""
        # Frame observation components (from lines 759-769)
        frame_components = [
            ("velocity", 3, "Body frame velocity [vx, vy, vz] * 25"),
            ("rotation_matrix", 9, "Rotation matrix flattened * 127"),
            ("goal_direction", 3, "Direction to goal * 127"),
            ("desired_speed", 1, "Desired speed * 25"),
            ("last_actions", 4, "Previous actions * 127"),
            ("left_tof", 1, "Left TOF distance * 30"),
            ("right_tof", 1, "Right TOF distance * 30"),
            ("back_tof", 1, "Back TOF distance * 30"),
            ("down_tof", 1, "Down TOF distance * 30"),
        ]

        # History and depth components (from lines 814-817)
        history_length = 5
        depth_dims = 8 * 8  # 8x8 depth camera

        # Build layout
        offset = 0

        # History of frame observations (5 frames)
        for i in range(history_length):
            for comp_name, comp_size, desc in frame_components:
                key = f"history_{i}_{comp_name}"
                self.obs_components[key] = (offset, offset + comp_size, f"Frame {i} {desc}")
                offset += comp_size

        # Depth history (1 frame of 64 pixels)
        self.obs_components["depth_image"] = (offset, offset + depth_dims, "Depth camera 8x8 * 30")

        # Action components (4D)
        self.action_components = {
            "roll": (0, 1, "Roll rate command"),
            "pitch": (1, 2, "Pitch rate command"),
            "yaw_rate": (2, 3, "Yaw rate command"),
            "thrust": (3, 4, "Thrust command")
        }

    def load_data(self, data_dir):
        """Load observation and action data from directory"""
        data_path = Path(data_dir)
        obs_dir = data_path / "obs"
        action_dir = data_path / "action"

        if not obs_dir.exists() or not action_dir.exists():
            raise FileNotFoundError("Directory must contain 'obs' and 'action' subdirectories")

        # Load all .npy files
        obs_files = sorted(obs_dir.glob("*.npy"))
        action_files = sorted(action_dir.glob("*.npy"))

        if not obs_files or not action_files:
            raise FileNotFoundError("No .npy files found")

        # Load and concatenate data
        obs_list = []
        action_list = []

        for obs_file in obs_files:
            obs_data = np.load(obs_file)
            if obs_data.ndim == 2:  # (batch, features)
                obs_list.extend(obs_data)
            else:  # single sample
                obs_list.append(obs_data)

        for action_file in action_files:
            action_data = np.load(action_file)
            if action_data.ndim == 2:
                action_list.extend(action_data)
            else:
                action_list.append(action_data)

        self.obs_data = np.array(obs_list)
        self.action_data = np.array(action_list)

        print(f"Loaded {len(obs_list)} observation samples with {self.obs_data.shape[1]} features")
        print(f"Loaded {len(action_list)} action samples with {self.action_data.shape[1]} features")

    def analyze_component(self, component_name):
        """Analyze a specific component and print statistics"""
        if component_name.startswith("obs_"):
            comp_key = component_name[4:]
            if comp_key not in self.obs_components:
                print(f"Unknown observation component: {comp_key}")
                return

            start, end, desc = self.obs_components[comp_key]
            data = self.obs_data[:, start:end]
            data_type = "Observation"

        elif component_name.startswith("action_"):
            comp_key = component_name[7:]
            if comp_key not in self.action_components:
                print(f"Unknown action component: {comp_key}")
                return

            start, end, desc = self.action_components[comp_key]
            data = self.action_data[:, start:end]
            data_type = "Action"
        else:
            print(f"Component name must start with 'obs_' or 'action_'")
            return

        print(f"\n{data_type} Component: {comp_key}")
        print(f"Description: {desc}")
        print(f"Shape: {data.shape}")
        print(f"Data type: {data.dtype}")

        # Statistics for each dimension
        for dim in range(data.shape[1]):
            dim_data = data[:, dim]
            print(f"\nDimension {dim}:")
            print(f"  Mean: {np.mean(dim_data):.6f}")
            print(f"  Std:  {np.std(dim_data):.6f}")
            print(f"  Min:  {np.min(dim_data):.6f}")
            print(f"  Max:  {np.max(dim_data):.6f}")
            print(f"  Median: {np.median(dim_data):.6f}")

            # Percentiles
            p25, p75 = np.percentile(dim_data, [25, 75])
            print(f"  Q1:   {p25:.6f}")
            print(f"  Q3:   {p75:.6f}")

            # Check for outliers (> 3 std from mean)
            outliers = np.abs(dim_data - np.mean(dim_data)) > 3 * np.std(dim_data)
            outlier_count = np.sum(outliers)
            if outlier_count > 0:
                print(f"  Outliers (>3σ): {outlier_count} ({100*outlier_count/len(dim_data):.2f}%)")

    def list_components(self):
        """List all available components"""
        print("Available observation components:")
        for name, (start, end, desc) in self.obs_components.items():
            if end <= self.obs_data.shape[1]:
                print(f"  obs_{name}: {desc} (indices {start}:{end})")

        print("\nAvailable action components:")
        for name, (start, end, desc) in self.action_components.items():
            if end <= self.action_data.shape[1]:
                print(f"  action_{name}: {desc} (indices {start}:{end})")

def main():
    """CLI interface"""
    parser = argparse.ArgumentParser(description="Analyze network I/O distributions")
    parser.add_argument("data_dir", help="Directory containing obs/ and action/ subdirectories with .npy files")
    parser.add_argument("--component", help="Component to analyze (e.g. obs_velocity, action_thrust)")
    parser.add_argument("--list", action="store_true", help="List all available components")

    args = parser.parse_args()

    analyzer = NetworkIOAnalyzer()

    try:
        analyzer.load_data(args.data_dir)
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1

    if args.list:
        analyzer.list_components()

    if args.component:
        analyzer.analyze_component(args.component)

    if not args.list and not args.component:
        print("Use --list to see available components or --component <name> to analyze specific component")
        analyzer.list_components()

if __name__ == "__main__":
    main()