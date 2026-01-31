"""
Global Planning Client for communicating with global_planning_server via REST API.
Provides functionality to load maps and request trajectory planning.
"""

import base64
import requests
import numpy as np
from typing import List, Dict, Optional, Any
import time
import logging

class GlobalPlanningClient:
    """Client for interacting with the global planning server."""

    def __init__(self, server_host: str = "localhost", server_port: int = 8080, timeout: float = 30.0):
        """
        Initialize the global planning client.

        Args:
            server_host: Hostname/IP of the global planning server
            server_port: Port number of the global planning server
            timeout: Default timeout for requests in seconds
        """
        self.base_url = f"http://{server_host}:{server_port}"
        self.timeout = timeout
        self.session = requests.Session()
        self.logger = logging.getLogger(self.__class__.__name__)

    def check_health(self) -> Dict[str, Any]:
        """
        Check server health status.

        Returns:
            Dictionary containing server health information

        Raises:
            requests.RequestException: If the request fails
        """
        try:
            response = self.session.get(
                f"{self.base_url}/api/health",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            self.logger.error(f"Health check failed: {e}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """
        Get server performance statistics.

        Returns:
            Dictionary containing performance statistics

        Raises:
            requests.RequestException: If the request fails
        """
        try:
            response = self.session.get(
                f"{self.base_url}/api/stats",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            self.logger.error(f"Failed to get stats: {e}")
            raise

    def load_map_from_file(self, map_file_path: str, voxel_width: float = 0.1,
                          map_bound: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Load a point cloud map from a file path on the server.

        Args:
            map_file_path: Path to the .pcd file on the server
            voxel_width: Voxel size for discretization
            map_bound: Map boundaries as [x_min, x_max, y_min, y_max, z_min, z_max]

        Returns:
            Dictionary containing the response from the server

        Raises:
            requests.RequestException: If the request fails
        """
        payload = {
            "map_file": map_file_path,
            "voxel_width": voxel_width
        }

        if map_bound is not None:
            if len(map_bound) != 6:
                raise ValueError("map_bound must have 6 elements: [x_min, x_max, y_min, y_max, z_min, z_max]")
            payload["map_bound"] = map_bound

        try:
            response = self.session.post(
                f"{self.base_url}/api/load_map",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            self.logger.error(f"Failed to load map from file {map_file_path}: {e}")
            raise

    def load_map_from_data(self, pcd_data: bytes, voxel_width: float = None,
                          map_bound: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Load a point cloud map from binary data.

        Args:
            pcd_data: Binary PCD file data
            voxel_width: Voxel width for processing the point cloud
            map_bound: Map boundaries as [x_min, x_max, y_min, y_max, z_min, z_max]

        Returns:
            Dictionary containing the response from the server

        Raises:
            requests.RequestException: If the request fails
        """
        # Encode binary data to base64
        encoded_data = base64.b64encode(pcd_data).decode('utf-8')

        payload = {
            "map_data": encoded_data,
            "format": "pcd_base64"
        }

        if voxel_width is not None:
            payload["voxel_width"] = voxel_width

        if map_bound is not None:
            if len(map_bound) != 6:
                raise ValueError("map_bound must have 6 elements: [x_min, x_max, y_min, y_max, z_min, z_max]")
            payload["map_bound"] = map_bound
            print(f"Map boundaries set to: {map_bound}")

        try:
            response = self.session.post(
                f"{self.base_url}/api/map",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            self.logger.error(f"Failed to load map from data: {e}")
            raise

    def plan_trajectory(self, start: List[float], goal: List[float],
                       max_wait_time: float = 30.0) -> Dict[str, Any]:
        """
        Request trajectory planning between start and goal points.

        Args:
            start: Start position as [x, y, z]
            goal: Goal position as [x, y, z]
            max_wait_time: Maximum time to wait for planning result (unused by server)

        Returns:
            Dictionary containing planning result with trajectory data

        Raises:
            requests.RequestException: If the request fails
            ValueError: If start/goal have invalid dimensions
        """
        if len(start) != 3 or len(goal) != 3:
            raise ValueError("Start and goal must be 3D coordinates [x, y, z]")

        payload = {
            "start": start,
            "goal": goal
        }

        try:
            response = self.session.post(
                f"{self.base_url}/api/plan",
                json=payload,
                timeout=max(self.timeout, max_wait_time + 5.0)  # Add buffer to request timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            self.logger.error(f"Failed to plan trajectory from {start} to {goal}: {e}")
            raise

    def extract_trajectory_arrays(self, planning_result: Dict[str, Any]) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract trajectory arrays from planning result.

        Args:
            planning_result: Result dictionary from plan_trajectory()

        Returns:
            Dictionary containing trajectory arrays as numpy arrays, or None if planning failed
        """
        if planning_result.get("status") != "success":
            error_msg = planning_result.get("error", "Unknown error")
            self.logger.warning(f"Planning failed: {error_msg}")
            return None

        trajectory_data = planning_result.get("trajectory", {})
        if "points" not in trajectory_data:
            self.logger.warning("No trajectory points found in response")
            return None

        # Convert trajectory points to separate arrays
        points = trajectory_data["points"]
        if not points:
            self.logger.warning("Empty trajectory points")
            return None

        # Extract individual arrays from trajectory points
        trajectory_arrays = {
            "positions": [],
            "velocities": [],
            "accelerations": [],
            "jerks": [],
            "quaternions": [],
            "angular_velocities": [],
            "thrusts": [],
            "timestamp": []
        }

        for point in points:
            trajectory_arrays["positions"].append(point.get("position", [0, 0, 0]))
            trajectory_arrays["velocities"].append(point.get("velocity", [0, 0, 0]))
            trajectory_arrays["accelerations"].append(point.get("acceleration", [0, 0, 0]))
            trajectory_arrays["jerks"].append(point.get("jerk", [0, 0, 0]))
            trajectory_arrays["quaternions"].append(point.get("quaternion", [0, 0, 0, 1]))
            trajectory_arrays["angular_velocities"].append(point.get("angular_velocity", [0, 0, 0]))
            trajectory_arrays["thrusts"].append(point.get("thrust", 0))
            trajectory_arrays["timestamp"].append(point.get("timestamp", 0))

        # Convert to numpy arrays
        for key, values in trajectory_arrays.items():
            trajectory_arrays[key] = np.array(values)

        return trajectory_arrays

    def is_server_available(self) -> bool:
        """
        Check if the global planning server is available.

        Returns:
            True if server is reachable and healthy, False otherwise
        """
        try:
            health = self.check_health()
            return health.get("status") == "healthy"
        except Exception as e:
            self.logger.debug(f"Server availability check failed: {e}")
            return False

    def wait_for_server(self, max_wait_time: float = 30.0, check_interval: float = 1.0) -> bool:
        """
        Wait for the global planning server to become available.

        Args:
            max_wait_time: Maximum time to wait in seconds
            check_interval: Time between availability checks

        Returns:
            True if server becomes available, False if timeout
        """
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            if self.is_server_available():
                return True
            time.sleep(check_interval)

        self.logger.warning(f"Server did not become available within {max_wait_time} seconds")
        return False


def create_maze_map_bounds(maze_size_x: float, maze_size_y: float, maze_height: float,
                          center_x: float = 0.0, center_y: float = 0.0) -> List[float]:
    """
    Create map boundaries that match maze dimensions.

    Args:
        maze_size_x: Size of maze in x direction
        maze_size_y: Size of maze in y direction
        maze_height: Height of maze
        center_x: Center x coordinate of maze
        center_y: Center y coordinate of maze

    Returns:
        Map boundaries as [x_min, x_max, y_min, y_max, z_min, z_max]
    """
    x_min = center_x - maze_size_x / 2.0
    x_max = center_x + maze_size_x / 2.0
    y_min = center_y - maze_size_y / 2.0
    y_max = center_y + maze_size_y / 2.0
    z_min = 0.0  # Ground level
    z_max = maze_height

    return [x_min, x_max, y_min, y_max, z_min, z_max]


def create_pcd_from_points(points) -> str:
    """
    Create a PCD format string from numpy points array.

    Args:
        points: Numpy array of 3D points [(x, y, z), ...]

    Returns:
        PCD format string ready for transmission
    """
    header = """# .PCD v.7 - Point Cloud Data file format
VERSION .7
FIELDS x y z
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
WIDTH {}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {}
DATA ascii
""".format(len(points), len(points))

    # Add point data
    point_data = ""
    for point in points:
        point_data += f"{point[0]} {point[1]} {point[2]}\n"

    return header + point_data


def filter_points_within_bounds(points, map_bounds: List[float]):
    """
    Filter points to only include those within the specified map boundaries.

    Args:
        points: Numpy array or list of 3D points [(x, y, z), ...]
        map_bounds: Map boundaries as [x_min, x_max, y_min, y_max, z_min, z_max]

    Returns:
        Filtered points array containing only points within bounds
    """
    if len(map_bounds) != 6:
        raise ValueError("map_bounds must have 6 elements: [x_min, x_max, y_min, y_max, z_min, z_max]")

    x_min, x_max, y_min, y_max, z_min, z_max = map_bounds

    # Convert to numpy array if it isn't already
    points_array = np.array(points)

    # Create boolean mask for points within bounds
    within_bounds = (
        (points_array[:, 0] >= x_min) & (points_array[:, 0] <= x_max) &  # x bounds
        (points_array[:, 1] >= y_min) & (points_array[:, 1] <= y_max) &  # y bounds
        (points_array[:, 2] >= z_min) & (points_array[:, 2] <= z_max)    # z bounds
    )

    # Filter points using the mask
    filtered_points = points_array[within_bounds]

    return filtered_points


class GlobalPlanningManager:
    """
    High-level manager for global planning operations.
    Provides simplified interfaces for environment integration.
    """

    def __init__(self, server_host: str = "localhost", server_port: int = 8080):
        """
        Initialize the global planning manager.

        Args:
            server_host: Hostname/IP of the global planning server
            server_port: Port number of the global planning server
        """
        self.client = GlobalPlanningClient(server_host, server_port)
        self.expert_trajectories = {}
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_environment_map(self, env_data: Dict[str, Any], env_spacing: float,
                           maze_height: float, center_x: float = 0.0, center_y: float = 0.0,
                           voxel_width: float = 0.1) -> bool:
        """
        Load environment map data into the global planning server.

        Args:
            env_data: Environment data containing "points" key with obstacle points
            env_spacing: Size of the environment space
            maze_height: Height of the maze walls
            center_x: X coordinate of maze center in world coordinates
            center_y: Y coordinate of maze center in world coordinates
            voxel_width: Voxel width for processing the point cloud

        Returns:
            True if map loaded successfully, False otherwise
        """
        try:
            # Check if server is available
            if not self.client.is_server_available():
                self.logger.warning("Global planning server is not available for map loading")
                return False

            # Get map boundaries that match maze dimensions
            map_bounds = create_maze_map_bounds(
                maze_size_x=env_spacing,
                maze_size_y=env_spacing,
                maze_height=maze_height,
                center_x=center_x,
                center_y=center_y
            )

            # Convert point cloud data to PCD format
            if "points" in env_data and len(env_data["points"]) > 0:
                points = env_data["points"]
                # Filter points within map boundaries
                filtered_points = filter_points_within_bounds(points, map_bounds)

                if len(filtered_points) == 0:
                    self.logger.warning("No points remain after filtering within map bounds")
                    return False

                pcd_content = create_pcd_from_points(filtered_points)

                # Load map from data
                response = self.client.load_map_from_data(
                    pcd_data=pcd_content.encode('utf-8'),
                    voxel_width=voxel_width,
                    map_bound=map_bounds
                )

                if response.get("status") == "success":
                    self.logger.info(f"Successfully loaded map to global planner")
                    self.logger.info(f"Points: {len(points)} total, {len(filtered_points)} within bounds")
                    self.logger.info(f"Map boundaries: {map_bounds}")
                    return True
                else:
                    error_msg = response.get("error", response.get("message", "Unknown error"))
                    self.logger.error(f"Failed to load map to global planner: {error_msg}")
                    return False

            else:
                self.logger.warning("No point cloud data available for global planner")
                return False

        except Exception as e:
            self.logger.error(f"Error loading map to global planner: {e}")
            return False

    def generate_expert_trajectories_batch(self, env_ids, start_positions, goal_positions) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Generate expert trajectories for multiple environments in batch.

        Args:
            env_ids: List/tensor of environment IDs
            start_positions: List/tensor of start positions [[x,y,z], ...]
            goal_positions: List/tensor of goal positions [[x,y,z], ...]

        Returns:
            Dictionary mapping env_id to trajectory data
        """
        results = {}

        try:
            # Check if server is available
            if not self.client.is_server_available():
                self.logger.warning("Global planning server is not available")
                return results

            # Generate trajectories for each environment
            for i, env_id in enumerate(env_ids):
                env_id_int = int(env_id) if hasattr(env_id, 'item') else int(env_id)

                # Convert positions to lists if they're tensors
                if hasattr(start_positions[i], 'cpu'):
                    start_pos = start_positions[i].cpu().numpy().tolist()
                else:
                    start_pos = list(start_positions[i])

                if hasattr(goal_positions[i], 'cpu'):
                    goal_pos = goal_positions[i].cpu().numpy().tolist()
                else:
                    goal_pos = list(goal_positions[i])

                # Request trajectory planning
                try:
                    planning_result = self.client.plan_trajectory(
                        start=start_pos,
                        goal=goal_pos,
                        max_wait_time=10.0  # Shorter timeout for training environments
                    )

                    if planning_result.get("status") == "success":
                        # Extract trajectory arrays
                        trajectory_arrays = self.client.extract_trajectory_arrays(planning_result)
                        if trajectory_arrays is not None:
                            # Store trajectory
                            results[env_id_int] = trajectory_arrays
                            self.expert_trajectories[env_id_int] = trajectory_arrays
                            self.logger.info(f"Generated expert trajectory for env {env_id_int}: {len(trajectory_arrays['positions'])} waypoints")
                        else:
                            self.logger.warning(f"Failed to extract trajectory arrays for env {env_id_int}")
                    else:
                        error_msg = planning_result.get("error", "Unknown planning error")
                        self.logger.warning(f"Planning failed for env {env_id_int}: {error_msg}")
                        # Clear any existing trajectory for this environment
                        self.expert_trajectories.pop(env_id_int, None)

                except Exception as e:
                    self.logger.error(f"Exception during trajectory planning for env {env_id_int}: {e}")
                    # Clear any existing trajectory for this environment
                    self.expert_trajectories.pop(env_id_int, None)

        except Exception as e:
            self.logger.error(f"Error in expert trajectory generation: {e}")

        return results

    def get_expert_trajectory(self, env_id: int) -> Optional[Dict[str, np.ndarray]]:
        """Get the expert trajectory for a specific environment ID."""
        return self.expert_trajectories.get(env_id, None)

    def has_expert_trajectory(self, env_id: int) -> bool:
        """Check if an expert trajectory exists for a specific environment ID."""
        return env_id in self.expert_trajectories

    def get_all_expert_trajectories(self) -> Dict[int, Dict[str, np.ndarray]]:
        """Get all expert trajectories as a dictionary."""
        return self.expert_trajectories.copy()

    def clear_expert_trajectory(self, env_id: int):
        """Clear expert trajectory for a specific environment."""
        self.expert_trajectories.pop(env_id, None)

    def clear_all_expert_trajectories(self):
        """Clear all expert trajectories."""
        self.expert_trajectories.clear()

    def is_server_available(self) -> bool:
        """Check if the global planning server is available."""
        return self.client.is_server_available()


# Test functions
def test_server_health(host: str = "localhost", port: int = 8080) -> bool:
    """Test server health and basic connectivity."""
    print(f"Testing server health at {host}:{port}")
    client = GlobalPlanningClient(host, port)

    try:
        health = client.check_health()
        print(f"✓ Health check successful: {health}")
        return True
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False


def test_server_stats(host: str = "localhost", port: int = 8080) -> bool:
    """Test server statistics endpoint."""
    print(f"Testing server stats at {host}:{port}")
    client = GlobalPlanningClient(host, port)

    try:
        stats = client.get_stats()
        print(f"✓ Stats retrieval successful: {stats}")
        return True
    except Exception as e:
        print(f"✗ Stats retrieval failed: {e}")
        return False


def test_map_loading(host: str = "localhost", port: int = 8080) -> bool:
    """Test map loading functionality with sample data."""
    print(f"Testing map loading at {host}:{port}")
    manager = GlobalPlanningManager(host, port)

    # Create sample point cloud data
    sample_points = np.array([
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 1.5],
        [3.0, 3.0, 2.0],
        [-1.0, -1.0, 0.5],
        [0.0, 0.0, 1.0]
    ])

    env_data = {"points": sample_points}

    try:
        success = manager.load_environment_map(
            env_data=env_data,
            env_spacing=10.0,
            maze_height=3.0,
            center_x=0.0,
            center_y=0.0
        )

        if success:
            print("✓ Map loading successful")
            return True
        else:
            print("✗ Map loading failed")
            return False
    except Exception as e:
        print(f"✗ Map loading failed with exception: {e}")
        return False


def test_trajectory_planning(host: str = "localhost", port: int = 8080) -> bool:
    """Test trajectory planning functionality."""
    print(f"Testing trajectory planning at {host}:{port}")
    client = GlobalPlanningClient(host, port)

    start = [0.0, 0.0, 1.0]
    goal = [5.0, 5.0, 1.0]

    try:
        result = client.plan_trajectory(start, goal, max_wait_time=15.0)

        if result.get("status") == "success":
            trajectory = client.extract_trajectory_arrays(result)
            if trajectory is not None:
                print(f"✓ Trajectory planning successful: {len(trajectory.get('positions', []))} waypoints")
                return True
            else:
                print("✗ Trajectory extraction failed")
                return False
        else:
            error_msg = result.get("error", "Unknown error")
            print(f"✗ Trajectory planning failed: {error_msg}")
            return False
    except Exception as e:
        print(f"✗ Trajectory planning failed with exception: {e}")
        return False


def test_global_planning_manager(host: str = "localhost", port: int = 8080) -> bool:
    """Test GlobalPlanningManager high-level interface."""
    print(f"Testing GlobalPlanningManager at {host}:{port}")
    manager = GlobalPlanningManager(host, port)

    # Test server availability
    if not manager.is_server_available():
        print("✗ Server not available for manager testing")
        return False

    # Create sample environment data with obstacles
    sample_points = np.array([
        [1.0, 1.0, 1.0],   # Inside bounds
        [2.0, 2.0, 1.5],   # Inside bounds
        [15.0, 15.0, 2.0], # Outside bounds (should be filtered)
        [-15.0, -15.0, 0.5], # Outside bounds (should be filtered)
        [0.5, 0.5, 1.0]    # Inside bounds
    ])

    env_data = {"points": sample_points}

    try:
        # Test map loading
        map_success = manager.load_environment_map(
            env_data=env_data,
            env_spacing=8.0,
            maze_height=3.0,
            center_x=0.0,
            center_y=0.0
        )

        if not map_success:
            print("✗ Manager map loading failed")
            return False

        # Test trajectory generation
        env_ids = [0, 1]
        start_positions = [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
        goal_positions = [[3.0, 3.0, 1.0], [2.0, 2.0, 1.0]]

        trajectories = manager.generate_expert_trajectories_batch(
            env_ids, start_positions, goal_positions
        )

        success_count = len(trajectories)
        print(f"✓ Manager test successful: {success_count}/{len(env_ids)} trajectories generated")

        # Test trajectory access methods
        for env_id in env_ids:
            has_traj = manager.has_expert_trajectory(env_id)
            traj = manager.get_expert_trajectory(env_id)
            print(f"  Env {env_id}: has_trajectory={has_traj}, trajectory={'available' if traj else 'none'}")

        return True

    except Exception as e:
        print(f"✗ Manager test failed with exception: {e}")
        return False


def run_comprehensive_tests(host: str = "localhost", port: int = 8080) -> Dict[str, bool]:
    """Run all tests and return results."""
    print(f"\n=== Comprehensive Global Planning API Tests ===")
    print(f"Target server: {host}:{port}\n")

    test_results = {}

    # Test 1: Health Check
    test_results["health"] = test_server_health(host, port)
    print()

    # Test 2: Stats Check
    test_results["stats"] = test_server_stats(host, port)
    print()

    # Test 3: Map Loading
    test_results["map_loading"] = test_map_loading(host, port)
    print()

    # Test 4: Trajectory Planning
    test_results["trajectory_planning"] = test_trajectory_planning(host, port)
    print()

    # Test 5: Global Planning Manager
    test_results["manager"] = test_global_planning_manager(host, port)
    print()

    # Summary
    print("=== Test Summary ===")
    passed = sum(test_results.values())
    total = len(test_results)

    for test_name, result in test_results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️ Some tests failed. Check server status and configuration.")

    return test_results


def print_usage():
    """Print usage information."""
    print("Global Planning Client Test Suite")
    print("Usage:")
    print("  python global_planning_client.py [OPTIONS]")
    print("")
    print("Options:")
    print("  --health              Check server health only")
    print("  --host HOST           Server hostname/IP (default: localhost)")
    print("  --port PORT           Server port (default: 8080)")
    print("  --test-all            Run comprehensive test suite (default)")
    print("  --help, -h            Show this help message")
    print("")
    print("Examples:")
    print("  python global_planning_client.py --health")
    print("  python global_planning_client.py --host 192.168.1.100 --port 9090")
    print("  python global_planning_client.py --test-all --host 143.89.46.145")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Global Planning Client Test Suite", add_help=False)
    parser.add_argument("--health", action="store_true", help="Check server health only")
    parser.add_argument("--host", type=str, default="143.89.46.145", help="Server hostname/IP")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--test-all", action="store_true", help="Run comprehensive test suite")
    parser.add_argument("--help", "-h", action="store_true", help="Show help message")

    args = parser.parse_args()

    if args.help:
        print_usage()
        exit(0)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if args.health:
        # Health check only
        print(f"Checking health of global planning server at {args.host}:{args.port}")
        success = test_server_health(args.host, args.port)
        exit(0 if success else 1)
    else:
        # Run comprehensive tests (default)
        results = run_comprehensive_tests(args.host, args.port)
        all_passed = all(results.values())
        exit(0 if all_passed else 1)