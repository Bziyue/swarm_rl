import torch
import matplotlib.pyplot as plt
import math

class JoystickGenerator:
    """
    Joystick command generator using Ornstein–Uhlenbeck processes for velocity (vx, vy)
    and height (z). Fully vectorized for num_env environments. Outputs are normalized
    to [-1,1] for vx/vy and [0,1] for z, then scaled by v_max and z_max.

    Parameters theta and sigma can randomly vary within specified ranges to create
    diverse command patterns.
    """
    def __init__(
        self,
        num_envs: int,
        dt: float = 0.02,
        theta_v: float = 0.3,
        sigma_v: float = 0.8,
        theta_z: float = 0.15,
        sigma_z: float = 0.3,
        mu_z: float = 1.0,
        v_max: float = 1.0,
        z_max: float = 1.0,
        device: torch.device = torch.device('cpu'),
        # Parameter ranges for randomization
        theta_v_range: tuple = None,  # (min, max) for theta_v
        sigma_v_range: tuple = None,  # (min, max) for sigma_v
        theta_z_range: tuple = None,  # (min, max) for theta_z
        sigma_z_range: tuple = None,  # (min, max) for sigma_z
        param_update_interval: int = 50,  # How often to update parameters (steps)
    ):
        self.num_envs = num_envs
        self.dt = dt
        self.device = device

        # Base OU process parameters
        self.base_theta_v = theta_v
        self.base_sigma_v = sigma_v
        self.base_theta_z = theta_z
        self.base_sigma_z = sigma_z

        # Current parameters (per environment)
        self.theta_v = torch.full((num_envs,), theta_v, device=device)
        self.sigma_v = torch.full((num_envs,), sigma_v, device=device)
        self.theta_z = torch.full((num_envs,), theta_z, device=device)
        self.sigma_z = torch.full((num_envs,), sigma_z, device=device)

        # Parameter ranges for randomization
        self.theta_v_range = theta_v_range or (0.1, 0.5)  # Default range if not specified
        self.sigma_v_range = sigma_v_range or (0.4, 1.2)
        self.theta_z_range = theta_z_range or (0.05, 0.3)
        self.sigma_z_range = sigma_z_range or (0.1, 0.6)

        # Randomization control
        self.param_update_interval = param_update_interval
        self.steps_since_update = 0

        # Mean targets
        self.mu_vx = 0.0
        self.mu_vy = 0.0
        self.mu_z = mu_z

        # Scaling factors
        self.v_max = v_max
        self.z_max = z_max

        self._init_state()
        self._randomize_parameters()  # Initialize with random parameters

    def _init_state(self):
        """Initialize internal state variables."""
        self.x_vx = torch.zeros(self.num_envs, device=self.device)
        self.x_vy = torch.zeros(self.num_envs, device=self.device)
        self.x_z  = torch.full((self.num_envs,), self.mu_z, device=self.device)

    def _randomize_parameters(self):
        """Randomize theta and sigma parameters within their specified ranges."""
        # Generate random values for each parameter and each environment
        self.theta_v = torch.rand(self.num_envs, device=self.device) * \
                     (self.theta_v_range[1] - self.theta_v_range[0]) + self.theta_v_range[0]
        self.sigma_v = torch.rand(self.num_envs, device=self.device) * \
                     (self.sigma_v_range[1] - self.sigma_v_range[0]) + self.sigma_v_range[0]
        self.theta_z = torch.rand(self.num_envs, device=self.device) * \
                     (self.theta_z_range[1] - self.theta_z_range[0]) + self.theta_z_range[0]
        self.sigma_z = torch.rand(self.num_envs, device=self.device) * \
                     (self.sigma_z_range[1] - self.sigma_z_range[0]) + self.sigma_z_range[0]

    def reset_idx(self, env_ids):
        """
        Reset the generator state for specific environment indices.

        Args:
            env_ids: Tensor of environment indices to reset
        """
        if len(env_ids) == 0:
            return

        # Reset only the specified environments to initial values
        self.x_vx[env_ids] = torch.zeros(len(env_ids), device=self.device)
        self.x_vy[env_ids] = torch.zeros(len(env_ids), device=self.device)
        self.x_z[env_ids] = torch.full((len(env_ids),), self.mu_z, device=self.device)

        # Reset their parameters to new random values
        self.theta_v[env_ids] = torch.rand(len(env_ids), device=self.device) * \
                              (self.theta_v_range[1] - self.theta_v_range[0]) + self.theta_v_range[0]
        self.sigma_v[env_ids] = torch.rand(len(env_ids), device=self.device) * \
                              (self.sigma_v_range[1] - self.sigma_v_range[0]) + self.sigma_v_range[0]
        self.theta_z[env_ids] = torch.rand(len(env_ids), device=self.device) * \
                              (self.theta_z_range[1] - self.theta_z_range[0]) + self.theta_z_range[0]
        self.sigma_z[env_ids] = torch.rand(len(env_ids), device=self.device) * \
                              (self.sigma_z_range[1] - self.sigma_z_range[0]) + self.sigma_z_range[0]

    def reset(self):
        """Reset the generator state to initial values for all environments."""
        self._init_state()
        self._randomize_parameters()
        self.steps_since_update = 0

    def step(self):
        """
        Advance the OU processes by one time step.
        Returns a dict of tensors:
          'ref_vx', 'ref_vy', 'ref_z' each of shape (num_envs,).
        """
        # Periodically update parameters
        self.steps_since_update += 1
        if self.steps_since_update >= self.param_update_interval:
            self._randomize_parameters()
            self.steps_since_update = 0

        # Draw noise
        noise_vx = torch.randn(self.num_envs, device=self.device)
        noise_vy = torch.randn(self.num_envs, device=self.device)
        noise_z  = torch.randn(self.num_envs, device=self.device)

        # Compute OU increments using current parameters for each environment
        sqrt_dt = torch.sqrt(torch.tensor(self.dt, device=self.device))
        dx_vx = self.theta_v * (self.mu_vx - self.x_vx) * self.dt + self.sigma_v * sqrt_dt * noise_vx
        dx_vy = self.theta_v * (self.mu_vy - self.x_vy) * self.dt + self.sigma_v * sqrt_dt * noise_vy
        dx_z  = self.theta_z * (self.mu_z - self.x_z)  * self.dt + self.sigma_z * sqrt_dt * noise_z

        # Update state
        self.x_vx = self.x_vx + dx_vx
        self.x_vy = self.x_vy + dx_vy
        self.x_z  = self.x_z  + dx_z

        # Normalize to desired ranges
        norm_vx = torch.tanh(self.x_vx)       # in [-1, 1]
        norm_vy = torch.tanh(self.x_vy)       # in [-1, 1]
        norm_z  = torch.sigmoid(self.x_z)     # in [0, 1]

        # Scale to real commands
        real_vx = norm_vx * self.v_max
        real_vy = norm_vy * self.v_max
        real_z  = norm_z  * self.z_max

        return {
            'ref_vx': real_vx,
            'ref_vy': real_vy,
            'ref_z':  real_z,
        }

    def generate(self, steps: int, visualize: bool = False):
        """
        Generate a sequence of joystick commands for a given number of steps.
        If visualize=True, plots the first environment's ref_vx, ref_vy, ref_z curves.
        Returns three tensors of shape (steps, num_envs): (traj_vx, traj_vy, traj_z).
        """
        traj_vx = []
        traj_vy = []
        traj_z  = []
        self.reset()

        for _ in range(steps):
            cmd = self.step()
            traj_vx.append(cmd['ref_vx'])
            traj_vy.append(cmd['ref_vy'])
            traj_z.append(cmd['ref_z'])

        traj_vx = torch.stack(traj_vx, dim=0)
        traj_vy = torch.stack(traj_vy, dim=0)
        traj_z  = torch.stack(traj_z,  dim=0)

        if visualize:
            # Convert all tensors to CPU for plotting
            t = torch.arange(steps, device=self.device).cpu().numpy() * self.dt
            vx = traj_vx[:, 0].cpu().numpy()
            vy = traj_vy[:, 0].cpu().numpy()
            z  = traj_z[:, 0].cpu().numpy()

            fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
            fig.suptitle('Joystick Command Trajectory')

            axs[0].plot(t, vx)
            axs[0].set_ylabel(f'ref_vx [|v|≤{self.v_max}]')
            axs[0].grid(True)

            axs[1].plot(t, vy)
            axs[1].set_ylabel(f'ref_vy [|v|≤{self.v_max}]')
            axs[1].grid(True)

            axs[2].plot(t, z)
            axs[2].set_ylabel(f'ref_z [0≤z≤{self.z_max}]')
            axs[2].set_xlabel('Time [s]')
            axs[2].grid(True)

            plt.tight_layout()
            plt.show()

        return traj_vx, traj_vy, traj_z

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Joystick command generator demo")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--steps", type=int, default=500, help="Number of steps to simulate")
    parser.add_argument("--dt", type=float, default=0.02, help="Simulation time step")
    parser.add_argument("--v_max", type=float, default=1.0, help="Maximum speed for vx and vy")
    parser.add_argument("--z_max", type=float, default=1.0, help="Maximum height for z")
    parser.add_argument("--visualize", action="store_true", help="Plot ref_vx, ref_vy, ref_z for env0")
    args = parser.parse_args()

    gen = JoystickGenerator(
        num_envs=args.num_envs,
        dt=args.dt,
        v_max=args.v_max,
        z_max=args.z_max,
        device=torch.device("cpu")
    )
    traj_vx, traj_vy, traj_z = gen.generate(
        args.steps,
        visualize=args.visualize
    )
