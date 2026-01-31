import time
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Slider, Button, RadioButtons
import requests
from collections import deque
import math

class QuadrotorVizClient:
    """
    Visualization client for the quadrotor controller.
    Connects to the controller via HTTP and displays real-time plots.
    """
    def __init__(self, server_url='http://localhost:8000'):
        self.server_url = server_url
        self.env_id = 0
        self.running = True
        self.paused = False
        self.last_param_update = 0
        self.param_update_interval = 0.5  # seconds between parameter updates
        self.last_draw_time = 0           # track last redraw time
        self.draw_interval = 0.1          # min seconds between redraws (10 Hz)

        # Time window configuration
        self.time_windows = {
            "0.1s": 10,    # 10 points at 100Hz
            "0.5s": 50,    # 50 points
            "1s": 100,     # 100 points
            "2s": 200,     # 200 points
            "5s": 500,     # 500 points
            "10s": 1000,   # 1000 points
        }
        self.current_detail_window = "1s"

        # Data storage for plotting
        self.max_samples = 1000  # 10 seconds at 100Hz
        self.times = deque(maxlen=self.max_samples)
        self.pos_data = {axis: deque(maxlen=self.max_samples) for axis in ['x', 'y', 'z']}
        self.vel_data = {axis: deque(maxlen=self.max_samples) for axis in ['x', 'y', 'z']}
        self.ang_vel_data = {axis: deque(maxlen=self.max_samples) for axis in ['x', 'y', 'z']}
        self.cmd_data = {cmd: deque(maxlen=self.max_samples) for cmd in ['roll', 'pitch', 'yaw_rate', 'thrust']}

        # Add storage for desired roll/pitch and Euler angles
        self.cmd_des_data = {cmd: deque(maxlen=self.max_samples) for cmd in ['roll_des', 'pitch_des', 'yaw_rate_des']}
        self.euler_data = {angle: deque(maxlen=self.max_samples) for angle in ['roll', 'pitch']}

        # Setup plots and controls
        self.setup_ui()

    def quaternion_to_euler(self, quat):
        """Convert quaternion to roll and pitch angles in degrees"""
        # Extract quaternion components
        w, x, y, z = quat

        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi/2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)

        return roll, pitch

    def setup_ui(self):
        """Set up the matplotlib plots and UI controls"""
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(14, 9))

        # Prevent window from always being on top
        try:
            manager = self.fig.canvas.manager
            if hasattr(manager, 'window'):
                if hasattr(manager.window, 'attributes'):
                    # Tkinter backend
                    manager.window.attributes('-topmost', 0)
                elif hasattr(manager.window, 'setWindowFlag'):
                    # Qt backend
                    from PyQt5.QtCore import Qt
                    manager.window.setWindowFlag(Qt.WindowStaysOnTopHint, False)
        except Exception:
            pass  # Ignore if window configuration is not supported

        self.fig.canvas.manager.set_window_title('Quadrotor Controller Visualization')

        # Create plot grid with adjusted layout for controls
        # Leave more space at the bottom for controls
        plt.subplots_adjust(bottom=0.35, left=0.1, right=0.9, hspace=0.3)

        # Replace position plots with roll and pitch comparison plots
        self.ax_roll = plt.subplot2grid((3, 6), (0, 0), colspan=3)
        self.ax_pitch = plt.subplot2grid((3, 6), (1, 0), colspan=3)
        self.ax_cmd_main = plt.subplot2grid((3, 6), (2, 0), colspan=3)

        self.ax_vel_detail = plt.subplot2grid((3, 6), (0, 3), colspan=3)
        self.ax_ang_vel_detail = plt.subplot2grid((3, 6), (1, 3), colspan=3)
        self.ax_yaw_rate = plt.subplot2grid((3, 6), (2, 3), colspan=3)  # New plot for yaw rate

        # Set titles and labels
        self.update_plot_titles()

        # Define color scheme - distinct colors for all lines
        self.velocity_colors = ['#9467bd', '#8c564b', '#e377c2']  # Purple, Brown, Pink
        self.ang_vel_colors = ['#17becf', '#bcbd22', '#ff7f0e']   # Cyan, Olive, Orange
        self.cmd_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red

        # Initialize empty plot lines
        self.lines_vel_detail = {}
        self.lines_ang_vel_detail = {}
        self.lines_cmd_main = {}

        # New lines for roll, pitch and yaw rate comparison
        self.lines_roll = {}
        self.lines_pitch = {}
        self.lines_yaw_rate = {}

        # Initialize angular data lines with distinct colors
        for i, axis in enumerate(['x', 'y', 'z']):
            self.lines_vel_detail[axis], = self.ax_vel_detail.plot([], [],
                                                                 color=self.velocity_colors[i],
                                                                 label=axis)
            self.lines_ang_vel_detail[axis], = self.ax_ang_vel_detail.plot([], [],
                                                                        color=self.ang_vel_colors[i],
                                                                        label=f'ω_{axis}')

        # Command data lines with distinct colors
        for i, cmd in enumerate(['roll', 'pitch', 'yaw_rate', 'thrust']):
            self.lines_cmd_main[cmd], = self.ax_cmd_main.plot([], [],
                                                           color=self.cmd_colors[i],
                                                           label=cmd)

        # Roll comparison lines (current and desired)
        self.lines_roll['actual'], = self.ax_roll.plot([], [],
                                                    color='#1f77b4', linestyle='-',
                                                    label='Roll Actual')
        self.lines_roll['desired'], = self.ax_roll.plot([], [],
                                                     color='#7f7fff', linestyle='--',
                                                     label='Roll Desired')

        # Pitch comparison lines (current and desired)
        self.lines_pitch['actual'], = self.ax_pitch.plot([], [],
                                                      color='#2ca02c', linestyle='-',
                                                      label='Pitch Actual')
        self.lines_pitch['desired'], = self.ax_pitch.plot([], [],
                                                       color='#98df8a', linestyle='--',
                                                       label='Pitch Desired')

        # Yaw rate comparison lines (current and desired)
        self.lines_yaw_rate['actual'], = self.ax_yaw_rate.plot([], [],
                                                            color='#d62728', linestyle='-',
                                                            label='Yaw Rate Actual')
        self.lines_yaw_rate['desired'], = self.ax_yaw_rate.plot([], [],
                                                             color='#ff9896', linestyle='--',
                                                             label='Yaw Rate Desired')

        # Add legends
        self.ax_roll.legend()
        self.ax_pitch.legend()
        self.ax_cmd_main.legend()
        self.ax_vel_detail.legend()
        self.ax_ang_vel_detail.legend()
        self.ax_yaw_rate.legend()

        # Add grid to all plots
        for ax in [self.ax_roll, self.ax_pitch, self.ax_cmd_main,
                  self.ax_vel_detail, self.ax_ang_vel_detail, self.ax_yaw_rate]:
            ax.grid(True)

        # ===== SIMPLIFIED UI LAYOUT =====

        # Environment ID input - moved to top right
        ax_env_id = plt.axes([0.10, 0.1, 0.15, 0.03])
        self.env_id_text = TextBox(ax_env_id, 'Environment ID', initial=str(self.env_id))
        self.env_id_text.on_submit(self.update_env_id)

        # ===== FIRST ROW OF CONTROLS =====

        # Controller parameter sliders - first column
        ax_krp_0 = plt.axes([0.10, 0.28, 0.35, 0.02])
        ax_krp_1 = plt.axes([0.10, 0.25, 0.35, 0.02])
        self.slider_krp_0 = Slider(ax_krp_0, 'Krp_ang[0]', 1.0, 15.0, valinit=6.5)
        self.slider_krp_1 = Slider(ax_krp_1, 'Krp_ang[1]', 1.0, 15.0, valinit=6.5)

        # Controller parameter sliders - second column
        ax_kinv_0 = plt.axes([0.55, 0.28, 0.35, 0.02])
        ax_kinv_1 = plt.axes([0.55, 0.25, 0.35, 0.02])
        ax_kinv_2 = plt.axes([0.55, 0.22, 0.35, 0.02])
        self.slider_kinv_0 = Slider(ax_kinv_0, 'Kinv_ang_vel_tau[0]', 5.0, 50.0, valinit=25.0)
        self.slider_kinv_1 = Slider(ax_kinv_1, 'Kinv_ang_vel_tau[1]', 5.0, 50.0, valinit=25.0)
        self.slider_kinv_2 = Slider(ax_kinv_2, 'Kinv_ang_vel_tau[2]', 5.0, 30.0, valinit=15.0)

        # Apply button for parameter changes - positioned after sliders
        ax_apply = plt.axes([0.80, 0.2, 0.10, 0.02])
        self.button_apply = Button(ax_apply, 'Apply')
        self.button_apply.on_clicked(self.send_param_update)

        # Play/pause button - moved to top left
        ax_play_pause = plt.axes([0.80, 0.1, 0.15, 0.03])
        self.button_play_pause = Button(ax_play_pause, 'Pause' if not self.paused else 'Play')
        self.button_play_pause.on_clicked(self.toggle_play_pause)

        # ===== SECOND ROW OF CONTROLS =====
        # Add time window selection for detail view - center
        ax_detail_window = plt.axes([0.35, 0.005, 0.30, 0.15])
        self.radio_detail_window = RadioButtons(
            ax_detail_window,
            list(self.time_windows.keys()),
            active=list(self.time_windows.keys()).index(self.current_detail_window)
        )
        self.radio_detail_window.on_clicked(self.set_detail_window)

        # Connect slider events for continuous feedback
        self.slider_krp_0.on_changed(self.on_param_changed)
        self.slider_krp_1.on_changed(self.on_param_changed)
        self.slider_kinv_0.on_changed(self.on_param_changed)
        self.slider_kinv_1.on_changed(self.on_param_changed)
        self.slider_kinv_2.on_changed(self.on_param_changed)

        # Status message display
        self.status_text = self.fig.text(0.5, 0.01, "", ha='center')

    def update_plot_titles(self):
        """Update plot titles with current time window settings"""
        window_text = f'({self.current_detail_window} window)'
        self.ax_roll.set_title(f'Roll Angle Comparison {window_text}')
        self.ax_pitch.set_title(f'Pitch Angle Comparison {window_text}')
        self.ax_cmd_main.set_title(f'Commands {window_text}')
        self.ax_vel_detail.set_title(f'Velocity {window_text}')
        self.ax_ang_vel_detail.set_title(f'Angular Velocity {window_text}')
        self.ax_yaw_rate.set_title(f'Yaw Rate Comparison {window_text}')

    def set_detail_window(self, label):
        """Set the time window for all plots"""
        self.current_detail_window = label
        self.update_plot_titles()
        self.fig.canvas.draw_idle()

    def toggle_play_pause(self, event):
        """Toggle between play and pause states"""
        self.paused = not self.paused
        self.button_play_pause.label.set_text('Play' if self.paused else 'Pause')
        self.set_status("Visualization " + ("paused" if self.paused else "resumed"))
        self.fig.canvas.draw_idle()

    def update_env_id(self, text):
        """Update the environment ID for visualization"""
        try:
            new_env_id = int(text)
            if new_env_id < 0:
                self.set_status(f"Invalid env_id: {new_env_id}. Must be non-negative.")
                self.env_id_text.set_val(str(self.env_id))
                return

            # Check if the env_id is within range by fetching current data
            response = requests.get(f"{self.server_url}/viz_data")
            if response.status_code == 200:
                data = response.json()
                # Updated to use batch_size instead of num_envs
                batch_size = data.get('batch_size', data.get('num_envs', 0))
                if new_env_id >= batch_size:
                    self.set_status(f"Invalid env_id: {new_env_id}. Max allowed: {batch_size-1}")
                    self.env_id_text.set_val(str(self.env_id))
                else:
                    self.env_id = new_env_id
                    self.clear_data()  # Clear data when switching environments
                    self.set_status(f"Switched to environment {self.env_id}")
            else:
                self.set_status(f"Error: Couldn't verify env_id. Server returned {response.status_code}")
        except ValueError:
            self.set_status(f"Invalid input: '{text}'. Please enter a number.")
            self.env_id_text.set_val(str(self.env_id))

    def clear_data(self):
        """Clear all stored data when switching environments"""
        self.times.clear()
        for axis in ['x', 'y', 'z']:
            self.pos_data[axis].clear()
            self.vel_data[axis].clear()
            self.ang_vel_data[axis].clear()
        for cmd in ['roll', 'pitch', 'yaw_rate', 'thrust']:
            self.cmd_data[cmd].clear()
        for cmd in ['roll_des', 'pitch_des', 'yaw_rate_des']:
            self.cmd_des_data[cmd].clear()
        for angle in ['roll', 'pitch']:
            self.euler_data[angle].clear()

    def on_param_changed(self, val):
        """Called when a slider value changes"""
        # This method allows for throttled parameter updates
        current_time = time.time()
        if current_time - self.last_param_update > self.param_update_interval:
            self.send_param_update(None)
            self.last_param_update = current_time

    def send_param_update(self, event):
        """Send parameter updates to the controller"""
        try:
            params = {
                'Krp_ang': [self.slider_krp_0.val, self.slider_krp_1.val],
                'Kinv_ang_vel_tau': [
                    self.slider_kinv_0.val,
                    self.slider_kinv_1.val,
                    self.slider_kinv_2.val
                ],
            }

            response = requests.post(
                f"{self.server_url}/update_params",
                data=json.dumps(params),
                headers={'Content-Type': 'application/json'}
            )

            if response.status_code == 200:
                self.set_status("Parameters updated successfully")
            else:
                self.set_status(f"Failed to update parameters: {response.status_code}")
        except Exception as e:
            self.set_status(f"Error: {str(e)}")

    def set_status(self, message):
        """Update status message in the UI"""
        self.status_text.set_text(message)

    def fetch_data(self):
        """Fetch data from the controller server"""
        try:
            response = requests.get(f"{self.server_url}/viz_data")
            if response.status_code == 200:
                data = response.json()
                return data
            else:
                self.set_status(f"Error fetching data: {response.status_code}")
                return None
        except Exception as e:
            self.set_status(f"Connection error: {str(e)}")
            return None

    def update_plots(self):
        """Update all plots with the latest data"""
        # Skip updates if paused
        if self.paused:
            return

        # Fetch the latest data
        data = self.fetch_data()
        if data is None:
            return

        # Check if we have data for the selected env_id
        # Support both batch_size and num_envs for backward compatibility
        batch_size = data.get('batch_size', data.get('num_envs', 0))
        if self.env_id >= batch_size:
            self.set_status(f"Selected env_id {self.env_id} is out of range. Max: {batch_size-1}")
            return

        # Get current timestamp
        current_time = data['timestamp']
        self.times.append(current_time)

        # Extract data for the selected environment
        pos = data['state']['pos'][self.env_id]
        vel = data['state']['vel'][self.env_id]
        ang_vel = data['state']['ang_vel'][self.env_id]
        quat = data['state']['quat'][self.env_id]

        # Convert quaternion to roll and pitch
        roll, pitch = self.quaternion_to_euler(quat)
        self.euler_data['roll'].append(roll)
        self.euler_data['pitch'].append(pitch)

        # Commands
        roll = data['cmd']['roll'][self.env_id]
        pitch = data['cmd']['pitch'][self.env_id]
        yaw_rate = data['cmd']['yaw_rate'][self.env_id]
        thrust = data['cmd']['thrust'][self.env_id]

        # Desired roll/pitch and yaw rate
        roll_des = data['cmd']['roll_des'][self.env_id]
        pitch_des = data['cmd']['pitch_des'][self.env_id]
        yaw_rate_des = data['cmd']['yaw_rate_des'][self.env_id]

        self.cmd_des_data['roll_des'].append(roll_des)
        self.cmd_des_data['pitch_des'].append(pitch_des)
        self.cmd_des_data['yaw_rate_des'].append(yaw_rate_des)

        # Update data queues
        for i, axis in enumerate(['x', 'y', 'z']):
            self.pos_data[axis].append(pos[i])
            self.vel_data[axis].append(vel[i])
            self.ang_vel_data[axis].append(ang_vel[i])

        self.cmd_data['roll'].append(roll)
        self.cmd_data['pitch'].append(pitch)
        self.cmd_data['yaw_rate'].append(yaw_rate)
        self.cmd_data['thrust'].append(thrust)

        # Create arrays for plotting
        times_array = np.array(self.times)
        if len(times_array) > 1:
            # Normalize times to start from 0
            times_array = times_array - times_array[0]

            # Calculate the number of points to show for each view
            detail_points = self.time_windows[self.current_detail_window]

            # Get the appropriate time window sections
            detail_start = max(0, len(times_array) - detail_points)
            detail_times = times_array[detail_start:]

            # Update commands plot with detail window
            for cmd in ['roll', 'pitch', 'yaw_rate', 'thrust']:
                cmd_detail = list(self.cmd_data[cmd])[detail_start:]
                self.lines_cmd_main[cmd].set_data(detail_times, cmd_detail)

            # Fix: manually set the x-axis limits for all windows to show correct time range
            self.ax_cmd_main.set_xlim(detail_times[0] if len(detail_times) > 0 else 0,
                                      detail_times[-1] if len(detail_times) > 0 else 1)

            # Update detail plots with selected window
            for axis in ['x', 'y', 'z']:
                vel_detail = list(self.vel_data[axis])[detail_start:]
                ang_vel_detail = list(self.ang_vel_data[axis])[detail_start:]

                self.lines_vel_detail[axis].set_data(detail_times, vel_detail)
                self.lines_ang_vel_detail[axis].set_data(detail_times, ang_vel_detail)

            # Roll comparison plot
            roll_actual = list(self.euler_data['roll'])[detail_start:]
            roll_des = list(self.cmd_des_data['roll_des'])[detail_start:]
            self.lines_roll['actual'].set_data(detail_times, roll_actual)
            self.lines_roll['desired'].set_data(detail_times, roll_des)

            # Pitch comparison plot
            pitch_actual = list(self.euler_data['pitch'])[detail_start:]
            pitch_des = list(self.cmd_des_data['pitch_des'])[detail_start:]
            self.lines_pitch['actual'].set_data(detail_times, pitch_actual)
            self.lines_pitch['desired'].set_data(detail_times, pitch_des)

            # Yaw rate comparison plot
            yaw_rate_actual = list(self.ang_vel_data['z'])[detail_start:]
            yaw_rate_des = list(self.cmd_des_data['yaw_rate_des'])[detail_start:]
            self.lines_yaw_rate['actual'].set_data(detail_times, yaw_rate_actual)
            self.lines_yaw_rate['desired'].set_data(detail_times, yaw_rate_des)

            # Fix: manually set the x-axis limits for detail views to show correct time range
            for ax in [self.ax_roll, self.ax_pitch, self.ax_vel_detail,
                       self.ax_ang_vel_detail, self.ax_yaw_rate]:
                ax.set_xlim(detail_times[0] if len(detail_times) > 0 else 0,
                           detail_times[-1] if len(detail_times) > 0 else 1)

            # Update y-axis limits for all plots
            for ax in [self.ax_roll, self.ax_pitch, self.ax_cmd_main,
                      self.ax_vel_detail, self.ax_ang_vel_detail, self.ax_yaw_rate]:
                ax.relim()
                ax.autoscale_view(scalex=False)  # Only autoscale y-axis, keep x-axis fixed

            # Update controller parameters on sliders
            if 'params' in data:
                params = data['params']
                if 'Krp_ang' in params:
                    krp = params['Krp_ang']
                    self.slider_krp_0.set_val(krp[0])
                    self.slider_krp_1.set_val(krp[1])
                if 'Kinv_ang_vel_tau' in params:
                    kinv = params['Kinv_ang_vel_tau']
                    self.slider_kinv_0.set_val(kinv[0])
                    self.slider_kinv_1.set_val(kinv[1])
                    self.slider_kinv_2.set_val(kinv[2])

            # Limit redraw frequency to reduce CPU usage and window activation
            current_time = time.time()
            if current_time - self.last_draw_time >= self.draw_interval:
                self.fig.canvas.draw_idle()
                self.last_draw_time = current_time

    def run(self):
        """Main loop for the visualization client"""
        plt.ion()  # Interactive mode for real-time updates
        try:
            while self.running:
                self.update_plots()
                # Increase pause time to reduce window activation frequency
                plt.pause(0.05)  # 20Hz update rate instead of 100Hz
        except KeyboardInterrupt:
            self.set_status("Client stopped by user")
        except Exception as e:
            self.set_status(f"Error: {str(e)}")
        finally:
            plt.ioff()
            plt.close('all')

if __name__ == "__main__":
    client = QuadrotorVizClient()
    client.run()
