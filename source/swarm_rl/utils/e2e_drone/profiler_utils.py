"""
Profiler utilities for performance analysis using torch.profiler.
Provides decorators and context managers for easy integration without code modification.
"""

import os
import time
import functools
import threading
from typing import Optional, Dict, Any, List, Callable, Union
from contextlib import contextmanager
from collections import defaultdict, deque
import torch


class ProfilerManager:
    """
    A comprehensive profiler manager using torch.profiler.
    Supports both method-level profiling with decorators and code-block profiling with context managers.
    """
    
    def __init__(
        self,
        log_dir: Optional[str] = None,
        enabled: bool = True,
        profile_memory: bool = True,
        profile_shapes: bool = False,
        with_stack: bool = False,
        with_flops: bool = False,
        with_modules: bool = False,
        max_trace_files: int = 10,
        auto_start: bool = True
    ):
        """
        Initialize the profiler manager.
        
        Args:
            log_dir: Directory to save profiling results and flame graphs
            enabled: Whether profiling is enabled
            profile_memory: Whether to profile memory usage
            profile_shapes: Whether to record tensor shapes
            with_stack: Whether to record stack traces
            with_flops: Whether to record FLOP counts
            with_modules: Whether to record module hierarchy
            max_trace_files: Maximum number of trace files to keep
            auto_start: Whether to automatically start profiling session
        """
        self.log_dir = log_dir
        self.enabled = enabled
        self.profile_memory = profile_memory
        self.profile_shapes = profile_shapes
        self.with_stack = with_stack
        self.with_flops = with_flops
        self.with_modules = with_modules
        self.max_trace_files = max_trace_files
        self.auto_start = auto_start
        
        # Thread-local storage for profiler instances
        self._local = threading.local()
        
        # Enhanced statistics tracking with separate CPU/GPU times
        self._method_stats = defaultdict(lambda: {
            'cpu_time': 0.0,
            'gpu_time': 0.0,
            'wall_time': 0.0,
            'call_count': 0,
            'avg_cpu_time': 0.0,
            'avg_gpu_time': 0.0,
            'avg_wall_time': 0.0,
            'min_cpu_time': float('inf'),
            'max_cpu_time': 0.0,
            'min_gpu_time': float('inf'),
            'max_gpu_time': 0.0,
            'min_wall_time': float('inf'),
            'max_wall_time': 0.0
        })
        
        # Recent timing history for moving averages
        self._timing_history = defaultdict(lambda: deque(maxlen=100))
        
        # Active profiling sessions
        self._active_sessions = {}
        self._global_session_active = False
        
        # CUDA timing support
        self._cuda_available = torch.cuda.is_available()
        
        # Create log directory if specified
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            os.makedirs(os.path.join(self.log_dir, "traces"), exist_ok=True)
            os.makedirs(os.path.join(self.log_dir, "reports"), exist_ok=True)
        
        # Auto-start global profiling session if enabled
        if self.enabled and self.auto_start:
            self.start_global_session()
    
    def start_global_session(self) -> None:
        """Start a global profiling session that runs continuously."""
        if not self.enabled or self._global_session_active:
            return
        
        try:
            # Initialize profiler with schedule for continuous profiling
            self._global_profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=1,
                    warmup=1, 
                    active=3,
                    repeat=1000000  # Very large number for continuous profiling
                ),
                record_shapes=self.profile_shapes,
                profile_memory=self.profile_memory,
                with_stack=self.with_stack,
                with_flops=self.with_flops,
                with_modules=self.with_modules,
                on_trace_ready=self._trace_handler if self.log_dir else None
            )
            self._global_profiler.start()
            self._global_session_active = True
            print(f"[Profiler] Started global profiling session")
        except Exception as e:
            print(f"[Profiler] Warning: Failed to start global session: {e}")
    
    def stop_global_session(self) -> None:
        """Stop the global profiling session."""
        if not self._global_session_active:
            return
        
        try:
            if hasattr(self, '_global_profiler'):
                self._global_profiler.stop()
                self._global_session_active = False
                print(f"[Profiler] Stopped global profiling session")
        except Exception as e:
            print(f"[Profiler] Warning: Failed to stop global session: {e}")
    
    def _trace_handler(self, profiler) -> None:
        """Handle trace ready event for continuous profiling."""
        if self.log_dir:
            try:
                timestamp = int(time.time())
                trace_path = os.path.join(self.log_dir, "traces", f"continuous_{timestamp}.json")
                profiler.export_chrome_trace(trace_path)
                # print(f"[Profiler] Saved continuous trace: {trace_path}")
                self._cleanup_old_files()
            except Exception as e:
                print(f"[Profiler] Warning: Failed to save continuous trace: {e}")

    def _get_profiler(self) -> Optional[torch.profiler.profile]:
        """Get or create a thread-local profiler instance."""
        if not self.enabled:
            return None
            
        if not hasattr(self._local, 'profiler') or self._local.profiler is None:
            self._local.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=self.profile_shapes,
                profile_memory=self.profile_memory,
                with_stack=self.with_stack,
                with_flops=self.with_flops,
                with_modules=self.with_modules,
            )
            self._local.profiler_started = False
            
        return self._local.profiler
    
    def start_session(self, session_name: str) -> None:
        """Start a profiling session."""
        if not self.enabled:
            return
            
        profiler = self._get_profiler()
        if profiler and not self._local.profiler_started:
            try:
                profiler.start()
                self._local.profiler_started = True
                self._active_sessions[session_name] = time.time()
                print(f"[Profiler] Started session: {session_name}")
            except Exception as e:
                print(f"[Profiler] Warning: Failed to start session {session_name}: {e}")
    
    def stop_session(self, session_name: str, save_trace: bool = True) -> None:
        """Stop a profiling session and optionally save results."""
        if not self.enabled:
            return
            
        profiler = self._get_profiler()
        if profiler and self._local.profiler_started:
            try:
                profiler.stop()
                self._local.profiler_started = False
                
                if session_name in self._active_sessions:
                    session_duration = time.time() - self._active_sessions[session_name]
                    del self._active_sessions[session_name]
                    print(f"[Profiler] Stopped session: {session_name} (duration: {session_duration:.2f}s)")
                
                if save_trace and self.log_dir:
                    self._save_profiling_results(profiler, session_name)
                
                # Reset profiler for next session
                self._local.profiler = None
                
            except Exception as e:
                print(f"[Profiler] Warning: Failed to stop session {session_name}: {e}")
    
    def _save_profiling_results(self, profiler: torch.profiler.profile, session_name: str) -> None:
        """Save profiling results including traces and reports."""
        try:
            timestamp = int(time.time())
            base_name = f"{session_name}_{timestamp}"
            
            # Save Chrome trace for flame graph visualization
            trace_path = os.path.join(self.log_dir, "traces", f"{base_name}.json")
            profiler.export_chrome_trace(trace_path)
            
            # Save detailed text report
            report_path = os.path.join(self.log_dir, "reports", f"{base_name}.txt")
            self._generate_detailed_report(profiler, report_path)
            
            # Clean up old files if too many
            self._cleanup_old_files()
            
            print(f"[Profiler] Saved trace: {trace_path}")
            print(f"[Profiler] Saved report: {report_path}")
            
        except Exception as e:
            print(f"[Profiler] Warning: Failed to save results for {session_name}: {e}")
    
    def _generate_detailed_report(self, profiler: torch.profiler.profile, report_path: str) -> None:
        """Generate a detailed text report from profiling results."""
        with open(report_path, 'w') as f:
            f.write("PyTorch Profiler Report\n")
            f.write("=" * 50 + "\n\n")
            
            # CPU time analysis
            f.write("TOP 20 OPERATIONS BY CPU TIME:\n")
            f.write("-" * 40 + "\n")
            try:
                cpu_table = profiler.key_averages().table(
                    sort_by="cpu_time_total",
                    row_limit=20,
                    max_name_column_width=60
                )
                f.write(cpu_table + "\n\n")
            except Exception as e:
                f.write(f"Error generating CPU table: {e}\n\n")
            
            # CUDA time analysis (if available)
            if torch.cuda.is_available():
                f.write("TOP 20 OPERATIONS BY CUDA TIME:\n")
                f.write("-" * 40 + "\n")
                try:
                    cuda_table = profiler.key_averages().table(
                        sort_by="cuda_time_total",
                        row_limit=20,
                        max_name_column_width=60
                    )
                    f.write(cuda_table + "\n\n")
                except Exception as e:
                    f.write(f"Error generating CUDA table: {e}\n\n")
            
            # Memory analysis (if enabled)
            if self.profile_memory:
                f.write("TOP 10 OPERATIONS BY MEMORY USAGE:\n")
                f.write("-" * 40 + "\n")
                try:
                    memory_table = profiler.key_averages().table(
                        sort_by="cpu_memory_usage",
                        row_limit=10,
                        max_name_column_width=60
                    )
                    f.write(memory_table + "\n\n")
                except Exception as e:
                    f.write(f"Error generating memory table: {e}\n\n")
    
    def _cleanup_old_files(self) -> None:
        """Remove old trace and report files to limit disk usage."""
        for subdir in ["traces", "reports"]:
            dir_path = os.path.join(self.log_dir, subdir)
            if not os.path.exists(dir_path):
                continue
                
            files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
            files.sort(key=lambda x: os.path.getctime(os.path.join(dir_path, x)), reverse=True)
            
            # Remove files beyond the limit
            for file_to_remove in files[self.max_trace_files:]:
                try:
                    os.remove(os.path.join(dir_path, file_to_remove))
                except Exception as e:
                    print(f"[Profiler] Warning: Failed to remove old file {file_to_remove}: {e}")
    
    def _measure_cuda_time(self, func, *args, **kwargs):
        """Measure CUDA execution time for a function."""
        if not self._cuda_available:
            return 0.0, func(*args, **kwargs)
        
        # Synchronize before starting measurement
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        result = func(*args, **kwargs)
        end_event.record()
        
        # Synchronize and get elapsed time
        torch.cuda.synchronize()
        gpu_time = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
        
        return gpu_time, result
    
    @contextmanager
    def profile_block(self, block_name: str):
        """
        Context manager for profiling code blocks with separate CPU/GPU timing.
        
        Usage:
            with profiler.profile_block("my_computation"):
                # Your code here
                result = expensive_computation()
        """
        if not self.enabled:
            yield
            return
        
        # Start wall clock timing
        wall_start = time.time()
        
        # Start CPU timing
        cpu_start = time.perf_counter()
        
        # Prepare CUDA timing
        gpu_time = 0.0
        if self._cuda_available:
            torch.cuda.synchronize()
            gpu_start_event = torch.cuda.Event(enable_timing=True)
            gpu_end_event = torch.cuda.Event(enable_timing=True)
            gpu_start_event.record()
        
        try:
            with torch.profiler.record_function(block_name):
                # Step the global profiler if active
                if self._global_session_active and hasattr(self, '_global_profiler'):
                    self._global_profiler.step()
                yield
        finally:
            # End timings
            cpu_end = time.perf_counter()
            wall_end = time.time()
            
            cpu_time = cpu_end - cpu_start
            wall_time = wall_end - wall_start
            
            # Get GPU time if available
            if self._cuda_available:
                gpu_end_event.record()
                torch.cuda.synchronize()
                gpu_time = gpu_start_event.elapsed_time(gpu_end_event) / 1000.0  # Convert to seconds
            
            # Update statistics
            self._update_timing_stats(block_name, cpu_time, gpu_time, wall_time)
    
    def profile_method(self, method_name: Optional[str] = None):
        """
        Decorator for profiling methods with separate CPU/GPU timing.
        
        Usage:
            @profiler.profile_method()
            def my_method(self):
                # Your code here
                pass
        """
        def decorator(func: Callable) -> Callable:
            name = method_name or f"{func.__module__}.{func.__qualname__}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)
                
                # Start wall clock timing
                wall_start = time.time()
                
                # Start CPU timing
                cpu_start = time.perf_counter()
                
                # Prepare CUDA timing
                gpu_time = 0.0
                if self._cuda_available:
                    torch.cuda.synchronize()
                    gpu_start_event = torch.cuda.Event(enable_timing=True)
                    gpu_end_event = torch.cuda.Event(enable_timing=True)
                    gpu_start_event.record()
                
                try:
                    with torch.profiler.record_function(name):
                        # Step the global profiler if active
                        if self._global_session_active and hasattr(self, '_global_profiler'):
                            self._global_profiler.step()
                        result = func(*args, **kwargs)
                    return result
                finally:
                    # End timings
                    cpu_end = time.perf_counter()
                    wall_end = time.time()
                    
                    cpu_time = cpu_end - cpu_start
                    wall_time = wall_end - wall_start
                    
                    # Get GPU time if available
                    if self._cuda_available:
                        gpu_end_event.record()
                        torch.cuda.synchronize()
                        gpu_time = gpu_start_event.elapsed_time(gpu_end_event) / 1000.0  # Convert to seconds
                    
                    # Update statistics
                    self._update_timing_stats(name, cpu_time, gpu_time, wall_time)
            
            return wrapper
        return decorator
    
    def _update_timing_stats(self, name: str, cpu_time: float, gpu_time: float, wall_time: float):
        """Update timing statistics for a given function/block."""
        stats = self._method_stats[name]
        
        # Update totals
        stats['cpu_time'] += cpu_time
        stats['gpu_time'] += gpu_time
        stats['wall_time'] += wall_time
        stats['call_count'] += 1
        
        # Update averages
        count = stats['call_count']
        stats['avg_cpu_time'] = stats['cpu_time'] / count
        stats['avg_gpu_time'] = stats['gpu_time'] / count
        stats['avg_wall_time'] = stats['wall_time'] / count
        
        # Update min/max
        stats['min_cpu_time'] = min(stats['min_cpu_time'], cpu_time)
        stats['max_cpu_time'] = max(stats['max_cpu_time'], cpu_time)
        stats['min_gpu_time'] = min(stats['min_gpu_time'], gpu_time)
        stats['max_gpu_time'] = max(stats['max_gpu_time'], gpu_time)
        stats['min_wall_time'] = min(stats['min_wall_time'], wall_time)
        stats['max_wall_time'] = max(stats['max_wall_time'], wall_time)
        
        # Update timing history
        self._timing_history[name].append({
            'cpu_time': cpu_time,
            'gpu_time': gpu_time,
            'wall_time': wall_time
        })
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get profiling statistics for all tracked methods/blocks."""
        return dict(self._method_stats)
    
    def print_statistics(self, top_n: int = 20, min_calls: int = 1, sort_by: str = "wall_time") -> None:
        """
        Print profiling statistics to console with separate CPU/GPU timing.
        
        Args:
            top_n: Number of top functions to display
            min_calls: Minimum number of calls to include
            sort_by: Sorting criteria ("wall_time", "cpu_time", or "gpu_time")
        """
        if not self._method_stats:
            print("[Profiler] No profiling data available.")
            return
        
        # Filter by minimum calls
        filtered_stats = {
            name: stats for name, stats in self._method_stats.items() 
            if stats['call_count'] >= min_calls
        }
        
        if not filtered_stats:
            print(f"[Profiler] No functions/blocks with at least {min_calls} calls.")
            return
        
        # Validate sort_by parameter
        valid_sorts = ["wall_time", "cpu_time", "gpu_time"]
        if sort_by not in valid_sorts:
            sort_by = "wall_time"
        
        print(f"\n[Profiler] Performance Statistics (Top {top_n}, min {min_calls} calls, sorted by {sort_by}):")
        print("=" * 130)
        
        if self._cuda_available:
            print(f"{'Function/Block':<40} {'Calls':<6} {'CPU(ms)':<10} {'GPU(ms)':<10} {'Wall(ms)':<10} {'CPU_Avg':<8} {'GPU_Avg':<8} {'Wall_Avg':<8}")
        else:
            print(f"{'Function/Block':<40} {'Calls':<6} {'CPU(ms)':<10} {'Wall(ms)':<10} {'CPU_Avg':<8} {'Wall_Avg':<8}")
        
        print("-" * 130)
        
        # Sort by specified criteria
        sorted_stats = sorted(
            filtered_stats.items(),
            key=lambda x: x[1][sort_by],
            reverse=True
        )
        
        for name, stats in sorted_stats[:top_n]:
            # Truncate long names
            display_name = name if len(name) <= 38 else name[:35] + "..."
            
            if self._cuda_available:
                print(f"{display_name:<40} "
                      f"{stats['call_count']:<6} "
                      f"{stats['cpu_time']*1000:<10.2f} "
                      f"{stats['gpu_time']*1000:<10.2f} "
                      f"{stats['wall_time']*1000:<10.2f} "
                      f"{stats['avg_cpu_time']*1000:<8.2f} "
                      f"{stats['avg_gpu_time']*1000:<8.2f} "
                      f"{stats['avg_wall_time']*1000:<8.2f}")
            else:
                print(f"{display_name:<40} "
                      f"{stats['call_count']:<6} "
                      f"{stats['cpu_time']*1000:<10.2f} "
                      f"{stats['wall_time']*1000:<10.2f} "
                      f"{stats['avg_cpu_time']*1000:<8.2f} "
                      f"{stats['avg_wall_time']*1000:<8.2f}")
        
        print("-" * 130)
        print(f"Total tracked functions/blocks: {len(filtered_stats)}")
        
        # Print summary statistics
        total_cpu_time = sum(stats['cpu_time'] for stats in filtered_stats.values())
        total_gpu_time = sum(stats['gpu_time'] for stats in filtered_stats.values())
        total_wall_time = sum(stats['wall_time'] for stats in filtered_stats.values())
        total_calls = sum(stats['call_count'] for stats in filtered_stats.values())
        
        print(f"Total CPU time: {total_cpu_time*1000:.2f} ms")
        if self._cuda_available:
            print(f"Total GPU time: {total_gpu_time*1000:.2f} ms")
        print(f"Total wall time: {total_wall_time*1000:.2f} ms")
        print(f"Total tracked calls: {total_calls}")
        
        if total_calls > 0:
            print(f"Average CPU time per call: {(total_cpu_time/total_calls)*1000:.2f} ms")
            if self._cuda_available:
                print(f"Average GPU time per call: {(total_gpu_time/total_calls)*1000:.2f} ms")
            print(f"Average wall time per call: {(total_wall_time/total_calls)*1000:.2f} ms")
        
        # Print GPU utilization if available
        if self._cuda_available and total_wall_time > 0:
            gpu_utilization = (total_gpu_time / total_wall_time) * 100
            print(f"GPU utilization: {gpu_utilization:.1f}%")
    
    def print_current_stats(self, top_n: int = 15, sort_by: str = "wall_time") -> None:
        """Print current profiling statistics - convenient function for immediate use."""
        print(f"\n[Profiler] Current Performance Statistics:")
        print(f"Profiler enabled: {self.enabled}")
        print(f"Global session active: {self._global_session_active}")
        print(f"CUDA available: {self._cuda_available}")
        self.print_statistics(top_n=top_n, min_calls=1, sort_by=sort_by)
    
    def get_summary(self) -> str:
        """Get a summary string of current profiling status."""
        if not self._method_stats:
            return "[Profiler] No data collected yet."
        
        total_functions = len(self._method_stats)
        total_cpu_time = sum(stats['cpu_time'] for stats in self._method_stats.values())
        total_gpu_time = sum(stats['gpu_time'] for stats in self._method_stats.values())
        total_wall_time = sum(stats['wall_time'] for stats in self._method_stats.values())
        total_calls = sum(stats['call_count'] for stats in self._method_stats.values())
        
        # Find top 3 time consumers (by wall time)
        top_3 = sorted(
            self._method_stats.items(),
            key=lambda x: x[1]['wall_time'],
            reverse=True
        )[:3]
        
        summary = f"[Profiler] Summary: {total_functions} functions, {total_calls} calls"
        summary += f"\nCPU: {total_cpu_time*1000:.1f}ms"
        if self._cuda_available:
            summary += f", GPU: {total_gpu_time*1000:.1f}ms"
        summary += f", Wall: {total_wall_time*1000:.1f}ms"
        
        if top_3:
            summary += f"\nTop consumers (wall time): "
            for i, (name, stats) in enumerate(top_3):
                short_name = name.split('.')[-1] if '.' in name else name
                summary += f"{short_name}({stats['wall_time']*1000:.1f}ms)"
                if i < len(top_3) - 1:
                    summary += ", "
        
        return summary

    def clear_statistics(self) -> None:
        """Clear all accumulated statistics."""
        self._method_stats.clear()
        self._timing_history.clear()
        print("[Profiler] Statistics cleared.")
    
    def enable(self) -> None:
        """Enable profiling."""
        self.enabled = True
        print("[Profiler] Profiling enabled.")
    
    def disable(self) -> None:
        """Disable profiling."""
        self.enabled = False
        print("[Profiler] Profiling disabled.")
    
    def __del__(self):
        """Cleanup when profiler is destroyed."""
        self.stop_global_session()


# Global profiler instance for convenience
_global_profiler: Optional[ProfilerManager] = None


def get_global_profiler() -> ProfilerManager:
    """Get or create the global profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = ProfilerManager(enabled=False)  # Disabled by default
    return _global_profiler


def set_global_profiler(profiler: ProfilerManager) -> None:
    """Set the global profiler instance."""
    global _global_profiler
    _global_profiler = profiler


# Convenience decorators using global profiler
def profile_method(method_name: Optional[str] = None):
    """Convenience decorator using global profiler."""
    return get_global_profiler().profile_method(method_name)


@contextmanager
def profile_block(block_name: str):
    """Convenience context manager using global profiler."""
    with get_global_profiler().profile_block(block_name):
        yield


# Additional utility functions
def configure_profiler(
    log_dir: Optional[str] = None,
    enabled: bool = True,
    **kwargs
) -> ProfilerManager:
    """Configure and return a new profiler instance."""
    profiler = ProfilerManager(log_dir=log_dir, enabled=enabled, **kwargs)
    set_global_profiler(profiler)
    return profiler


def print_profiling_stats(top_n: int = 20, sort_by: str = "wall_time") -> None:
    """Print profiling statistics using global profiler."""
    get_global_profiler().print_current_stats(top_n, sort_by)


def clear_profiling_stats() -> None:
    """Clear profiling statistics using global profiler."""
    get_global_profiler().clear_statistics()


def get_profiling_summary() -> str:
    """Get profiling summary using global profiler."""
    return get_global_profiler().get_summary()
