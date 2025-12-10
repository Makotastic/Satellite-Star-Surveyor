"""Plotting utilities for visualization and analysis."""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from matplotlib.figure import Figure


def plot_state_trajectory(
    times: np.ndarray,
    states: np.ndarray,
    references: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None,
    title: str = "State Trajectory",
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot state trajectory over time.
    
    Args:
        times: Time array [N]
        states: State trajectory [N, state_dim]
        references: Reference trajectory [N, state_dim] (optional)
        labels: State component labels (optional)
        title: Plot title
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    state_dim = states.shape[1]
    
    if labels is None:
        labels = [f'State {i+1}' for i in range(state_dim)]
    
    # Create subplots
    fig, axes = plt.subplots(state_dim, 1, figsize=(10, 2*state_dim), sharex=True)
    
    if state_dim == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        ax.plot(times, states[:, i], 'b-', label='Actual', linewidth=2)
        
        if references is not None:
            ax.plot(times, references[:, i], 'r--', label='Reference', linewidth=1.5)
        
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_control_inputs(
    times: np.ndarray,
    controls: np.ndarray,
    labels: Optional[List[str]] = None,
    limits: Optional[tuple] = None,
    title: str = "Control Inputs",
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot control inputs over time.
    
    Args:
        times: Time array [N]
        controls: Control trajectory [N, control_dim]
        labels: Control component labels (optional)
        limits: Control limits (u_min, u_max) (optional)
        title: Plot title
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    control_dim = controls.shape[1]
    
    if labels is None:
        labels = [f'Control {i+1}' for i in range(control_dim)]
    
    fig, axes = plt.subplots(control_dim, 1, figsize=(10, 2*control_dim), sharex=True)
    
    if control_dim == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        ax.plot(times, controls[:, i], 'g-', linewidth=2)
        
        if limits is not None:
            ax.axhline(y=limits[0][i], color='r', linestyle='--', alpha=0.5, label='Limits' if i == 0 else '')
            ax.axhline(y=limits[1][i], color='r', linestyle='--', alpha=0.5)
        
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.3)
        if limits is not None and i == 0:
            ax.legend()
    
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_control_inputs_comparison(
    results: Dict[str, Dict[str, np.ndarray]],
    colors: Optional[Dict[str, str]] = None,
    limits: Optional[tuple] = None,
    axes: Optional[List[plt.Axes]] = None,
    title: str = "Control Inputs Comparison",
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot comparison of control inputs across controllers for each axis.
    
    Args:
        results: Dictionary of {controller_name: {times, controls}}
        colors: Dict of controller to color
        limits: Control limits (u_min, u_max) (optional)
        axes: List of 3 axes for each control component (optional)
        title: Plot title
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    if axes is None:
        fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
        create_fig = True
    else:
        fig = axes[0].figure
        create_fig = False
    
    if colors is None:
        colors = {name: plt.cm.Set1(i) for i, name in enumerate(results.keys())}
    
    labels = ['u_x', 'u_y', 'u_z']
    u_min, u_max = limits if limits else (np.zeros(3), np.zeros(3))
    
    for i, (ax, label) in enumerate(zip(axes, labels)):
        for name, data in results.items():
            times = data['times'][:-1]  # Controls are one less than states
            controls = data['controls']
            line_color = colors.get(name, 'blue')
            ax.plot(times, controls[:, i], label=f'{name} {label}', linewidth=2, color=line_color)
            
            if limits is not None:
                ax.axhline(y=u_min[i], color='r', linestyle='--', alpha=0.5)
                ax.axhline(y=u_max[i], color='r', linestyle='--', alpha=0.5)
        
        ax.set_ylabel(f'{label} (N·m)')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc='upper right')
    
    axes[-1].set_xlabel('Time (s)')
    
    if create_fig:
        fig.suptitle(title, fontsize=14, fontweight='bold')
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_comparison(
    results: Dict[str, Dict[str, np.ndarray]],
    metric: str = 'states',
    component_idx: int = 0,
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> Figure:
    """
    Compare multiple controller results.
    
    Args:
        results: Dictionary of {controller_name: {times, states, controls}}
        metric: Which metric to plot ('states' or 'controls')
        component_idx: Which component to plot
        title: Plot title (optional)
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for name, data in results.items():
        times = data['times']
        values = data[metric][:, component_idx]
        ax.plot(times, values, label=name, linewidth=2)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(f'{metric.capitalize()} Component {component_idx}')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title(f'Controller Comparison: {metric.capitalize()}',
                     fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_quaternion_trajectory(
    times: np.ndarray,
    quaternions: np.ndarray,
    references: Optional[np.ndarray] = None,
    title: str = "Quaternion Trajectory",
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot quaternion components over time.
    
    Args:
        times: Time array [N]
        quaternions: Quaternion trajectory [N, 4] (w, x, y, z)
        references: Reference quaternions [N, 4] (optional)
        title: Plot title
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    labels = ['q_w', 'q_x', 'q_y', 'q_z']
    
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    
    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(times, quaternions[:, i], 'b-', label='Actual', linewidth=2)
        
        if references is not None:
            ax.plot(times, references[:, i], 'r--', label='Reference', linewidth=1.5)
        
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_angular_velocity(
    times: np.ndarray,
    omega: np.ndarray,
    references: Optional[np.ndarray] = None,
    title: str = "Angular Velocity",
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot angular velocity components over time.
    
    Args:
        times: Time array [N]
        omega: Angular velocity trajectory [N, 3]
        references: Reference angular velocities [N, 3] (optional)
        title: Plot title
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    labels = ['ω_x', 'ω_y', 'ω_z']
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    
    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(times, omega[:, i], 'b-', label='Actual', linewidth=2)
        
        if references is not None:
            ax.plot(times, references[:, i], 'r--', label='Reference', linewidth=1.5)
        
        ax.set_ylabel(f'{label} (rad/s)')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_performance_comparison(
    metrics: Dict[str, Dict[str, float]],
    colors: Optional[Dict[str, str]] = None,
    metric_names: Optional[List[str]] = None,
    axes: Optional[List[plt.Axes]] = None,
    title: str = "Performance Comparison",
    save_path: Optional[str] = None
) -> Figure:
    """
    Create bar chart comparing controller performance metrics.
    
    Args:
        metrics: Dictionary of {controller_name: {metric_name: value}}
        colors: Dict of controller to color
        metric_names: List of metrics to plot (optional, plots all if None)
        axes: List of axes for each metric (optional)
        title: Plot title
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    if metric_names is None:
        metric_names = list(set(
            key for controller_metrics in metrics.values()
            for key in controller_metrics.keys()
        ))
    
    controller_names = list(metrics.keys())
    n_metrics = len(metric_names)
    n_controllers = len(controller_names)
    
    if axes is None:
        fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 3*n_metrics))
        if n_metrics == 1:
            axes = [axes]
        create_fig = True
    else:
        fig = axes[0].figure
        create_fig = False
    
    if colors is None:
        colors = {name: plt.cm.Set3(i) for i, name in enumerate(controller_names)}
    
    x = np.arange(n_controllers)
    width = 0.6
    
    for i, (ax, metric_name) in enumerate(zip(axes, metric_names)):
        values = [metrics[name].get(metric_name, 0) for name in controller_names]
        bar_colors = [colors[name] for name in controller_names]
        bars = ax.bar(x, values, width, color=bar_colors)
        
        # Add value labels
        y_pos = max(values) * 0.01 if values else 0
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + y_pos,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel(metric_name)
        ax.set_xticks(x)
        ax.set_xticklabels(controller_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Controller legend (only on first subplot to avoid repetition)
        if i == 0:
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=colors[name], label=name) for name in controller_names]
            ax.legend(handles=legend_elements, loc='upper right')
    
    if create_fig:
        fig.suptitle(title, fontsize=14, fontweight='bold')
        fig.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_all_comparisons(
    results: Dict[str, Dict[str, np.ndarray]],
    metrics: Dict[str, Dict[str, float]],
    colors: Dict[str, str],
    metric_names: List[str] = ['rmse_total', 'quat_rmse', 'vel_rmse'],
    limits: tuple = (None, None),
    title: str = "Controller Comparison Dashboard",
    save_path: Optional[str] = None
) -> Figure:
    """
    Create a combined figure with all comparison plots on one page.
    
    Args:
        results: Dictionary of simulation results
        metrics: Dictionary of performance metrics
        colors: Dict of controller to color
        metric_names: Metrics for bar chart
        limits: Control limits
        title: Overall title
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    import matplotlib.gridspec as gridspec
    
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid: 2 rows, 2 columns, but bottom-left for controls (3 sub), bottom-right for performance (3 sub)
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1])
    
    # Top-left: Quaternion errors
    ax_quat = fig.add_subplot(gs[0, 0])
    plot_error_comparison(results, colors=colors, error_type='quaternion', ax=ax_quat)
    
    # Top-right: Velocity errors
    ax_vel = fig.add_subplot(gs[0, 1])
    plot_error_comparison(results, colors=colors, error_type='velocity', ax=ax_vel)
    
    # Bottom-left: Control inputs (3 subplots)
    ax_controls = []
    gs_controls = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1, 0], hspace=0.3)
    for i in range(3):
        ax_controls.append(fig.add_subplot(gs_controls[i]))
    plot_control_inputs_comparison(results, colors=colors, limits=limits, axes=ax_controls)
    
    # Bottom-right: Performance bars (3 subplots for each metric)
    ax_perf = []
    gs_perf = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1, 1], hspace=0.3)
    for i in range(3):
        ax_perf.append(fig.add_subplot(gs_perf[i]))
    plot_performance_comparison(metrics, colors=colors, metric_names=metric_names, axes=ax_perf)
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_error_comparison(
    results: Dict[str, Dict[str, np.ndarray]],
    colors: Optional[Dict[str, str]] = None,
    error_type: str = 'quaternion',
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot comparison of errors (quaternion angle or velocity) across controllers.
    
    Args:
        results: Dictionary of {controller_name: {times, quaternion_errors or velocity_errors, states}}
        colors: Dict of controller to color
        error_type: 'quaternion' or 'velocity'
        ax: Axes to plot on (optional)
        title: Plot title (optional)
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        create_fig = True
    else:
        fig = ax.figure
        create_fig = False
    
    if colors is None:
        colors = {name: plt.cm.Set1(i) for i, name in enumerate(results.keys())}
    
    goal_quat = np.array([1.0, 0.0, 0.0, 0.0])
    goal_omega = np.zeros(3)
    ylabel = None
    
    for name, data in results.items():
        times = data['times']
        states = data['states']
        
        if error_type == 'quaternion':
            if 'quaternion_errors' in data:
                errors = data['quaternion_errors']
            else:
                from .metrics import compute_quaternion_trajectory_errors
                errors = compute_quaternion_trajectory_errors(states, goal_quat)
            if ylabel is None:
                ylabel = 'Quaternion Angle Error (rad)'
        else:
            if 'velocity_errors' in data:
                errors = data['velocity_errors']
            else:
                from .metrics import compute_velocity_errors
                errors = compute_velocity_errors(states, goal_omega)
            if ylabel is None:
                ylabel = 'Angular Velocity Error (rad/s)'
        
        line_color = colors.get(name, 'blue')
        ax.plot(times, errors, label=name, linewidth=2, color=line_color)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    else:
        ax.set_title(f'{error_type.capitalize()} Errors', fontsize=12, fontweight='bold')
    
    if create_fig:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig