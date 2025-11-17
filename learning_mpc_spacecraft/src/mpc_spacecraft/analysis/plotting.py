"""Plotting utilities for visualization and analysis."""

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
            ax.axhline(y=limits[0], color='r', linestyle='--', alpha=0.5, label='Limits')
            ax.axhline(y=limits[1], color='r', linestyle='--', alpha=0.5)
        
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
    metric_names: Optional[List[str]] = None,
    title: str = "Performance Comparison",
    save_path: Optional[str] = None
) -> Figure:
    """
    Create bar chart comparing controller performance metrics.
    
    Args:
        metrics: Dictionary of {controller_name: {metric_name: value}}
        metric_names: List of metrics to plot (optional, plots all if None)
        title: Plot title
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    if metric_names is None:
        # Get all unique metric names
        metric_names = list(set(
            key for controller_metrics in metrics.values() 
            for key in controller_metrics.keys()
        ))
    
    controller_names = list(metrics.keys())
    n_metrics = len(metric_names)
    n_controllers = len(controller_names)
    
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 3*n_metrics))
    
    if n_metrics == 1:
        axes = [axes]
    
    x = np.arange(n_controllers)
    width = 0.6
    
    for i, (ax, metric_name) in enumerate(zip(axes, metric_names)):
        values = [metrics[name].get(metric_name, 0) for name in controller_names]
        ax.bar(x, values, width, label=metric_name)
        ax.set_ylabel(metric_name)
        ax.set_xticks(x)
        ax.set_xticklabels(controller_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig