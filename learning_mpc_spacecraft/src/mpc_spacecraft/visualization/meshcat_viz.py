"""Meshcat-based 3D visualization for spacecraft."""

import numpy as np
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
from typing import Optional
import time


class MeshcatVisualizer:
    """
    Real-time 3D visualization using Meshcat.
    
    Displays spacecraft, reference frames, and trajectories in a browser-based viewer.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 7000,
        open_browser: bool = True
    ):
        """
        Initialize Meshcat visualizer.
        
        Args:
            host: Meshcat server host
            port: Meshcat server port
            open_browser: Whether to open browser automatically
        """
        self.vis = meshcat.Visualizer()
        
        if open_browser:
            self.vis.open()
        
        self._setup_scene()
    
    def _setup_scene(self):
        """Setup the visualization scene with default objects."""
        # Clear any existing scene
        self.vis.delete()
        
        # Add grid
        self.vis["/Grid"].set_object(
            g.LineSegments(
                g.PointsGeometry(
                    position=self._create_grid(size=10, divisions=20),
                    color=np.array([[0.5, 0.5, 0.5]] * 2 * 21 * 21).T
                ),
                g.LineBasicMaterial(vertexColors=True)
            )
        )
        
        # Add coordinate frame at origin
        self._add_frame("/World", scale=1.0)
        
        # Add spacecraft
        self._add_spacecraft("/Spacecraft")
        
        # Add reference frame
        self._add_frame("/Reference", scale=0.8, alpha=0.5)
        
        # Set camera
        self.vis["/Cameras/default"].set_transform(
            tf.translation_matrix([5, 5, 3])
        )
    
    def _create_grid(self, size: float = 10, divisions: int = 20) -> np.ndarray:
        """Create grid lines."""
        step = size / divisions
        points = []
        
        for i in range(divisions + 1):
            # Lines parallel to x-axis
            points.append([-size/2 + i*step, -size/2, 0])
            points.append([-size/2 + i*step, size/2, 0])
            
            # Lines parallel to y-axis
            points.append([-size/2, -size/2 + i*step, 0])
            points.append([size/2, -size/2 + i*step, 0])
        
        return np.array(points).T
    
    def _add_frame(
        self,
        path: str,
        scale: float = 1.0,
        alpha: float = 1.0
    ):
        """Add a coordinate frame (RGB = XYZ)."""
        # X-axis (red)
        self.vis[path]["x"].set_object(
            g.Cylinder(height=scale, radius=0.01),
            g.MeshLambertMaterial(color=0xff0000, opacity=alpha)
        )
        self.vis[path]["x"].set_transform(
            tf.rotation_matrix(np.pi/2, [0, 1, 0]) @
            tf.translation_matrix([scale/2, 0, 0])
        )
        
        # Y-axis (green)
        self.vis[path]["y"].set_object(
            g.Cylinder(height=scale, radius=0.01),
            g.MeshLambertMaterial(color=0x00ff00, opacity=alpha)
        )
        self.vis[path]["y"].set_transform(
            tf.rotation_matrix(-np.pi/2, [1, 0, 0]) @
            tf.translation_matrix([0, scale/2, 0])
        )
        
        # Z-axis (blue)
        self.vis[path]["z"].set_object(
            g.Cylinder(height=scale, radius=0.01),
            g.MeshLambertMaterial(color=0x0000ff, opacity=alpha)
        )
        self.vis[path]["z"].set_transform(
            tf.translation_matrix([0, 0, scale/2])
        )
    
    def _add_spacecraft(self, path: str):
        """Add spacecraft mesh (simplified box representation)."""
        # Main body
        self.vis[path]["body"].set_object(
            g.Box([0.5, 0.3, 0.2]),
            g.MeshLambertMaterial(color=0x888888)
        )
        
        # Solar panels
        self.vis[path]["panel_left"].set_object(
            g.Box([0.05, 0.8, 0.4]),
            g.MeshLambertMaterial(color=0x0066cc)
        )
        self.vis[path]["panel_left"].set_transform(
            tf.translation_matrix([-0.3, 0, 0])
        )
        
        self.vis[path]["panel_right"].set_object(
            g.Box([0.05, 0.8, 0.4]),
            g.MeshLambertMaterial(color=0x0066cc)
        )
        self.vis[path]["panel_right"].set_transform(
            tf.translation_matrix([0.3, 0, 0])
        )
        
        # Add body frame
        self._add_frame(f"{path}/frame", scale=0.5)
    
    def update_spacecraft_pose(
        self,
        quaternion: np.ndarray,
        position: Optional[np.ndarray] = None
    ):
        """
        Update spacecraft pose.
        
        Args:
            quaternion: Attitude quaternion [w, x, y, z]
            position: Position vector [x, y, z] (optional)
        """
        # Convert quaternion to rotation matrix
        w, x, y, z = quaternion
        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])
        
        # Create transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        
        if position is not None:
            T[:3, 3] = position
        
        self.vis["/Spacecraft"].set_transform(T)
    
    def update_reference_pose(
        self,
        quaternion: np.ndarray,
        position: Optional[np.ndarray] = None
    ):
        """
        Update reference frame pose.
        
        Args:
            quaternion: Reference quaternion [w, x, y, z]
            position: Position vector [x, y, z] (optional)
        """
        # Convert quaternion to rotation matrix
        w, x, y, z = quaternion
        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])
        
        # Create transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        
        if position is not None:
            T[:3, 3] = position
        
        self.vis["/Reference"].set_transform(T)
    
    def add_trajectory(
        self,
        states: np.ndarray,
        name: str = "trajectory",
        color: int = 0xff0000
    ):
        """
        Add a trajectory visualization.
        
        Args:
            states: State trajectory array [N, state_dim]
            name: Trajectory name
            color: Line color (hex)
        """
        # Extract positions (if available) or use quaternion visualization
        # For now, visualize angular velocity magnitude
        if states.shape[1] >= 7:
            omega = states[:, 4:7]
            omega_mag = np.linalg.norm(omega, axis=1)
            
            # Create line segments
            points = []
            for i, mag in enumerate(omega_mag):
                points.append([i * 0.1, mag, 0])
            
            if len(points) > 1:
                points_array = np.array(points).T
                self.vis[f"/Trajectories/{name}"].set_object(
                    g.Line(
                        g.PointsGeometry(position=points_array),
                        g.LineBasicMaterial(color=color)
                    )
                )
    
    def animate_trajectory(
        self,
        states: np.ndarray,
        dt: float,
        references: Optional[np.ndarray] = None
    ):
        """
        Animate a trajectory in real-time.
        
        Args:
            states: State trajectory [N, state_dim]
            dt: Timestep between states
            references: Reference trajectory (optional)
        """
        for i, state in enumerate(states):
            q = state[:4]
            self.update_spacecraft_pose(q)
            
            if references is not None and i < len(references):
                q_ref = references[i][:4]
                self.update_reference_pose(q_ref)
            
            time.sleep(dt)
    
    def clear_trajectories(self):
        """Clear all trajectory visualizations."""
        self.vis["/Trajectories"].delete()
    
    def close(self):
        """Close the visualizer."""
        self.vis.delete()