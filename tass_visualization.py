"""
TASS Motion Analysis Visualization Module
==========================================

Provides 2D and 3D visualization of towed array sonar system simulation results.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib import cm
from typing import Optional


def plot_cable_3d(results: dict, time_indices: Optional[list] = None,
                  title: str = "3D TASS Cable Configuration",
                  save_path: Optional[str] = None):
    """
    Plot 3D cable configurations at selected time steps.

    Args:
        results: simulation results dictionary
        time_indices: list of time indices to plot (None = auto-select)
        save_path: path to save figure (None = show)
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    positions = results['positions']
    times = results['time']

    if time_indices is None:
        n_frames = min(10, len(positions))
        time_indices = np.linspace(0, len(positions) - 1, n_frames, dtype=int)

    cmap = cm.viridis
    norm = Normalize(vmin=times[time_indices[0]], vmax=times[time_indices[-1]])

    for idx in time_indices:
        pos = positions[idx]
        t = times[idx]
        color = cmap(norm(t))
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
                color=color, alpha=0.7, linewidth=1.5,
                label=f't = {t:.0f} s')
        # Mark tow point
        ax.scatter(*pos[0], color=color, s=30, marker='o')
        # Mark tail
        ax.scatter(*pos[-1], color=color, s=20, marker='v')

    # Plot tow point trajectory
    tow_pts = results['tow_points']
    ax.plot(tow_pts[:, 0], tow_pts[:, 1], tow_pts[:, 2],
            'r--', linewidth=2, label='Tow point path')

    ax.set_xlabel('X [m] (Forward)')
    ax.set_ylabel('Y [m] (Lateral)')
    ax.set_zlabel('Z [m] (Depth)')
    ax.set_title(title)
    ax.legend(fontsize=8, loc='upper left')

    # Invert Z axis (depth increases downward)
    ax.invert_zaxis()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_cable_xy(results: dict, time_indices: Optional[list] = None,
                  title: str = "TASS Cable Plan View (X-Y)",
                  save_path: Optional[str] = None):
    """Plot cable configuration in plan view (X-Y plane)."""
    fig, ax = plt.subplots(figsize=(12, 8))

    positions = results['positions']
    times = results['time']

    if time_indices is None:
        n_frames = min(10, len(positions))
        time_indices = np.linspace(0, len(positions) - 1, n_frames, dtype=int)

    cmap = cm.viridis
    norm = Normalize(vmin=times[time_indices[0]], vmax=times[time_indices[-1]])

    for idx in time_indices:
        pos = positions[idx]
        t = times[idx]
        color = cmap(norm(t))
        ax.plot(pos[:, 0], pos[:, 1], color=color, linewidth=1.5,
                label=f't = {t:.0f} s')
        ax.plot(pos[0, 0], pos[0, 1], 'o', color=color, markersize=6)
        ax.plot(pos[-1, 0], pos[-1, 1], 'v', color=color, markersize=5)

    # Tow point trajectory
    tow_pts = results['tow_points']
    ax.plot(tow_pts[:, 0], tow_pts[:, 1], 'r--', linewidth=2,
            label='Tow point path')

    ax.set_xlabel('X [m] (Forward)')
    ax.set_ylabel('Y [m] (Lateral)')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_cable_xz(results: dict, time_indices: Optional[list] = None,
                  title: str = "TASS Cable Side View (X-Z)",
                  save_path: Optional[str] = None):
    """Plot cable configuration in side view (X-Z plane)."""
    fig, ax = plt.subplots(figsize=(12, 6))

    positions = results['positions']
    times = results['time']

    if time_indices is None:
        n_frames = min(10, len(positions))
        time_indices = np.linspace(0, len(positions) - 1, n_frames, dtype=int)

    cmap = cm.viridis
    norm = Normalize(vmin=times[time_indices[0]], vmax=times[time_indices[-1]])

    for idx in time_indices:
        pos = positions[idx]
        t = times[idx]
        color = cmap(norm(t))
        ax.plot(pos[:, 0], pos[:, 2], color=color, linewidth=1.5,
                label=f't = {t:.0f} s')

    ax.set_xlabel('X [m] (Forward)')
    ax.set_ylabel('Z [m] (Depth)')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_tension_distribution(results: dict, time_indices: Optional[list] = None,
                               title: str = "Tension Distribution Along Cable",
                               save_path: Optional[str] = None):
    """Plot tension distribution along cable at selected times."""
    fig, ax = plt.subplots(figsize=(10, 6))

    tensions = results['tensions']
    times = results['time']
    cable = results['cable_props']
    n_nodes = len(tensions[0])
    s = np.linspace(0, cable.length, n_nodes)

    if time_indices is None:
        n_frames = min(8, len(tensions))
        time_indices = np.linspace(0, len(tensions) - 1, n_frames, dtype=int)

    cmap = cm.plasma
    norm = Normalize(vmin=times[time_indices[0]], vmax=times[time_indices[-1]])

    for idx in time_indices:
        T = tensions[idx]
        t = times[idx]
        color = cmap(norm(t))
        ax.plot(s, T, color=color, linewidth=1.5, label=f't = {t:.0f} s')

    ax.set_xlabel('Arc Length s [m]')
    ax.set_ylabel('Tension [N]')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_tension_time_history(results: dict,
                               node_indices: Optional[list] = None,
                               title: str = "Tension Time History",
                               save_path: Optional[str] = None):
    """Plot tension time history at selected nodes."""
    fig, ax = plt.subplots(figsize=(10, 6))

    tensions = results['tensions']
    times = results['time']
    cable = results['cable_props']
    n_nodes = len(tensions[0])

    if node_indices is None:
        node_indices = [0, n_nodes // 4, n_nodes // 2,
                        3 * n_nodes // 4, n_nodes - 1]

    for ni in node_indices:
        T_history = [tensions[i][ni] for i in range(len(tensions))]
        s_pos = ni * cable.length / (n_nodes - 1)
        ax.plot(times, T_history, linewidth=1.5,
                label=f's = {s_pos:.0f} m')

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Tension [N]')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_tail_trajectory(results: dict,
                          title: str = "Array Tail Trajectory",
                          save_path: Optional[str] = None):
    """Plot the trajectory of the cable tail end."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    positions = results['positions']
    times = results['time']
    tow_pts = results['tow_points']

    tail_x = [pos[-1, 0] for pos in positions]
    tail_y = [pos[-1, 1] for pos in positions]
    tail_z = [pos[-1, 2] for pos in positions]

    # X-Y plan view
    axes[0].plot(tow_pts[:, 0], tow_pts[:, 1], 'r--', linewidth=1.5,
                 label='Tow point')
    axes[0].plot(tail_x, tail_y, 'b-', linewidth=1.5, label='Array tail')
    axes[0].set_xlabel('X [m]')
    axes[0].set_ylabel('Y [m]')
    axes[0].set_title('Plan View (X-Y)')
    axes[0].legend()
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)

    # Time history of lateral offset
    lateral_offset = [pos[-1, 1] - tow_pts[i, 1]
                      for i, pos in enumerate(positions)]
    axes[1].plot(times, lateral_offset, 'b-', linewidth=1.5)
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('Lateral Offset [m]')
    axes[1].set_title('Tail Lateral Offset')
    axes[1].grid(True, alpha=0.3)

    # Depth history
    axes[2].plot(times, tail_z, 'b-', linewidth=1.5, label='Array tail')
    axes[2].plot(times, tow_pts[:, 2], 'r--', linewidth=1.5, label='Tow point')
    axes[2].set_xlabel('Time [s]')
    axes[2].set_ylabel('Depth [m]')
    axes[2].set_title('Depth History')
    axes[2].legend()
    axes[2].invert_yaxis()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_quasi_static_results(qs_results: dict,
                               title: str = "Quasi-Static Steady-State Solution",
                               save_path: Optional[str] = None):
    """Plot results from quasi-static analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    pos = qs_results['positions']
    T = qs_results['tensions']
    phi = np.degrees(qs_results['phi'])
    theta = np.degrees(qs_results['theta'])
    s = qs_results['arc_length']

    # 3D cable shape
    ax3d = fig.add_subplot(221, projection='3d')
    ax3d.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'b-', linewidth=2)
    ax3d.scatter(*pos[0], color='red', s=100, marker='o', label='Tow point')
    ax3d.scatter(*pos[-1], color='green', s=100, marker='v', label='Tail')
    ax3d.set_xlabel('X [m]')
    ax3d.set_ylabel('Y [m]')
    ax3d.set_zlabel('Z [m]')
    ax3d.set_title('Cable Configuration')
    ax3d.legend()
    ax3d.invert_zaxis()

    # Remove the flat subplot at position 221 (replaced by 3D)
    axes[0, 0].remove()

    # Tension distribution
    axes[0, 1].plot(s, T, 'b-', linewidth=2)
    axes[0, 1].set_xlabel('Arc Length s [m]')
    axes[0, 1].set_ylabel('Tension [N]')
    axes[0, 1].set_title('Tension Distribution')
    axes[0, 1].grid(True, alpha=0.3)

    # Inclination angle
    axes[1, 0].plot(s, phi, 'r-', linewidth=2)
    axes[1, 0].set_xlabel('Arc Length s [m]')
    axes[1, 0].set_ylabel('Inclination φ [deg]')
    axes[1, 0].set_title('Inclination Angle')
    axes[1, 0].grid(True, alpha=0.3)

    # Azimuth angle
    axes[1, 1].plot(s, theta, 'g-', linewidth=2)
    axes[1, 1].set_xlabel('Arc Length s [m]')
    axes[1, 1].set_ylabel('Azimuth θ [deg]')
    axes[1, 1].set_title('Azimuth Angle')
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def generate_all_plots(results: dict, prefix: str = "tass"):
    """Generate all standard plots and save to files."""
    plot_cable_3d(results, save_path=f"{prefix}_3d_config.png")
    plot_cable_xy(results, save_path=f"{prefix}_plan_view.png")
    plot_cable_xz(results, save_path=f"{prefix}_side_view.png")
    plot_tension_distribution(results, save_path=f"{prefix}_tension_dist.png")
    plot_tension_time_history(results, save_path=f"{prefix}_tension_history.png")
    plot_tail_trajectory(results, save_path=f"{prefix}_tail_trajectory.png")
    print(f"\nAll plots saved with prefix '{prefix}_'")
