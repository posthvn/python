"""
TASS Motion Analysis Visualization Module
==========================================

Provides 2D and 3D visualization of towed array sonar system simulation results.

Coordinate convention:
  - X: forward (tow direction)
  - Y: lateral (starboard positive)
  - Z: depth (downward positive, Z=0 at tow point)
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
    Z+ = depth (downward). Tow point at origin.
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
        # Plot with -Z so depth appears downward visually in 3D
        ax.plot(pos[:, 0], pos[:, 1], -pos[:, 2],
                color=color, alpha=0.7, linewidth=1.5,
                label=f't = {t:.0f} s')
        ax.scatter(pos[0, 0], pos[0, 1], -pos[0, 2],
                   color=color, s=30, marker='o')
        ax.scatter(pos[-1, 0], pos[-1, 1], -pos[-1, 2],
                   color=color, s=20, marker='v')

    # Plot tow point trajectory
    tow_pts = results['tow_points']
    ax.plot(tow_pts[:, 0], tow_pts[:, 1], -tow_pts[:, 2],
            'r--', linewidth=2, label='Tow point path')

    # Mark origin
    ax.scatter(0, 0, 0, color='red', s=100, marker='*', zorder=10,
               label='Origin (Tow Point t=0)')

    ax.set_xlabel('X [m] (Forward)')
    ax.set_ylabel('Y [m] (Lateral)')
    ax.set_zlabel('Depth [m]')
    ax.set_title(title)
    ax.legend(fontsize=7, loc='upper left')

    # Set Z tick labels to show positive depth values
    zticks = ax.get_zticks()
    ax.set_zticks(zticks)
    ax.set_zticklabels([f'{-z:.0f}' for z in zticks])

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

    # Mark origin
    ax.plot(0, 0, 'r*', markersize=15, label='Origin')

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
    """Plot cable configuration in side view (X-Z plane). Z+ = depth downward."""
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

    # Mark origin
    ax.plot(0, 0, 'r*', markersize=15, label='Origin (Tow Point)')

    ax.set_xlabel('X [m] (Forward)')
    ax.set_ylabel('Depth Z [m] (+ downward)')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.invert_yaxis()  # So depth increases downward visually
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
    n_elem = len(tensions[0])

    # Build arc-length array at element midpoints
    ep = results.get('elem_props')
    if ep is not None and 'node_s' in ep:
        node_s = ep['node_s']
        s = 0.5 * (node_s[:n_elem] + node_s[1:n_elem + 1])
    else:
        total_length = results.get('total_length', 1.0)
        if hasattr(total_length, 'length'):
            total_length = total_length.length
        s = np.linspace(0, total_length, n_elem)

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

    ax.set_xlabel('Arc Length s [m] (from tow point)')
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
    n_elem = len(tensions[0])

    # Build arc-length at element midpoints
    ep = results.get('elem_props')
    if ep is not None and 'node_s' in ep:
        node_s = ep['node_s']
        s_mid = 0.5 * (node_s[:n_elem] + node_s[1:n_elem + 1])
    else:
        total_length = results.get('total_length', 1.0)
        if hasattr(total_length, 'length'):
            total_length = total_length.length
        s_mid = np.linspace(0, total_length, n_elem)

    if node_indices is None:
        node_indices = [0, n_elem // 4, n_elem // 2,
                        3 * n_elem // 4, n_elem - 1]

    for ni in node_indices:
        T_history = [tensions[i][ni] for i in range(len(tensions))]
        s_pos = s_mid[ni]
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
    axes[0].plot(0, 0, 'r*', markersize=12, label='Origin')
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

    # Depth history (Z+ = depth)
    axes[2].plot(times, tail_z, 'b-', linewidth=1.5, label='Array tail')
    axes[2].plot(times, tow_pts[:, 2], 'r--', linewidth=1.5, label='Tow point')
    axes[2].set_xlabel('Time [s]')
    axes[2].set_ylabel('Depth Z [m] (+ downward)')
    axes[2].set_title('Depth History')
    axes[2].legend()
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
    """
    Plot results from quasi-static analysis.
    Tow point at origin (0,0,0). Z+ = depth (downward).
    """
    fig = plt.figure(figsize=(14, 10))

    pos = qs_results['positions']
    T = qs_results['tensions']
    phi = np.degrees(qs_results['phi'])
    theta = np.degrees(qs_results['theta'])
    s = qs_results['arc_length']

    # 3D cable shape
    ax3d = fig.add_subplot(221, projection='3d')
    ax3d.plot(pos[:, 0], pos[:, 1], -pos[:, 2], 'b-', linewidth=2)
    ax3d.scatter(pos[0, 0], pos[0, 1], -pos[0, 2],
                 color='red', s=100, marker='*', label='Tow Point (0,0,0)')
    ax3d.scatter(pos[-1, 0], pos[-1, 1], -pos[-1, 2],
                 color='green', s=100, marker='v', label='Tail')
    ax3d.set_xlabel('X [m]')
    ax3d.set_ylabel('Y [m]')
    ax3d.set_zlabel('Depth [m]')
    ax3d.set_title('Cable Configuration')
    ax3d.legend(fontsize=7)
    # Z tick labels show positive depth
    zticks = ax3d.get_zticks()
    ax3d.set_zticklabels([f'{-z:.0f}' for z in zticks])

    # Tension distribution
    ax_t = fig.add_subplot(222)
    ax_t.plot(s, T, 'b-', linewidth=2)
    ax_t.set_xlabel('Arc Length s [m]')
    ax_t.set_ylabel('Tension [N]')
    ax_t.set_title('Tension Distribution')
    ax_t.grid(True, alpha=0.3)

    # Inclination angle
    ax_phi = fig.add_subplot(223)
    ax_phi.plot(s, phi, 'r-', linewidth=2)
    ax_phi.set_xlabel('Arc Length s [m]')
    ax_phi.set_ylabel('Inclination φ [deg]')
    ax_phi.set_title('Inclination Angle')
    ax_phi.grid(True, alpha=0.3)

    # Side view (X-Z)
    ax_xz = fig.add_subplot(224)
    ax_xz.plot(pos[:, 0], pos[:, 2], 'b-', linewidth=2)
    ax_xz.plot(pos[0, 0], pos[0, 2], 'r*', markersize=15, label='Tow Point (0,0,0)')
    ax_xz.plot(pos[-1, 0], pos[-1, 2], 'gv', markersize=10, label='Tail')
    ax_xz.set_xlabel('X [m] (Forward)')
    ax_xz.set_ylabel('Depth Z [m] (+ downward)')
    ax_xz.set_title('Side View (X-Z)')
    ax_xz.invert_yaxis()  # Depth increases downward visually
    ax_xz.legend(fontsize=8)
    ax_xz.grid(True, alpha=0.3)

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
