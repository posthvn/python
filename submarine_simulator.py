"""
Submarine Underwater Motion Simulator
======================================
Based on Gertler-Hagen equations of motion for a 60m class submarine.
- 6-DOF rigid body dynamics (surge, sway, heave, roll, pitch, yaw)
- Cross-shaped rudder configuration (vertical rudder + horizontal stern planes)
- PID controller for depth keeping (cascaded: depth->pitch->stern plane)

Coordinate system (body-fixed, right-hand):
  x: forward (surge)
  y: starboard (sway)
  z: downward (heave)
  phi: roll, theta: pitch, psi: yaw

Reference:
  Gertler, M. & Hagen, G.R. (1967), "Standard Equations of Motion for
  Submarine Simulation", NSRDC Report 2510.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple
import json


# ---------------------------------------------------------------------------
# Physical & geometric parameters for a generic 60 m submarine
# ---------------------------------------------------------------------------
@dataclass
class SubmarineParams:
    """Hull and hydrodynamic parameters for a ~60 m submarine.

    All hydrodynamic derivatives are stored in DIMENSIONAL form to avoid
    repeated re-dimensionalisation at every time step.  The values below
    are pre-computed from typical non-dimensional primes for a torpedo-
    shaped hull (L/D ~ 9) using:

        Force  = C' * 0.5 * rho * L^2 * U_ref^2   (velocity-dep.)
        Moment = C' * 0.5 * rho * L^3 * U_ref^2

    with U_ref = 5 m/s,  rho = 1025 kg/m^3,  L = 60 m.
    """
    # --- Geometry ---
    L: float = 60.0           # length [m]
    D: float = 6.5            # max diameter [m]

    # --- Mass / inertia ---
    mass: float = 2.8e6       # displacement mass [kg] (~2800 t)
    W: float = 2.8e6 * 9.81   # weight [N]
    B: float = 2.8e6 * 9.81   # buoyancy [N] (neutral buoyancy)

    x_G: float = 0.0          # CG longitudinal offset from CB [m]
    y_G: float = 0.0          # CG lateral offset [m]
    z_G: float = 0.15         # CG below CB [m] (z down in NED, so positive
                               #   means CG is below CB -> stabilising)

    Ixx: float = 5.0e7        # roll  MOI [kg m^2]
    Iyy: float = 3.5e9        # pitch MOI [kg m^2]
    Izz: float = 3.5e9        # yaw   MOI [kg m^2]
    Ixz: float = 0.0

    # --- Water ---
    rho: float = 1025.0       # [kg/m^3]

    # --- Speed regime ---
    U_design: float = 5.0     # design cruise speed [m/s] (~10 knots)

    # --- Control limits ---
    delta_r_max: float = 30.0    # max rudder angle [deg]
    delta_s_max: float = 25.0    # max stern-plane angle [deg]
    delta_rate_max: float = 5.0  # max actuator rate [deg/s]


# ---------------------------------------------------------------------------
# PID Controller
# ---------------------------------------------------------------------------
@dataclass
class PIDController:
    """Discrete PID controller with anti-windup."""
    Kp: float = 0.0
    Ki: float = 0.0
    Kd: float = 0.0
    integral: float = field(default=0.0, repr=False)
    prev_error: float = field(default=0.0, repr=False)
    output_min: float = -25.0
    output_max: float = 25.0
    integral_limit: float = 50.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, error: float, dt: float) -> float:
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit,
                                self.integral_limit)
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error
        out = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        return float(np.clip(out, self.output_min, self.output_max))


# ---------------------------------------------------------------------------
# Autopilot (depth + heading hold)
# ---------------------------------------------------------------------------
@dataclass
class Autopilot:
    """Cascaded depth / pitch controller + heading controller."""
    depth_pid: PIDController = field(default_factory=lambda: PIDController(
        Kp=0.8, Ki=0.03, Kd=12.0, output_min=-15.0, output_max=15.0
    ))
    pitch_pid: PIDController = field(default_factory=lambda: PIDController(
        Kp=2.0, Ki=0.02, Kd=8.0, output_min=-25.0, output_max=25.0
    ))
    heading_pid: PIDController = field(default_factory=lambda: PIDController(
        Kp=1.5, Ki=0.01, Kd=4.0, output_min=-30.0, output_max=30.0
    ))

    depth_cmd: float = 100.0     # [m], positive downward
    heading_cmd: float = 0.0     # [deg]

    def compute_controls(self, depth: float, theta: float, psi: float,
                         dt: float) -> Tuple[float, float]:
        """Return (delta_s_cmd [deg], delta_r_cmd [deg])."""
        # Outer loop: depth error -> desired pitch angle [deg]
        # In NED Euler convention, positive theta = nose UP.
        # To dive (increase depth), we need negative theta (nose down).
        depth_error = self.depth_cmd - depth   # positive when too shallow
        theta_cmd_deg = -self.depth_pid.compute(depth_error, dt)  # negate: dive->nose down
        theta_cmd_deg = np.clip(theta_cmd_deg, -15.0, 15.0)

        # Inner loop: pitch error -> stern plane command
        pitch_error_deg = theta_cmd_deg - np.degrees(theta)
        delta_s = self.pitch_pid.compute(pitch_error_deg, dt)

        # Heading controller
        heading_error = self._wrap(self.heading_cmd - np.degrees(psi))
        delta_r = self.heading_pid.compute(heading_error, dt)

        return delta_s, delta_r

    @staticmethod
    def _wrap(a: float) -> float:
        return (a + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------------------
# Submarine dynamics
# ---------------------------------------------------------------------------
class SubmarineDynamics:
    """6-DOF Gertler-Hagen equations for a 60 m submarine.

    State vector  x = [x_n, y_n, z_n, phi, theta, psi, u, v, w, p, q, r]

    All hydrodynamic coefficients are pre-dimensionalised once in __init__
    for the design speed U_d so that the RHS evaluation is efficient.
    Speed-dependent scaling (U/U_d)^2 is applied at run time.
    """

    def __init__(self, params: SubmarineParams | None = None):
        self.p = params or SubmarineParams()
        sp = self.p
        rho = sp.rho
        L = sp.L
        Ud = sp.U_design

        # Pre-compute dimensional reference values
        qL2 = 0.5 * rho * L**2          # for force coefficients
        qL3 = 0.5 * rho * L**3          # for moment coefficients / added mass (force)
        qL4 = 0.5 * rho * L**4          # for added mass (moment coupling)
        qL5 = 0.5 * rho * L**5          # for added inertia

        # --- Added mass (dimensional) ---
        # Non-dim primes (* 1e0 for clarity)
        self.Xud = -1.0e-3 * qL3        # surge added mass
        self.Yvd = -1.2e-2 * qL3        # sway added mass
        self.Zwd = -1.2e-2 * qL3        # heave added mass
        self.Kpd = -1.0e-4 * qL5        # roll added inertia
        self.Mqd = -8.0e-4 * qL5        # pitch added inertia
        self.Nrd = -8.0e-4 * qL5        # yaw added inertia
        self.Yrd = 1.2e-4 * qL4
        self.Zqd = -1.2e-4 * qL4
        self.Mwd = -1.2e-4 * qL4
        self.Nvd = 1.2e-4 * qL4

        # --- Velocity / rate derivatives (dimensional at Ud) ---
        # These scale as U^2 at runtime
        Ud2 = Ud**2
        self.Xuu = -1.5e-3 * qL2        # dimensional per U^2

        self.Yv = -1.2e-2 * qL2 * Ud    # Y_v' * qL2 * U (linear in v)
        self.Yr = 2.0e-3 * qL3 * Ud     # Y_r' * qL3 * U
        self.Zw = -1.2e-2 * qL2 * Ud
        self.Zq = -2.0e-3 * qL3 * Ud
        self.Kv = -1.0e-4 * qL3 * Ud
        self.Kp = -3.0e-4 * qL4 * Ud
        self.Mw = -4.0e-3 * qL3 * Ud
        self.Mq = -5.0e-3 * qL4 * Ud
        self.Nv = -4.0e-3 * qL3 * Ud
        self.Nr = -5.0e-3 * qL4 * Ud

        # Quadratic (|·|·) coefficients – these scale as U^2 inherently
        self.Yvv = -0.09 * qL2
        self.Zww = -0.09 * qL2
        self.Mww = 0.02 * qL3
        self.Nvv = 0.02 * qL3

        # Control surface derivatives (scale as U^2)
        # Sign convention:
        #   +delta_s  -> pitch nose-down (dive)  -> Z>0, M>0
        #   +delta_r  -> yaw starboard (turn right) -> N>0
        self.Ydr = 4.0e-3 * qL2 * Ud2
        self.Ndr = 2.0e-3 * qL3 * Ud2
        self.Kdr = -0.5e-4 * qL3 * Ud2
        self.Zds = 4.0e-3 * qL2 * Ud2
        self.Mds = 2.0e-3 * qL3 * Ud2

        # Thrust: balanced so that at Ud the net surge force = 0
        # X_drag(Ud) = Xuu * Ud^2, so thrust = -Xuu * Ud^2
        self.T0 = -self.Xuu * Ud2       # constant thrust [N]

        # Build constant part of 6x6 mass matrix
        m = sp.mass
        self.M_mat = np.zeros((6, 6))
        self.M_mat[0, 0] = m - self.Xud
        self.M_mat[1, 1] = m - self.Yvd
        self.M_mat[1, 5] = -m * sp.x_G - self.Yrd
        self.M_mat[2, 2] = m - self.Zwd
        self.M_mat[2, 4] = m * sp.x_G - self.Zqd
        self.M_mat[3, 3] = sp.Ixx - self.Kpd
        self.M_mat[3, 5] = -sp.Ixz
        self.M_mat[4, 2] = -m * sp.z_G - self.Mwd
        self.M_mat[4, 4] = sp.Iyy - self.Mqd
        self.M_mat[5, 1] = m * sp.x_G - self.Nvd
        self.M_mat[5, 3] = -sp.Ixz
        self.M_mat[5, 5] = sp.Izz - self.Nrd

    def derivatives(self, state: np.ndarray, delta_s: float,
                    delta_r: float) -> np.ndarray:
        """Compute d(state)/dt.  delta_s, delta_r in radians."""
        sp = self.p
        _, _, _, phi, theta, psi, u, v, w, p, q, r = state

        U = max(np.sqrt(u**2 + v**2 + w**2), 0.5)

        # --- Kinematics ---
        cphi, sphi = np.cos(phi), np.sin(phi)
        cth, sth = np.cos(theta), np.sin(theta)
        cpsi, spsi = np.cos(psi), np.sin(psi)
        cth_safe = max(abs(cth), 1e-6) * np.sign(cth) if abs(cth) > 1e-6 else 1e-6

        dx = cth*cpsi*u + (sphi*sth*cpsi - cphi*spsi)*v + (cphi*sth*cpsi + sphi*spsi)*w
        dy = cth*spsi*u + (sphi*sth*spsi + cphi*cpsi)*v + (cphi*sth*spsi - sphi*cpsi)*w
        dz = -sth*u + sphi*cth*v + cphi*cth*w

        dphi = p + (sphi*sth/cth_safe)*q + (cphi*sth/cth_safe)*r
        dtheta = cphi*q - sphi*r
        dpsi = (sphi/cth_safe)*q + (cphi/cth_safe)*r

        # --- Hydrodynamic forces (dimensional) ---
        m = sp.mass

        # Surge
        X_hyd = self.Xuu * u * abs(u)
        X_thrust = self.T0
        X_rigid = m * (v*r - w*q + sp.x_G*(q**2 + r**2))
        X_hs = -(sp.W - sp.B) * sth

        # Sway
        Y_hyd = self.Yv*v + self.Yr*r + self.Yvv*abs(v)*v + self.Ydr*delta_r
        Y_rigid = m * (-u*r + w*p)
        Y_hs = (sp.W - sp.B) * cphi * cth  # 0 for neutral buoyancy

        # Heave
        Z_hyd = self.Zw*w + self.Zq*q + self.Zww*abs(w)*w + self.Zds*delta_s
        Z_rigid = m * (u*q - v*p)
        Z_hs = (sp.W - sp.B) * cphi * cth

        # Roll
        K_hyd = self.Kv*v + self.Kp*p + self.Kdr*delta_r
        K_rigid = (sp.Iyy - sp.Izz)*q*r
        K_hs = -(sp.z_G * sp.W) * sphi * cth

        # Pitch
        M_hyd = self.Mw*w + self.Mq*q + self.Mww*abs(w)*w + self.Mds*delta_s
        M_rigid = (sp.Izz - sp.Ixx)*p*r
        M_hs = -(sp.z_G * sp.W)*sth - (sp.x_G * sp.W)*cphi*cth

        # Yaw
        N_hyd = self.Nv*v + self.Nr*r + self.Nvv*abs(v)*v + self.Ndr*delta_r
        N_rigid = (sp.Ixx - sp.Iyy)*p*q
        N_hs = (sp.x_G * sp.W)*sphi*cth

        F = np.array([
            X_hyd + X_thrust + X_rigid + X_hs,
            Y_hyd + Y_rigid + Y_hs,
            Z_hyd + Z_rigid + Z_hs,
            K_hyd + K_rigid + K_hs,
            M_hyd + M_rigid + M_hs,
            N_hyd + N_rigid + N_hs,
        ])

        acc = np.linalg.solve(self.M_mat, F)

        return np.array([dx, dy, dz, dphi, dtheta, dpsi,
                         acc[0], acc[1], acc[2], acc[3], acc[4], acc[5]])


# ---------------------------------------------------------------------------
# Actuator model
# ---------------------------------------------------------------------------
def actuator(cmd_deg: float, cur_deg: float, max_deg: float,
             rate_max: float, dt: float) -> float:
    cmd_deg = np.clip(cmd_deg, -max_deg, max_deg)
    change = np.clip(cmd_deg - cur_deg, -rate_max * dt, rate_max * dt)
    return cur_deg + change


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def run_simulation(
    total_time: float = 600.0,
    dt: float = 0.05,
    initial_speed: float = 5.0,
    initial_depth: float = 50.0,
    target_depth: float = 100.0,
    target_heading: float = 0.0,
    heading_change_time: float = 200.0,
    heading_change_deg: float = 30.0,
) -> dict:
    """Run the submarine simulation and return time-history results."""
    params = SubmarineParams()
    dyn = SubmarineDynamics(params)
    ap = Autopilot()
    ap.depth_cmd = target_depth
    ap.heading_cmd = target_heading

    state = np.array([
        0.0, 0.0, initial_depth,
        0.0, 0.0, np.radians(target_heading),
        initial_speed, 0.0, 0.0,
        0.0, 0.0, 0.0,
    ])

    n = int(total_time / dt)
    t_arr = np.zeros(n)
    s_arr = np.zeros((n, 12))
    ds_arr = np.zeros(n)
    dr_arr = np.zeros(n)
    dcmd = np.zeros(n)
    hcmd = np.zeros(n)

    ds_act = 0.0
    dr_act = 0.0

    for i in range(n):
        t = i * dt
        t_arr[i] = t
        s_arr[i] = state

        if t >= heading_change_time:
            ap.heading_cmd = heading_change_deg

        dcmd[i] = ap.depth_cmd
        hcmd[i] = ap.heading_cmd

        ds_cmd, dr_cmd = ap.compute_controls(state[2], state[4], state[5], dt)

        ds_act = actuator(ds_cmd, ds_act, params.delta_s_max,
                          params.delta_rate_max, dt)
        dr_act = actuator(dr_cmd, dr_act, params.delta_r_max,
                          params.delta_rate_max, dt)

        ds_arr[i] = ds_act
        dr_arr[i] = dr_act

        ds_r = np.radians(ds_act)
        dr_r = np.radians(dr_act)

        # RK4
        k1 = dyn.derivatives(state, ds_r, dr_r)
        k2 = dyn.derivatives(state + 0.5*dt*k1, ds_r, dr_r)
        k3 = dyn.derivatives(state + 0.5*dt*k2, ds_r, dr_r)
        k4 = dyn.derivatives(state + dt*k3, ds_r, dr_r)
        state = state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

        # Normalise angles
        for idx in [3, 4, 5]:
            state[idx] = (state[idx] + np.pi) % (2*np.pi) - np.pi

    return {
        "time": t_arr, "state": s_arr,
        "delta_s": ds_arr, "delta_r": dr_arr,
        "depth_cmd": dcmd, "heading_cmd": hcmd,
        "params": {
            "total_time": total_time, "dt": dt,
            "initial_speed": initial_speed, "initial_depth": initial_depth,
            "target_depth": target_depth, "target_heading": target_heading,
            "heading_change_time": heading_change_time,
            "heading_change_deg": heading_change_deg,
        },
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_results(results: dict, save_path: str = "submarine_simulation.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available – skipping plots.")
        return

    t = results["time"]
    s = results["state"]
    ds = results["delta_s"]
    dr = results["delta_r"]

    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    fig.suptitle("Submarine Motion Simulation (Gertler-Hagen EOM, 60 m hull)",
                 fontsize=14, fontweight="bold")

    # Depth
    ax = axes[0, 0]
    ax.plot(t, s[:, 2], "b-", lw=1.5, label="Depth")
    ax.plot(t, results["depth_cmd"], "r--", lw=1, label="Cmd")
    ax.set_ylabel("Depth [m]"); ax.set_xlabel("Time [s]")
    ax.legend(); ax.set_title("Depth"); ax.invert_yaxis(); ax.grid(True, alpha=0.3)

    # Pitch
    ax = axes[0, 1]
    ax.plot(t, np.degrees(s[:, 4]), "g-", lw=1.5)
    ax.set_ylabel("Pitch [deg]"); ax.set_xlabel("Time [s]")
    ax.set_title("Pitch Angle"); ax.grid(True, alpha=0.3)

    # Heading
    ax = axes[1, 0]
    ax.plot(t, np.degrees(s[:, 5]), "b-", lw=1.5, label="Heading")
    ax.plot(t, results["heading_cmd"], "r--", lw=1, label="Cmd")
    ax.set_ylabel("Heading [deg]"); ax.set_xlabel("Time [s]")
    ax.legend(); ax.set_title("Heading"); ax.grid(True, alpha=0.3)

    # Roll
    ax = axes[1, 1]
    ax.plot(t, np.degrees(s[:, 3]), "m-", lw=1.5)
    ax.set_ylabel("Roll [deg]"); ax.set_xlabel("Time [s]")
    ax.set_title("Roll Angle"); ax.grid(True, alpha=0.3)

    # Surge speed
    ax = axes[2, 0]
    ax.plot(t, s[:, 6], "b-", lw=1.5)
    ax.set_ylabel("u [m/s]"); ax.set_xlabel("Time [s]")
    ax.set_title("Surge Velocity"); ax.grid(True, alpha=0.3)

    # Sway / heave
    ax = axes[2, 1]
    ax.plot(t, s[:, 7], "c-", lw=1.5, label="v (sway)")
    ax.plot(t, s[:, 8], "m-", lw=1.5, label="w (heave)")
    ax.set_ylabel("[m/s]"); ax.set_xlabel("Time [s]")
    ax.legend(); ax.set_title("Sway & Heave"); ax.grid(True, alpha=0.3)

    # Control surfaces
    ax = axes[3, 0]
    ax.plot(t, ds, "b-", lw=1.5, label="Stern plane")
    ax.plot(t, dr, "r-", lw=1.5, label="Rudder")
    ax.set_ylabel("[deg]"); ax.set_xlabel("Time [s]")
    ax.legend(); ax.set_title("Control Surfaces"); ax.grid(True, alpha=0.3)

    # Trajectory top-view
    ax = axes[3, 1]
    ax.plot(s[:, 0], s[:, 1], "b-", lw=1.5)
    ax.plot(s[0, 0], s[0, 1], "go", ms=8, label="Start")
    ax.plot(s[-1, 0], s[-1, 1], "rs", ms=8, label="End")
    ax.set_xlabel("X North [m]"); ax.set_ylabel("Y East [m]")
    ax.legend(); ax.set_title("Horizontal Trajectory")
    ax.set_aspect("equal", adjustable="datalim"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {save_path}")
    plt.close()


def plot_3d_trajectory(results: dict,
                       save_path: str = "submarine_trajectory_3d.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    s = results["state"]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(s[:, 0], s[:, 1], -s[:, 2], "b-", lw=1.5)
    ax.plot([s[0, 0]], [s[0, 1]], [-s[0, 2]], "go", ms=8, label="Start")
    ax.plot([s[-1, 0]], [s[-1, 1]], [-s[-1, 2]], "rs", ms=8, label="End")
    ax.set_xlabel("X North [m]"); ax.set_ylabel("Y East [m]")
    ax.set_zlabel("Altitude [m]")
    ax.set_title("3D Submarine Trajectory"); ax.legend()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"3D plot saved to {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("  Submarine Motion Simulator")
    print("  Gertler-Hagen 6-DOF EOM  |  60 m Hull  |  Cross Rudder")
    print("=" * 65)

    print("\nScenario: depth 50 m -> 100 m,  heading 0 -> 30 deg at t=200 s")
    print("Running (600 s, dt=0.05 s) ...")

    results = run_simulation(
        total_time=600.0, dt=0.05,
        initial_speed=5.0, initial_depth=50.0,
        target_depth=100.0, target_heading=0.0,
        heading_change_time=200.0, heading_change_deg=30.0,
    )

    f = results["state"][-1]
    print("\n--- Final State (t = 600 s) ---")
    print(f"  Position (NED): x={f[0]:.1f} m, y={f[1]:.1f} m, depth={f[2]:.1f} m")
    print(f"  Euler:  roll={np.degrees(f[3]):.2f}°  pitch={np.degrees(f[4]):.2f}°"
          f"  heading={np.degrees(f[5]):.2f}°")
    print(f"  Speed:  u={f[6]:.2f} m/s  v={f[7]:.3f} m/s  w={f[8]:.3f} m/s")
    print(f"  Depth error:   {abs(results['depth_cmd'][-1] - f[2]):.2f} m")
    print(f"  Heading error: {abs(results['heading_cmd'][-1] - np.degrees(f[5])):.2f}°")

    print("\nGenerating plots...")
    plot_results(results)
    plot_3d_trajectory(results)

    summary = {
        "scenario": results["params"],
        "final_state": {
            "x_north_m": round(f[0], 2), "y_east_m": round(f[1], 2),
            "depth_m": round(f[2], 2),
            "roll_deg": round(np.degrees(f[3]), 3),
            "pitch_deg": round(np.degrees(f[4]), 3),
            "heading_deg": round(np.degrees(f[5]), 3),
            "u_ms": round(f[6], 3), "v_ms": round(f[7], 4),
            "w_ms": round(f[8], 4),
        },
        "depth_error_m": round(abs(results["depth_cmd"][-1] - f[2]), 3),
        "heading_error_deg": round(
            abs(results["heading_cmd"][-1] - np.degrees(f[5])), 3),
    }
    with open("simulation_summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)
    print("Summary saved to simulation_summary.json")
    print("\nDone.")


if __name__ == "__main__":
    main()
