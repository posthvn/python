"""
3D Towed Array Sonar System (TASS) Cable Dynamics Model
========================================================

Based on Sanders, J.V. (1982) "A three-dimensional dynamic analysis of a towed system."
Ocean Engineering, Vol. 9, No. 5, pp. 483-499.

This module implements both quasi-static and dynamic lumped-parameter models
for 3D towed cable/array motion analysis.

Supports multi-section cables (HWC, LWC, AM, TR) with different
physical properties and element lengths per section.

Coordinate System:
  - X: forward (tow direction)
  - Y: lateral (starboard positive)
  - Z: depth (downward positive, tow point at origin)

Author: Auto-generated based on Sanders (1982) formulation
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List


@dataclass
class CableSection:
    """Physical properties of one cable section."""
    name: str = "cable"
    length: float = 1000.0          # Section length [m]
    diameter: float = 0.05          # Outer diameter [m]
    mass_per_length: float = 2.0    # Mass per unit length in air [kg/m]
    EA: float = 2.0e4               # Axial stiffness [N]
    Cd_n: float = 1.2               # Normal drag coefficient
    Cd_t: float = 0.02              # Tangential drag coefficient
    added_mass_coeff: float = 1.0   # Added mass coefficient (Ca)
    damping_coeff: float = 100.0    # Structural damping [N·s/m]
    buoyancy_type: str = "negative" # "negative", "neutral", "positive"
    ds: float = 0.0                 # Element length [m] (0 = auto)


@dataclass
class EnvironmentProperties:
    """Environmental conditions."""
    rho_water: float = 1025.0
    gravity: float = 9.81
    current_velocity: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0])
    )


class MultiSectionCable:
    """
    Multi-section TASS cable definition.

    A TASS typically consists of:
      - HWC (Heavy Weight Cable): negatively buoyant, tow cable
      - LWC (Light Weight Cable): neutrally buoyant, vibration isolation
      - AM  (Acoustic Module):    neutrally buoyant, hydrophone array
      - TR  (Tail Rope):          neutrally buoyant, drogue/stabilizer

    Order: Tow point → HWC → LWC → AM → TR (free end)

    Each section can have its own element length (ds) for FDM discretization.
    """

    def __init__(self, sections: List[CableSection], env: EnvironmentProperties):
        self.sections = sections
        self.env = env
        self.total_length = sum(s.length for s in sections)

        # Precompute per-section submerged weight
        self.section_weights = []
        for s in sections:
            area = np.pi * (s.diameter / 2) ** 2
            if s.buoyancy_type == "neutral":
                w = 0.0
            else:
                w = (s.mass_per_length - env.rho_water * area) * env.gravity
            self.section_weights.append(w)

        # Compute element layout per section
        self._compute_element_layout()

    def _compute_element_layout(self):
        """Compute the element layout: per-section n_elements and ds."""
        self.section_n_elements = []
        self.section_ds = []

        for sec in self.sections:
            if sec.ds > 0:
                n = max(1, round(sec.length / sec.ds))
            else:
                # Auto: default ~10m elements
                n = max(1, round(sec.length / 10.0))
            ds_actual = sec.length / n
            self.section_n_elements.append(n)
            self.section_ds.append(ds_actual)

        self.n_elements_total = sum(self.section_n_elements)
        self.n_nodes_total = self.n_elements_total + 1

    def get_element_properties(self) -> dict:
        """
        Compute per-element properties for the discretized cable.
        Each section uses its own element length.

        Returns:
            dict with per-element arrays and section boundary info
        """
        N = self.n_elements_total

        # Per-element arrays
        ds_array = np.zeros(N)
        diameter = np.zeros(N)
        mass_per_length = np.zeros(N)
        EA = np.zeros(N)
        Cd_n = np.zeros(N)
        Cd_t = np.zeros(N)
        added_mass = np.zeros(N)
        damping = np.zeros(N)
        submerged_weight = np.zeros(N)
        section_id = np.zeros(N, dtype=int)

        # Fill per-element properties section by section
        elem_offset = 0
        section_boundaries = [0]  # Node indices where sections start

        for sec_idx, sec in enumerate(self.sections):
            n_elem = self.section_n_elements[sec_idx]
            ds_sec = self.section_ds[sec_idx]
            w = self.section_weights[sec_idx]
            area = np.pi * (sec.diameter / 2) ** 2

            for j in range(n_elem):
                i = elem_offset + j
                ds_array[i] = ds_sec
                diameter[i] = sec.diameter
                mass_per_length[i] = sec.mass_per_length
                EA[i] = sec.EA
                Cd_n[i] = sec.Cd_n
                Cd_t[i] = sec.Cd_t
                added_mass[i] = self.env.rho_water * area * sec.added_mass_coeff
                damping[i] = sec.damping_coeff
                submerged_weight[i] = w
                section_id[i] = sec_idx

            elem_offset += n_elem
            section_boundaries.append(elem_offset)  # Node index

        # Node arc-length positions (cumulative ds from tow point)
        node_s = np.zeros(self.n_nodes_total)
        for i in range(N):
            node_s[i + 1] = node_s[i] + ds_array[i]

        return {
            'ds': ds_array,
            'diameter': diameter,
            'mass_per_length': mass_per_length,
            'EA': EA,
            'Cd_n': Cd_n,
            'Cd_t': Cd_t,
            'added_mass': added_mass,
            'damping': damping,
            'submerged_weight': submerged_weight,
            'section_id': section_id,
            'section_boundaries': section_boundaries,
            'node_s': node_s,
        }


class TASSCableModel:
    """
    3D TASS Cable Dynamics Model with multi-section support.

    Each element can have a different length (ds), diameter, mass,
    drag coefficients, and submerged weight.

    Governing equations (Sanders, 1982):
      Tangential:  dT/ds = f_t - w·sin(φ)
      Normal (V):  T·dφ/ds = f_n,v - w·cos(φ)
      Normal (H):  T·cos(φ)·dθ/ds = f_n,h
    """

    def __init__(self, cable: MultiSectionCable, env: EnvironmentProperties):
        self.env = env
        self.cable = cable
        self.total_length = cable.total_length
        self.elem_props = cable.get_element_properties()
        self.n_elements = cable.n_elements_total
        self.n_nodes = cable.n_nodes_total

        # State arrays
        self.positions = np.zeros((self.n_nodes, 3))
        self.velocities = np.zeros((self.n_nodes, 3))
        self.tensions = np.zeros(self.n_elements)

    def compute_element_vectors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute element tangent vectors, lengths, and strains."""
        ds = self.elem_props['ds']
        t_hat = np.zeros((self.n_elements, 3))
        seg_lengths = np.zeros(self.n_elements)
        strains = np.zeros(self.n_elements)

        for i in range(self.n_elements):
            seg = self.positions[i + 1] - self.positions[i]
            seg_len = np.linalg.norm(seg)
            seg_lengths[i] = seg_len
            if seg_len < 1e-12:
                t_hat[i] = np.array([1.0, 0.0, 0.0])
            else:
                t_hat[i] = seg / seg_len
            strains[i] = (seg_len - ds[i]) / ds[i]

        return t_hat, seg_lengths, strains

    def compute_forces(self, tow_point_vel: np.ndarray,
                        fluid_velocities: Optional[np.ndarray] = None
                        ) -> np.ndarray:
        """Compute total forces at each node using per-element properties."""
        rho = self.env.rho_water
        ep = self.elem_props
        ds = ep['ds']

        t_hat, seg_lengths, strains = self.compute_element_vectors()

        if fluid_velocities is None:
            fluid_velocities = np.tile(
                self.env.current_velocity, (self.n_elements, 1)
            )

        forces = np.zeros((self.n_nodes, 3))

        # --- Tension forces (per-element EA and ds) ---
        for i in range(self.n_elements):
            tension = ep['EA'][i] * max(strains[i], 0.0)
            self.tensions[i] = tension

            dv = self.velocities[i + 1] - self.velocities[i]
            dv_axial = np.dot(dv, t_hat[i]) * t_hat[i]
            f_damp = ep['damping'][i] * dv_axial

            force = tension * t_hat[i] + f_damp
            forces[i] += force
            forces[i + 1] -= force

        # --- Hydrodynamic drag forces (per-element Cd, diameter, ds) ---
        for i in range(self.n_elements):
            d = ep['diameter'][i]
            Cd_n = ep['Cd_n'][i]
            Cd_t = ep['Cd_t'][i]

            mid_vel = 0.5 * (self.velocities[i] + self.velocities[i + 1])
            v_rel = fluid_velocities[i] - mid_vel

            v_t_scalar = np.dot(v_rel, t_hat[i])
            v_t = v_t_scalar * t_hat[i]
            v_n = v_rel - v_t

            v_t_mag = abs(v_t_scalar)
            v_n_mag = np.linalg.norm(v_n)

            f_tang = 0.5 * rho * Cd_t * np.pi * d * v_t_mag * v_t * ds[i]
            f_norm = 0.5 * rho * Cd_n * d * v_n_mag * v_n * ds[i]

            f_element = f_tang + f_norm
            forces[i] += 0.5 * f_element
            forces[i + 1] += 0.5 * f_element

        # --- Gravity and buoyancy (per-element submerged weight and ds) ---
        for i in range(self.n_elements):
            w = ep['submerged_weight'][i] * ds[i]
            forces[i, 2] += 0.5 * w
            forces[i + 1, 2] += 0.5 * w

        return forces

    def compute_accelerations(self, tow_point_pos: np.ndarray,
                               tow_point_vel: np.ndarray,
                               fluid_velocities: Optional[np.ndarray] = None
                               ) -> np.ndarray:
        """Compute accelerations at each node."""
        self.positions[0] = tow_point_pos.copy()
        self.velocities[0] = tow_point_vel.copy()

        f_total = self.compute_forces(tow_point_vel, fluid_velocities)
        ep = self.elem_props
        ds = ep['ds']

        accelerations = np.zeros((self.n_nodes, 3))
        for i in range(1, self.n_nodes):
            # Mass from adjacent elements
            if i < self.n_elements:
                m1 = (ep['mass_per_length'][i - 1] + ep['added_mass'][i - 1]) * ds[i - 1] * 0.5
                m2 = (ep['mass_per_length'][i] + ep['added_mass'][i]) * ds[i] * 0.5
                m_total = m1 + m2
            else:
                m_total = (ep['mass_per_length'][-1] + ep['added_mass'][-1]) * ds[-1] * 0.5
            accelerations[i] = f_total[i] / max(m_total, 1e-6)

        return accelerations

    def time_step_rk4(self, dt: float, tow_point_pos: np.ndarray,
                       tow_point_vel: np.ndarray,
                       fluid_velocities: Optional[np.ndarray] = None):
        """Advance solution by one time step using RK4."""
        pos_save = self.positions.copy()
        vel_save = self.velocities.copy()

        a1 = self.compute_accelerations(tow_point_pos, tow_point_vel, fluid_velocities)
        k1_v = a1 * dt
        k1_x = self.velocities * dt

        self.positions = pos_save + 0.5 * k1_x
        self.velocities = vel_save + 0.5 * k1_v
        self.positions[0] = tow_point_pos
        self.velocities[0] = tow_point_vel
        a2 = self.compute_accelerations(tow_point_pos, tow_point_vel, fluid_velocities)
        k2_v = a2 * dt
        k2_x = self.velocities * dt

        self.positions = pos_save + 0.5 * k2_x
        self.velocities = vel_save + 0.5 * k2_v
        self.positions[0] = tow_point_pos
        self.velocities[0] = tow_point_vel
        a3 = self.compute_accelerations(tow_point_pos, tow_point_vel, fluid_velocities)
        k3_v = a3 * dt
        k3_x = self.velocities * dt

        self.positions = pos_save + k3_x
        self.velocities = vel_save + k3_v
        self.positions[0] = tow_point_pos
        self.velocities[0] = tow_point_vel
        a4 = self.compute_accelerations(tow_point_pos, tow_point_vel, fluid_velocities)
        k4_v = a4 * dt
        k4_x = self.velocities * dt

        self.positions = pos_save + (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6
        self.velocities = vel_save + (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6

        self.positions[0] = tow_point_pos
        self.velocities[0] = tow_point_vel

    def solve_quasi_static(self, tow_velocity: np.ndarray,
                            depth: float = 0.0) -> dict:
        """
        Quasi-static steady-state solution (Sanders 1982).

        Supports multi-section cables with varying w(s), Cd(s), d(s), ds(s).
        Integration from free end (tail) to tow point using per-element ds.
        """
        V = np.linalg.norm(tow_velocity[:2])
        rho = self.env.rho_water
        N = self.n_nodes
        ep = self.elem_props
        ds_arr = ep['ds']

        # Node arc-length from tow point
        node_s = ep['node_s']

        T = np.zeros(N)
        phi = np.zeros(N)
        theta = np.zeros(N)
        x = np.zeros(N)
        y = np.zeros(N)
        z = np.zeros(N)

        # Initial conditions at free end (node index 0 in integration = last element)
        d_tail = ep['diameter'][-1]
        Cd_t_tail = ep['Cd_t'][-1]
        w_tail = ep['submerged_weight'][-1]
        f_t_tail = 0.5 * rho * Cd_t_tail * np.pi * d_tail * V ** 2

        T[0] = max(f_t_tail * 0.5, 1.0)

        if V > 0.1 and abs(w_tail) > 0.01:
            phi[0] = np.arctan2(abs(w_tail), max(f_t_tail, 0.1))
            phi[0] = min(phi[0], np.radians(45))
            if w_tail < 0:
                phi[0] = -phi[0]
        else:
            phi[0] = np.radians(1)

        theta[0] = np.arctan2(tow_velocity[1], tow_velocity[0]) if V > 0.1 else 0.0
        x[0] = 0.0
        y[0] = 0.0
        z[0] = depth

        # Integrate from free end (i=0) to tow point (i=N-1).
        # Integration step i uses element (n_elements - 1 - i) properties.
        for i in range(N - 1):
            elem_idx = max(0, min(self.n_elements - 1 - i, self.n_elements - 1))

            d_i = ep['diameter'][elem_idx]
            Cd_t_i = ep['Cd_t'][elem_idx]
            w_i = ep['submerged_weight'][elem_idx]
            ds_step = ds_arr[elem_idx]

            f_t_i = 0.5 * rho * Cd_t_i * np.pi * d_i * V ** 2

            T_i, phi_i, theta_i = T[i], phi[i], theta[i]

            def qs_rhs(T_v, phi_v, _theta_v, _w=w_i, _ft=f_t_i):
                cp = np.cos(phi_v)
                sp = np.sin(phi_v)
                dT_ds = _ft + _w * sp
                dphi_ds = (-_w * cp) / max(T_v, 10.0)
                dphi_ds = np.clip(dphi_ds, -0.05, 0.05)
                dtheta_ds = 0.0
                return dT_ds, dphi_ds, dtheta_ds

            # RK4 integration with this element's ds
            k1_T, k1_p, k1_t = qs_rhs(T_i, phi_i, theta_i)
            k2_T, k2_p, k2_t = qs_rhs(
                T_i + 0.5 * ds_step * k1_T,
                phi_i + 0.5 * ds_step * k1_p,
                theta_i + 0.5 * ds_step * k1_t)
            k3_T, k3_p, k3_t = qs_rhs(
                T_i + 0.5 * ds_step * k2_T,
                phi_i + 0.5 * ds_step * k2_p,
                theta_i + 0.5 * ds_step * k2_t)
            k4_T, k4_p, k4_t = qs_rhs(
                T_i + ds_step * k3_T,
                phi_i + ds_step * k3_p,
                theta_i + ds_step * k3_t)

            T[i + 1] = T_i + ds_step * (k1_T + 2 * k2_T + 2 * k3_T + k4_T) / 6
            phi[i + 1] = phi_i + ds_step * (k1_p + 2 * k2_p + 2 * k3_p + k4_p) / 6
            theta[i + 1] = theta_i + ds_step * (k1_t + 2 * k2_t + 2 * k3_t + k4_t) / 6
            phi[i + 1] = np.clip(phi[i + 1], -np.pi / 2, np.pi / 2)

            cp = np.cos(phi[i + 1])
            sp = np.sin(phi[i + 1])
            ct = np.cos(theta[i + 1])
            st = np.sin(theta[i + 1])
            x[i + 1] = x[i] + ds_step * cp * ct
            y[i + 1] = y[i] + ds_step * cp * st
            z[i + 1] = z[i] - ds_step * sp

            T[i + 1] = max(T[i + 1], 1.0)

        # Reverse so index 0 = tow point
        positions = np.column_stack([x[::-1], y[::-1], z[::-1]])

        # Shift to tow point at origin, Z+ = depth
        tow_pt = positions[0].copy()
        positions -= tow_pt
        positions[:, 2] = -positions[:, 2]

        self.positions = positions

        return {
            'positions': positions,
            'tensions': T[::-1],
            'phi': phi[::-1],
            'theta': theta[::-1],
            'arc_length': node_s,
        }


class TowShipTrajectory:
    """
    Generate tow ship trajectories for simulation.
    Supports: straight-line, steady turning, S-turn maneuver.
    """

    def __init__(self, speed: float = 5.0, heading_init: float = 0.0,
                 depth: float = 0.0):
        self.speed = speed
        self.heading = heading_init
        self.depth = depth

    def straight_line(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        pos = np.array([self.speed * t, 0.0, self.depth])
        vel = np.array([self.speed, 0.0, 0.0])
        return pos, vel

    def steady_turn(self, t: float, turn_radius: float = 500.0
                     ) -> Tuple[np.ndarray, np.ndarray]:
        omega = self.speed / turn_radius
        angle = omega * t
        pos = np.array([
            turn_radius * np.sin(angle),
            turn_radius * (1 - np.cos(angle)),
            self.depth
        ])
        vel = np.array([
            self.speed * np.cos(angle),
            self.speed * np.sin(angle),
            0.0
        ])
        return pos, vel

    def s_turn(self, t: float, turn_radius: float = 500.0,
               turn_duration: float = 60.0) -> Tuple[np.ndarray, np.ndarray]:
        omega = self.speed / turn_radius
        t1 = turn_duration
        t2 = t1 + turn_duration
        t3 = t2 + turn_duration

        if t < t1:
            pos = np.array([self.speed * t, 0.0, self.depth])
            vel = np.array([self.speed, 0.0, 0.0])
        elif t < t2:
            dt = t - t1
            angle = omega * dt
            x0 = self.speed * t1
            pos = np.array([
                x0 + turn_radius * np.sin(angle),
                turn_radius * (1 - np.cos(angle)),
                self.depth
            ])
            vel = np.array([
                self.speed * np.cos(angle),
                self.speed * np.sin(angle),
                0.0
            ])
        elif t < t3:
            dt = t - t2
            angle_end = omega * turn_duration
            x1 = self.speed * t1 + turn_radius * np.sin(angle_end)
            y1 = turn_radius * (1 - np.cos(angle_end))
            heading1 = angle_end
            angle = -omega * dt
            pos = np.array([
                x1 + turn_radius * (np.sin(heading1 + angle) - np.sin(heading1)),
                y1 + turn_radius * (-np.cos(heading1 + angle) + np.cos(heading1)),
                self.depth
            ])
            vel = np.array([
                self.speed * np.cos(heading1 + angle),
                self.speed * np.sin(heading1 + angle),
                0.0
            ])
        else:
            dt = t - t3
            pos_s, vel_s = self.s_turn(t3 - 0.001, turn_radius, turn_duration)
            heading_final = np.arctan2(vel_s[1], vel_s[0])
            pos = pos_s + np.array([
                self.speed * np.cos(heading_final) * dt,
                self.speed * np.sin(heading_final) * dt,
                0.0
            ])
            vel = np.array([
                self.speed * np.cos(heading_final),
                self.speed * np.sin(heading_final),
                0.0
            ])

        return pos, vel


class TASSSimulation:
    """
    Main simulation driver for 3D TASS motion analysis.
    Supports multi-section cable models with per-section element lengths.
    """

    def __init__(self, cable: MultiSectionCable, env: EnvironmentProperties,
                 dt: float = 0.1):
        self.cable = cable
        self.env = env
        self.dt = dt
        self.model = TASSCableModel(cable, env)
        self.trajectory = TowShipTrajectory()
        self.time = 0.0

        self.time_history: List[float] = []
        self.position_history: List[np.ndarray] = []
        self.tension_history: List[np.ndarray] = []
        self.tow_point_history: List[np.ndarray] = []

    def set_trajectory(self, trajectory: TowShipTrajectory):
        self.trajectory = trajectory

    def initialize_steady_state(self, tow_velocity: np.ndarray,
                                 depth: float = 0.0):
        """
        Initialize with straight-line configuration. Tow point at origin.
        Per-element catenary angle varies by section weight.
        """
        tow_pos_0, tow_vel_0 = self.trajectory.straight_line(0.0)
        speed = np.linalg.norm(tow_velocity[:2])
        ep = self.model.elem_props
        ds = ep['ds']

        direction = tow_velocity / max(np.linalg.norm(tow_velocity), 1e-10)

        cum_x = 0.0
        cum_z = 0.0
        self.model.positions[0] = tow_pos_0.copy()
        self.model.velocities[0] = tow_vel_0.copy()

        for i in range(self.model.n_elements):
            d_i = ep['diameter'][i]
            Cd_t_i = ep['Cd_t'][i]
            w_i = ep['submerged_weight'][i]

            f_t = 0.5 * self.env.rho_water * Cd_t_i * np.pi * d_i * speed ** 2
            if abs(w_i) > 0.01 and f_t > 0.01:
                cat_angle = np.arctan2(abs(w_i), f_t)
                cat_angle = min(cat_angle, np.radians(20))
            else:
                cat_angle = np.radians(0.5)

            cum_x += ds[i] * np.cos(cat_angle)
            cum_z += ds[i] * np.sin(cat_angle)

            self.model.positions[i + 1] = tow_pos_0.copy()
            self.model.positions[i + 1, 0] -= cum_x * direction[0]
            self.model.positions[i + 1, 1] -= cum_x * direction[1]
            self.model.positions[i + 1, 2] += cum_z
            self.model.velocities[i + 1] = tow_vel_0.copy()

        # Quasi-static for reference
        qs_model = TASSCableModel(self.cable, self.env)
        result = qs_model.solve_quasi_static(tow_velocity, depth)
        return result

    def run(self, t_end: float, save_interval: int = 10,
            maneuver: str = 'straight', **kwargs) -> dict:
        n_steps = int(t_end / self.dt)
        self.time = 0.0

        self.time_history = []
        self.position_history = []
        self.tension_history = []
        self.tow_point_history = []

        if maneuver == 'straight':
            traj_func = self.trajectory.straight_line
        elif maneuver == 'turn':
            turn_radius = kwargs.get('turn_radius', 500.0)
            traj_func = lambda t: self.trajectory.steady_turn(t, turn_radius)
        elif maneuver == 's_turn':
            turn_radius = kwargs.get('turn_radius', 500.0)
            turn_duration = kwargs.get('turn_duration', 60.0)
            traj_func = lambda t: self.trajectory.s_turn(
                t, turn_radius, turn_duration)
        else:
            raise ValueError(f"Unknown maneuver type: {maneuver}")

        total_length = self.model.total_length
        n_elem = self.model.n_elements
        print(f"Running TASS simulation: {maneuver} maneuver")
        print(f"  Total cable length: {total_length} m, Elements: {n_elem}")
        print(f"  Time step: {self.dt} s, Duration: {t_end} s, Steps: {n_steps}")

        for step in range(n_steps):
            self.time = step * self.dt
            tow_pos, tow_vel = traj_func(self.time)
            self.model.time_step_rk4(self.dt, tow_pos, tow_vel)

            if step % save_interval == 0:
                self.time_history.append(self.time)
                self.position_history.append(self.model.positions.copy())
                self.tension_history.append(self.model.tensions.copy())
                self.tow_point_history.append(tow_pos.copy())

                if step % (save_interval * 50) == 0:
                    max_T = np.max(self.model.tensions)
                    print(f"  t = {self.time:.1f} s, max tension = {max_T:.1f} N")

        print("Simulation complete.")

        return {
            'time': np.array(self.time_history),
            'positions': self.position_history,
            'tensions': self.tension_history,
            'tow_points': np.array(self.tow_point_history),
            'cable': self.cable,
            'env': self.env,
            'elem_props': self.model.elem_props,
            'total_length': total_length,
        }
