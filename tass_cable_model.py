"""
3D Towed Array Sonar System (TASS) Cable Dynamics Model
========================================================

Based on Sanders, J.V. (1982) "A three-dimensional dynamic analysis of a towed system."
Ocean Engineering, Vol. 9, No. 5, pp. 483-499.

This module implements both quasi-static and dynamic lumped-parameter models
for 3D towed cable/array motion analysis.

Coordinate System:
  - X: forward (tow direction)
  - Y: lateral (starboard positive)
  - Z: vertical (downward positive)

Author: Auto-generated based on Sanders (1982) formulation
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List


@dataclass
class CableProperties:
    """Physical properties of the towed cable/array."""
    length: float = 1000.0          # Total cable length [m]
    diameter: float = 0.05          # Cable outer diameter [m]
    mass_per_length: float = 2.0    # Mass per unit length in air [kg/m]
    EA: float = 1.0e6               # Axial stiffness [N]
    EI: float = 0.0                 # Bending stiffness [N·m²] (0 for flexible cable)
    Cd_n: float = 1.2               # Normal drag coefficient
    Cd_t: float = 0.02              # Tangential drag coefficient
    added_mass_coeff: float = 1.0   # Added mass coefficient (Ca)
    damping_coeff: float = 50.0     # Structural damping coefficient [N·s/m]


@dataclass
class TowedBodyProperties:
    """Properties of the towed body (depressor, sensor module, etc.)."""
    mass: float = 50.0              # Mass in air [kg]
    volume: float = 0.02            # Displaced volume [m³]
    Cd: float = 1.0                 # Drag coefficient
    area: float = 0.1               # Projected area [m²]


@dataclass
class EnvironmentProperties:
    """Environmental conditions."""
    rho_water: float = 1025.0       # Seawater density [kg/m³]
    rho_cable: float = 1500.0       # Cable material density [kg/m³]
    gravity: float = 9.81           # Gravitational acceleration [m/s²]
    current_velocity: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0])
    )


@dataclass
class TowShipState:
    """Tow ship motion state."""
    position: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0])
    )
    velocity: np.ndarray = field(
        default_factory=lambda: np.array([5.0, 0.0, 0.0])
    )
    heading: float = 0.0            # Ship heading [rad]
    turn_rate: float = 0.0          # Turn rate [rad/s]


class TASSCableModel:
    """
    3D Towed Array Sonar System Cable Dynamics Model.

    Implements the Sanders (1982) lumped-parameter approach for 3D cable dynamics.
    The cable is discretized into N segments, each represented as a lumped mass
    connected by massless elastic elements.

    Governing equations (Sanders, 1982):
      Tangential:  dT/ds = f_t - w·sin(φ)
      Normal (V):  T·dφ/ds = f_n,v - w·cos(φ)
      Normal (H):  T·cos(φ)·dθ/ds = f_n,h

    where:
      T: tension, s: arc length, φ: inclination, θ: azimuth
      f_t: tangential drag/unit length
      f_n,v, f_n,h: normal drag/unit length (vertical, horizontal)
      w: submerged weight/unit length
    """

    def __init__(self, cable: CableProperties, env: EnvironmentProperties,
                 n_elements: int = 100):
        self.cable = cable
        self.env = env
        self.n_elements = n_elements
        self.n_nodes = n_elements + 1
        self.ds = cable.length / n_elements  # Segment length

        # Compute derived quantities
        cable_area = np.pi * (cable.diameter / 2) ** 2
        self.mass_per_length = cable.mass_per_length
        self.submerged_weight = (
            cable.mass_per_length - env.rho_water * cable_area
        ) * env.gravity  # w [N/m]
        self.added_mass_n = env.rho_water * cable_area * cable.added_mass_coeff

        # State arrays (n_nodes x 3): positions and velocities
        self.positions = np.zeros((self.n_nodes, 3))
        self.velocities = np.zeros((self.n_nodes, 3))
        self.tensions = np.zeros(self.n_nodes)

        # Initialize straight-line configuration
        self._initialize_straight_line(np.array([5.0, 0.0, 0.0]))

    def _initialize_straight_line(self, tow_velocity: np.ndarray,
                                   depth: float = 10.0):
        """Initialize cable in a straight-line catenary-like configuration."""
        speed = np.linalg.norm(tow_velocity)
        if speed < 1e-10:
            direction = np.array([1.0, 0.0, 0.0])
        else:
            direction = tow_velocity / speed

        # Compute catenary angle from drag/weight balance
        f_t = 0.5 * self.env.rho_water * self.cable.Cd_t * np.pi * \
              self.cable.diameter * speed ** 2
        if abs(self.submerged_weight) > 1e-10 and speed > 0.1:
            cat_angle = np.arctan2(abs(self.submerged_weight), f_t)
        else:
            cat_angle = 0.05  # Small angle for near-neutral buoyancy

        # Limit catenary angle
        cat_angle = min(cat_angle, np.radians(30))

        for i in range(self.n_nodes):
            s = i * self.ds
            self.positions[i, 0] = -s * np.cos(cat_angle) * direction[0]
            self.positions[i, 1] = -s * np.cos(cat_angle) * direction[1]
            self.positions[i, 2] = depth + s * np.sin(cat_angle)
            self.velocities[i] = tow_velocity.copy()

        # Estimate initial tension
        for i in range(self.n_nodes):
            remaining = self.cable.length - i * self.ds
            self.tensions[i] = f_t * remaining

    def compute_element_vectors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute element direction vectors and lengths.

        Returns:
            t_hat: unit tangent vectors (n_elements, 3)
            seg_lengths: segment lengths (n_elements,)
            strains: segment strains (n_elements,)
        """
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
            strains[i] = (seg_len - self.ds) / self.ds

        return t_hat, seg_lengths, strains

    def compute_forces(self, tow_point_vel: np.ndarray,
                        fluid_velocities: Optional[np.ndarray] = None
                        ) -> np.ndarray:
        """
        Compute total forces at each node.

        Args:
            tow_point_vel: velocity of tow point (3,)
            fluid_velocities: fluid velocity at each element (n_elements, 3)

        Returns:
            forces: total force at each node (n_nodes, 3)
        """
        rho = self.env.rho_water
        d = self.cable.diameter
        Cd_n = self.cable.Cd_n
        Cd_t = self.cable.Cd_t
        EA = self.cable.EA
        c_damp = self.cable.damping_coeff

        # Get element geometry
        t_hat, seg_lengths, strains = self.compute_element_vectors()

        if fluid_velocities is None:
            fluid_velocities = np.tile(
                self.env.current_velocity, (self.n_elements, 1)
            )

        forces = np.zeros((self.n_nodes, 3))

        # --- Tension forces (elastic spring between nodes) ---
        for i in range(self.n_elements):
            tension = EA * max(strains[i], 0.0)  # No compression
            self.tensions[i] = tension

            # Damping along segment direction
            dv = self.velocities[i + 1] - self.velocities[i]
            dv_axial = np.dot(dv, t_hat[i]) * t_hat[i]
            f_damp = c_damp * dv_axial

            force = tension * t_hat[i] + f_damp
            forces[i] += force
            forces[i + 1] -= force

        # --- Hydrodynamic drag forces ---
        for i in range(self.n_elements):
            mid_vel = 0.5 * (self.velocities[i] + self.velocities[i + 1])
            v_rel = fluid_velocities[i] - mid_vel

            # Decompose into tangential and normal
            v_t_scalar = np.dot(v_rel, t_hat[i])
            v_t = v_t_scalar * t_hat[i]
            v_n = v_rel - v_t

            v_t_mag = abs(v_t_scalar)
            v_n_mag = np.linalg.norm(v_n)

            # Tangential drag (friction)
            f_tang = 0.5 * rho * Cd_t * np.pi * d * v_t_mag * v_t * self.ds
            # Normal drag (pressure)
            f_norm = 0.5 * rho * Cd_n * d * v_n_mag * v_n * self.ds

            f_element = f_tang + f_norm
            forces[i] += 0.5 * f_element
            forces[i + 1] += 0.5 * f_element

        # --- Gravity and buoyancy ---
        cable_area = np.pi * (d / 2) ** 2
        w_node = self.submerged_weight * self.ds  # Submerged weight per segment
        for i in range(self.n_nodes):
            weight = w_node
            if i == 0 or i == self.n_nodes - 1:
                weight *= 0.5
            forces[i, 2] += weight  # Z positive downward

        return forces

    def compute_accelerations(self, tow_point_pos: np.ndarray,
                               tow_point_vel: np.ndarray,
                               fluid_velocities: Optional[np.ndarray] = None
                               ) -> np.ndarray:
        """
        Compute accelerations at each node.

        Dynamic equation (extending Sanders' quasi-static formulation):
          (m + m_a) * a = F_tension + F_drag + F_gravity

        Args:
            tow_point_pos: position of tow point (3,)
            tow_point_vel: velocity of tow point (3,)
            fluid_velocities: fluid velocity field (n_elements, 3), optional

        Returns:
            accelerations: acceleration vectors (n_nodes, 3)
        """
        # Set tow point boundary condition
        self.positions[0] = tow_point_pos.copy()
        self.velocities[0] = tow_point_vel.copy()

        f_total = self.compute_forces(tow_point_vel, fluid_velocities)

        # Mass per node (including added mass)
        cable_area = np.pi * (self.cable.diameter / 2) ** 2
        m_node = self.mass_per_length * self.ds
        m_added = self.env.rho_water * cable_area * self.cable.added_mass_coeff * self.ds
        m_total = m_node + m_added

        # Compute accelerations (skip tow point node 0)
        accelerations = np.zeros((self.n_nodes, 3))
        for i in range(1, self.n_nodes):
            accelerations[i] = f_total[i] / m_total

        return accelerations

    def time_step_rk4(self, dt: float, tow_point_pos: np.ndarray,
                       tow_point_vel: np.ndarray,
                       fluid_velocities: Optional[np.ndarray] = None):
        """
        Advance solution by one time step using 4th-order Runge-Kutta method.
        """
        pos_save = self.positions.copy()
        vel_save = self.velocities.copy()

        # k1
        a1 = self.compute_accelerations(tow_point_pos, tow_point_vel,
                                          fluid_velocities)
        k1_v = a1 * dt
        k1_x = self.velocities * dt

        # k2
        self.positions = pos_save + 0.5 * k1_x
        self.velocities = vel_save + 0.5 * k1_v
        self.positions[0] = tow_point_pos
        self.velocities[0] = tow_point_vel
        a2 = self.compute_accelerations(tow_point_pos, tow_point_vel,
                                          fluid_velocities)
        k2_v = a2 * dt
        k2_x = self.velocities * dt

        # k3
        self.positions = pos_save + 0.5 * k2_x
        self.velocities = vel_save + 0.5 * k2_v
        self.positions[0] = tow_point_pos
        self.velocities[0] = tow_point_vel
        a3 = self.compute_accelerations(tow_point_pos, tow_point_vel,
                                          fluid_velocities)
        k3_v = a3 * dt
        k3_x = self.velocities * dt

        # k4
        self.positions = pos_save + k3_x
        self.velocities = vel_save + k3_v
        self.positions[0] = tow_point_pos
        self.velocities[0] = tow_point_vel
        a4 = self.compute_accelerations(tow_point_pos, tow_point_vel,
                                          fluid_velocities)
        k4_v = a4 * dt
        k4_x = self.velocities * dt

        # Update
        self.positions = pos_save + (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6
        self.velocities = vel_save + (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6

        # Enforce tow point boundary
        self.positions[0] = tow_point_pos
        self.velocities[0] = tow_point_vel

    def solve_quasi_static(self, tow_velocity: np.ndarray,
                            depth: float = 50.0) -> dict:
        """
        Quasi-static steady-state solution (original Sanders 1982 approach).

        Solves the ODE system along the cable from the free end to the tow point:
          dT/ds = f_t - w·sin(φ)
          dφ/ds = (f_n,v - w·cos(φ)) / T
          dθ/ds = f_n,h / (T·cos(φ))
          dx/ds = cos(φ)·cos(θ)
          dy/ds = cos(φ)·sin(θ)
          dz/ds = -sin(φ)

        Args:
            tow_velocity: tow ship velocity vector [Vx, Vy, Vz]
            depth: tow depth [m]

        Returns:
            dict with keys: 'positions', 'tensions', 'phi', 'theta', 'arc_length'
        """
        V = np.linalg.norm(tow_velocity[:2])  # Horizontal speed
        rho = self.env.rho_water
        d = self.cable.diameter
        Cd_n = self.cable.Cd_n
        Cd_t = self.cable.Cd_t
        w = self.submerged_weight  # Submerged weight per unit length
        L = self.cable.length
        N = self.n_nodes

        # Arrays for solution (index 0 = free end, index N-1 = tow point)
        s = np.linspace(0, L, N)
        T = np.zeros(N)
        phi = np.zeros(N)
        theta = np.zeros(N)
        x = np.zeros(N)
        y = np.zeros(N)
        z = np.zeros(N)

        # Tangential drag per unit length at tow speed
        f_t_base = 0.5 * rho * Cd_t * np.pi * d * V ** 2

        # Boundary condition at free end
        T[0] = max(f_t_base * 0.5, 1.0)  # Small residual tension

        # Initial inclination: balance of weight and drag
        if V > 0.1 and abs(w) > 0.01:
            phi[0] = np.arctan2(abs(w), f_t_base) if f_t_base > 0 else np.pi / 4
            phi[0] = min(phi[0], np.radians(45))
            if w < 0:
                phi[0] = -phi[0]  # Positively buoyant → cable goes up
        else:
            phi[0] = np.radians(5)

        # Initial azimuth (direction of tow)
        theta[0] = np.arctan2(tow_velocity[1], tow_velocity[0]) if V > 0.1 else 0.0

        x[0] = 0.0
        y[0] = 0.0
        z[0] = depth

        ds_step = L / (N - 1)

        for i in range(N - 1):
            T_i, phi_i, theta_i = T[i], phi[i], theta[i]

            def qs_rhs(T_v, phi_v, theta_v):
                cp = np.cos(phi_v)
                sp = np.sin(phi_v)

                # Tangential drag along cable
                dT_ds = f_t_base + w * sp

                # Normal force balance (vertical plane)
                # In vertical plane: weight component perpendicular to cable
                f_n_v = 0.0  # In steady straight tow, normal velocity is small
                dphi_ds = (f_n_v - w * cp) / max(T_v, 10.0)

                # Clamp angle rate to prevent divergence
                dphi_ds = np.clip(dphi_ds, -0.05, 0.05)

                # Horizontal curvature
                dtheta_ds = 0.0  # Zero for straight tow

                return dT_ds, dphi_ds, dtheta_ds

            # RK4 step
            k1_T, k1_p, k1_t = qs_rhs(T_i, phi_i, theta_i)
            k2_T, k2_p, k2_t = qs_rhs(
                T_i + 0.5 * ds_step * k1_T,
                phi_i + 0.5 * ds_step * k1_p,
                theta_i + 0.5 * ds_step * k1_t
            )
            k3_T, k3_p, k3_t = qs_rhs(
                T_i + 0.5 * ds_step * k2_T,
                phi_i + 0.5 * ds_step * k2_p,
                theta_i + 0.5 * ds_step * k2_t
            )
            k4_T, k4_p, k4_t = qs_rhs(
                T_i + ds_step * k3_T,
                phi_i + ds_step * k3_p,
                theta_i + ds_step * k3_t
            )

            T[i + 1] = T_i + ds_step * (k1_T + 2 * k2_T + 2 * k3_T + k4_T) / 6
            phi[i + 1] = phi_i + ds_step * (k1_p + 2 * k2_p + 2 * k3_p + k4_p) / 6
            theta[i + 1] = theta_i + ds_step * (k1_t + 2 * k2_t + 2 * k3_t + k4_t) / 6

            # Clamp angles
            phi[i + 1] = np.clip(phi[i + 1], -np.pi / 2, np.pi / 2)

            # Position integration
            cp = np.cos(phi[i + 1])
            sp = np.sin(phi[i + 1])
            ct = np.cos(theta[i + 1])
            st = np.sin(theta[i + 1])
            x[i + 1] = x[i] + ds_step * cp * ct
            y[i + 1] = y[i] + ds_step * cp * st
            z[i + 1] = z[i] - ds_step * sp  # Integrating upward from depth

            # Enforce positive tension
            T[i + 1] = max(T[i + 1], 1.0)

        # Reverse so index 0 = tow point
        positions = np.column_stack([x[::-1], y[::-1], z[::-1]])

        # Shift so tow point is at origin (0,0,0)
        tow_pt = positions[0].copy()
        positions -= tow_pt
        # Negate Z so that Z+ = depth (downward from tow point)
        positions[:, 2] = -positions[:, 2]

        self.positions = positions

        return {
            'positions': positions,
            'tensions': T[::-1],
            'phi': phi[::-1],
            'theta': theta[::-1],
            'arc_length': s
        }


class TowShipTrajectory:
    """
    Generate tow ship trajectories for simulation.

    Supports:
      - Straight-line tow
      - Steady turning (circular)
      - S-turn maneuver
    """

    def __init__(self, speed: float = 5.0, heading_init: float = 0.0,
                 depth: float = 0.0):
        self.speed = speed
        self.heading = heading_init
        self.depth = depth  # Z=0 means tow point at origin
        self.position = np.array([0.0, 0.0, depth])

    def straight_line(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Straight-line trajectory at constant speed."""
        pos = np.array([self.speed * t, 0.0, self.depth])
        vel = np.array([self.speed, 0.0, 0.0])
        return pos, vel

    def steady_turn(self, t: float, turn_radius: float = 500.0
                     ) -> Tuple[np.ndarray, np.ndarray]:
        """Steady circular turning trajectory."""
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
        """
        S-turn maneuver: straight → turn right → turn left → straight.
        """
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

    Integrates the cable dynamics model with ship trajectory and produces
    time-history results.
    """

    def __init__(self, cable: CableProperties, env: EnvironmentProperties,
                 n_elements: int = 100, dt: float = 0.1):
        self.cable_props = cable
        self.env = env
        self.n_elements = n_elements
        self.dt = dt
        self.model = TASSCableModel(cable, env, n_elements)
        self.trajectory = TowShipTrajectory()
        self.time = 0.0

        # Results storage
        self.time_history: List[float] = []
        self.position_history: List[np.ndarray] = []
        self.tension_history: List[np.ndarray] = []
        self.tow_point_history: List[np.ndarray] = []

    def set_trajectory(self, trajectory: TowShipTrajectory):
        """Set the tow ship trajectory generator."""
        self.trajectory = trajectory

    def initialize_steady_state(self, tow_velocity: np.ndarray,
                                 depth: float = 0.0):
        """
        Initialize the dynamic model with a straight-line configuration
        behind the tow point at t=0.

        Tow point starts at origin (0,0,0). Cable extends in -X direction
        and +Z direction (deeper). Z+ = depth (downward).
        """
        tow_pos_0, tow_vel_0 = self.trajectory.straight_line(0.0)
        speed = np.linalg.norm(tow_velocity[:2])
        ds = self.model.ds

        # Compute catenary angle from drag/weight balance
        f_t = 0.5 * self.env.rho_water * self.cable_props.Cd_t * np.pi * \
              self.cable_props.diameter * speed ** 2
        w = self.model.submerged_weight
        if abs(w) > 0.01 and f_t > 0.01:
            cat_angle = np.arctan2(abs(w), f_t)
            cat_angle = min(cat_angle, np.radians(20))
        else:
            cat_angle = np.radians(2)

        # Place nodes behind the tow point (tow point at origin)
        direction = tow_velocity / max(np.linalg.norm(tow_velocity), 1e-10)
        for i in range(self.model.n_nodes):
            s = i * ds
            self.model.positions[i] = tow_pos_0.copy()
            self.model.positions[i, 0] -= s * np.cos(cat_angle) * direction[0]
            self.model.positions[i, 1] -= s * np.cos(cat_angle) * direction[1]
            self.model.positions[i, 2] += s * np.sin(cat_angle)  # +Z = deeper
            self.model.velocities[i] = tow_vel_0.copy()

        # Also run quasi-static for reference
        qs_model = TASSCableModel(self.cable_props, self.env, self.n_elements)
        result = qs_model.solve_quasi_static(tow_velocity, depth)
        return result

    def run(self, t_end: float, save_interval: int = 10,
            maneuver: str = 'straight', **kwargs) -> dict:
        """
        Run the full dynamic simulation.

        Args:
            t_end: simulation end time [s]
            save_interval: save results every N steps
            maneuver: trajectory type ('straight', 'turn', 's_turn')
            **kwargs: additional arguments for trajectory

        Returns:
            dict with simulation results
        """
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
                t, turn_radius, turn_duration
            )
        else:
            raise ValueError(f"Unknown maneuver type: {maneuver}")

        print(f"Running TASS simulation: {maneuver} maneuver")
        print(f"  Cable length: {self.cable_props.length} m")
        print(f"  Elements: {self.n_elements}")
        print(f"  Time step: {self.dt} s")
        print(f"  Duration: {t_end} s")
        print(f"  Steps: {n_steps}")

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
            'cable_props': self.cable_props,
            'env': self.env,
        }
