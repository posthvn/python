#!/usr/bin/env python3
"""
TASS 3D Motion Analysis - Simulation Runner
=============================================

Based on Sanders, J.V. (1982) "A three-dimensional dynamic analysis of a towed system."
Ocean Engineering, Vol. 9, No. 5, pp. 483-499.

This script runs four simulation scenarios:
  1. Quasi-static steady-state analysis
  2. Dynamic straight-line tow (with cross-current)
  3. Dynamic steady turning maneuver
  4. S-turn maneuver

Usage:
  python run_tass_simulation.py
"""

import numpy as np
import os

from tass_cable_model import (
    CableProperties, EnvironmentProperties,
    TASSCableModel, TowShipTrajectory, TASSSimulation
)
from tass_visualization import (
    plot_cable_3d, plot_cable_xy, plot_cable_xz,
    plot_tension_distribution, plot_tension_time_history,
    plot_tail_trajectory, plot_quasi_static_results,
)


def make_cable(length=300.0):
    """Create standard cable properties."""
    return CableProperties(
        length=length,
        diameter=0.05,
        mass_per_length=2.5,
        EA=2.0e4,           # Stiffness chosen for explicit RK4 stability
        Cd_n=1.2,
        Cd_t=0.025,
        damping_coeff=100.0,
    )


def scenario_quasi_static():
    """Scenario 1: Quasi-static steady-state solution."""
    print("=" * 70)
    print("SCENARIO 1: Quasi-Static Steady-State Analysis")
    print("=" * 70)

    cable = make_cable(length=500.0)
    env = EnvironmentProperties(
        rho_water=1025.0,
        current_velocity=np.array([0.0, 0.0, 0.0]),
    )

    model = TASSCableModel(cable, env, n_elements=200)

    tow_speed = 5.0  # m/s (~10 knots)
    tow_velocity = np.array([tow_speed, 0.0, 0.0])

    print(f"  Tow speed: {tow_speed} m/s ({tow_speed * 1.944:.1f} knots)")
    print(f"  Cable length: {cable.length} m")
    print(f"  Tow point: (0, 0, 0), Z+ = depth downward")

    qs_result = model.solve_quasi_static(tow_velocity, depth=0.0)

    print(f"\n  Results:")
    print(f"    Tow point tension: {qs_result['tensions'][0]:.1f} N")
    print(f"    Tail tension: {qs_result['tensions'][-1]:.1f} N")
    print(f"    Max inclination: {np.degrees(np.max(np.abs(qs_result['phi']))):.1f} deg")
    print(f"    Cable depth range: {np.min(qs_result['positions'][:, 2]):.1f} ~ "
          f"{np.max(qs_result['positions'][:, 2]):.1f} m")

    plot_quasi_static_results(qs_result,
                               title="Quasi-Static Steady-State (V=5 m/s, L=500 m)",
                               save_path="results/qs_steady_state.png")

    return qs_result


def scenario_straight_tow():
    """Scenario 2: Dynamic straight-line tow simulation."""
    print("\n" + "=" * 70)
    print("SCENARIO 2: Dynamic Straight-Line Tow")
    print("=" * 70)

    cable = make_cable(length=300.0)
    env = EnvironmentProperties(
        rho_water=1025.0,
        current_velocity=np.array([0.5, 0.2, 0.0]),  # Cross-current
    )

    sim = TASSSimulation(cable, env, n_elements=50, dt=0.02)
    trajectory = TowShipTrajectory(speed=5.0, depth=0.0)
    sim.set_trajectory(trajectory)
    sim.initialize_steady_state(np.array([5.0, 0.0, 0.0]))

    results = sim.run(t_end=60.0, save_interval=10, maneuver='straight')

    plot_cable_3d(results, title="Dynamic Straight Tow (V=5 m/s)",
                  save_path="results/straight_3d.png")
    plot_cable_xy(results, title="Dynamic Straight Tow - Plan View",
                  save_path="results/straight_plan.png")
    plot_cable_xz(results, title="Dynamic Straight Tow - Side View",
                  save_path="results/straight_side.png")
    plot_tension_distribution(results,
                               title="Tension Distribution - Straight Tow",
                               save_path="results/straight_tension.png")
    plot_tension_time_history(results,
                               title="Tension History - Straight Tow",
                               save_path="results/straight_tension_hist.png")

    return results


def scenario_turning():
    """Scenario 3: Dynamic turning maneuver simulation."""
    print("\n" + "=" * 70)
    print("SCENARIO 3: Dynamic Turning Maneuver")
    print("=" * 70)

    cable = make_cable(length=300.0)
    env = EnvironmentProperties(
        rho_water=1025.0,
        current_velocity=np.array([0.0, 0.0, 0.0]),
    )

    sim = TASSSimulation(cable, env, n_elements=50, dt=0.02)
    trajectory = TowShipTrajectory(speed=5.0, depth=0.0)
    sim.set_trajectory(trajectory)
    sim.initialize_steady_state(np.array([5.0, 0.0, 0.0]))

    turn_radius = 500.0
    print(f"  Turn radius: {turn_radius} m")
    results = sim.run(t_end=120.0, save_interval=10,
                      maneuver='turn', turn_radius=turn_radius)

    plot_cable_3d(results, title=f"Turning Maneuver (R={turn_radius}m)",
                  save_path="results/turn_3d.png")
    plot_cable_xy(results, title=f"Turning Maneuver - Plan View (R={turn_radius}m)",
                  save_path="results/turn_plan.png")
    plot_tension_distribution(results,
                               title="Tension Distribution - Turning",
                               save_path="results/turn_tension.png")
    plot_tail_trajectory(results,
                          title="Array Tail During Turn",
                          save_path="results/turn_tail.png")

    return results


def scenario_s_turn():
    """Scenario 4: S-turn maneuver simulation."""
    print("\n" + "=" * 70)
    print("SCENARIO 4: S-Turn Maneuver")
    print("=" * 70)

    cable = make_cable(length=300.0)
    env = EnvironmentProperties(
        rho_water=1025.0,
        current_velocity=np.array([0.0, 0.0, 0.0]),
    )

    sim = TASSSimulation(cable, env, n_elements=50, dt=0.02)
    trajectory = TowShipTrajectory(speed=5.0, depth=0.0)
    sim.set_trajectory(trajectory)
    sim.initialize_steady_state(np.array([5.0, 0.0, 0.0]))

    turn_radius = 400.0
    turn_duration = 50.0
    print(f"  Turn radius: {turn_radius} m")
    print(f"  Turn phase duration: {turn_duration} s")

    results = sim.run(t_end=200.0, save_interval=10,
                      maneuver='s_turn',
                      turn_radius=turn_radius,
                      turn_duration=turn_duration)

    plot_cable_3d(results, title="S-Turn Maneuver",
                  save_path="results/sturn_3d.png")
    plot_cable_xy(results, title="S-Turn Maneuver - Plan View",
                  save_path="results/sturn_plan.png")
    plot_tail_trajectory(results, title="Array Tail During S-Turn",
                          save_path="results/sturn_tail.png")
    plot_tension_time_history(results, title="Tension History - S-Turn",
                               save_path="results/sturn_tension_hist.png")

    return results


def main():
    """Run all simulation scenarios."""
    os.makedirs("results", exist_ok=True)

    print("=" * 70)
    print("  3D TASS Motion Analysis Code")
    print("  Based on Sanders, J.V. (1982)")
    print("  'A three-dimensional dynamic analysis of a towed system'")
    print("  Ocean Engineering, Vol. 9, No. 5, pp. 483-499")
    print("=" * 70)

    qs_results = scenario_quasi_static()
    straight_results = scenario_straight_tow()
    turn_results = scenario_turning()
    sturn_results = scenario_s_turn()

    print("\n" + "=" * 70)
    print("All simulations completed. Results saved to 'results/' directory.")
    print("=" * 70)

    return {
        'quasi_static': qs_results,
        'straight': straight_results,
        'turning': turn_results,
        's_turn': sturn_results,
    }


if __name__ == '__main__':
    main()
