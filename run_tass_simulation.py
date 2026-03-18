#!/usr/bin/env python3
"""
TASS 3D Motion Analysis - Multi-Section Simulation
====================================================

Based on Sanders, J.V. (1982) "A three-dimensional dynamic analysis of a towed system."

TASS configuration (with per-section FDM element lengths):
  Tow Point → HWC (1000m, ds=100m) → LWC (300m, ds=50m) → AM (100m, ds=20m) → TR (80m, ds=20m)

Total length: 1480 m
Total elements: 10 + 6 + 5 + 4 = 25

Usage:
  python run_tass_simulation.py
"""

import numpy as np
import os

from tass_cable_model import (
    CableSection, MultiSectionCable, EnvironmentProperties,
    TASSCableModel, TowShipTrajectory, TASSSimulation
)
from tass_visualization import (
    plot_cable_3d, plot_cable_xy, plot_cable_xz,
    plot_tension_distribution, plot_tension_time_history,
    plot_tail_trajectory, plot_quasi_static_results,
)


def make_tass_cable(env: EnvironmentProperties) -> MultiSectionCable:
    """
    Create standard 4-section TASS cable with per-section element lengths.

    Sections (from tow point to tail):
      1. HWC: 1000m, ds=100m → 10 elements, negatively buoyant
      2. LWC:  300m, ds= 50m →  6 elements, neutrally buoyant
      3. AM:   100m, ds= 20m →  5 elements, neutrally buoyant
      4. TR:    80m, ds= 20m →  4 elements, neutrally buoyant
    """
    hwc = CableSection(
        name="HWC",
        length=1000.0,
        diameter=0.035,
        mass_per_length=2.8,
        EA=3.0e4,
        Cd_n=1.2,
        Cd_t=0.025,
        added_mass_coeff=1.0,
        damping_coeff=120.0,
        buoyancy_type="negative",
        ds=100.0,               # 1000m / 100m = 10 elements
    )

    lwc = CableSection(
        name="LWC",
        length=300.0,
        diameter=0.045,
        mass_per_length=1.63,
        EA=2.0e4,
        Cd_n=1.2,
        Cd_t=0.020,
        added_mass_coeff=1.0,
        damping_coeff=80.0,
        buoyancy_type="neutral",
        ds=50.0,                # 300m / 50m = 6 elements
    )

    am = CableSection(
        name="AM",
        length=100.0,
        diameter=0.070,
        mass_per_length=3.95,
        EA=1.5e4,
        Cd_n=1.0,
        Cd_t=0.015,
        added_mass_coeff=1.0,
        damping_coeff=60.0,
        buoyancy_type="neutral",
        ds=20.0,                # 100m / 20m = 5 elements
    )

    tr = CableSection(
        name="TR",
        length=80.0,
        diameter=0.030,
        mass_per_length=0.72,
        EA=1.0e4,
        Cd_n=1.2,
        Cd_t=0.020,
        added_mass_coeff=1.0,
        damping_coeff=50.0,
        buoyancy_type="neutral",
        ds=20.0,                # 80m / 20m = 4 elements
    )

    return MultiSectionCable([hwc, lwc, am, tr], env)


def print_tass_config(cable: MultiSectionCable):
    """Print TASS cable configuration summary."""
    print("\n  TASS Cable Configuration (FDM Discretization):")
    print("  " + "-" * 72)
    print(f"  {'Section':<8} {'Length':>8} {'ds':>6} {'Elems':>6} {'Dia':>8} "
          f"{'Buoyancy':<10} {'w':>10}")
    print(f"  {'':8} {'[m]':>8} {'[m]':>6} {'':>6} {'[mm]':>8} "
          f"{'':10} {'[N/m]':>10}")
    print("  " + "-" * 72)

    total_elem = 0
    for i, sec in enumerate(cable.sections):
        n = cable.section_n_elements[i]
        ds = cable.section_ds[i]
        w = cable.section_weights[i]
        total_elem += n
        print(f"  {sec.name:<8} {sec.length:>8.0f} {ds:>6.0f} {n:>6d} "
              f"{sec.diameter*1000:>8.1f} {sec.buoyancy_type:<10} {w:>10.3f}")

    print("  " + "-" * 72)
    print(f"  {'TOTAL':<8} {cable.total_length:>8.0f} {'':>6} {total_elem:>6d}")
    print(f"  Nodes: {cable.n_nodes_total}")
    print()


def scenario_quasi_static(cable, env):
    """Scenario 1: Quasi-static steady-state."""
    print("=" * 70)
    print("SCENARIO 1: Quasi-Static Steady-State Analysis")
    print("=" * 70)

    model = TASSCableModel(cable, env)

    tow_speed = 5.0
    tow_velocity = np.array([tow_speed, 0.0, 0.0])

    print(f"  Tow speed: {tow_speed} m/s ({tow_speed * 1.944:.1f} knots)")
    print(f"  Total length: {cable.total_length} m")
    print(f"  Elements: {cable.n_elements_total}, Nodes: {cable.n_nodes_total}")
    print(f"  Tow point: (0, 0, 0), Z+ = depth")

    qs_result = model.solve_quasi_static(tow_velocity, depth=0.0)

    pos = qs_result['positions']
    print(f"\n  Results:")
    print(f"    Tow point tension: {qs_result['tensions'][0]:.1f} N")
    print(f"    Tail tension: {qs_result['tensions'][-1]:.1f} N")
    print(f"    Max depth: {np.max(pos[:, 2]):.1f} m")
    print(f"    Tail position: X={pos[-1, 0]:.0f}m, Z={pos[-1, 2]:.1f}m")

    qs_result['cable'] = cable

    plot_quasi_static_results(qs_result,
                               title="Quasi-Static: HWC(1000)+LWC(300)+AM(100)+TR(80)",
                               save_path="results/qs_steady_state.png")

    return qs_result


def scenario_straight_tow(cable, env):
    """Scenario 2: Dynamic straight-line tow."""
    print("\n" + "=" * 70)
    print("SCENARIO 2: Dynamic Straight-Line Tow")
    print("=" * 70)

    sim = TASSSimulation(cable, env, dt=0.05)
    trajectory = TowShipTrajectory(speed=5.0, depth=0.0)
    sim.set_trajectory(trajectory)
    sim.initialize_steady_state(np.array([5.0, 0.0, 0.0]))

    results = sim.run(t_end=120.0, save_interval=4, maneuver='straight')

    plot_cable_3d(results, title="Straight Tow (V=5 m/s, 4-section TASS)",
                  save_path="results/straight_3d.png")
    plot_cable_xy(results, title="Straight Tow - Plan View",
                  save_path="results/straight_plan.png")
    plot_cable_xz(results, title="Straight Tow - Side View",
                  save_path="results/straight_side.png")
    plot_tension_distribution(results,
                               title="Tension Distribution - Straight Tow",
                               save_path="results/straight_tension.png")
    plot_tension_time_history(results,
                               title="Tension History - Straight Tow",
                               save_path="results/straight_tension_hist.png")

    return results


def scenario_turning(cable, env):
    """Scenario 3: Dynamic turning maneuver."""
    print("\n" + "=" * 70)
    print("SCENARIO 3: Dynamic Turning Maneuver")
    print("=" * 70)

    sim = TASSSimulation(cable, env, dt=0.05)
    trajectory = TowShipTrajectory(speed=5.0, depth=0.0)
    sim.set_trajectory(trajectory)
    sim.initialize_steady_state(np.array([5.0, 0.0, 0.0]))

    turn_radius = 800.0
    print(f"  Turn radius: {turn_radius} m")
    results = sim.run(t_end=200.0, save_interval=4,
                      maneuver='turn', turn_radius=turn_radius)

    plot_cable_3d(results, title=f"Turning (R={turn_radius}m, 4-section TASS)",
                  save_path="results/turn_3d.png")
    plot_cable_xy(results, title=f"Turning - Plan View (R={turn_radius}m)",
                  save_path="results/turn_plan.png")
    plot_tension_distribution(results,
                               title="Tension Distribution - Turning",
                               save_path="results/turn_tension.png")
    plot_tail_trajectory(results,
                          title="Array Tail During Turn",
                          save_path="results/turn_tail.png")

    return results


def scenario_s_turn(cable, env):
    """Scenario 4: S-turn maneuver."""
    print("\n" + "=" * 70)
    print("SCENARIO 4: S-Turn Maneuver")
    print("=" * 70)

    sim = TASSSimulation(cable, env, dt=0.05)
    trajectory = TowShipTrajectory(speed=5.0, depth=0.0)
    sim.set_trajectory(trajectory)
    sim.initialize_steady_state(np.array([5.0, 0.0, 0.0]))

    turn_radius = 600.0
    turn_duration = 60.0
    print(f"  Turn radius: {turn_radius} m, phase duration: {turn_duration} s")

    results = sim.run(t_end=300.0, save_interval=4,
                      maneuver='s_turn',
                      turn_radius=turn_radius,
                      turn_duration=turn_duration)

    plot_cable_3d(results, title="S-Turn (4-section TASS)",
                  save_path="results/sturn_3d.png")
    plot_cable_xy(results, title="S-Turn - Plan View",
                  save_path="results/sturn_plan.png")
    plot_tail_trajectory(results, title="Array Tail During S-Turn",
                          save_path="results/sturn_tail.png")
    plot_tension_time_history(results, title="Tension History - S-Turn",
                               save_path="results/sturn_tension_hist.png")

    return results


def main():
    os.makedirs("results", exist_ok=True)

    print("=" * 70)
    print("  3D TASS Motion Analysis Code (Multi-Section, Variable ds)")
    print("  Based on Sanders, J.V. (1982)")
    print("  'A three-dimensional dynamic analysis of a towed system'")
    print("  Ocean Engineering, Vol. 9, No. 5, pp. 483-499")
    print("=" * 70)

    env = EnvironmentProperties(
        rho_water=1025.0,
        current_velocity=np.array([0.0, 0.0, 0.0]),
    )

    cable = make_tass_cable(env)
    print_tass_config(cable)

    qs = scenario_quasi_static(cable, env)
    straight = scenario_straight_tow(cable, env)
    turn = scenario_turning(cable, env)
    sturn = scenario_s_turn(cable, env)

    print("\n" + "=" * 70)
    print("All simulations completed. Results saved to 'results/' directory.")
    print("=" * 70)


if __name__ == '__main__':
    main()
