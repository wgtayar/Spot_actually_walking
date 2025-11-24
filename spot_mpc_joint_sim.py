#!/usr/bin/env python3
"""
spot_mpc_joint_sim.py

Closed-loop MPC simulation for Spot standing controller.

This script:
1. Builds the joint-space MPC model (Phase 2)
2. Creates a Drake LeafSystem MPC controller (Phase 4)
3. Simulates Spot in Meshcat with full dynamics + contacts
4. Compares MPC performance against LQR baseline

The simulation uses:
- MultibodyPlant with contacts for Spot + ground
- JointMPCController wrapper that solves QP at each step
- Meshcat visualization
- Joint-space state [q_act; v_act] (24-dimensional)

Based on the architecture from spot_lqr_standing.py but with MPC control.
"""

from __future__ import annotations

import numpy as np

from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    MeshcatVisualizer,
    MultibodyPlant,
    Parser,
    Simulator,
    StartMeshcat,
    LogVectorOutput,
    MatrixGain,
    JointActuatorIndex,
    BasicVector,
    Context,
)

from underactuated import ConfigureParser

from mpc_model_joint_space import build_spot_mpc_model_joint_space
from joint_mpc_controller import JointMPCController
from spot_lqr_standing import get_default_standing_state


def build_spot_mpc_simulation(
    time_step: float = 0.01,
    N_horizon: int = 20,
    dt_mpc: float = 0.01,
    verbose: bool = False,
):
    """
    Build Drake diagram for closed-loop MPC simulation of Spot.

    Args:
        time_step: Simulation timestep (should match dt_mpc for consistency).
        N_horizon: MPC prediction horizon length.
        dt_mpc: Discretization timestep for MPC model.
        verbose: If True, print MPC solver output.

    Returns:
        diagram: Drake Diagram with plant, controller, and visualizer.
        plant: MultibodyPlant instance.
        controller: JointMPCController instance.
    """
    builder = DiagramBuilder()

    # Plant + scene graph
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=time_step)
    parser = Parser(plant)
    ConfigureParser(parser)

    # Load Spot + ground
    parser.AddModelsFromUrl("package://underactuated/models/spot/spot.dmd.yaml")
    parser.AddModelsFromUrl("package://underactuated/models/littledog/ground.urdf")

    plant.Finalize()

    # Meshcat visualization
    meshcat = StartMeshcat()
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat=meshcat)

    # Build MPC model (joint-space)
    print("\nBuilding MPC model...")
    mpc_model = build_spot_mpc_model_joint_space(
        dt_mpc=dt_mpc, 
        use_zoh=True, 
        plant_timestep=time_step  # Match simulation plant timestep!
    )
    print(f"  State dimension: {mpc_model.n_x}")
    print(f"  Control dimension: {mpc_model.n_u}")
    print(f"  Output dimension: {mpc_model.n_y}")
    print(f"  Horizon: N = {N_horizon} (predicts {N_horizon * dt_mpc:.2f}s ahead)")
    print(f"  Discretization: dt = {dt_mpc}s")

    # CRITICAL FIX: Reuse model's joint indices instead of recomputing!
    # This ensures the state selector extracts joints in the SAME ORDER
    # that the MPC model expects.
    idx_q_act = mpc_model.idx_q_act
    idx_v_act = mpc_model.idx_v_act
    n_act = mpc_model.n_act
    n_x_joint = mpc_model.n_x
    
    # DEBUG: Print simulation plant joint ordering
    print(f"\n    *** SIMULATION PLANT JOINT ORDER ***")
    for i in range(plant.num_actuators()):
        actuator = plant.get_joint_actuator(JointActuatorIndex(i))
        joint = actuator.joint()
        q_start = joint.position_start()
        v_start = joint.velocity_start()
        print(f"    [{i}] {joint.name():30s} q_idx={q_start:2d}, v_idx={v_start:2d}")
    
    # Get full state dimensions
    n_q = plant.num_positions()
    n_v = plant.num_velocities()
    n_x_full = n_q + n_v
    
    print(f"\nState dimensions:")
    print(f"  Full state dimension: {n_x_full} (includes base)")
    print(f"  Joint state dimension: {n_x_joint}")
    print(f"  MPC will receive full state and extract joints internally")
    
    # CRITICAL FIX: Use y_ref=None so controller uses model's own equilibrium
    # This avoids any confusion about joint ordering
    print(f"\nController setup:")
    print(f"  Using model's equilibrium: y* = {mpc_model.C_y @ mpc_model.x_star}")
    print(f"  u* norm: {np.linalg.norm(mpc_model.u_star):.3f} Nm")
    
    # Torque bounds strategy:
    # The equilibrium requires large torque ||u*|| ≈ 844 Nm (total across all joints)
    # This is because Spot's standing pose is a crouched position requiring active muscle tension
    # 
    # We set bounds on DEVIATION δu (not total torque) to allow:
    # 1. The controller to apply the necessary equilibrium torque u*
    # 2. Reasonable deviations ±δu for stabilization
    #
    # Since we're bounding DEVIATIONS, we can use modest bounds like ±100 Nm
    # This allows total torques in range: u* - 100 to u* + 100
    
    # Bounds on DEVIATION from equilibrium
    # CRITICAL: Use very loose bounds to prevent infeasibility
    # The cost function will naturally keep controls reasonable
    du_max = 200.0 * np.ones(mpc_model.n_u)  # Allow ±200 Nm deviation per joint
    du_min = -du_max
    
    # Compute implied total torque bounds
    u_total_min = mpc_model.u_star + du_min
    u_total_max = mpc_model.u_star + du_max
    
    print(f"  Equilibrium torque: ||u*|| = {np.linalg.norm(mpc_model.u_star):.1f} Nm")
    print(f"  Deviation bounds: ±{du_max[0]:.0f} Nm per joint (loose to prevent infeasibility)")
    print(f"  Total torque range: [{u_total_min.min():.0f}, {u_total_max.max():.0f}] Nm")
    
    # Create MPC controller with model's equilibrium (y_ref=None)
    # CRITICAL FIX: Pass full state dimension so controller can receive x_full
    controller = JointMPCController(
        model=mpc_model,
        N_horizon=N_horizon,
        n_x_full=n_x_full,  # NEW: Pass full state dimension
        y_ref=None,  # Use model's equilibrium - cleaner and safer!
        u_min=du_min,  # Pass DEVIATION bounds, not total bounds!
        u_max=du_max,
        verbose=verbose,
    )
    controller_sys = builder.AddSystem(controller)
    controller_sys.set_name("mpc_controller")

    # Connect: plant -> controller -> plant
    # Controller now receives FULL state directly (includes base!)
    builder.Connect(
        plant.get_state_output_port(),
        controller_sys.get_input_port(),
    )
    builder.Connect(
        controller_sys.get_output_port(),
        plant.get_actuation_input_port(),
    )

    # Log state for analysis
    state_logger = LogVectorOutput(plant.get_state_output_port(), builder)
    state_logger.set_name("state_logger")

    # Log control for analysis
    control_logger = LogVectorOutput(controller_sys.get_output_port(), builder)
    control_logger.set_name("control_logger")

    # Build diagram
    diagram = builder.Build()
    diagram.set_name("spot_mpc_simulation")

    return diagram, plant, controller, state_logger, control_logger, mpc_model


def run_mpc_simulation(
    duration: float = 5.0,
    time_step: float = 0.01,
    N_horizon: int = 20,
    dt_mpc: float = 0.01,
    initial_perturbation: float = 0.1,
    verbose: bool = False,
):
    """
    Run closed-loop MPC simulation with Spot.

    Args:
        duration: Simulation duration in seconds.
        time_step: Simulation timestep.
        N_horizon: MPC prediction horizon.
        dt_mpc: MPC model discretization timestep.
        initial_perturbation: Magnitude of initial joint position perturbation (radians).
        verbose: If True, print MPC solver output.
    """
    print("\n" + "=" * 70)
    print("Spot MPC Joint-Space Simulation")
    print("=" * 70)

    # Build simulation diagram
    diagram, plant, controller, state_logger, control_logger, mpc_model = build_spot_mpc_simulation(
        time_step=time_step,
        N_horizon=N_horizon,
        dt_mpc=dt_mpc,
        verbose=verbose,
    )

    # Create simulator
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    plant_context = plant.GetMyMutableContextFromRoot(context)

    # CRITICAL FIX: Initialize from model's full equilibrium for perfect consistency
    n_q_plant = plant.num_positions()
    n_v_plant = plant.num_velocities()
    x_full_star = mpc_model.x_full_star
    
    q_star = x_full_star[:n_q_plant]
    v_star = x_full_star[n_q_plant:]
    
    print(f"\nInitialized from MPC model equilibrium:")
    print(f"  q_star shape: {q_star.shape}, v_star shape: {v_star.shape}")
    
    if initial_perturbation > 0:
        # Perturb joint positions (not base)
        n_actuated = plant.num_actuators()
        perturbation = np.random.uniform(
            -initial_perturbation, 
            initial_perturbation, 
            size=n_actuated
        )
        
        # Apply perturbation to actuated joints only
        for i in range(n_actuated):
            actuator = plant.get_joint_actuator(JointActuatorIndex(i))
            joint = actuator.joint()
            q_start = joint.position_start()
            for k in range(joint.num_positions()):
                q_star[q_start + k] += perturbation[i]
        
        print(f"\nInitial perturbation applied: ±{initial_perturbation:.3f} rad")

    plant.SetPositions(plant_context, q_star)
    plant.SetVelocities(plant_context, v_star)

    print(f"\nSimulation parameters:")
    print(f"  Duration: {duration}s")
    print(f"  Timestep: {time_step}s")
    print(f"  MPC horizon: N = {N_horizon}")
    print(f"  MPC dt: {dt_mpc}s")

    # Run simulation
    print("\nStarting simulation...")
    print("(Check Meshcat for visualization)")
    
    try:
        simulator.AdvanceTo(duration)
        print(f"\n✓ Simulation completed successfully!")
    except Exception as e:
        print(f"\n✗ Simulation failed: {e}")
        raise

    # ------------------------------------------------------------------
    # Diagnostic 1: per-joint error between final plant joints and model
    # ------------------------------------------------------------------
    print("\nJoint-space tracking diagnostics (final time):")
    # Final full state from log
    x_final = state_logger.FindLog(context).data()[:, -1]
    q_final = x_final[:n_q_plant]
    v_final = x_final[n_q_plant:]

    # Nominal full state from model
    x_nom = mpc_model.x_full_star
    q_nom = x_nom[:n_q_plant]
    v_nom = x_nom[n_q_plant:]

    # Extract actuated joint positions using the SAME indices as the model
    q_act_final = q_final[mpc_model.idx_q_act]
    q_act_nom = q_nom[mpc_model.idx_q_act]

    joint_names = []
    for i in range(plant.num_actuators()):
        actuator = plant.get_joint_actuator(JointActuatorIndex(i))
        joint_names.append(actuator.joint().name())

    pos_errors = q_act_final - q_act_nom
    print("  Per-joint position error (final - nominal):")
    for name, e in zip(joint_names, pos_errors):
        print(f"    {name:25s}: {e:+.3f} rad ({np.rad2deg(e):+6.1f} deg)")

    print(f"  ||q_act_final - q_act_nom|| = {np.linalg.norm(pos_errors):.3f} rad")

    # ------------------------------------------------------------------
    # Diagnostic 2: check full-plant drift at (x*, u*)
    # ------------------------------------------------------------------
    print("\nEquilibrium consistency diagnostics (full plant at x*, u*):")

    # Set full plant to model's equilibrium
    plant.SetPositions(plant_context, q_nom)
    plant.SetVelocities(plant_context, v_nom)

    # Apply equilibrium torque u* on actuators
    u_star = mpc_model.u_star
    plant.get_actuation_input_port().FixValue(plant_context, u_star)

    # Compute time derivatives
    derivs = plant.AllocateTimeDerivatives()
    plant.CalcTimeDerivatives(plant_context, derivs)
    xdot = derivs.get_vector().CopyToVector()

    qdot = xdot[:n_q_plant]
    vdot = xdot[n_q_plant:]

    print(f"  ||qdot(x*, u*)|| = {np.linalg.norm(qdot):.3e}")
    print(f"  ||vdot(x*, u*)|| = {np.linalg.norm(vdot):.3e}")

    # Reset actuation to zero after diagnostic so we don't surprise users
    plant.get_actuation_input_port().FixValue(plant_context, np.zeros(plant.num_actuators()))

    # ------------------------------------------------------------------
    # Original performance analysis
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Performance Analysis")
    print("-" * 70)

    # Extract logged data
    state_log = state_logger.FindLog(context)
    control_log = control_logger.FindLog(context)

    times = state_log.sample_times()
    states = state_log.data()
    controls = control_log.data()

    # Use model's equilibrium for consistency
    n_q = plant.num_positions()
    n_v = plant.num_velocities()
    x_nom = mpc_model.x_full_star
    q_nom = x_nom[:n_q]
    v_nom = x_nom[n_q:]

    # Compute tracking error over time
    errors_q = np.zeros(len(times))
    errors_v = np.zeros(len(times))
    
    for i, t in enumerate(times):
        x_t = states[:, i]
        q_t = x_t[:n_q]
        v_t = x_t[n_q:]
        errors_q[i] = np.linalg.norm(q_t - q_nom)
        errors_v[i] = np.linalg.norm(v_t - v_nom)

    # Final errors
    final_error_q = errors_q[-1]
    final_error_v = errors_v[-1]

    # Control effort statistics
    control_norm = np.linalg.norm(controls, axis=0)
    avg_control = np.mean(control_norm)
    max_control = np.max(control_norm)

    print(f"\nTracking Performance:")
    print(f"  Initial position error: {errors_q[0]:.4f} rad")
    print(f"  Final position error:   {final_error_q:.4f} rad")
    print(f"  Initial velocity error: {errors_v[0]:.4f} rad/s")
    print(f"  Final velocity error:   {final_error_v:.4f} rad/s")

    print(f"\nControl Effort:")
    print(f"  Average ||u||: {avg_control:.1f} Nm")
    print(f"  Maximum ||u||: {max_control:.1f} Nm")
    print(f"  Equilibrium ||u*||: {np.linalg.norm(controller.model.u_star):.1f} Nm")

    # Convergence check
    position_threshold = 0.05  # 0.05 rad
    velocity_threshold = 0.1   # 0.1 rad/s
    
    if final_error_q < position_threshold and final_error_v < velocity_threshold:
        print(f"\n✓ Controller successfully stabilized Spot around standing pose")
    else:
        print(f"\n⚠ Controller did not fully stabilize (errors above threshold)")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run MPC simulation for Spot standing controller"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Simulation duration in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=50,
        help="MPC prediction horizon length (default: 20)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="MPC discretization timestep in seconds (default: 0.01)",
    )
    parser.add_argument(
        "--perturbation",
        type=float,
        default=0.0,
        help="Initial joint perturbation magnitude in radians (default: 0.0)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose MPC solver output",
    )

    args = parser.parse_args()

    run_mpc_simulation(
        duration=args.duration,
        time_step=args.dt,
        N_horizon=args.horizon,
        dt_mpc=args.dt,
        initial_perturbation=args.perturbation,
        verbose=args.verbose,
    )
