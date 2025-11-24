"""
MPC-based Trajectory Planner for Spot (Joint Space Only)

This module uses MPC as an OFFLINE trajectory planner:
- Plans smooth joint trajectories from start to goal configurations
- Uses simplified joint-only dynamics (base fixed during planning)
- Returns full trajectory for LQR to track

Design philosophy:
- MPC plans in joint space (what it's good at)
- LQR tracks with full-state feedback (what it's good at)
- Clean separation of concerns
"""

from __future__ import annotations

import dataclasses
import numpy as np
from scipy.linalg import expm
import cvxpy as cp

from pydrake.all import JointActuatorIndex

# Reuse plant building utilities
from spot_lqr_standing import (
    build_spot_design_diagram,
    get_default_standing_state,
)


@dataclasses.dataclass
class JointOnlyMPCModel:
    """
    MPC model with ONLY joint states (no base).
    Matches LQR's linearization approach.
    
    State: x = [q_joints; v_joints] (24-dim for Spot)
    """
    
    # Discrete dynamics: x[k+1] = A_d @ x[k] + B_d @ u[k]
    A_d: np.ndarray  # (n_x, n_x)
    B_d: np.ndarray  # (n_x, n_u)
    
    # Continuous dynamics (for reference)
    A_c: np.ndarray  # (n_x, n_x)
    B_c: np.ndarray  # (n_x, n_u)
    
    # Equilibrium
    x_star: np.ndarray  # (n_x,) - joint positions and velocities
    u_star: np.ndarray  # (n_u,) - equilibrium torques
    
    # Output: y = q_joints (first half of state)
    C_y: np.ndarray  # (n_act, n_x)
    
    # Joint mapping (for reference)
    idx_q_act: list  # Position indices in full state
    idx_v_act: list  # Velocity indices in full state
    
    # Dimensions
    n_x: int   # State dimension (2 * n_act)
    n_u: int   # Control dimension
    n_act: int # Number of actuated joints
    
    # Discretization
    dt: float
    
    # Cost weights for planning
    Q_stage: np.ndarray   # Stage cost on positions (n_act, n_act)
    Q_terminal: np.ndarray # Terminal cost on positions (n_act, n_act)
    R: np.ndarray         # Control cost (n_u, n_u)
    
    # Diagnostics
    equilibrium_drift: float
    max_eigenvalue: float


def build_joint_only_mpc_model(dt: float = 0.01) -> JointOnlyMPCModel:
    """
    Build MPC model with ONLY joint states (no base).
    
    This matches the LQR linearization approach:
    - Fix base at nominal pose during linearization
    - Only model actuated joint dynamics
    - Clean, no quaternion drift issues
    
    Args:
        dt: Discretization timestep
        
    Returns:
        JointOnlyMPCModel for trajectory planning
    """
    
    print("\n" + "="*70)
    print("Building Joint-Only MPC Model for Trajectory Planning")
    print("="*70)
    
    # 1. Build continuous plant
    print("\n[1] Building plant...")
    diagram, plant = build_spot_design_diagram(time_step=0.0)
    
    diagram_context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(diagram_context)
    
    # 2. Get equilibrium state
    print("\n[2] Setting equilibrium state...")
    q_full_star, v_full_star = get_default_standing_state(plant)
    plant.SetPositions(plant_context, q_full_star)
    plant.SetVelocities(plant_context, v_full_star)
    
    n_q = plant.num_positions()
    n_v = plant.num_velocities()
    n_u = plant.num_actuators()
    
    print(f"    Full plant: n_q={n_q}, n_v={n_v}, n_u={n_u}")
    
    # 3. Get actuated joint indices
    print("\n[3] Identifying actuated joints...")
    idx_q_act = []
    idx_v_act = []
    
    for i in range(n_u):
        actuator = plant.get_joint_actuator(JointActuatorIndex(i))
        joint = actuator.joint()
        q0 = joint.position_start()
        v0 = joint.velocity_start()
        nq = joint.num_positions()
        nv = joint.num_velocities()
        
        for k in range(nq):
            idx_q_act.append(q0 + k)
        for k in range(nv):
            idx_v_act.append(v0 + k)
    
    n_act = len(idx_q_act)
    print(f"    Found {n_act} actuated joints")
    
    # 4. Build joint-only state
    q_act_star = q_full_star[idx_q_act]
    v_act_star = v_full_star[idx_v_act]
    x_star = np.concatenate([q_act_star, v_act_star])
    n_x = x_star.shape[0]
    
    print(f"    Joint-only state dimension: n_x={n_x}")
    
    # 5. Define joint-space dynamics function
    u_zero = np.zeros(n_u)
    derivs = plant.AllocateTimeDerivatives()
    
    def f_joint(x_joint: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Compute joint-space dynamics with base FIXED at nominal pose.
        This matches LQR's linearization approach.
        """
        # Start from nominal full state
        q_full = q_full_star.copy()
        v_full = v_full_star.copy()
        
        # Only modify joint DOFs
        q_full[idx_q_act] = x_joint[:n_act]
        v_full[idx_v_act] = x_joint[n_act:]
        
        # Set state and compute derivatives
        plant.SetPositions(plant_context, q_full)
        plant.SetVelocities(plant_context, v_full)
        plant.get_actuation_input_port().FixValue(plant_context, u)
        
        plant.CalcTimeDerivatives(plant_context, derivs)
        xdot_full = derivs.get_vector().CopyToVector()
        
        # Extract joint derivatives
        qdot_act = xdot_full[idx_q_act]
        vdot_act = xdot_full[n_q:][idx_v_act]
        
        return np.concatenate([qdot_act, vdot_act])
    
    # 6. Linearize around equilibrium
    print("\n[4] Linearizing joint dynamics...")
    xdot_star = f_joint(x_star, u_zero)
    drift = np.linalg.norm(xdot_star)
    print(f"    Drift at equilibrium: {drift:.3e}")
    
    # Finite differences
    eps_x = 1e-6
    eps_u = 1e-4
    
    A_c = np.zeros((n_x, n_x))
    B_c = np.zeros((n_x, n_u))
    
    print(f"    Computing A_c ({n_x}x{n_x})...")
    for i in range(n_x):
        x_pert = x_star.copy()
        x_pert[i] += eps_x
        xdot_pert = f_joint(x_pert, u_zero)
        A_c[:, i] = (xdot_pert - xdot_star) / eps_x
    
    print(f"    Computing B_c ({n_x}x{n_u})...")
    for j in range(n_u):
        u_pert = u_zero.copy()
        u_pert[j] += eps_u
        xdot_pert = f_joint(x_star, u_pert)
        B_c[:, j] = (xdot_pert - xdot_star) / eps_u
    
    print(f"    Max |A_c|: {np.abs(A_c).max():.3e}")
    print(f"    Max |B_c|: {np.abs(B_c).max():.3e}")
    
    # 7. Compute equilibrium torque
    print("\n[5] Computing equilibrium torque...")
    if drift < 1e-3:
        u_star = u_zero
        print(f"    ✓ Already at equilibrium with u=0")
    else:
        u_star, _, _, _ = np.linalg.lstsq(B_c, -xdot_star, rcond=None)
        residual = np.linalg.norm(B_c @ u_star + xdot_star)
        print(f"    ||u*||: {np.linalg.norm(u_star):.3f} Nm")
        print(f"    Residual: {residual:.3e}")
    
    # 8. Discretize with ZOH
    print(f"\n[6] Discretizing with dt={dt}s (ZOH)...")
    M = np.zeros((n_x + n_u, n_x + n_u))
    M[:n_x, :n_x] = A_c * dt
    M[:n_x, n_x:] = B_c * dt
    
    expM = expm(M)
    A_d = expM[:n_x, :n_x]
    B_d = expM[:n_x, n_x:]
    
    # 9. Check stability
    print("\n[7] Checking stability...")
    eigvals = np.linalg.eigvals(A_d)
    max_eig = np.max(np.abs(eigvals))
    print(f"    Max |eigenvalue| of A_d: {max_eig:.4f}")
    
    if max_eig < 1.0:
        print(f"    ✓ Discrete system is STABLE")
    elif max_eig < 1.02:
        print(f"    ⚠ Marginally unstable (OK for MPC planning)")
    else:
        print(f"    ✗ UNSTABLE - may need tuning")
    
    # 10. Output matrix - track joint positions only
    print("\n[8] Defining output matrix...")
    C_y = np.zeros((n_act, n_x))
    C_y[:n_act, :n_act] = np.eye(n_act)  # y = q_joints
    
    print(f"    Output y = q_joints (dim={n_act})")
    
    # 11. Cost weights for trajectory planning
    print("\n[9] Setting cost weights...")
    
    # Stage cost - LOW to allow exploration
    Q_stage = np.eye(n_act) * 10.0
    
    # Terminal cost - moderate goal reaching (not too aggressive)
    Q_terminal = np.eye(n_act) * 1000.0
    
    # Control cost - smooth trajectories
    R = np.eye(n_u) * 1.0
    
    print(f"    Q_stage: {Q_stage[0,0]:.0f} * I (low - allows deviation)")
    print(f"    Q_terminal: {Q_terminal[0,0]:.0f} * I (moderate goal reaching)")
    print(f"    R: {R[0,0]:.2f} * I")
    
    # 12. Package model
    print("\n" + "="*70)
    print("✓ Joint-Only MPC Model Built Successfully")
    print("="*70)
    print(f"  State dimension: {n_x}")
    print(f"  Control dimension: {n_u}")
    print(f"  Output dimension: {n_act}")
    print(f"  Discretization: dt={dt}s")
    print()
    
    return JointOnlyMPCModel(
        A_d=A_d,
        B_d=B_d,
        A_c=A_c,
        B_c=B_c,
        x_star=x_star,
        u_star=u_star,
        C_y=C_y,
        idx_q_act=idx_q_act,
        idx_v_act=idx_v_act,
        n_x=n_x,
        n_u=n_u,
        n_act=n_act,
        dt=dt,
        Q_stage=Q_stage,
        Q_terminal=Q_terminal,
        R=R,
        equilibrium_drift=drift,
        max_eigenvalue=max_eig,
    )


@dataclasses.dataclass
class TrajectoryPlan:
    """Result from trajectory planning."""
    times: np.ndarray      # (N+1,) timesteps
    y_traj: np.ndarray     # (N+1, n_act) joint positions
    x_traj: np.ndarray     # (N+1, n_x) full states
    u_traj: np.ndarray     # (N, n_u) control inputs
    success: bool
    cost: float


def plan_joint_trajectory(
    model: JointOnlyMPCModel,
    y_start: np.ndarray,
    y_goal: np.ndarray,
    N_horizon: int = 100,
    duration: float = 2.0,
    u_max: float = 100.0,
    verbose: bool = True
) -> TrajectoryPlan:
    """
    Plan a smooth joint trajectory from y_start to y_goal using MPC.
    
    This is OFFLINE planning - solves QP once to get full trajectory.
    
    Args:
        model: Joint-only MPC model
        y_start: Initial joint configuration (n_act,)
        y_goal: Target joint configuration (n_act,)
        N_horizon: Planning horizon length
        duration: Total trajectory duration (seconds)
        u_max: Torque limit (Nm)
        verbose: Print progress
        
    Returns:
        TrajectoryPlan with full trajectory
    """
    
    if verbose:
        print("\n" + "="*70)
        print("Planning Joint Trajectory with MPC")
        print("="*70)
        print(f"  Horizon: N={N_horizon}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Timestep: {model.dt:.4f}s")
        print(f"  Torque limit: ±{u_max:.0f} Nm")
    
    # Setup QP variables
    n_x = model.n_x
    n_u = model.n_u
    n_act = model.n_act
    
    x = cp.Variable((n_x, N_horizon + 1))
    u = cp.Variable((n_u, N_horizon))
    
    # Convert y_start to full state (append zero velocities)
    x_start = np.concatenate([y_start, np.zeros(n_act)])
    
    if verbose:
        print(f"\n  Start config: ||y_start - y*|| = {np.linalg.norm(y_start - model.C_y @ model.x_star):.3f} rad")
        print(f"  Goal config:  ||y_goal - y*||  = {np.linalg.norm(y_goal - model.C_y @ model.x_star):.3f} rad")
    
    # Build QP
    constraints = []
    cost = 0.0
    
    # Initial condition
    constraints += [x[:, 0] == x_start]
    
    # Stage costs and dynamics
    for k in range(N_horizon):
        # Dynamics constraint
        constraints += [x[:, k+1] == model.A_d @ x[:, k] + model.B_d @ u[:, k]]
        
        # Torque limits
        constraints += [u[:, k] >= -u_max]
        constraints += [u[:, k] <= u_max]
        
        # Stage cost - penalize velocity (smooth motion)
        # Don't track goal yet - let terminal cost handle that
        v_k = x[n_act:, k]  # Joint velocities
        cost += 0.1 * cp.sum_squares(v_k)  # Light penalty on velocity
        
        # Control cost - smooth control
        du_k = u[:, k] - model.u_star
        cost += cp.quad_form(du_k, model.R)
    
    # Terminal cost - strong goal reaching
    y_final = model.C_y @ x[:, N_horizon]
    dy_final = y_final - y_goal
    cost += cp.quad_form(dy_final, model.Q_terminal)
    
    # Solve QP
    if verbose:
        print(f"\n  Solving QP...")
    
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-5, eps_rel=1e-5)
    
    success = prob.status == cp.OPTIMAL
    
    if verbose:
        if success:
            print(f"  ✓ Solution found!")
            print(f"  Cost: {prob.value:.3e}")
            print(f"  Final error: ||y_final - y_goal|| = {np.linalg.norm(y_final.value - y_goal):.4f} rad")
            print(f"  Max |u|: {np.abs(u.value).max():.1f} Nm")
        else:
            print(f"  ✗ Solver failed: {prob.status}")
    
    # Extract trajectory
    times = np.linspace(0, duration, N_horizon + 1)
    y_traj = (model.C_y @ x.value).T  # (N+1, n_act)
    x_traj = x.value.T                # (N+1, n_x)
    u_traj = u.value.T                # (N, n_u)
    
    if verbose:
        print("="*70)
        print()
    
    return TrajectoryPlan(
        times=times,
        y_traj=y_traj,
        x_traj=x_traj,
        u_traj=u_traj,
        success=success,
        cost=prob.value if success else float('inf')
    )


def visualize_trajectory_plan(plan: TrajectoryPlan, model: JointOnlyMPCModel):
    """
    Visualize the planned trajectory.
    
    Creates plots showing:
    - Joint positions over time
    - Control inputs over time
    - Goal convergence
    """
    import matplotlib.pyplot as plt
    
    N = len(plan.times)
    n_act = model.n_act
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Joint positions
    ax = axes[0]
    for i in range(n_act):
        ax.plot(plan.times, plan.y_traj[:, i], label=f'Joint {i}')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Joint Position (rad)')
    ax.set_title('Planned Joint Trajectory')
    ax.legend(ncol=4, fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Control inputs
    ax = axes[1]
    for i in range(model.n_u):
        ax.plot(plan.times[:-1], plan.u_traj[:, i], label=f'Joint {i}')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Torque (Nm)')
    ax.set_title('Control Inputs')
    ax.legend(ncol=4, fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Distance to goal
    ax = axes[2]
    y_goal = plan.y_traj[-1, :]
    errors = np.linalg.norm(plan.y_traj - y_goal, axis=1)
    ax.plot(plan.times, errors, linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Distance to Goal (rad)')
    ax.set_title('Convergence to Goal Configuration')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('/home/hassan/Underactuated-Biped/external/spot/trajectory_plan.png', dpi=150)
    print("Trajectory visualization saved to: trajectory_plan.png")
    plt.close()


if __name__ == "__main__":
    # Test trajectory planning
    print("Testing MPC Trajectory Planner\n")
    
    # Build model
    model = build_joint_only_mpc_model(dt=0.01)
    
    # Define simple trajectory: lean forward
    y_start = model.C_y @ model.x_star  # Current standing pose
    y_goal = y_start.copy()
    
    # Small forward lean by adjusting hip angles
    # Indices: [FL_hip_x, FL_hip_y, FL_knee, FR_hip_x, FR_hip_y, FR_knee, ...]
    # Adjust all hip_y angles slightly to shift weight forward
    for i in [1, 4, 7, 10]:  # hip_y indices
        y_goal[i] += 0.15  # ~8.6 degrees more abduction
    
    print(f"\nPlanning trajectory:")
    print(f"  Start: hip_y = {y_start[1]:.3f} rad")
    print(f"  Goal:  hip_y = {y_goal[1]:.3f} rad")
    
    # Plan trajectory
    plan = plan_joint_trajectory(
        model=model,
        y_start=y_start,
        y_goal=y_goal,
        N_horizon=200,
        duration=2.0,
        u_max=100.0,
        verbose=True
    )
    
    if plan.success:
        visualize_trajectory_plan(plan, model)
