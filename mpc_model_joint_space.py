# mpc_model_joint_space.py
#
# Joint-space Linear MPC model for Spot around a standing equilibrium.
#
# Key idea: Instead of linearizing the full floating-base dynamics
# (which has quaternion drift issues), we:
#   1) Fix the floating base at the standing pose
#   2) Linearize only the actuated joint dynamics
#   3) Use this reduced model for MPC
#
# This approach is more tractable and avoids the quaternion drift problem.

from __future__ import annotations

import dataclasses
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy.linalg import expm

from pydrake.all import (
    JointActuatorIndex,
)

# Reuse utilities from the LQR script
from spot_lqr_standing import (
    build_spot_design_diagram,
    get_default_standing_state,
)


@dataclasses.dataclass
class SpotMPCModelJointSpace:
    """Container for joint-space linear MPC data around a standing equilibrium."""

    # Discrete time linear model: x_{k+1} = A_d x_k + B_d u_k
    # where x = [q_joints; v_joints] (actuated joints only)
    A_d: np.ndarray        # shape (n_x, n_x)
    B_d: np.ndarray        # shape (n_x, n_u)

    # Continuous time model (for reference): xdot = A x + B u
    A_c: np.ndarray        # shape (n_x, n_x)
    B_c: np.ndarray        # shape (n_x, n_u)

    # Output model y = C_y x for joint positions
    C_y: np.ndarray        # shape (n_y, n_x) with n_y = n_act

    # Equilibrium state and input (in joint space)
    x_star: np.ndarray     # shape (n_x,) - joint positions and velocities
    u_star: np.ndarray     # shape (n_u,) - equilibrium torques

    # Full state equilibrium (for reference)
    x_full_star: np.ndarray  # shape (n_q + n_v,)
    
    # Joint indices mapping
    idx_q_act: list        # indices of actuated positions in full state
    idx_v_act: list        # indices of actuated velocities in full state

    # Dimensions
    n_x: int               # joint-space state dimension (2 * n_act)
    n_u: int               # number of actuators
    n_y: int               # output dimension
    n_act: int             # number of actuated joints

    # MPC discretization and cost weights
    dt_mpc: float
    Q_y: np.ndarray        # weight on y tracking, shape (n_y, n_y)
    R_u: np.ndarray        # weight on u, shape (n_u, n_u)
    Qf_y: np.ndarray       # terminal weight on y, shape (n_y, n_y)

    # Diagnostics
    equilibrium_drift: float  # ||xdot|| at equilibrium
    max_eigenvalue: float     # max |eigenvalue| of A_d

    def stage_cost(self, y: np.ndarray, y_target: np.ndarray, u: np.ndarray) -> float:
        """Helper for debugging: compute instantaneous stage cost."""
        dy = y - y_target
        du = u - self.u_star
        return float(dy.T @ self.Q_y @ dy + du.T @ self.R_u @ du)

    def is_stable(self) -> bool:
        """Check if discrete system is stable."""
        return self.max_eigenvalue < 1.0
    
    def is_usable_for_mpc(self, tolerance=0.02) -> bool:
        """
        Check if system is usable for MPC.
        Allows small eigenvalues outside unit circle since MPC replans frequently.
        """
        return self.max_eigenvalue < (1.0 + tolerance)


def build_spot_mpc_model_joint_space(dt_mpc: float = 0.01, use_zoh: bool = True) -> SpotMPCModelJointSpace:
    """
    Build a joint-space linear MPC model around the nominal standing pose.

    This approach:
    - Fixes the floating base at the standing pose
    - Linearizes only the actuated joint dynamics
    - Avoids quaternion drift issues
    - Results in a smaller, more tractable MPC problem

    Args:
        dt_mpc: MPC control timestep (seconds)
        use_zoh: If True, use zero-order hold discretization.

    Returns:
        SpotMPCModelJointSpace with linearized joint dynamicsF
    """
    print("\n" + "="*60)
    print("Building Joint-Space MPC Model")
    print("="*60)

    # 1. Build design diagram and plant
    diagram, plant = build_spot_design_diagram(time_step=0.0)
    
    # 2. Find equilibrium state
    print("\n[1] Finding equilibrium state...")
    diagram_context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(diagram_context)

    q_full_star, v_full_star = get_default_standing_state(plant)
    plant.SetPositions(plant_context, q_full_star)
    plant.SetVelocities(plant_context, v_full_star)

    n_q = plant.num_positions()
    n_v = plant.num_velocities()
    n_u = plant.num_actuators()

    x_full_star = np.concatenate([q_full_star, v_full_star])
    
    print(f"    Full state dimension:  n_x_full = {n_q + n_v}")
    print(f"    Input dimension:  n_u = {n_u}")
    print(f"    Base position: {q_full_star[4:7]}")

    # 3. Get actuated joint indices
    print("\n[2] Identifying actuated joints...")
    idx_q_act = []
    idx_v_act = []

    for i in range(plant.num_actuators()):
        actuator = plant.get_joint_actuator(JointActuatorIndex(i))
        joint = actuator.joint()
        nq_j = joint.num_positions()
        nv_j = joint.num_velocities()
        q0 = joint.position_start()
        v0 = joint.velocity_start()

        for k in range(nq_j):
            idx_q_act.append(q0 + k)
        for k in range(nv_j):
            idx_v_act.append(v0 + k)

    n_act = len(idx_q_act)
    assert n_act == len(idx_v_act), "Position/velocity index lists must match"
    
    print(f"    Number of actuated joints: {n_act}")
    print(f"    Joint-space state dimension: {2 * n_act}")

    # 4. Extract joint-space equilibrium
    q_act_star = q_full_star[idx_q_act]
    v_act_star = v_full_star[idx_v_act]
    x_joint_star = np.concatenate([q_act_star, v_act_star])
    n_x = 2 * n_act

    # 5. Linearize joint-space dynamics FIRST (to get B matrix)
    print("\n[3] Linearizing joint-space dynamics (to compute B matrix)...")
    derivs = plant.AllocateTimeDerivatives()
    u_zero = np.zeros(n_u)
    
    # Define joint-space dynamics function
    def f_joint(x_joint: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Compute joint-space dynamics: xdot_joint = f(x_joint, u)
        by fixing the floating base at the equilibrium.
        """
        # Start from full equilibrium state
        q_full = q_full_star.copy()
        v_full = v_full_star.copy()
        
        # Override joint states
        q_full[idx_q_act] = x_joint[:n_act]
        v_full[idx_v_act] = x_joint[n_act:]
        
        # Set state and input
        plant.SetPositions(plant_context, q_full)
        plant.SetVelocities(plant_context, v_full)
        plant.get_actuation_input_port().FixValue(plant_context, u)
        
        # Compute derivatives
        plant.CalcTimeDerivatives(plant_context, derivs)
        xdot_full = derivs.get_vector().CopyToVector()
        
        # Extract joint-space derivatives
        qdot_full = xdot_full[:n_q]
        vdot_full = xdot_full[n_q:]
        qdot_act = qdot_full[idx_q_act]
        vdot_act = vdot_full[idx_v_act]
        
        return np.concatenate([qdot_act, vdot_act])
    
    # Compute initial drift at (x_star, u=0)
    xdot_joint_zero = f_joint(x_joint_star, u_zero)
    drift_zero = np.linalg.norm(xdot_joint_zero)
    
    print(f"    Initial joint-space drift with u=0: {drift_zero:.3e}")
    
    # Finite differences for linearization around (x_star, u=0)
    eps_x = 1e-6
    eps_u = 1e-4
    
    A_c_temp = np.zeros((n_x, n_x))
    B_c = np.zeros((n_x, n_u))
    
    print(f"    Computing B matrix ({n_x}x{n_u}) at u=0...")
    for j in range(n_u):
        u_pert = u_zero.copy()
        u_pert[j] += eps_u
        xdot_pert = f_joint(x_joint_star, u_pert)
        B_c[:, j] = (xdot_pert - xdot_joint_zero) / eps_u
    
    print(f"    Max |B_c|: {np.abs(B_c).max():.3e}")
    
    # 6. Solve for equilibrium torque u_star
    print("\n[4] Computing equilibrium torque...")
    
    if drift_zero < 1e-3:
        # Already close to equilibrium
        u_star = u_zero
        residual_norm = drift_zero
        print(f"    ✓ System is already at equilibrium with u=0")
        print(f"    ||xdot||: {drift_zero:.3e}")
    else:
        # Solve B_c @ u_star ≈ -xdot_joint_zero
        print(f"    Solving for u* such that B u* + xdot0 ≈ 0...")
        u_star, residuals, rank, svals = np.linalg.lstsq(
            B_c, -xdot_joint_zero, rcond=None
        )
        residual_norm = np.linalg.norm(B_c @ u_star + xdot_joint_zero)
        
        print(f"    ||xdot0|| (before):      {drift_zero:.3e}")
        print(f"    ||B u* + xdot0|| (after): {residual_norm:.3e}")
        print(f"    ||u*||:                  {np.linalg.norm(u_star):.3f} Nm")
        print(f"    Rank(B):                 {rank}/{n_u}")
        
        if residual_norm > 1e-2:
            print(f"    ⚠ WARNING: Residual is large - equilibrium may not be achievable")
    
    # 7. Re-linearize around (x_star, u_star) for accurate dynamics
    print("\n[5] Re-linearizing around equilibrium point (x*, u*)...")
    
    xdot_joint_star = f_joint(x_joint_star, u_star)
    drift_final = np.linalg.norm(xdot_joint_star)
    
    print(f"    Drift at (x*, u*): {drift_final:.3e}")
    
    # Compute A matrix around equilibrium
    A_c = np.zeros((n_x, n_x))
    
    print(f"    Computing A matrix ({n_x}x{n_x}) at equilibrium...")
    for i in range(n_x):
        x_pert = x_joint_star.copy()
        x_pert[i] += eps_x
        xdot_pert = f_joint(x_pert, u_star)
        A_c[:, i] = (xdot_pert - xdot_joint_star) / eps_x
    
    # Re-compute B matrix at equilibrium (more accurate)
    print(f"    Re-computing B matrix at equilibrium...")
    for j in range(n_u):
        u_pert = u_star.copy()
        u_pert[j] += eps_u
        xdot_pert = f_joint(x_joint_star, u_pert)
        B_c[:, j] = (xdot_pert - xdot_joint_star) / eps_u
    
    print(f"    Max |A_c|: {np.abs(A_c).max():.3e}")
    print(f"    Max |B_c|: {np.abs(B_c).max():.3e}")

    # 8. Discretize
    print(f"\n[6] Discretizing with dt = {dt_mpc} s...")
    
    if use_zoh:
        # Zero-order hold
        M = np.zeros((n_x + n_u, n_x + n_u))
        M[:n_x, :n_x] = A_c * dt_mpc
        M[:n_x, n_x:] = B_c * dt_mpc
        
        expM = expm(M)
        A_d = expM[:n_x, :n_x]
        B_d = expM[:n_x, n_x:]
        print(f"    Using Zero-Order Hold (ZOH)")
    else:
        # Forward Euler
        A_d = np.eye(n_x) + dt_mpc * A_c
        B_d = dt_mpc * B_c
        print(f"    Using Forward Euler")
    
    # 9. Check stability
    print("\n[7] Checking stability...")
    eigvals = np.linalg.eigvals(A_d)
    max_eig = np.max(np.abs(eigvals))
    print(f"    Max |eigenvalue| of A_d: {max_eig:.4f}")
    
    if max_eig < 1.0:
        print(f"    ✓ Discrete system is STABLE")
    elif max_eig < 1.02:
        print(f"    ⚠ Discrete system is marginally unstable (acceptable for MPC)")
    else:
        print(f"    ✗ Discrete system is UNSTABLE")

    # 10. Output matrix (just extract joint positions)
    print("\n[8] Defining output matrix...")
    n_y = n_act
    C_y = np.zeros((n_y, n_x))
    C_y[:n_act, :n_act] = np.eye(n_act)  # Output = joint positions
    
    y_star = C_y @ x_joint_star
    print(f"    Output y = joint positions (dim={n_y})")

    # 11. Cost weights
    print("\n[9] Setting up cost weights...")
    Q_y = np.eye(n_y) * 100.0  # Penalize joint position deviations
    R_u = np.eye(n_u) * 1e-2    # Penalize control effort
    Qf_y = Q_y * 10.0           # Terminal cost
    
    print(f"    Q_y: {Q_y[0,0]:.1f} * I_{n_y}")
    print(f"    R_u: {R_u[0,0]:.3e} * I_{n_u}")

    # 12. Package model
    model = SpotMPCModelJointSpace(
        A_d=A_d,
        B_d=B_d,
        A_c=A_c,
        B_c=B_c,
        C_y=C_y,
        x_star=x_joint_star,
        u_star=u_star,
        x_full_star=x_full_star,
        idx_q_act=idx_q_act,
        idx_v_act=idx_v_act,
        n_x=n_x,
        n_u=n_u,
        n_y=n_y,
        n_act=n_act,
        dt_mpc=dt_mpc,
        Q_y=Q_y,
        R_u=R_u,
        Qf_y=Qf_y,
        equilibrium_drift=drift_final,
        max_eigenvalue=max_eig,
    )

    print("\n" + "="*60)
    print("Joint-Space MPC Model Built Successfully!")
    print("="*60 + "\n")

    return model


if __name__ == "__main__":
    print("\nBuilding joint-space MPC model...")
    # Use small timestep for stability with unmodified stiff contact dynamics
    # dt=0.001s (1000 Hz) is typical for stiff contact systems
    model = build_spot_mpc_model_joint_space(dt_mpc=0.01, use_zoh=True)
    
    print("\n" + "="*60)
    print("Model Summary")
    print("="*60)
    print(f"Joint-space state dimension: {model.n_x}")
    print(f"Number of actuators: {model.n_u}")
    print(f"Equilibrium drift: {model.equilibrium_drift:.3e}")
    print(f"Max eigenvalue: {model.max_eigenvalue:.4f}")
    print(f"Strictly stable: {model.is_stable()}")
    print(f"Usable for MPC: {model.is_usable_for_mpc()}")
    print("="*60)
