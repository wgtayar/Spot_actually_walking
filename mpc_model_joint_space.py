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
    """Container for joint-space linear MPC data around a standing equilibrium.
    
    Now includes base state in optimization for proper feedback control.
    State vector: x = [base_roll, base_pitch, base_z, base_vx, base_vy, base_vz, 
                       q_joints, v_joints]
    """

    # Discrete time linear model: x_{k+1} = A_d x_k + B_d u_k
    # where x = [base_state; q_joints; v_joints]
    A_d: np.ndarray        # shape (n_x, n_x)
    B_d: np.ndarray        # shape (n_x, n_u)

    # Continuous time model (for reference): xdot = A x + B u
    A_c: np.ndarray        # shape (n_x, n_x)
    B_c: np.ndarray        # shape (n_x, n_u)

    # Output model y = C_y x for tracking
    C_y: np.ndarray        # shape (n_y, n_x)

    # Equilibrium state and input
    x_star: np.ndarray     # shape (n_x,) - base + joint states
    u_star: np.ndarray     # shape (n_u,) - equilibrium torques

    # Full state equilibrium (for reference)
    x_full_star: np.ndarray  # shape (n_q + n_v,)
    
    # Joint indices mapping
    idx_q_act: list        # indices of actuated positions in full state
    idx_v_act: list        # indices of actuated velocities in full state
    
    # Base state indices IN MPC STATE VECTOR
    idx_base_roll: int     # index 0
    idx_base_pitch: int    # index 1
    idx_base_z: int        # index 2
    idx_base_vx: int       # index 3
    idx_base_vy: int       # index 4
    idx_base_vz: int       # index 5
    n_base: int            # number of base states (6)

    # Dimensions
    n_x: int               # full state dimension (n_base + 2 * n_act)
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


def build_spot_mpc_model_joint_space(dt_mpc: float = 0.001, use_zoh: bool = True, plant_timestep: float = 0.0) -> SpotMPCModelJointSpace:
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
        plant_timestep: Timestep for the design plant. MUST be 0.0 for linearization!
                        (Discrete plants cannot be linearized via CalcTimeDerivatives)

    Returns:
        SpotMPCModelJointSpace with linearized joint dynamicsF
    """
    print("\n" + "="*60)
    print("Building Joint-Space MPC Model")
    print("="*60)

    # 1. Build design diagram and plant (MUST be continuous for linearization!)
    if plant_timestep != 0.0:
        print(f"\n⚠ WARNING: plant_timestep={plant_timestep} but must be 0.0 for linearization!")
        print(f"  Forcing plant_timestep=0.0 (continuous plant required for CalcTimeDerivatives)")
        plant_timestep = 0.0
    
    diagram, plant = build_spot_design_diagram(time_step=plant_timestep)
    
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
    joint_names = []

    for i in range(plant.num_actuators()):
        actuator = plant.get_joint_actuator(JointActuatorIndex(i))
        joint = actuator.joint()
        joint_names.append(joint.name())
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
    print(f"\n    *** MPC MODEL JOINT ORDER ***")
    for i, (name, q_idx, v_idx) in enumerate(zip(joint_names, idx_q_act, idx_v_act)):
        print(f"    [{i}] {name:30s} q_idx={q_idx:2d}, v_idx={v_idx:2d}")

    # 4. Extract equilibrium with base state
    q_act_star = q_full_star[idx_q_act]
    v_act_star = v_full_star[idx_v_act]
    
    # Base equilibrium: roll=0, pitch=0, z=0.55, velocities=0
    # We'll use simplified representation: [roll, pitch, z, vx, vy, vz]
    n_base = 6
    base_star = np.array([0.0, 0.0, q_full_star[6], 0.0, 0.0, 0.0])  # z from full state
    
    # MPC state: [base_state; q_joints; v_joints]
    x_mpc_star = np.concatenate([base_star, q_act_star, v_act_star])
    n_x = n_base + 2 * n_act
    
    print(f"    MPC state includes base: n_x = {n_x} (base: {n_base}, joints: {2*n_act})")

    # 5. Linearize FULL floating-base dynamics and extract submatrix
    print("\n[3] Linearizing full floating-base dynamics...")
    derivs = plant.AllocateTimeDerivatives()
    u_zero = np.zeros(n_u)
    
    # First, linearize the FULL system: xdot_full = f_full(x_full, u)
    # This gives us A_full and B_full that properly capture base-joint coupling
    
    def f_full(x_full: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Compute full system dynamics."""
        q_full = x_full[:n_q]
        v_full = x_full[n_q:]
        
        plant.SetPositions(plant_context, q_full)
        plant.SetVelocities(plant_context, v_full)
        plant.get_actuation_input_port().FixValue(plant_context, u)
        plant.CalcTimeDerivatives(plant_context, derivs)
        
        return derivs.get_vector().CopyToVector()
    
    # Evaluate at equilibrium
    xdot_full_star = f_full(x_full_star, u_zero)
    drift_full = np.linalg.norm(xdot_full_star)
    print(f"    Full system drift at equilibrium: {drift_full:.3e}")
    
    # Finite differences for full system
    eps_x = 1e-6
    eps_u = 1e-4
    
    n_x_full = n_q + n_v
    A_full = np.zeros((n_x_full, n_x_full))
    B_full = np.zeros((n_x_full, n_u))
    
    print(f"    Computing A_full ({n_x_full}x{n_x_full})...")
    for i in range(n_x_full):
        x_pert = x_full_star.copy()
        x_pert[i] += eps_x
        xdot_pert = f_full(x_pert, u_zero)
        A_full[:, i] = (xdot_pert - xdot_full_star) / eps_x
    
    print(f"    Computing B_full ({n_x_full}x{n_u})...")
    for j in range(n_u):
        u_pert = u_zero.copy()
        u_pert[j] += eps_u
        xdot_pert = f_full(x_full_star, u_pert)
        B_full[:, j] = (xdot_pert - xdot_full_star) / eps_u
    
    print(f"    Max |A_full|: {np.abs(A_full).max():.3e}")
    print(f"    Max |B_full|: {np.abs(B_full).max():.3e}")
    
    # Now extract the MPC-relevant submatrices
    # MPC state: [roll, pitch, z, vx, vy, vz, q_joints, v_joints]
    # We need to select rows/columns corresponding to these states
    
    # For now, approximate: assume base is weakly coupled to joints
    # A_c maps MPC state to MPC state derivative
    # B_c maps control to MPC state derivative
    
    print("\n[4] Extracting MPC-relevant dynamics...")
    
    # Strategy: Build transformation matrices to map between full state and MPC state
    # MPC state: [roll, pitch, z, vx, vy, vz, q_joints, v_joints] (30-dim)
    # Full state: [quat(4), x, y, z, q_joints(12); wx, wy, wz, vx, vy, vz, v_joints(12)] (37-dim)
    
    # Create selection/transformation matrix S: x_mpc ≈ S @ x_full
    # This is approximate for roll/pitch (nonlinear from quaternion)
    S = np.zeros((n_x, n_x_full))
    
    # Base orientation: roll ≈ 2*qx (small angle), pitch ≈ 2*qy
    S[0, 1] = 2.0  # roll from qx
    S[1, 2] = 2.0  # pitch from qy
    S[2, 6] = 1.0  # z position
    
    # Base linear velocities
    S[3, n_q + 3] = 1.0  # vx
    S[4, n_q + 4] = 1.0  # vy
    S[5, n_q + 5] = 1.0  # vz
    
    # Joint positions
    for i, qi in enumerate(idx_q_act):
        S[n_base + i, qi] = 1.0
    
    # Joint velocities
    for i, vi in enumerate(idx_v_act):
        S[n_base + n_act + i, n_q + vi] = 1.0
    
    # Linearized dynamics in MPC coordinates:
    # xdot_mpc ≈ S @ xdot_full = S @ (A_full @ x_full + B_full @ u)
    #          ≈ S @ A_full @ S^+ @ x_mpc + S @ B_full @ u
    # where S^+ is the pseudoinverse
    
    # For our specific S, the pseudoinverse is straightforward
    # (S has orthogonal rows for the most part)
    S_pinv = np.linalg.pinv(S)
    
    # Compute reduced dynamics
    A_c = S @ A_full @ S_pinv
    B_c = S @ B_full
    
    # Add small damping to base angular rates for stability
    # (Since we're using small-angle approximation, add some numerical damping)
    A_c[0, 0] = max(A_c[0, 0], -0.1)  # Light damping on roll
    A_c[1, 1] = max(A_c[1, 1], -0.1)  # Light damping on pitch    
    print(f"    Max |A_c|: {np.abs(A_c).max():.3e}")
    print(f"    Max |B_c|: {np.abs(B_c).max():.3e}")

    # 5. Solve for equilibrium torque
    print("\n[5] Computing equilibrium torque...")
    
    # The full system drift is non-zero due to underactuation
    # We find u* to minimize joint-space drift only
    # Extract joint derivatives from full drift
    xdot_joints_zero = np.concatenate([
        xdot_full_star[idx_q_act],  # qdot for actuated joints
        xdot_full_star[n_q:][idx_v_act]  # vdot for actuated joints (offset by n_q)
    ])
    drift_joints = np.linalg.norm(xdot_joints_zero)
    
    print(f"    Joint-space drift: {drift_joints:.3e}")
    
    if drift_joints < 1e-3:
        u_star = u_zero
        residual_norm = drift_joints
        print(f"    ✓ Joints already at equilibrium with u=0")
    else:
        # Solve B_joints @ u_star ≈ -xdot_joints_zero
        # Extract joint rows from B_c
        B_joints = B_c[n_base:, :]  # Only joint dynamics affected by control
        
        u_star, residuals, rank, svals = np.linalg.lstsq(
            B_joints, -xdot_joints_zero, rcond=None
        )
        residual_norm = np.linalg.norm(B_joints @ u_star + xdot_joints_zero)
        
        print(f"    ||u*||: {np.linalg.norm(u_star):.3f} Nm")
        print(f"    Residual: {residual_norm:.3e}")
    
    drift_final = residual_norm

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

    # 10. Output matrix - track base state AND joint positions
    print("\n[8] Defining output matrix...")
    
    # Track: base_roll, base_pitch, base_z, joint positions
    # (Don't track linear velocities or joint velocities in output cost)
    n_y = 3 + n_act  # roll, pitch, z + joint positions
    C_y = np.zeros((n_y, n_x))
    
    # Base tracking (first 3 states)
    C_y[0, 0] = 1.0  # roll
    C_y[1, 1] = 1.0  # pitch
    C_y[2, 2] = 1.0  # z
    
    # Joint position tracking
    C_y[3:, n_base:n_base+n_act] = np.eye(n_act)
    
    y_star = C_y @ x_mpc_star
    print(f"    Output y = [base_roll, base_pitch, base_z, joint_positions] (dim={n_y})")
    print(f"    Now tracking base orientation AND joint positions!")

    # 11. Cost weights
    # CRITICAL: These weights determine controller behavior!
    # - Q_y penalizes state deviations (higher = stay closer to reference)
    # - R_u penalizes control effort (higher = smoother, less aggressive)
    # - Qf_y terminal cost (higher = more conservative at horizon end)
    print("\n[9] Setting up cost weights...")
    
    Q_y = np.eye(n_y)
    
    # Restored to working configuration values (achieved z≈0.18m, roll < 2°)
    Q_y[0, 0] = 50000.0   # roll
    Q_y[1, 1] = 50000.0   # pitch
    Q_y[2, 2] = 10000.0   # height
    
    # Moderate weights on joint tracking
    Q_y[3:, 3:] *= 1000.0
    
    R_u = np.eye(n_u) * 0.5       # Restored: lower control penalty
    Qf_y = Q_y * 5.0              # Terminal cost
    
    print(f"    Q_y[base]: roll={Q_y[0,0]:.0f}, pitch={Q_y[1,1]:.0f}, z={Q_y[2,2]:.0f}")
    print(f"    Q_y[joints]: {Q_y[3,3]:.0f}")
    print(f"    R_u: {R_u[0,0]:.3e} * I_{n_u}")

    # 12. Package model
    model = SpotMPCModelJointSpace(
        A_d=A_d,
        B_d=B_d,
        A_c=A_c,
        B_c=B_c,
        C_y=C_y,
        x_star=x_mpc_star,
        u_star=u_star,
        x_full_star=x_full_star,
        idx_q_act=idx_q_act,
        idx_v_act=idx_v_act,
        idx_base_roll=0,      # Index in MPC state vector
        idx_base_pitch=1,     # Index in MPC state vector
        idx_base_z=2,         # Index in MPC state vector
        idx_base_vx=3,        # Index in MPC state vector
        idx_base_vy=4,        # Index in MPC state vector
        idx_base_vz=5,        # Index in MPC state vector
        n_base=n_base,
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
    # dt=0.001s (100 Hz) is typical for stiff contact systems
    model = build_spot_mpc_model_joint_space(dt_mpc=0.001, use_zoh=True)
    
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
