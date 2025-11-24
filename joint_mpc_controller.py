#!/usr/bin/env python3
"""
joint_mpc_controller.py

Drake LeafSystem wrapper for joint-space MPC controller.

This module provides a Drake controller system that:
1. Reads the current joint state from Drake
2. Solves the MPC QP at each control step
3. Outputs the optimal torque command

The controller is designed to be connected in a Drake diagram:
    - Input port: joint state (positions + velocities)
    - Output port: joint torques

This follows the same pattern as the LQR controller in spot_lqr_standing.py.
"""

import numpy as np
from pydrake.all import LeafSystem, BasicVector

from mpc_model_joint_space import SpotMPCModelJointSpace
from joint_mpc_qp import solve_joint_mpc_qp


class JointMPCController(LeafSystem):
    """
    Drake system that implements joint-space MPC control.

    At each time step, this controller:
    1. Reads the current joint state x = [q_act; v_act]
    2. Solves a finite-horizon QP to compute optimal control u0
    3. Outputs u0 as the torque command

    The QP is resolved at each step (receding horizon).
    """

    def __init__(
        self,
        model: SpotMPCModelJointSpace,
        N_horizon: int,
        n_x_full: int,
        y_ref: np.ndarray | None = None,
        u_min: np.ndarray | None = None,
        u_max: np.ndarray | None = None,
        verbose: bool = False,
    ):
        """
        Args:
            model: Joint-space MPC model with dynamics and cost.
            N_horizon: Prediction horizon length.
            n_x_full: Dimension of full state (needed for input port).
            y_ref: Target joint positions. If None, uses nominal y*.
            u_min: Optional lower torque bounds, shape (n_u,).
            u_max: Optional upper torque bounds, shape (n_u,).
            verbose: If True, print QP solver output.
        """
        LeafSystem.__init__(self)

        self.model = model
        self.N_horizon = N_horizon
        self.n_x_full = n_x_full
        self.verbose = verbose

        # Target reference
        if y_ref is None:
            self.y_ref = model.C_y @ model.x_star  # nominal standing configuration
        else:
            assert y_ref.shape == (model.n_y,), f"y_ref shape mismatch"
            self.y_ref = y_ref

        # Optional input bounds
        self.u_min = u_min
        self.u_max = u_max

        # Input port: FULL state [q_full; v_full] including base!
        self.state_input_port = self.DeclareVectorInputPort(
            "full_state",
            BasicVector(n_x_full),
        )

        # Counter for verbose output
        self._call_count = 0

        # Output port: joint torques
        self.DeclareVectorOutputPort(
            "joint_torques",
            BasicVector(model.n_u),
            self.CalcOptimalControl,
        )

    def CalcOptimalControl(self, context, output):
        """
        Compute optimal control by solving MPC QP.

        This is the Drake output port callback function.
        
        The QP returns a control DEVIATION δu, and we add the equilibrium
        torque u* to get the total torque:
            u_total = u* + δu
        """
        # Read FULL state from plant (includes base!)
        x_full = self.state_input_port.Eval(context)
        
        # Drake state is [q_full; v_full] where q and v may have different dimensions
        # (quaternions vs angular velocity)
        q_full = x_full[:19]  # 19 positions (7 base + 12 joints)
        v_full = x_full[19:]  # 18 velocities (6 base + 12 joints)
        
        # Extract base state
        # CRITICAL: Drake floating base is [qw, qx, qy, qz, x, y, z]
        base_quat = q_full[0:4]  # [qw, qx, qy, qz]
        base_pos = q_full[4:7]   # [x, y, z]
        base_lin_vel = v_full[3:6]  # [vx, vy, vz] (after angular vel)
        
        # Convert quaternion to Euler angles
        qw, qx, qy, qz = base_quat
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        base_roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (qw * qy - qz * qx)
        if abs(sinp) >= 1:
            base_pitch = np.copysign(np.pi / 2, sinp)
        else:
            base_pitch = np.arcsin(sinp)
        
        # Build MPC state vector: [roll, pitch, z, vx, vy, vz, q_joints, v_joints]
        base_state = np.array([
            base_roll, 
            base_pitch, 
            base_pos[2],        # z
            base_lin_vel[0],    # vx
            base_lin_vel[1],    # vy
            base_lin_vel[2],    # vz
        ])
        
        # Extract actuated joint positions and velocities
        q_act = q_full[self.model.idx_q_act]
        v_act = v_full[self.model.idx_v_act]
        
        # Build full MPC state: [base; joints]
        x_mpc = np.concatenate([base_state, q_act, v_act])
        
        # Debug: print state every 10 steps
        self._call_count += 1
        if self._call_count % 10 == 0:
            y0 = self.model.C_y @ x_mpc
            print(
                f"[Step {self._call_count:4d}] "
                f"||y - y_ref|| = {np.linalg.norm(y0 - self.y_ref):.4f}, "
                f"||v_joints|| = {np.linalg.norm(v_act):.4f}, "
                f"base: roll={np.rad2deg(base_roll):+.1f}°, "
                f"pitch={np.rad2deg(base_pitch):+.1f}°, z={base_pos[2]:.3f}m"
            )
        
        # Solve MPC QP for control DEVIATION δu
        try:
            u_delta = solve_joint_mpc_qp(
                self.model,
                x_mpc,  # Pass FULL MPC state (including base)
                self.y_ref,
                self.N_horizon,
                self.u_min,
                self.u_max,
                verbose=self.verbose,
            )
        except RuntimeError as e:
            print(f"[JointMPCController] QP solve failed: {e}")
            print(f"[JointMPCController] Falling back to equilibrium torque u*")
            u_delta = np.zeros(self.model.n_u)

        # CRITICAL: Add equilibrium torque feedforward
        # The QP works in deviation coordinates, so we must add u* back
        u_total = self.model.u_star + u_delta

        # Output the TOTAL control
        output.SetFromVector(u_total)

        # Verbose QP output
        if self.verbose and self._call_count % 100 == 0:
            y0 = self.model.C_y @ x0
            print(
                f"[MPC Step {self._call_count}] "
                f"||y - y_ref|| = {np.linalg.norm(y0 - self.y_ref):.4f}, "
                f"||u_total|| = {np.linalg.norm(u_total):.1f} Nm, "
                f"||δu|| = {np.linalg.norm(u_delta):.1f} Nm"
            )

    def set_target(self, y_ref: np.ndarray):
        """
        Update the target joint configuration.

        Args:
            y_ref: New target joint positions, shape (n_y,).
        """
        assert y_ref.shape == (self.model.n_y,), "y_ref shape mismatch"
        self.y_ref = y_ref


if __name__ == "__main__":
    """
    Standalone test of the MPC controller wrapper.

    This verifies:
    1. Controller can be instantiated
    2. Control computation runs without errors
    3. Output has correct dimensions
    """
    from pydrake.systems.framework import DiagramBuilder
    from pydrake.systems.analysis import Simulator
    from pydrake.systems.primitives import ConstantVectorSource
    from mpc_model_joint_space import build_spot_mpc_model_joint_space

    print("\n" + "=" * 60)
    print("Testing Joint MPC Controller (Drake System)")
    print("=" * 60)

    # Build model
    model = build_spot_mpc_model_joint_space(dt_mpc=0.01, use_zoh=True)

    # Create controller
    N_horizon = 20
    controller = JointMPCController(
        model=model,
        N_horizon=N_horizon,
        y_ref=None,  # use nominal target
        verbose=False,
    )

    print(f"\n✓ Controller created with horizon N = {N_horizon}")
    print(f"  State dimension: {model.n_x}")
    print(f"  Control dimension: {model.n_u}")
    print(f"  Output dimension: {model.n_y}")

    # Create a simple diagram to test
    builder = DiagramBuilder()

    # Add controller
    controller_sys = builder.AddSystem(controller)

    # Feed perturbed state as input
    x0 = model.x_star.copy()
    x0[0] += 0.1  # perturb first joint
    state_source = builder.AddSystem(ConstantVectorSource(x0))

    # Connect state source to controller
    builder.Connect(
        state_source.get_output_port(),
        controller_sys.get_input_port(),
    )

    # Build and simulate
    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()

    # Run one step
    print("\nRunning controller for one step...")
    simulator.AdvanceTo(0.001)

    # Extract output
    controller_context = controller_sys.GetMyContextFromRoot(context)
    u_output = controller_sys.get_output_port().Eval(controller_context)

    print(f"\n✓ Controller executed successfully")
    print(f"  Output torque: ||u|| = {np.linalg.norm(u_output):.3f} Nm")
    print(f"  Deviation from equilibrium: ||u - u*|| = {np.linalg.norm(u_output - model.u_star):.3f} Nm")

    # Sanity checks
    assert u_output.shape == (model.n_u,), f"Output shape mismatch"
    assert np.all(np.isfinite(u_output)), "Output contains NaN or Inf"

    print("\n" + "=" * 60)
    print("MPC Controller Test Passed")
    print("=" * 60 + "\n")
