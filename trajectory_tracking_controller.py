"""
Time-Varying LQR Controller for Trajectory Tracking

Uses fixed LQR gain to track time-varying reference trajectory:
  u(t) = u_ref(t) - K * (x(t) - x_ref(t))

Where:
- x_ref(t), u_ref(t) come from MPC trajectory plan
- K is the LQR gain from standing controller
- x(t) is the current full plant state
"""

from __future__ import annotations

import numpy as np
from pydrake.all import LeafSystem, BasicVector


class TrajectoryTrackingController(LeafSystem):
    """
    LQR controller that tracks a time-varying reference trajectory.
    
    Input: x_full (full plant state)
    Output: u (control torques)
    
    Control law:
        x_joint = S @ x_full  (extract joint state)
        x_joint_ref = interpolate from trajectory at current time
        u = u_ref - K @ (x_joint - x_joint_ref)
    """
    
    def __init__(
        self,
        K_joint: np.ndarray,        # LQR gain matrix
        times: np.ndarray,          # Trajectory timesteps (N+1,)
        y_ref_traj: np.ndarray,     # Reference joint positions (N+1, n_act)
        u_ref_traj: np.ndarray,     # Reference torques (N, n_u)
        x_full_star: np.ndarray,    # Nominal full state
        idx_q_act: list,            # Joint position indices
        idx_v_act: list,            # Joint velocity indices
        n_q: int,                   # Number of positions in full state
    ):
        LeafSystem.__init__(self)
        
        self.K_joint = K_joint
        self.times = times
        self.y_ref_traj = y_ref_traj
        self.u_ref_traj = u_ref_traj
        self.x_full_star = x_full_star
        self.idx_q_act = idx_q_act
        self.idx_v_act = idx_v_act
        self.n_q = n_q
        
        self.n_act = len(idx_q_act)
        self.n_x_full = len(x_full_star)
        self.n_u = K_joint.shape[1] // 2  # K is (n_u, 2*n_act)
        
        # Declare ports
        self.DeclareVectorInputPort("x_full", self.n_x_full)
        self.DeclareVectorOutputPort("u", self.n_u, self.CalcControl)
        
        # Build selection matrix S: x_joint = S @ x_full
        self.S = self._build_selection_matrix()
        
        print(f"\n[TrajectoryTrackingController] Initialized")
        print(f"  Trajectory duration: {times[-1]:.2f}s")
        print(f"  Number of waypoints: {len(times)}")
        print(f"  Joint state dimension: {2 * self.n_act}")
        print(f"  Control dimension: {self.n_u}")
    
    def _build_selection_matrix(self) -> np.ndarray:
        """Build matrix S such that x_joint = S @ x_full."""
        n_x_joint = 2 * self.n_act
        S = np.zeros((n_x_joint, self.n_x_full))
        
        # Joint positions
        for i, q_idx in enumerate(self.idx_q_act):
            S[i, q_idx] = 1.0
        
        # Joint velocities (offset by n_q in full state)
        for i, v_idx in enumerate(self.idx_v_act):
            S[self.n_act + i, self.n_q + v_idx] = 1.0
        
        return S
    
    def _interpolate_reference(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Interpolate reference trajectory at time t.
        
        Returns:
            x_joint_ref: Reference joint state (2*n_act,)
            u_ref: Reference control (n_u,)
        """
        # Clamp time to trajectory bounds
        t_clamped = np.clip(t, self.times[0], self.times[-1])
        
        # Find nearest timestep (simple nearest-neighbor for now)
        idx = np.searchsorted(self.times, t_clamped)
        idx = min(idx, len(self.times) - 1)
        
        # Get reference joint positions at this time
        y_ref = self.y_ref_traj[idx, :]
        
        # Reference joint state (zero velocities for now)
        # TODO: Could compute velocities from finite differences
        x_joint_ref = np.concatenate([y_ref, np.zeros(self.n_act)])
        
        # Get reference control (if past end, use last value)
        if idx >= len(self.u_ref_traj):
            u_ref = self.u_ref_traj[-1, :]
        else:
            u_ref = self.u_ref_traj[idx, :]
        
        return x_joint_ref, u_ref
    
    def CalcControl(self, context, output):
        """
        Compute control output at current time.
        
        u = u_ref(t) - K @ (x_joint(t) - x_joint_ref(t))
        """
        # Get current time and state
        t = context.get_time()
        x_full = self.get_input_port(0).Eval(context)
        
        # Extract joint state from full state
        x_joint = self.S @ x_full
        
        # Interpolate reference
        x_joint_ref, u_ref = self._interpolate_reference(t)
        
        # LQR feedback
        u = u_ref - self.K_joint @ (x_joint - x_joint_ref)
        
        output.SetFromVector(u)


def build_trajectory_tracking_diagram(
    K_joint: np.ndarray,
    times: np.ndarray,
    y_ref_traj: np.ndarray,
    u_ref_traj: np.ndarray,
    x_full_star: np.ndarray,
    idx_q_act: list,
    idx_v_act: list,
):
    """
    Build closed-loop diagram with trajectory tracking controller.
    
    Args:
        K_joint: LQR gain from standing controller
        times: Trajectory timesteps (N+1,)
        y_ref_traj: Reference joint positions (N+1, n_act)
        u_ref_traj: Reference torques (N, n_u)
        x_full_star: Nominal full state
        idx_q_act: Joint position indices
        idx_v_act: Joint velocity indices
        
    Returns:
        root_diagram: Complete system with controller and plant
        plant: MultibodyPlant reference
        q_star: Initial positions
        v_star: Initial velocities
    """
    from pydrake.all import DiagramBuilder
    from spot_lqr_standing import build_spot_runtime_diagram, get_default_standing_state
    
    # Build runtime plant with Meshcat
    diagram, plant = build_spot_runtime_diagram(time_step=0.0)
    
    q_star, v_star = get_default_standing_state(plant)
    n_q = plant.num_positions()
    n_v = plant.num_velocities()
    n_x_full = n_q + n_v
    
    # Build system with controller
    builder = DiagramBuilder()
    spot_sys = builder.AddSystem(diagram)
    
    state_port = spot_sys.get_output_port(0)
    actuation_port = spot_sys.get_input_port(0)
    
    # Add trajectory tracking controller
    controller = TrajectoryTrackingController(
        K_joint=K_joint,
        times=times,
        y_ref_traj=y_ref_traj,
        u_ref_traj=u_ref_traj,
        x_full_star=x_full_star,
        idx_q_act=idx_q_act,
        idx_v_act=idx_v_act,
        n_q=n_q,
    )
    controller_sys = builder.AddSystem(controller)
    
    # Connect
    builder.Connect(state_port, controller_sys.get_input_port(0))
    builder.Connect(controller_sys.get_output_port(0), actuation_port)
    
    root_diagram = builder.Build()
    return root_diagram, plant, q_star, v_star


if __name__ == "__main__":
    print("This module should be imported, not run directly.")
    print("See demo_trajectory_tracking.py for usage example.")
