# external/spot/spot_lqr_standing.py
#
# Joint-space LQR regulator around a nominal standing pose for Spot.
#
# Design model:
#   - MultibodyPlant + SceneGraph (Spot + ground, with contact),
#   - we define a reduced joint state
#         x_joint = [q_act; v_act] ∈ R^{2 * n_act}
#     where q_act, v_act are the actuated joint positions/velocities,
#   - we keep the floating base DOFs fixed at a nominal standing pose
#     when computing f_joint(x_joint, u),
#   - we numerically approximate A_joint, B_joint of
#         xdot_joint = f_joint(x_joint, u)
#     via finite differences on CalcTimeDerivatives (contact included).
#
# We then:
#   1) Find a "gravity/contact–compensating" torque u_star by solving
#        B_joint u_star ≈ -xdot0_joint
#      at the nominal joint state.
#   2) Design continuous-time LQR on (A_joint, B_joint) with weights Q_joint, R.
#
# Runtime model:
#   - MultibodyPlant + SceneGraph + ground + Meshcat (full Spot model),
#   - we use a stateless affine controller
#         u = u_star - K_joint (S x_full - x_joint_star),
#     where S selects joint positions/velocities from the full state.
#   - At x_full ≈ x_full_star, the torques are ≈ u_star (feedforward);
#     the feedback term only corrects deviations from the nominal joint
#     configuration.

from __future__ import annotations

import numpy as np

from pydrake.all import (
    AffineSystem,
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    LinearQuadraticRegulator,
    MeshcatVisualizer,
    MultibodyPlant,
    Parser,
    Simulator,
    StartMeshcat,
    JointActuatorIndex,
)

from underactuated import ConfigureParser


def get_default_standing_state(plant: MultibodyPlant):
    q = plant.GetDefaultPositions().copy()
    # Index 6 is base z
    q[6] -= 0.02889683
    v = np.zeros(plant.num_velocities())
    return q, v


def build_spot_design_diagram(time_step: float = 0.0):
    # Build a "design" diagram used for numerical joint-space linearization
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(
        builder, time_step=time_step)
    parser = Parser(plant)

    ConfigureParser(parser)

    # Spot + ground (same as Antonio's setup)
    parser.AddModelsFromUrl(
        "package://underactuated/models/spot/spot.dmd.yaml")
    parser.AddModelsFromUrl(
        "package://underactuated/models/littledog/ground.urdf")

    plant.Finalize()

    diagram = builder.Build()
    return diagram, plant


def build_spot_runtime_diagram(time_step: float = 0.0):
    # Build a "runtime" diagram with Meshcat visualizer
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(
        builder, time_step=time_step)
    parser = Parser(plant)

    ConfigureParser(parser)

    # Spot + ground (same as Antonio's setup)
    parser.AddModelsFromUrl(
        "package://underactuated/models/spot/spot.dmd.yaml")
    parser.AddModelsFromUrl(
        "package://underactuated/models/littledog/ground.urdf")

    plant.Finalize()

    # Meshcat
    meshcat = StartMeshcat()
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat=meshcat)

    # Export actuation and state ports at the diagram level
    builder.ExportInput(plant.get_actuation_input_port(), "actuation")
    builder.ExportOutput(plant.get_state_output_port(), "x")

    diagram = builder.Build()
    return diagram, plant


def compute_joint_space_lqr_gain():
    # Build Spot design diagram (with ground + contact), define a reduced joint state to accomodate Drake
    diagram, plant = build_spot_design_diagram(time_step=0.0)

    # Contexts
    diagram_context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(diagram_context)

    # Full-state nominal standing pose
    q_full_star, v_full_star = get_default_standing_state(plant)
    plant.SetPositions(plant_context, q_full_star)
    plant.SetVelocities(plant_context, v_full_star)

    n_q = plant.num_positions()
    n_v = plant.num_velocities()
    n_x_full = n_q + n_v
    n_u = plant.num_actuators()

    x_full_star = np.concatenate([q_full_star, v_full_star])
    u_zero = np.zeros(n_u)

    # Fix nominal input at zero to measure drift
    plant.get_actuation_input_port().FixValue(plant_context, u_zero)

    # Actuated joint indices
    idx_q_act = []
    idx_v_act = []
    joint_names_act = []

    for i in range(plant.num_actuators()):
        actuator = plant.get_joint_actuator(JointActuatorIndex(i))
        joint = actuator.joint()
        nq_j = joint.num_positions()
        nv_j = joint.num_velocities()
        q0 = joint.position_start()
        v0 = joint.velocity_start()

        for k in range(nq_j):
            idx_q_act.append(q0 + k)
            joint_names_act.append(joint.name())
        for k in range(nv_j):
            idx_v_act.append(v0 + k)

    n_act = len(idx_q_act)
    assert n_act == len(
        idx_v_act), "Position/velocity index lists must match length"

    print("\n" + "=" * 80)
    print("[LQR] Joint-space linearization setup")
    print("=" * 80)
    print(f"  Total generalized coordinates n_q        = {n_q}")
    print(f"  Total generalized velocities n_v        = {n_v}")
    print(f"  Total full state dimension n_x_full    = {n_x_full}")
    print(f"  Number of actuators n_u                = {n_u}")
    print(f"  Number of actuated joint positions     = {n_act}")
    print(f"  Joint-space state dimension n_x_joint  = {2 * n_act}")
    print("")
    print("  Actuated joints (design model):")
    for k, name in enumerate(joint_names_act):
        print(
            f"    [{k:2d}] '{name}' "
            f"(q index {idx_q_act[k]}, v index {idx_v_act[k]})"
        )
    print("=" * 80 + "\n")

    # Joint-space nominal state
    q_act_star = q_full_star[idx_q_act]
    v_act_star = v_full_star[idx_v_act]
    x_joint_star = np.concatenate([q_act_star, v_act_star])
    n_x_joint = x_joint_star.shape[0]

    derivs = plant.AllocateTimeDerivatives()

    def f_joint(x_joint: np.ndarray, u: np.ndarray) -> np.ndarray:
        # Compute xdot_joint given x_joint, GRF, and joint torques u, in the nominal pose
        assert x_joint.shape[0] == n_x_joint
        assert u.shape[0] == n_u

        # Start from nominal full state
        q_full = q_full_star.copy()
        v_full = v_full_star.copy()

        q_full[idx_q_act] = x_joint[:n_act]
        v_full[idx_v_act] = x_joint[n_act:]

        plant.SetPositions(plant_context, q_full)
        plant.SetVelocities(plant_context, v_full)
        plant.get_actuation_input_port().FixValue(plant_context, u)

        plant.CalcTimeDerivatives(plant_context, derivs)
        xdot_full = derivs.get_vector().CopyToVector()

        qdot_full = xdot_full[:n_q]
        vdot_full = xdot_full[n_q:]

        qdot_act = qdot_full[idx_q_act]
        vdot_act = vdot_full[idx_v_act]

        return np.concatenate([qdot_act, vdot_act])

    xdot0_joint = f_joint(x_joint_star, u_zero)

    # Finite difference step sizes
    eps_x = 1e-6
    eps_u = 1e-4

    A_joint = np.zeros((n_x_joint, n_x_joint))
    B_joint = np.zeros((n_x_joint, n_u))

    # A_joint
    for i in range(n_x_joint):
        xj_pert = x_joint_star.copy()
        xj_pert[i] += eps_x
        xdot = f_joint(xj_pert, u_zero)
        A_joint[:, i] = (xdot - xdot0_joint) / eps_x

    # B_joint
    for j in range(n_u):
        uj_pert = u_zero.copy()
        uj_pert[j] += eps_u
        xdot = f_joint(x_joint_star, uj_pert)
        B_joint[:, j] = (xdot - xdot0_joint) / eps_u

    print(
        f"[LQR] Joint-space numerical model: n_x_joint = {n_x_joint}, n_u = {n_u}")

    # Solve for a static torque u_star that (approximately) cancels the joint drift
    # in the linear model: xdot ≈ xdot0_joint + B_joint u_star ≈ 0.
    u_star, residuals, rank, svals = np.linalg.lstsq(
        B_joint, -xdot0_joint, rcond=None
    )

    print(
        f"[LQR] ‖xdot0_joint‖ = {np.linalg.norm(xdot0_joint):.3e}, "
        f"‖B u* + xdot0‖ = {np.linalg.norm(B_joint @ u_star + xdot0_joint):.3e}"
    )

    Q_joint = np.eye(n_x_joint)
    pos_weight = 100.0
    vel_weight = 1000.0

    Q_joint[:n_act, :n_act] *= pos_weight
    Q_joint[n_act:, n_act:] *= vel_weight

    R_scale = 0.01  # smaller = allow stronger torques
    R = np.eye(n_u) * R_scale

    print("[LQR] Cost weights:")
    print(f"  Position weight (per DOF): {pos_weight}")
    print(f"  Velocity weight (per DOF): {vel_weight}")
    print(f"  Torque cost scaling (R_scale): {R_scale}")
    print(f"  Q_joint shape: {Q_joint.shape}, R shape: {R.shape}")

    # Drake's continuous-time LQR on the joint-space model
    K_joint, _ = LinearQuadraticRegulator(A_joint, B_joint, Q_joint, R)

    print("\n[LQR] LQR gain matrix K_joint:")
    print(f"  K_joint shape: {K_joint.shape}")
    print(f"  ‖K_joint‖_∞ = {np.max(np.abs(K_joint)):.3e}")
    print("=" * 80 + "\n")

    return K_joint, x_full_star, x_joint_star, u_star, n_x_full, n_u, idx_q_act, idx_v_act


def build_lqr_closed_loop_diagram(K_joint,
                                  x_full_star,
                                  x_joint_star,
                                  u_star,
                                  n_x_full,
                                  n_u,
                                  idx_q_act,
                                  idx_v_act):
    # Build closed-loop diagram with joint-space LQR controller
    diagram, plant = build_spot_runtime_diagram(time_step=0.0)

    q_star, v_star = get_default_standing_state(plant)

    n_q = plant.num_positions()
    n_v = plant.num_velocities()
    assert n_x_full == n_q + n_v

    n_act = len(idx_q_act)
    n_x_joint = 2 * n_act

    # x_joint = S x_full
    S = np.zeros((n_x_joint, n_x_full))

    for r, q_idx in enumerate(idx_q_act):
        S[r, q_idx] = 1.0

    # Velocities offset by n_q in full state
    for r, v_idx in enumerate(idx_v_act):
        S[n_act + r, n_q + v_idx] = 1.0

    builder = DiagramBuilder()
    spot_sys = builder.AddSystem(diagram)

    state_port = spot_sys.get_output_port(0)
    actuation_port = spot_sys.get_input_port(0)

    assert state_port.size() == n_x_full
    assert actuation_port.size() == n_u

    # We want: u = u_star - K_joint (S x_full - x_joint_star)
    # We model this as an AffineSystem with:
    #   - no internal state (n_x_affine = 0),
    #   - input u_affine = x_full (dim = n_x_full),
    #   - output y = u (dim = n_u)
    #
    # Drake's AffineSystem uses:
    #   xdot = A x + B u_affine + f0   (x has dim 0 → trivial)
    #   y    = C x + D u_affine + y0
    #
    # We encode all dependence on x_full via D, with C = 0
    C_full = -K_joint @ S
    y0 = u_star + K_joint @ x_joint_star

    A_affine = np.zeros((0, 0))

    B_affine = np.zeros((0, n_x_full))
    f0_affine = np.zeros((0,))

    C_affine = np.zeros((n_u, 0))
    D_affine = C_full
    y0_affine = y0

    controller = AffineSystem(
        A=A_affine,
        B=B_affine,
        f0=f0_affine,
        C=C_affine,
        D=D_affine,
        y0=y0_affine,
    )
    controller_sys = builder.AddSystem(controller)

    builder.Connect(
        state_port,
        controller_sys.get_input_port(0),
    )
    builder.Connect(
        controller_sys.get_output_port(0),
        actuation_port,
    )

    joint_names = []
    for i in range(plant.num_actuators()):
        actuator = plant.get_joint_actuator(JointActuatorIndex(i))
        joint = actuator.joint()
        nq_j = joint.num_positions()
        for _ in range(nq_j):
            joint_names.append(joint.name())

    root_diagram = builder.Build()

    return root_diagram, plant, q_star, v_star, S, joint_names

def run_lqr_sim_with_logging(
    root_diagram,
    plant,
    q_star,
    v_star,
    K_joint,
    x_joint_star,
    u_star,
    S,
    joint_names,
    t_final: float = 10.0,
    dt: float = 0.05,
    log_interval: float | None = None,
):
    """
    Run the closed-loop LQR simulation and continuously log:
      - joint-space errors (q_act - q_act* and v_act - v_act*),
      - LQR joint torques u.

    This does NOT change the controller wiring. The AffineSystem built in
    build_lqr_closed_loop_diagram is still driving the actuation port.

    We only RECONSTRUCT the same control law for logging:
        u = u_star - K_joint (x_joint - x_joint_star),
    with x_joint = S x_full, where S and joint_names come from
    build_lqr_closed_loop_diagram.
    """
    import sys

    if log_interval is None:
        log_interval = dt  # "continuous-ish" logging

    # --- Set up simulator exactly like in the PD script ---
    simulator = Simulator(root_diagram)
    root_context = simulator.get_mutable_context()

    plant_context = plant.GetMyMutableContextFromRoot(root_context)
    plant.SetPositions(plant_context, q_star)
    plant.SetVelocities(plant_context, v_star)

    simulator.set_target_realtime_rate(1.0)
    simulator.Initialize()

    # --- Dimension checks using S and runtime plant ---
    n_q = plant.num_positions()
    n_v = plant.num_velocities()
    n_x_full_runtime = n_q + n_v

    n_x_joint, n_x_full_S = S.shape
    assert n_x_full_S == n_x_full_runtime, \
        "S width must match runtime full state dimension [q; v]"
    n_act = len(joint_names)
    assert n_x_joint == 2 * n_act, \
        "S must map full state to [q_act; v_act] of length 2 * n_act"

    # Nominal joint state
    q_act_star = x_joint_star[:n_act]
    v_act_star = x_joint_star[n_act:]

    print("=" * 80)
    print("LQR CLOSED-LOOP STANDING SIMULATION", flush=True)
    print("=" * 80)
    print(f"  Number of actuated joint DOFs: {n_act}")
    print("  Actuated joints (runtime model):")
    for k, name in enumerate(joint_names):
        print(f"    [{k:2d}] '{name}'")
    print("")
    print("  Nominal actuated joint pose (q*):")
    for k, name in enumerate(joint_names):
        print(f"    [{k:2d}] {name:>24s}: q* = {q_act_star[k]:+7.4f} rad")
    print("=" * 80)
    print("  Starting from nominal standing pose.")
    print("  LQR controller regulates deviations around this pose using:")
    print("    u(t) = u* - K_joint (x_joint(t) - x_joint*)", flush=True)
    print("")

    # --- Main simulation loop (PD-style pattern) ---
    sim_duration = t_final
    last_log_time = -1e9

    try:
        while True:
            root_context = simulator.get_mutable_context()
            t = root_context.get_time()

            if t >= sim_duration:
                print("\nSimulation finished.", flush=True)
                break

            # Plant state at current time
            plant_context = plant.GetMyMutableContextFromRoot(root_context)
            q = plant.GetPositions(plant_context)
            v = plant.GetVelocities(plant_context)
            x_full = np.concatenate([q, v])

            # Project to actuated joint state
            x_joint = S @ x_full
            q_act = x_joint[:n_act]
            v_act = x_joint[n_act:]

            e_q = q_act - q_act_star
            e_v = v_act - v_act_star

            # Same law as AffineSystem
            u = u_star - K_joint @ (x_joint - x_joint_star)

            if t - last_log_time >= log_interval:
                max_eq = float(np.max(np.abs(e_q)))
                max_ev = float(np.max(np.abs(e_v)))
                max_u = float(np.max(np.abs(u)))

                print(f"\n[LQR] t = {t:6.3f} s")
                print(f"  max |position error| (rad):   {max_eq:.4e}")
                print(f"  max |velocity error| (rad/s): {max_ev:.4e}")
                print(f"  max |joint torque|   (N·m):   {max_u:.4e}")
                for idx, name in enumerate(joint_names):
                    print(
                        f"    {name:>24s}: "
                        f"e_q = {e_q[idx]:+7.4f} rad, "
                        f"e_v = {e_v[idx]:+7.4f} rad/s, "
                        f"u = {u[idx]:+7.4f} N·m"
                    )
                sys.stdout.flush()
                last_log_time = t

            # Advance one step, like in the PD script
            target_time = min(sim_duration, t + dt)
            simulator.AdvanceTo(target_time)

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.", flush=True)

def main():
    # 1) Design LQR on the contact-inclusive model
    (K_joint,
     x_full_star,
     x_joint_star,
     u_star,
     n_x_full,
     n_u,
     idx_q_act,
     idx_v_act) = compute_joint_space_lqr_gain()

    # 2) Build closed-loop runtime diagram (Spot + ground + Meshcat + LQR)
    (root_diagram,
     plant,
     q_star,
     v_star,
     S,
     joint_names) = build_lqr_closed_loop_diagram(
        K_joint,
        x_full_star,
        x_joint_star,
        u_star,
        n_x_full,
        n_u,
        idx_q_act,
        idx_v_act,
    )

    # 3) Run simulation with continuous logging of LQR behavior
    run_lqr_sim_with_logging(
        root_diagram=root_diagram,
        plant=plant,
        q_star=q_star,
        v_star=v_star,
        K_joint=K_joint,
        x_joint_star=x_joint_star,
        u_star=u_star,
        S=S,
        joint_names=joint_names,
        t_final=10.0,     # you can tune this
        dt=0.05,          # sim logging step
        log_interval=0.10 # log every 0.1 s of sim time
    )


if __name__ == "__main__":
    main()
