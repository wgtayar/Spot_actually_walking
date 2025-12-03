# external/spot/spot_lqr_standing.py
#
# Joint-space and full-state LQR regulators around a nominal standing pose for Spot.
#
# Mode 1 (default): JOINT-SPACE LQR (existing behavior)
#   - Design model:
#       x_joint = [q_act; v_act]
#       Base DOFs are frozen at the nominal standing pose when computing f_joint.
#       We linearize xdot_joint = f_joint(x_joint, u).
#   - Runtime model:
#       u = u_star - K_joint (S x_full - x_joint_star)
#       where S selects joint positions/velocities from the full state.
#
# Mode 2: FULL-STATE LQR (floating-base)
#   - Design model:
#       x_full = [q_full; v_full] includes base + joints.
#       We linearize xdot_full = f_full(x_full, u) around (x_full_star, u_star).
#       We use the same contact-inclusive plant.
#   - Runtime model:
#       u = u_star - K_full (x_full - x_full_star)
#       with the full state fed into the controller.
#
# You can choose the mode from the command line:
#   python spot_lqr_standing.py             # joint-space LQR (default)
#   python spot_lqr_standing.py --mode full {--t_final 10} # full-state (floating-base) LQR

from __future__ import annotations

import argparse
import numpy as np
from scipy.optimize import minimize

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
from scipy.optimize import minimize

def solve_full_state_standing_equilibrium(
    plant: MultibodyPlant,
    plant_context,
    q_initial: np.ndarray,
    v_initial: np.ndarray | None = None,
    model_instance_name: str = "spot",
    ground_height: float = 0.0,
    verbose: bool = True,
):
    """
    Solve for a static full-state equilibrium (q*, v* = 0, u*) for Spot
    in contact with the ground:

        x* = [q*; v* = 0],  such that  xdot = f(x*, u*) ≈ 0.

    Strategy:
      - Keep base orientation (quaternion) and base x,y fixed to q_initial.
      - Allow base z and all joint positions to move slightly.
      - Set v* = 0.
      - Optimize over:
            theta = [delta_q_free; u]
        to minimize:
            ||xdot_full||^2
          + alpha_q * ||delta_q_free||^2
          + beta_u  * ||u||^2
          + gamma_feet * (min_foot_z - ground_height)^2

    This uses the contact-inclusive dynamics via plant.CalcTimeDerivatives.
    """

    n_q = plant.num_positions()
    n_v = plant.num_velocities()
    n_u = plant.num_actuators()

    assert q_initial.shape[0] == n_q
    if v_initial is None:
        v_initial = np.zeros(n_v)
    else:
        assert v_initial.shape[0] == n_v

    # We'll only allow base z and joint positions to vary.
    # Spot floating-base layout (from this model):
    #   q[0:4]  = base orientation quaternion (w, x, y, z)
    #   q[4:6]  = base x, y
    #   q[6]    = base z
    #   q[7:]   = joint positions
    base_z_index = 6
    joint_pos_indices = list(range(7, n_q))
    free_q_indices = [base_z_index] + joint_pos_indices
    n_free_q = len(free_q_indices)

    # Allocate derivatives once for speed
    derivs = plant.AllocateTimeDerivatives()

    # Foot frames for ground penalty
    foot_frames = _find_spot_foot_frames(plant, model_instance_name)

    # Regularization weights
    alpha_q = 1e-2   # keep q near q_initial
    beta_u  = 1e-3   # keep u small
    gamma_feet = 1e1 # encourage feet to stay on ground

    # Initial guess for u: free-space gravity comp, like you had before
    tau_g = plant.CalcGravityGeneralizedForces(plant_context)
    u0 = np.zeros(n_u)
    for i in range(n_u):
        actuator = plant.get_joint_actuator(JointActuatorIndex(i))
        joint = actuator.joint()
        assert joint.num_velocities() == 1
        v_index = joint.velocity_start()
        u0[i] = -tau_g[v_index]

    theta0 = np.concatenate([np.zeros(n_free_q), u0])

    def objective(theta: np.ndarray) -> float:
        """
        theta = [delta_q_free; u]
        """
        delta_q_free = theta[:n_free_q]
        u = theta[n_free_q:]

        # Build q from q_initial + delta_q on free indices
        q = q_initial.copy()
        q[free_q_indices] = q_initial[free_q_indices] + delta_q_free
        v = np.zeros(n_v)  # static equilibrium: v* = 0

        # Set state and actuation
        plant.SetPositions(plant_context, q)
        plant.SetVelocities(plant_context, v)
        plant.get_actuation_input_port().FixValue(plant_context, u)

        # Compute xdot = [qdot; vdot]
        plant.CalcTimeDerivatives(plant_context, derivs)
        xdot = derivs.get_vector().CopyToVector()

        qdot = xdot[:n_q]
        vdot = xdot[n_q:]

        # Equilibrium cost: want both qdot and vdot ≈ 0
        cost_eq = float(qdot @ qdot + vdot @ vdot)

        # Regularize deviation from original pose
        cost_reg_q = alpha_q * float(delta_q_free @ delta_q_free)

        # Regularize torque magnitude
        cost_reg_u = beta_u * float(u @ u)

        # Keep feet near ground
        cost_feet = 0.0
        if foot_frames:
            min_foot_z = _compute_min_foot_height(plant, plant_context, foot_frames)
            dz = min_foot_z - ground_height
            cost_feet = gamma_feet * float(dz * dz)

        return cost_eq + cost_reg_q + cost_reg_u + cost_feet

    if verbose:
        # Evaluate at the initial guess to show how bad it is
        plant.SetPositions(plant_context, q_initial)
        plant.SetVelocities(plant_context, v_initial)
        plant.get_actuation_input_port().FixValue(plant_context, u0)
        plant.CalcTimeDerivatives(plant_context, derivs)
        xdot_init = derivs.get_vector().CopyToVector()
        print("\n[EQ FULL-STATE] Initial (q_initial, u0) equilibrium residual:")
        print(f"  ||xdot|| = {np.linalg.norm(xdot_init):.3e}")
        print(f"  max |qdot| = {np.max(np.abs(xdot_init[:n_q])):.3e}")
        print(f"  max |vdot| = {np.max(np.abs(xdot_init[n_q:])):.3e}")

    # Solve with BFGS
    res = minimize(
        objective,
        theta0,
        method="BFGS",
        options={"maxiter": 400, "gtol": 1e-6, "disp": verbose},
    )

    theta_star = res.x
    delta_q_free_star = theta_star[:n_free_q]
    u_star = theta_star[n_free_q:]

    # Build final q*, v* = 0
    q_star = q_initial.copy()
    q_star[free_q_indices] = q_initial[free_q_indices] + delta_q_free_star
    v_star = np.zeros(n_v)

    # Evaluate xdot at the solution for reporting
    plant.SetPositions(plant_context, q_star)
    plant.SetVelocities(plant_context, v_star)
    plant.get_actuation_input_port().FixValue(plant_context, u_star)
    plant.CalcTimeDerivatives(plant_context, derivs)
    xdot_star = derivs.get_vector().CopyToVector()

    if verbose:
        print("\n[EQ FULL-STATE] Optimized standing equilibrium:")
        print(f"  success = {res.success},  message = {res.message}")
        print(f"  ||xdot|| = {np.linalg.norm(xdot_star):.3e}")
        print(f"  max |qdot| = {np.max(np.abs(xdot_star[:n_q])):.3e}")
        print(f"  max |vdot| = {np.max(np.abs(xdot_star[n_q:])):.3e}")
        # Optional: check foot heights
        if foot_frames:
            min_foot_z = _compute_min_foot_height(plant, plant_context, foot_frames)
            print(f"  min foot z at equilibrium: {min_foot_z:+.4f} m")

    return q_star, v_star, u_star

def solve_fixed_pose_joint_equilibrium(
    plant: MultibodyPlant,
    base_context,
    q_standing: np.ndarray,
    idx_v_act,
    torque_init: np.ndarray | None = None,
    torque_reg: float = 1e-4,
    verbose: bool = True,
):
    """
    Given a fixed standing pose q_standing and v = 0, solve for actuator torques u
    that make the full generalized accelerations vdot as small as possible:

        u* = argmin_u  ||vdot(q_standing, v=0, u)||^2 + torque_reg * ||u||^2

    where vdot is the full vector of generalized accelerations (base + joints)
    returned by the MultibodyPlant.

    We *do not* change q_standing in this solver – it stays exactly as passed in.
    """

    n_q = plant.num_positions()
    n_v = plant.num_velocities()
    n_u = plant.num_actuators()

    assert q_standing.shape[0] == n_q

    derivs = plant.AllocateTimeDerivatives()

    if torque_init is None:
        u0 = np.zeros(n_u)
    else:
        u0 = np.asarray(torque_init).copy()

    def eval_xdot(u_vec: np.ndarray):
        """
        Evaluate xdot = [qdot; vdot] at (q_standing, v=0, u_vec).

        We reuse the same plant context (base_context) and simply overwrite
        positions, velocities, and actuation each call.
        """
        # Use the existing plant subcontext directly (no Clone).
        ctx = base_context

        # Set the fixed standing pose and zero velocities
        plant.SetPositions(ctx, q_standing)
        plant.SetVelocities(ctx, np.zeros(n_v))

        # Apply candidate torques
        plant.get_actuation_input_port().FixValue(ctx, u_vec)

        # Compute time derivatives
        plant.CalcTimeDerivatives(ctx, derivs)
        xdot = derivs.get_vector().CopyToVector()
        qdot = xdot[:n_q]
        vdot = xdot[n_q:]
        return qdot, vdot, xdot

    def objective(u_vec: np.ndarray) -> float:
        # qdot is always zero since v=0, so we only penalize vdot
        _, vdot, _ = eval_xdot(u_vec)
        cost_eq = float(vdot @ vdot)
        cost_reg = torque_reg * float(u_vec @ u_vec)
        return cost_eq + cost_reg

    if verbose:
        qdot0, vdot0, xdot0 = eval_xdot(u0)
        print("\n[EQ FIXED-POSE] Initial equilibrium residual at fixed pose:")
        print(f"  ||vdot||        = {np.linalg.norm(vdot0):.3e}")
        print(f"  max |vdot|      = {np.max(np.abs(vdot0)):.3e}")
        if len(idx_v_act) > 0:
            vdot_act0 = vdot0[idx_v_act]
            print(f"  max |vdot_act|  = {np.max(np.abs(vdot_act0)):.3e}")
        print(f"  max |xdot_full| = {np.max(np.abs(xdot0)):.3e}")

    res = minimize(
        objective,
        u0,
        method="BFGS",
        options={"maxiter": 200, "gtol": 1e-6, "disp": verbose},
    )

    u_star = res.x
    qdot_star, vdot_star, xdot_star = eval_xdot(u_star)

    if verbose:
        print("\n[EQ FIXED-POSE] Joint equilibrium solve at fixed standing pose:")
        print(f"  success         = {res.success},  message = {res.message}")
        print(f"  final cost      = {res.fun:.3e}")
        print(f"  ||vdot||        = {np.linalg.norm(vdot_star):.3e}")
        print(f"  max |vdot|      = {np.max(np.abs(vdot_star)):.3e}")
        if len(idx_v_act) > 0:
            vdot_act_star = vdot_star[idx_v_act]
            print(f"  max |vdot_act|  = {np.max(np.abs(vdot_act_star)):.3e}")
        print(f"  max |xdot_full| = {np.max(np.abs(xdot_star)):.3e}")

    return u_star, vdot_star, xdot_star


def _find_spot_foot_frames(
    plant: MultibodyPlant,
    model_instance_name: str = "spot",
    substrings: tuple[str, ...] = ("foot", "toe"),
):
    """
    Heuristically find Spot's foot frames by scanning all frames whose name
    contains 'foot' or 'toe' in the 'spot' model instance.
    """
    try:
        model_instance = plant.GetModelInstanceByName(model_instance_name)
        frame_indices = plant.GetFrameIndices(model_instance)
    except RuntimeError:
        # Fallback: scan all frames if we don't know the instance name.
        model_instance = None
        frame_indices = range(plant.num_frames())

    foot_frames = []
    for frame_index in frame_indices:
        frame = plant.get_frame(frame_index)
        name = frame.name().lower()
        if any(s in name for s in substrings):
            foot_frames.append(frame)

    return foot_frames


def _compute_min_foot_height(plant: MultibodyPlant, context, foot_frames):
    """
    Returns the minimum z height of the given frames (in world coordinates).
    """
    min_z = np.inf
    for frame in foot_frames:
        X_WF = plant.CalcRelativeTransform(context, plant.world_frame(), frame)
        z = float(X_WF.translation()[2])
        min_z = min(min_z, z)
    return min_z


def check_foot_contacts_at_pose(
    plant: MultibodyPlant,
    context,
    model_instance_name: str = "spot",
    ground_height: float = 0.0,
    tol: float = 5e-3,
) -> None:
    """
    Debug utility: prints the world z of all detected foot frames
    and reports whether they are on the ground (z ≈ ground_height).
    """
    foot_frames = _find_spot_foot_frames(plant, model_instance_name)
    if not foot_frames:
        print(
            "[check_foot_contacts_at_pose] WARNING: Could not find any frames "
            "with 'foot' or 'toe' in their name; skipping contact check."
        )
        return

    print("\n[check_foot_contacts_at_pose] Foot frame heights at current q:")
    min_foot_z = np.inf
    for frame in foot_frames:
        X_WF = plant.CalcRelativeTransform(context, plant.world_frame(), frame)
        z = float(X_WF.translation()[2])
        min_foot_z = min(min_foot_z, z)
        print(f"  {frame.name():>32s}: z = {z:+.4f} m")
    print(f"  -> min foot z = {min_foot_z:+.4f} m")

    if abs(min_foot_z - ground_height) < tol:
        print(f"  Feet are on the ground within ±{tol:.3e} m tolerance.\n")
    elif min_foot_z > ground_height + tol:
        print(
            f"  WARNING: all feet are ABOVE the ground by at least "
            f"{min_foot_z - ground_height:.4f} m.\n"
        )
    else:  # min_foot_z < ground_height - tol
        print(
            f"  WARNING: at least one foot appears BELOW the ground "
            f"(penetration of {ground_height - min_foot_z:.4f} m).\n"
        )


def get_default_standing_state(
    plant: MultibodyPlant,
    model_instance_name: str = "spot",
    ground_height: float = 0.0,
):
    """
    Returns a 'nice' standing state (q0, v0) for Spot:
      - Start from the URDF's default positions.
      - Automatically shift base z (q[6]) so that the lowest foot frame lies
        on the ground (z = ground_height).
      - v0 = 0.
    """
    q0 = plant.GetDefaultPositions().copy()
    v0 = np.zeros(plant.num_velocities())

    # Use a temporary context just for kinematics.
    context = plant.CreateDefaultContext()
    plant.SetPositions(context, q0)
    plant.SetVelocities(context, v0)

    # Find candidate foot frames.
    foot_frames = _find_spot_foot_frames(plant, model_instance_name)
    if not foot_frames:
        print(
            "[get_default_standing_state] WARNING: Could not automatically "
            "find foot frames. Returning raw default positions from the URDF."
        )
        return q0, v0

    # Measure min foot height in the raw default pose.
    min_foot_z = _compute_min_foot_height(plant, context, foot_frames)

    # Index 6 is base z for Spot's floating-base quaternion joint:
    # [quat_w, quat_x, quat_y, quat_z, x, y, z]
    base_z_index = 6

    # Shift the whole body so that the lowest foot touches the ground.
    # New min_foot_z ≈ ground_height.
    delta_z = min_foot_z - ground_height
    q0[base_z_index] -= delta_z

    return q0, v0


def build_spot_design_diagram(time_step: float = 0.0):
    # Build a "design" diagram used for numerical linearization
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


# =============================================================================
# JOINT-SPACE LQR (existing behavior)
# =============================================================================

def compute_joint_space_lqr_gain():
    # Build Spot design diagram (with ground + contact)
    diagram, plant = build_spot_design_diagram(time_step=0.0)

    # Contexts
    diagram_context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(diagram_context)

    # Full-state nominal "guess" pose
    q_full_star, v_full_star = get_default_standing_state(plant)
    plant.SetPositions(plant_context, q_full_star)
    plant.SetVelocities(plant_context, v_full_star)

    n_q = plant.num_positions()
    n_v = plant.num_velocities()
    n_x_full = n_q + n_v
    n_u = plant.num_actuators()

    x_full_star = np.concatenate([q_full_star, v_full_star])

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
    assert n_act == len(idx_v_act)

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

    # Joint-space nominal guess
    q_act_guess = q_full_star[idx_q_act]
    v_act_guess = v_full_star[idx_v_act]
    x_joint_guess = np.concatenate([q_act_guess, v_act_guess])
    n_x_joint = x_joint_guess.shape[0]

    derivs = plant.AllocateTimeDerivatives()

    def f_joint(x_joint: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Return [qdot_act; vdot_act] for given joint state x_joint and torque u."""
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

    # ----------------------------------------------------------------------
    # Nonlinear equilibrium solve in joint space
    # ----------------------------------------------------------------------
    def equilibrium_objective(theta: np.ndarray) -> float:
        """
        theta = [x_joint; u], we minimize:
            ||f_joint(x_joint, u)||^2
          + α ||x_joint - x_joint_guess||^2
          + β ||u||^2
        """
        xj = theta[:n_x_joint]
        u = theta[n_x_joint:]

        xdot = f_joint(xj, u)
        # Split for debugging / weighting if desired
        qdot = xdot[:n_act]
        vdot = xdot[n_act:]

        alpha = 1e-2   # regularization on deviation from initial x_guess
        beta = 1e-3    # regularization on torque magnitude

        cost_eq = qdot @ qdot + vdot @ vdot
        cost_reg_x = alpha * np.sum((xj - x_joint_guess) ** 2)
        cost_reg_u = beta * np.sum(u ** 2)

        return cost_eq + cost_reg_x + cost_reg_u

    # Initial guess for optimizer: (x_joint_guess, u=0)
    theta0 = np.concatenate([x_joint_guess, np.zeros(n_u)])

    print("[LQR] Solving for nearby joint-space equilibrium (x_joint*, u*) ...")
    res = minimize(
        equilibrium_objective,
        theta0,
        method="BFGS",
        options={"maxiter": 200, "gtol": 1e-6, "disp": False},
    )

    theta_star = res.x
    x_joint_star = theta_star[:n_x_joint]
    u_star = theta_star[n_x_joint:]

    # Update full-state nominal with equilibrium joint state
    q_full_star[idx_q_act] = x_joint_star[:n_act]
    v_full_star[idx_v_act] = x_joint_star[n_act:]
    x_full_star = np.concatenate([q_full_star, v_full_star])

    plant.SetPositions(plant_context, q_full_star)
    plant.SetVelocities(plant_context, v_full_star)

    xdot0_joint = f_joint(x_joint_star, u_star)

    print("[DEBUG] After equilibrium solve:")
    print("  ||xdot0_joint|| =", np.linalg.norm(xdot0_joint))
    print("  qdot_act part:", xdot0_joint[:n_act])
    print("  vdot_act part:", xdot0_joint[n_act:])

    # ----------------------------------------------------------------------
    # Linearization around (x_joint_star, u_star)
    # ----------------------------------------------------------------------
    eps_x = 1e-6
    eps_u = 1e-4

    A_joint = np.zeros((n_x_joint, n_x_joint))
    B_joint = np.zeros((n_x_joint, n_u))

    # A_joint (derivative wrt x_joint)
    for i in range(n_x_joint):
        xj_pert = x_joint_star.copy()
        xj_pert[i] += eps_x
        xdot = f_joint(xj_pert, u_star)
        A_joint[:, i] = (xdot - xdot0_joint) / eps_x

    # B_joint (derivative wrt u)
    for j in range(n_u):
        uj_pert = u_star.copy()
        uj_pert[j] += eps_u
        xdot = f_joint(x_joint_star, uj_pert)
        B_joint[:, j] = (xdot - xdot0_joint) / eps_u

    print(
        f"[LQR] Joint-space numerical model: n_x_joint = {n_x_joint}, n_u = {n_u}"
    )
    print(
        f"[LQR] ||xdot0_joint|| at equilibrium ≈ {np.linalg.norm(xdot0_joint):.3e}"
    )

    # ----------------------------------------------------------------------
    # LQR weights
    # ----------------------------------------------------------------------
    Q_joint = np.eye(n_x_joint)
    pos_weight = 50.0
    vel_weight = 100.0

    Q_joint[:n_act, :n_act] *= pos_weight
    Q_joint[n_act:, n_act:] *= vel_weight

    R_scale = 0.1
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

    # Return the updated x_full_star and x_joint_star and u_star
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

    # u = u_star - K_joint (S x_full - x_joint_star)
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
    Closed-loop simulation for the joint-space LQR controller.

    We reconstruct the law:
        u = u_star - K_joint (x_joint - x_joint_star),
    with x_joint = S x_full, where S and joint_names come from
    build_lqr_closed_loop_diagram.
    """
    import sys

    if log_interval is None:
        log_interval = dt  # "continuous-ish" logging

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
    print("LQR CLOSED-LOOP STANDING SIMULATION (JOINT-SPACE)", flush=True)
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
    print("  LQR controller (joint-space) regulates deviations around this pose using:")
    print("    u(t) = u* - K_joint (x_joint(t) - x_joint*)", flush=True)
    print("")

    # --- Main simulation loop ---
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

            target_time = min(sim_duration, t + dt)
            simulator.AdvanceTo(target_time)

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.", flush=True)


# =============================================================================
# FULL-STATE (FLOATING-BASE) LQR
# =============================================================================

# def compute_full_joint_space_lqr_gain():
#     """
#     Full-state (floating-base) LQR, but now using a clean standing pose:

#       1) Choose a standing equilibrium:
#            x* = [q*; 0],  u* = gravity-comp torques at q*,
#          where q* is obtained from get_default_standing_state, with the base
#          z shifted so that all feet are on the ground.

#       2) Numerically linearize xdot = f(x, u) around (x*, u*).

#       3) Build a REDUCED state (dropping base x,y) to avoid uncontrollable
#          modes, regularize any uncontrollable zero modes, solve LQR on the
#          reduced system, then embed the gain back into the full 37D state.

#     Returns:
#         K_full:          (n_u, n_x_full) LQR gain (embedded from reduced state)
#         x_full_star:     (n_x_full,) equilibrium state [q*; v* = 0]
#         u_star:          (n_u,) equilibrium torques
#         n_x_full:        full state dimension
#         n_u:             number of actuators
#         idx_q_act:       indices of actuated joint positions in q
#         idx_v_act:       indices of actuated joint velocities in v
#         joint_names_act: list of actuated joint names
#     """
#     # ------------------------------------------------------------------
#     # 1) Build design diagram and choose standing equilibrium (q*, v* = 0)
#     # ------------------------------------------------------------------
#     diagram, plant = build_spot_design_diagram(time_step=0.0)
#     diagram_context = diagram.CreateDefaultContext()
#     plant_context = plant.GetMyMutableContextFromRoot(diagram_context)

#     # Nice standing pose with all feet on the ground and v = 0.
#     # q_full_star, v_full_star = get_default_standing_state(plant)
#     # plant.SetPositions(plant_context, q_full_star)
#     # plant.SetVelocities(plant_context, v_full_star)

#     # n_q = plant.num_positions()
#     # n_v = plant.num_velocities()
#     # n_x_full = n_q + n_v
#     # n_u = plant.num_actuators()

#     # # Actuated joint indices (for logging and for S later)
#     # idx_q_act = []
#     # idx_v_act = []
#     # joint_names_act = []
#     # for i in range(plant.num_actuators()):
#     #     actuator = plant.get_joint_actuator(JointActuatorIndex(i))
#     #     joint = actuator.joint()
#     #     nq_j = joint.num_positions()
#     #     nv_j = joint.num_velocities()
#     #     q0_j = joint.position_start()
#     #     v0_j = joint.velocity_start()
#     #     for k in range(nq_j):
#     #         idx_q_act.append(q0_j + k)
#     #         joint_names_act.append(joint.name())
#     #     for k in range(nv_j):
#     #         idx_v_act.append(v0_j + k)
#     # n_act = len(idx_q_act)
#     # assert n_act == len(idx_v_act)

#     # print("\n" + "=" * 80)
#     # print("[LQR FULL-STATE] Linearization + equilibrium setup")
#     # print("=" * 80)
#     # print(f"  Total generalized coordinates n_q        = {n_q}")
#     # print(f"  Total generalized velocities n_v        = {n_v}")
#     # print(f"  Total full state dimension n_x_full    = {n_x_full}")
#     # print(f"  Number of actuators n_u                = {n_u}")
#     # print(f"  Number of actuated joint positions     = {n_act}")
#     # print("\n  Actuated joints (design model):")
#     # for k, name in enumerate(joint_names_act):
#     #     print(
#     #         f"    [{k:2d}] '{name}' "
#     #         f"(q index {idx_q_act[k]}, v index {idx_v_act[k]})"
#     #     )
#     # print("=" * 80 + "\n")

#     # # --- Check that feet are actually on the ground at the chosen q* ---
#     # check_foot_contacts_at_pose(plant, plant_context, model_instance_name="spot")

#     # # ------------------------------------------------------------------
#     # # 2) Define equilibrium torques u* (gravity-comp at q*, v* = 0)
#     # # ------------------------------------------------------------------
#     # derivs = plant.AllocateTimeDerivatives()

#     # # Gravity generalized forces (size n_v).  For each 1-DoF joint, we map
#     # # the corresponding entry of tau_g to that joint's actuator torque.
#     # tau_g = plant.CalcGravityGeneralizedForces(plant_context)
#     # u_star = np.zeros(n_u)
#     # for i in range(n_u):
#     #     actuator = plant.get_joint_actuator(JointActuatorIndex(i))
#     #     joint = actuator.joint()
#     #     assert joint.num_velocities() == 1, "Assuming 1-DoF joints for Spot"
#     #     v_index = joint.velocity_start()
#     #     u_star[i] = -tau_g[v_index]

#     # # Full equilibrium state
#     # x_full_star = np.concatenate([q_full_star, v_full_star])
    
    
#     # Nice standing pose with all feet on the ground and v = 0 (geometric guess).
#     q_full_guess, v_full_guess = get_default_standing_state(plant)
#     plant.SetPositions(plant_context, q_full_guess)
#     plant.SetVelocities(plant_context, v_full_guess)

#     n_q = plant.num_positions()
#     n_v = plant.num_velocities()
#     n_x_full = n_q + n_v
#     n_u = plant.num_actuators()

#     # Actuated joint indices (for logging and for S later)
#     idx_q_act = []
#     idx_v_act = []
#     joint_names_act = []
#     for i in range(plant.num_actuators()):
#         actuator = plant.get_joint_actuator(JointActuatorIndex(i))
#         joint = actuator.joint()
#         nq_j = joint.num_positions()
#         nv_j = joint.num_velocities()
#         q0_j = joint.position_start()
#         v0_j = joint.velocity_start()
#         for k in range(nq_j):
#             idx_q_act.append(q0_j + k)
#             joint_names_act.append(joint.name())
#         for k in range(nv_j):
#             idx_v_act.append(v0_j + k)
#     n_act = len(idx_q_act)
#     assert n_act == len(idx_v_act)

#     print("\n" + "=" * 80)
#     print("[LQR FULL-STATE] Linearization + equilibrium setup")
#     print("=" * 80)
#     print(f"  Total generalized coordinates n_q        = {n_q}")
#     print(f"  Total generalized velocities n_v        = {n_v}")
#     print(f"  Total full state dimension n_x_full    = {n_x_full}")
#     print(f"  Number of actuators n_u                = {n_u}")
#     print(f"  Number of actuated joint positions     = {n_act}")
#     print("\n  Actuated joints (design model):")
#     for k, name in enumerate(joint_names_act):
#         print(
#             f"    [{k:2d}] '{name}' "
#             f"(q index {idx_q_act[k]}, v index {idx_v_act[k]})"
#         )
#     print("=" * 80 + "\n")

#     # --- Check that feet are actually on the ground at the geometric guess ---
#     check_foot_contacts_at_pose(plant, plant_context, model_instance_name="spot")

#     # ------------------------------------------------------------------
#     # 2) Solve for a true static equilibrium (q*, v* = 0, u*)
#     # ------------------------------------------------------------------
#     q_full_star, v_full_star, u_star = solve_full_state_standing_equilibrium(
#         plant=plant,
#         plant_context=plant_context,
#         q_initial=q_full_guess,
#         v_initial=v_full_guess,
#         model_instance_name="spot",
#         ground_height=0.0,
#         verbose=True,
#     )

#     # Set context to the solved equilibrium
#     plant.SetPositions(plant_context, q_full_star)
#     plant.SetVelocities(plant_context, v_full_star)

#     # Full equilibrium state
#     x_full_star = np.concatenate([q_full_star, v_full_star])

#     # Allocate derivatives for later use in f_full
#     derivs = plant.AllocateTimeDerivatives()

    

#     def f_full(x_full: np.ndarray, u: np.ndarray) -> np.ndarray:
#         """
#         Continuous-time dynamics xdot = f(x, u) for the full floating-base
#         Spot model with contact.
#         """
#         assert x_full.shape[0] == n_x_full
#         assert u.shape[0] == n_u
#         q = x_full[:n_q]
#         v = x_full[n_q:]
#         plant.SetPositions(plant_context, q)
#         plant.SetVelocities(plant_context, v)
#         plant.get_actuation_input_port().FixValue(plant_context, u)
#         plant.CalcTimeDerivatives(plant_context, derivs)
#         return derivs.get_vector().CopyToVector()

#     xdot0_full = f_full(x_full_star, u_star)

#     print("[LQR FULL-STATE] Standing equilibrium used for LQR:")
#     print(f"  ||u_star||_inf          = {np.max(np.abs(u_star)):.3e} N·m")
#     print(f"  ||xdot0_full||          = {np.linalg.norm(xdot0_full):.3e}")
#     print(f"  max |qdot|              = {np.max(np.abs(xdot0_full[:n_q])):.3e}")
#     print(f"  max |vdot|              = {np.max(np.abs(xdot0_full[n_q:])):.3e}\n")

#     # ------------------------------------------------------------------
#     # 3) Numerical linearization around (x_full_star, u_star)
#     # ------------------------------------------------------------------
#     eps_x = 1e-6
#     eps_u = 1e-4

#     A_full = np.zeros((n_x_full, n_x_full))
#     B_full = np.zeros((n_x_full, n_u))

#     # A_full = ∂f/∂x
#     for i in range(n_x_full):
#         x_pert = x_full_star.copy()
#         x_pert[i] += eps_x
#         xdot_pert = f_full(x_pert, u_star)
#         A_full[:, i] = (xdot_pert - xdot0_full) / eps_x

#     # B_full = ∂f/∂u
#     for j in range(n_u):
#         u_pert = u_star.copy()
#         u_pert[j] += eps_u
#         xdot_pert = f_full(x_full_star, u_pert)
#         B_full[:, j] = (xdot_pert - xdot0_full) / eps_u

#     print(f"[LQR FULL-STATE] Numerical model: n_x_full = {n_x_full}, n_u = {n_u}")
#     print(f"[LQR FULL-STATE] ||xdot0_full|| at equilibrium ≈ "
#           f"{np.linalg.norm(xdot0_full):.3e}\n")

#     # ------------------------------------------------------------------
#     # 4) LQR weights on FULL state (for diagnostics; we'll reduce later)
#     # ------------------------------------------------------------------
#     Q_full = np.eye(n_x_full)

#     # Position indices
#     idx_quat = [0, 1, 2, 3]      # base orientation (quaternion)
#     idx_base_xy = [4, 5]         # base x, y
#     idx_base_z = [6]             # base z
#     idx_joint_pos = list(range(7, n_q))

#     # Velocity indices
#     idx_base_vel = list(range(n_q, n_q + 6))           # base angular + linear
#     idx_joint_vel = list(range(n_q + 6, n_q + n_v))    # joint velocities

#     base_pos_weight = 15.0
#     joint_pos_weight = 15.0
#     base_vel_weight = 10.0
#     joint_vel_weight = 10.0

#     # Base orientation + z
#     for i in idx_quat + idx_base_z:
#         Q_full[i, i] *= base_pos_weight

#     # Do NOT penalize x,y translation directly (unactuated symmetries)
#     for i in idx_base_xy:
#         Q_full[i, i] *= 0.0

#     # Joints
#     for i in idx_joint_pos:
#         Q_full[i, i] *= joint_pos_weight

#     # Base velocities
#     for i in idx_base_vel:
#         Q_full[i, i] *= base_vel_weight

#     # Joint velocities
#     for i in idx_joint_vel:
#         Q_full[i, i] *= joint_vel_weight

#     # Strong penalty on torque to keep gains moderate
#     R_scale = 10.0
#     R = np.eye(n_u) * R_scale

#     print("[LQR FULL-STATE] Cost weights:")
#     print(f"  Base orientation/z weight:   {base_pos_weight}")
#     print(f"  Joint position weight:       {joint_pos_weight}")
#     print(f"  Base velocity weight:        {base_vel_weight}")
#     print(f"  Joint velocity weight:       {joint_vel_weight}")
#     print(f"  Torque cost scaling (R_scale): {R_scale}")
#     print(f"  Q_full shape: {Q_full.shape}, R shape: {R.shape}")

#     # --- Eigenvalue analysis for FULL state ---
#     eigvals, eigvecs = np.linalg.eig(A_full)
#     print("\n[LQR FULL-STATE] Eigenvalues of A_full:")
#     for i, lam in enumerate(eigvals):
#         print(f"  mode {i:2d}: λ = {lam.real:+8.4e} + {lam.imag:+8.4e}j")

#     # --- Approximate controllability / observability per mode ---
#     print("\n[LQR FULL-STATE] Mode controllability / observability (full state):")
#     for i, lam in enumerate(eigvals):
#         v = eigvecs[:, i]
#         ctr = np.linalg.norm(B_full.T @ v)
#         obs = np.linalg.norm(Q_full @ v)
#         flag = ""
#         if lam.real >= -1e-6 and ctr < 1e-6:
#             flag += " UNCONTROLLABLE_NONSTABLE"
#         if lam.real >= -1e-6 and obs < 1e-6:
#             flag += " UNOBSERVABLE_NONSTABLE"
#         print(
#             f"  mode {i:2d}: λ = {lam.real:+8.4e} + {lam.imag:+8.4e}j, "
#             f"||B^T v|| = {ctr:.3e}, ||Q v|| = {obs:.3e}{flag}"
#         )

#     print("\n[LQR FULL-STATE] Dominant state components for uncontrollable nonstable modes (full state):")
#     bad_threshold_ctr = 1e-6
#     for i, lam in enumerate(eigvals):
#         v = eigvecs[:, i]
#         ctr = np.linalg.norm(B_full.T @ v)
#         if lam.real >= -1e-6 and ctr < bad_threshold_ctr:
#             print(f"\n  >>> Suspect mode {i} (λ = {lam})")
#             idxs = np.argsort(-np.abs(v))[:8]
#             for idx in idxs:
#                 print(f"    state {idx:2d}: v[{idx}] = {v[idx]:+8.4e}")

#     # ------------------------------------------------------------------
#     # 5) REDUCED-STATE LQR: drop base x,y and regularize bad zero modes
#     # ------------------------------------------------------------------
#     drop_state_indices = np.array([4, 5], dtype=int)
#     keep_state_indices = np.setdiff1d(np.arange(n_x_full), drop_state_indices)

#     A_red = A_full[np.ix_(keep_state_indices, keep_state_indices)]
#     B_red = B_full[keep_state_indices, :]
#     Q_red = Q_full[np.ix_(keep_state_indices, keep_state_indices)]

#     n_x_red = A_red.shape[0]
#     print("\n[LQR FULL-STATE] Reduced-state model for LQR:")
#     print(f"  Dropping base translation positions from LQR state: indices {drop_state_indices.tolist()}")
#     print(f"  n_x_red = {n_x_red}, n_u = {B_red.shape[1]}")

#     def describe_state_index_full(k: int) -> str:
#         """Helper to print which coordinate a full state index corresponds to."""
#         if k < n_q:
#             if k == 0:
#                 return "q[0] (base quat w)"
#             elif k in [1, 2, 3]:
#                 return f"q[{k}] (base quat imag)"
#             elif k == 4:
#                 return "q[4] (base x)"
#             elif k == 5:
#                 return "q[5] (base y)"
#             elif k == 6:
#                 return "q[6] (base z)"
#             else:
#                 return f"q[{k}] (joint position)"
#         else:
#             idx_v = k - n_q
#             if idx_v < 6:
#                 return f"v[{idx_v}] (base velocity)"
#             else:
#                 return f"v[{idx_v}] (joint velocity)"

#     # Eigen-analysis on reduced system
#     eigvals_red, eigvecs_red = np.linalg.eig(A_red)
#     print("\n[LQR FULL-STATE] Eigenvalues of A_red (reduced state):")
#     for i, lam in enumerate(eigvals_red):
#         print(f"  red mode {i:2d}: λ = {lam.real:+8.4e} + {lam.imag:+8.4e}j")

#     print("\n[LQR FULL-STATE] Mode controllability / observability (reduced state):")
#     for i, lam in enumerate(eigvals_red):
#         v = eigvecs_red[:, i]
#         ctr = np.linalg.norm(B_red.T @ v)
#         obs = np.linalg.norm(Q_red @ v)
#         flag = ""
#         if lam.real >= -1e-6 and ctr < 1e-6:
#             flag += " UNCONTROLLABLE_NONSTABLE"
#         if lam.real >= -1e-6 and obs < 1e-6:
#             flag += " UNOBSERVABLE_NONSTABLE"
#         print(
#             f"  red mode {i:2d}: λ = {lam.real:+8.4e} + {lam.imag:+8.4e}j, "
#             f"||B_red^T v|| = {ctr:.3e}, ||Q_red v|| = {obs:.3e}{flag}"
#         )

#     # --- Identify uncontrollable zero modes and regularize them ---
#     zero_tol = 1e-8
#     ctr_tol = 1e-6
#     leak_rate = 1e-3  # small negative leak

#     print("\n[LQR FULL-STATE] Regularizing uncontrollable zero modes in A_red (if any):")
#     leak_targets = []

#     for i, lam in enumerate(eigvals_red):
#         if abs(lam.real) < zero_tol and abs(lam.imag) < zero_tol:
#             v = eigvecs_red[:, i]
#             ctr = np.linalg.norm(B_red.T @ v)
#             if ctr < ctr_tol:
#                 # This eigenmode is (numerically) uncontrollable and zero
#                 j_max = int(np.argmax(np.abs(v)))
#                 full_idx = int(keep_state_indices[j_max])
#                 desc = describe_state_index_full(full_idx)
#                 leak_targets.append(j_max)
#                 print(
#                     f"  -> red mode {i}: λ ≈ 0, uncontrollable. "
#                     f"Dominant reduced state index {j_max} "
#                     f"(full index {full_idx}: {desc})."
#                 )

#     if leak_targets:
#         leak_targets = sorted(set(leak_targets))
#         for j in leak_targets:
#             full_idx = int(keep_state_indices[j])
#             desc = describe_state_index_full(full_idx)
#             print(f"     Applying leak: A_red[{j},{j}] -= {leak_rate:.1e} "
#                   f"(full index {full_idx}: {desc})")
#             A_red[j, j] -= leak_rate
#     else:
#         print("  No uncontrollable zero modes detected; no regularization applied.")

#     # Recompute eigenvalues after regularization
#     eigvals_red_reg, _ = np.linalg.eig(A_red)
#     print("\n[LQR FULL-STATE] Eigenvalues of A_red AFTER regularization:")
#     for i, lam in enumerate(eigvals_red_reg):
#         print(f"  red-reg mode {i:2d}: λ = {lam.real:+8.4e} + {lam.imag:+8.4e}j")

#     # ------------------------------------------------------------------
#     # 6) Continuous-time LQR on regularized REDUCED (A_red, B_red)
#     # ------------------------------------------------------------------
#     print("\n[LQR FULL-STATE] Computing continuous-time LQR on reduced state (A_red, B_red) ...")
#     try:
#         K_red, _ = LinearQuadraticRegulator(A_red, B_red, Q_red, R)
#         print("[LQR FULL-STATE] CARE on reduced system succeeded.")
#     except RuntimeError as e:
#         print("[LQR FULL-STATE] CARE STILL FAILED on (A_red, B_red).")
#         print("  ", e)
#         raise

#     # Embed K_red back into full state: columns for dropped indices are zero.
#     K_full = np.zeros((n_u, n_x_full))
#     K_full[:, keep_state_indices] = K_red

#     # Diagnostics on closed-loop full system
#     eig_cl = np.linalg.eigvals(A_full - B_full @ K_full)

#     print("\n[LQR FULL-STATE] Embedded LQR gain matrix K_full (full state):")
#     print(f"  K_full shape: {K_full.shape}")
#     print(f"  ‖K_full‖_∞ = {np.max(np.abs(K_full)):.3e}")
#     print("[LQR FULL-STATE] Closed-loop eigenvalues of (A_full - B_full K_full):")
#     for i, lam in enumerate(eig_cl):
#         print(f"  cl mode {i:2d}: λ = {lam.real:+8.4e} + {lam.imag:+8.4e}j")
#     print(f"  max Re(eig(A_full - B_full K_full)) = {np.max(eig_cl.real):.3e}")
#     print("=" * 80 + "\n")

#     return (
#         K_full,
#         x_full_star,
#         u_star,
#         n_x_full,
#         n_u,
#         idx_q_act,
#         idx_v_act,
#         joint_names_act,
#     )

def compute_full_joint_space_lqr_gain():
    """
    Full-state (floating-base) LQR around a clean standing pose:

      1) Choose a standing pose q* using get_default_standing_state so that
         all feet lie on the ground (base z shifted accordingly), with v* = 0.

      2) Solve for actuator torques u* that best support this fixed pose,
         by minimizing the generalized accelerations vdot at (q*, v=0, u).

      3) Numerically linearize xdot = f(x, u) around (x*, u*), where
           x* = [q*; v* = 0].

      4) Build a REDUCED state (dropping base x,y), regularize uncontrollable
         zero modes, solve LQR on the reduced system, then embed the gain
         back into the full 37D state.

    Returns:
        K_full:          (n_u, n_x_full) LQR gain (embedded from reduced state)
        x_full_star:     (n_x_full,) equilibrium state [q*; v* = 0]
        u_star:          (n_u,) equilibrium torques
        n_x_full:        full state dimension
        n_u:             number of actuators
        idx_q_act:       indices of actuated joint positions in q
        idx_v_act:       indices of actuated joint velocities in v
        joint_names_act: list of actuated joint names
    """
    # ------------------------------------------------------------------
    # 1) Build design diagram and choose standing pose (q*, v* = 0)
    # ------------------------------------------------------------------
    diagram, plant = build_spot_design_diagram(time_step=0.0)
    diagram_context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(diagram_context)

    # Geometric standing pose with all feet on the ground and v = 0.
    q_full_guess, v_full_guess = get_default_standing_state(plant)
    plant.SetPositions(plant_context, q_full_guess)
    plant.SetVelocities(plant_context, v_full_guess)

    n_q = plant.num_positions()
    n_v = plant.num_velocities()
    n_x_full = n_q + n_v
    n_u = plant.num_actuators()

    # Actuated joint indices (for logging and for S later)
    idx_q_act = []
    idx_v_act = []
    joint_names_act = []
    for i in range(plant.num_actuators()):
        actuator = plant.get_joint_actuator(JointActuatorIndex(i))
        joint = actuator.joint()
        nq_j = joint.num_positions()
        nv_j = joint.num_velocities()
        q0_j = joint.position_start()
        v0_j = joint.velocity_start()
        for k in range(nq_j):
            idx_q_act.append(q0_j + k)
            joint_names_act.append(joint.name())
        for k in range(nv_j):
            idx_v_act.append(v0_j + k)
    n_act = len(idx_q_act)
    assert n_act == len(idx_v_act)

    print("\n" + "=" * 80)
    print("[LQR FULL-STATE] Linearization + equilibrium setup")
    print("=" * 80)
    print(f"  Total generalized coordinates n_q        = {n_q}")
    print(f"  Total generalized velocities n_v        = {n_v}")
    print(f"  Total full state dimension n_x_full    = {n_x_full}")
    print(f"  Number of actuators n_u                = {n_u}")
    print(f"  Number of actuated joint positions     = {n_act}")
    print("\n  Actuated joints (design model):")
    for k, name in enumerate(joint_names_act):
        print(
            f"    [{k:2d}] '{name}' "
            f"(q index {idx_q_act[k]}, v index {idx_v_act[k]})"
        )
    print("=" * 80 + "\n")

    # --- Check that feet are actually on the ground at the geometric guess ---
    check_foot_contacts_at_pose(plant, plant_context, model_instance_name="spot")

    # ------------------------------------------------------------------
    # 2) Solve for torques u* that best support this FIXED pose
    #    (we DO NOT move q; we only adjust actuator torques).
    # ------------------------------------------------------------------
    q_full_star = q_full_guess.copy()
    v_full_star = np.zeros(n_v)

    # Gravity generalized forces as an initial guess for torques
    tau_g = plant.CalcGravityGeneralizedForces(plant_context)
    u0 = np.zeros(n_u)
    for i in range(n_u):
        actuator = plant.get_joint_actuator(JointActuatorIndex(i))
        joint = actuator.joint()
        assert joint.num_velocities() == 1, "Assuming 1-DoF joints for Spot"
        v_index = joint.velocity_start()
        u0[i] = -tau_g[v_index]

    u_star, vdot_star, xdot_full_star = solve_fixed_pose_joint_equilibrium(
        plant=plant,
        base_context=plant_context,
        q_standing=q_full_star,
        idx_v_act=idx_v_act,
        torque_init=u0,
        torque_reg=1e-4,
        verbose=True,
    )

    # Set context to the equilibrium pose (q*, v* = 0)
    plant.SetPositions(plant_context, q_full_star)
    plant.SetVelocities(plant_context, v_full_star)

    # Full equilibrium state
    x_full_star = np.concatenate([q_full_star, v_full_star])

    # Allocate derivatives for later use in f_full
    derivs = plant.AllocateTimeDerivatives()

    # ------------------------------------------------------------------
    # 3) Full dynamics function for numerical linearization
    # ------------------------------------------------------------------
    def f_full(x_full: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Continuous-time dynamics xdot = f(x, u) for the full floating-base
        Spot model with contact.
        """
        assert x_full.shape[0] == n_x_full
        assert u.shape[0] == n_u
        q = x_full[:n_q]
        v = x_full[n_q:]
        plant.SetPositions(plant_context, q)
        plant.SetVelocities(plant_context, v)
        plant.get_actuation_input_port().FixValue(plant_context, u)
        plant.CalcTimeDerivatives(plant_context, derivs)
        return derivs.get_vector().CopyToVector()

    xdot0_full = f_full(x_full_star, u_star)

    print("[LQR FULL-STATE] Standing equilibrium used for LQR:")
    print(f"  ||u_star||_inf          = {np.max(np.abs(u_star)):.3e} N·m")
    print(f"  ||xdot0_full||          = {np.linalg.norm(xdot0_full):.3e}")
    print(f"  max |qdot|              = {np.max(np.abs(xdot0_full[:n_q])):.3e}")
    print(f"  max |vdot|              = {np.max(np.abs(xdot0_full[n_q:])):.3e}\n")

    # ------------------------------------------------------------------
    # 4) Numerical linearization around (x_full_star, u_star)
    # ------------------------------------------------------------------
    eps_x = 1e-6
    eps_u = 1e-4

    A_full = np.zeros((n_x_full, n_x_full))
    B_full = np.zeros((n_x_full, n_u))

    # A_full = ∂f/∂x
    for i in range(n_x_full):
        x_pert = x_full_star.copy()
        x_pert[i] += eps_x
        xdot_pert = f_full(x_pert, u_star)
        A_full[:, i] = (xdot_pert - xdot0_full) / eps_x

    # B_full = ∂f/∂u
    for j in range(n_u):
        u_pert = u_star.copy()
        u_pert[j] += eps_u
        xdot_pert = f_full(x_full_star, u_pert)
        B_full[:, j] = (xdot_pert - xdot0_full) / eps_u

    print(f"[LQR FULL-STATE] Numerical model: n_x_full = {n_x_full}, n_u = {n_u}")
    print(f"[LQR FULL-STATE] ||xdot0_full|| at equilibrium ≈ "
          f"{np.linalg.norm(xdot0_full):.3e}\n")

    # ------------------------------------------------------------------
    # 5) LQR weights on FULL state (same as before)
    # ------------------------------------------------------------------
    Q_full = np.eye(n_x_full)

    # Position indices
    idx_quat = [0, 1, 2, 3]      # base orientation (quaternion)
    idx_base_xy = [4, 5]         # base x, y
    idx_base_z = [6]             # base z
    idx_joint_pos = list(range(7, n_q))

    # Velocity indices
    idx_base_vel = list(range(n_q, n_q + 6))           # base angular + linear
    idx_joint_vel = list(range(n_q + 6, n_q + n_v))    # joint velocities

    base_pos_weight = 15.0
    joint_pos_weight = 15.0
    base_vel_weight = 10.0
    joint_vel_weight = 10.0

    # Base orientation + z
    for i in idx_quat + idx_base_z:
        Q_full[i, i] *= base_pos_weight

    # Do NOT penalize x,y translation directly (unactuated symmetries)
    for i in idx_base_xy:
        Q_full[i, i] *= 0.0

    # Joints
    for i in idx_joint_pos:
        Q_full[i, i] *= joint_pos_weight

    # Base velocities
    for i in idx_base_vel:
        Q_full[i, i] *= base_vel_weight

    # Joint velocities
    for i in idx_joint_vel:
        Q_full[i, i] *= joint_vel_weight

    # Strong penalty on torque to keep gains moderate
    R_scale = 10.0
    R = np.eye(n_u) * R_scale

    print("[LQR FULL-STATE] Cost weights:")
    print(f"  Base orientation/z weight:   {base_pos_weight}")
    print(f"  Joint position weight:       {joint_pos_weight}")
    print(f"  Base velocity weight:        {base_vel_weight}")
    print(f"  Joint velocity weight:       {joint_vel_weight}")
    print(f"  Torque cost scaling (R_scale): {R_scale}")
    print(f"  Q_full shape: {Q_full.shape}, R shape: {R.shape}")

    # --- Eigenvalue analysis for FULL state ---
    eigvals, eigvecs = np.linalg.eig(A_full)
    print("\n[LQR FULL-STATE] Eigenvalues of A_full:")
    for i, lam in enumerate(eigvals):
        print(f"  mode {i:2d}: λ = {lam.real:+8.4e} + {lam.imag:+8.4e}j")

    # --- Approximate controllability / observability per mode ---
    print("\n[LQR FULL-STATE] Mode controllability / observability (full state):")
    for i, lam in enumerate(eigvals):
        v = eigvecs[:, i]
        ctr = np.linalg.norm(B_full.T @ v)
        obs = np.linalg.norm(Q_full @ v)
        flag = ""
        if lam.real >= -1e-6 and ctr < 1e-6:
            flag += " UNCONTROLLABLE_NONSTABLE"
        if lam.real >= -1e-6 and obs < 1e-6:
            flag += " UNOBSERVABLE_NONSTABLE"
        print(
            f"  mode {i:2d}: λ = {lam.real:+8.4e} + {lam.imag:+8.4e}j, "
            f"||B^T v|| = {ctr:.3e}, ||Q v|| = {obs:.3e}{flag}"
        )

    print("\n[LQR FULL-STATE] Dominant state components for uncontrollable nonstable modes (full state):")
    bad_threshold_ctr = 1e-6
    def describe_state_index_full(k: int) -> str:
        """Helper to print which coordinate a full state index corresponds to."""
        if k < n_q:
            if k == 0:
                return "q[0] (base quat w)"
            elif k in [1, 2, 3]:
                return f"q[{k}] (base quat imag)"
            elif k == 4:
                return "q[4] (base x)"
            elif k == 5:
                return "q[5] (base y)"
            elif k == 6:
                return "q[6] (base z)"
            else:
                return f"q[{k}] (joint position)"
        else:
            idx_v = k - n_q
            if idx_v < 6:
                return f"v[{idx_v}] (base velocity)"
            else:
                return f"v[{idx_v}] (joint velocity)"

    for i, lam in enumerate(eigvals):
        v = eigvecs[:, i]
        ctr = np.linalg.norm(B_full.T @ v)
        if lam.real >= -1e-6 and ctr < bad_threshold_ctr:
            print(f"\n  >>> Suspect mode {i} (λ = {lam})")
            idxs = np.argsort(-np.abs(v))[:8]
            for idx in idxs:
                print(f"    state {idx:2d}: v[{idx}] = {v[idx]:+8.4e}")

    # ------------------------------------------------------------------
    # 5b) REDUCED-STATE LQR: drop base x,y and regularize bad zero modes
    # ------------------------------------------------------------------
    drop_state_indices = np.array([4, 5], dtype=int)
    keep_state_indices = np.setdiff1d(np.arange(n_x_full), drop_state_indices)

    A_red = A_full[np.ix_(keep_state_indices, keep_state_indices)]
    B_red = B_full[keep_state_indices, :]
    Q_red = Q_full[np.ix_(keep_state_indices, keep_state_indices)]

    n_x_red = A_red.shape[0]
    print("\n[LQR FULL-STATE] Reduced-state model for LQR:")
    print(f"  Dropping base translation positions from LQR state: indices {drop_state_indices.tolist()}")
    print(f"  n_x_red = {n_x_red}, n_u = {B_red.shape[1]}")

    eigvals_red, eigvecs_red = np.linalg.eig(A_red)
    print("\n[LQR FULL-STATE] Eigenvalues of A_red (reduced state):")
    for i, lam in enumerate(eigvals_red):
        print(f"  red mode {i:2d}: λ = {lam.real:+8.4e} + {lam.imag:+8.4e}j")

    print("\n[LQR FULL-STATE] Mode controllability / observability (reduced state):")
    for i, lam in enumerate(eigvals_red):
        v = eigvecs_red[:, i]
        ctr = np.linalg.norm(B_red.T @ v)
        obs = np.linalg.norm(Q_red @ v)
        flag = ""
        if lam.real >= -1e-6 and ctr < 1e-6:
            flag += " UNCONTROLLABLE_NONSTABLE"
        if lam.real >= -1e-6 and obs < 1e-6:
            flag += " UNOBSERVABLE_NONSTABLE"
        print(
            f"  red mode {i:2d}: λ = {lam.real:+8.4e} + {lam.imag:+8.4e}j, "
            f"||B_red^T v|| = {ctr:.3e}, ||Q_red v|| = {obs:.3e}{flag}"
        )

    # --- Identify uncontrollable zero modes and regularize them ---
    zero_tol = 1e-8
    ctr_tol = 1e-6
    leak_rate = 1e-3  # small negative leak

    print("\n[LQR FULL-STATE] Regularizing uncontrollable zero modes in A_red (if any):")
    leak_targets = []

    for i, lam in enumerate(eigvals_red):
        if abs(lam.real) < zero_tol and abs(lam.imag) < zero_tol:
            v = eigvecs_red[:, i]
            ctr = np.linalg.norm(B_red.T @ v)
            if ctr < ctr_tol:
                # This eigenmode is (numerically) uncontrollable and zero
                j_max = int(np.argmax(np.abs(v)))
                full_idx = int(keep_state_indices[j_max])
                desc = describe_state_index_full(full_idx)
                leak_targets.append(j_max)
                print(
                    f"  -> red mode {i}: λ ≈ 0, uncontrollable. "
                    f"Dominant reduced state index {j_max} "
                    f"(full index {full_idx}: {desc})."
                )

    if leak_targets:
        leak_targets = sorted(set(leak_targets))
        for j in leak_targets:
            full_idx = int(keep_state_indices[j])
            desc = describe_state_index_full(full_idx)
            print(f"     Applying leak: A_red[{j},{j}] -= {leak_rate:.1e} "
                  f"(full index {full_idx}: {desc})")
            A_red[j, j] -= leak_rate
    else:
        print("  No uncontrollable zero modes detected; no regularization applied.")

    # Recompute eigenvalues after regularization
    eigvals_red_reg, _ = np.linalg.eig(A_red)
    print("\n[LQR FULL-STATE] Eigenvalues of A_red AFTER regularization:")
    for i, lam in enumerate(eigvals_red_reg):
        print(f"  red-reg mode {i:2d}: λ = {lam.real:+8.4e} + {lam.imag:+8.4e}j")

    # ------------------------------------------------------------------
    # 6) Continuous-time LQR on regularized REDUCED (A_red, B_red)
    # ------------------------------------------------------------------
    print("\n[LQR FULL-STATE] Computing continuous-time LQR on reduced state (A_red, B_red) ...")
    try:
        K_red, _ = LinearQuadraticRegulator(A_red, B_red, Q_red, R)
        print("[LQR FULL-STATE] CARE on reduced system succeeded.")
    except RuntimeError as e:
        print("[LQR FULL-STATE] CARE STILL FAILED on (A_red, B_red).")
        print("  ", e)
        raise

    # Embed K_red back into full state: columns for dropped indices are zero.
    K_full = np.zeros((n_u, n_x_full))
    K_full[:, keep_state_indices] = K_red

    # Diagnostics on closed-loop full system
    eig_cl = np.linalg.eigvals(A_full - B_full @ K_full)

    print("\n[LQR FULL-STATE] Embedded LQR gain matrix K_full (full state):")
    print(f"  K_full shape: {K_full.shape}")
    print(f"  ‖K_full‖_∞ = {np.max(np.abs(K_full)):.3e}")
    print("[LQR FULL-STATE] Closed-loop eigenvalues of (A_full - B_full K_full):")
    for i, lam in enumerate(eig_cl):
        print(f"  cl mode {i:2d}: λ = {lam.real:+8.4e} + {lam.imag:+8.4e}j")
    print(f"  max Re(eig(A_full - B_full K_full)) = {np.max(eig_cl.real):.3e}")
    print("=" * 80 + "\n")

    return (
        K_full,
        x_full_star,
        u_star,
        n_x_full,
        n_u,
        idx_q_act,
        idx_v_act,
        joint_names_act,
    )


def build_full_lqr_closed_loop_diagram(K_full,
                                       x_full_star,
                                       u_star,
                                       n_x_full,
                                       n_u,
                                       idx_q_act,
                                       idx_v_act):
    """
    Build closed-loop diagram for FULL-STATE LQR:
        u = u_star - K_full (x_full - x_full_star)
    """
    diagram, plant = build_spot_runtime_diagram(time_step=0.0)

    n_q = plant.num_positions()
    n_v = plant.num_velocities()
    assert n_x_full == n_q + n_v

    n_act = len(idx_q_act)
    n_x_joint = 2 * n_act

    # S maps full state [q; v] -> [q_act; v_act] for logging
    S = np.zeros((n_x_joint, n_x_full))
    for r, q_idx in enumerate(idx_q_act):
        S[r, q_idx] = 1.0
    for r, v_idx in enumerate(idx_v_act):
        S[n_act + r, n_q + v_idx] = 1.0

    builder = DiagramBuilder()
    spot_sys = builder.AddSystem(diagram)

    state_port = spot_sys.get_output_port(0)    # "x"
    actuation_port = spot_sys.get_input_port(0) # "actuation"

    assert state_port.size() == n_x_full
    assert actuation_port.size() == n_u

    # u = u_star - K_full (x_full - x_full_star)
    # y = -K_full x_full + (u_star + K_full x_full_star)
    D_affine = -K_full
    y0_affine = u_star + K_full @ x_full_star

    A_affine = np.zeros((0, 0))
    B_affine = np.zeros((0, n_x_full))
    f0_affine = np.zeros((0,))

    C_affine = np.zeros((n_u, 0))

    controller = AffineSystem(
        A=A_affine,
        B=B_affine,
        f0=f0_affine,
        C=C_affine,
        D=D_affine,
        y0=y0_affine,
    )
    controller_sys = builder.AddSystem(controller)

    builder.Connect(state_port, controller_sys.get_input_port(0))
    builder.Connect(controller_sys.get_output_port(0), actuation_port)

    # Joint names for logging
    joint_names = []
    for i in range(plant.num_actuators()):
        actuator = plant.get_joint_actuator(JointActuatorIndex(i))
        joint = actuator.joint()
        nq_j = joint.num_positions()
        for _ in range(nq_j):
            joint_names.append(joint.name())

    root_diagram = builder.Build()

    # Extract q*, v* from x_full_star (used to initialize plant state)
    q_star = x_full_star[:n_q]
    v_star = x_full_star[n_q:]

    return root_diagram, plant, q_star, v_star, S, joint_names


def run_full_lqr_sim_with_logging(
    root_diagram,
    plant,
    x_full_star,
    K_full,
    u_star,
    S,
    joint_names,
    t_final: float = 10.0,
    dt: float = 0.05,
    log_interval: float | None = None,
):
    """
    Closed-loop simulation for the FULL-STATE LQR controller:

        u = u_star - K_full (x_full - x_full_star)

    We still project to joint space via S for logging of joint errors.
    """
    import sys

    if log_interval is None:
        log_interval = dt

    simulator = Simulator(root_diagram)
    root_context = simulator.get_mutable_context()

    plant_context = plant.GetMyMutableContextFromRoot(root_context)
    n_q = plant.num_positions()
    n_v = plant.num_velocities()
    n_x_full_runtime = n_q + n_v

    assert x_full_star.shape[0] == n_x_full_runtime
    q_star = x_full_star[:n_q]
    v_star = x_full_star[n_q:]
    plant.SetPositions(plant_context, q_star)
    plant.SetVelocities(plant_context, v_star)

    simulator.set_target_realtime_rate(1.0)
    simulator.Initialize()

    # Dimension checks
    n_x_joint, n_x_full_S = S.shape
    assert n_x_full_S == n_x_full_runtime
    n_act = len(joint_names)
    assert n_x_joint == 2 * n_act

    # Nominal joint state for logging
    x_joint_star = S @ x_full_star
    q_act_star = x_joint_star[:n_act]
    v_act_star = x_joint_star[n_act:]

    print("=" * 80)
    print("LQR CLOSED-LOOP STANDING SIMULATION (FULL-STATE)", flush=True)
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
    print("  LQR controller (full-state) regulates deviations using:")
    print("    u(t) = u* - K_full (x_full(t) - x_full*)", flush=True)
    print("")

    sim_duration = t_final
    last_log_time = -1e9

    try:
        while True:
            root_context = simulator.get_mutable_context()
            t = root_context.get_time()

            if t >= sim_duration:
                print("\nSimulation finished.", flush=True)
                break

            plant_context = plant.GetMyMutableContextFromRoot(root_context)
            q = plant.GetPositions(plant_context)
            v = plant.GetVelocities(plant_context)
            x_full = np.concatenate([q, v])

            # Joint-space projection for logging
            x_joint = S @ x_full
            q_act = x_joint[:n_act]
            v_act = x_joint[n_act:]

            e_q = q_act - q_act_star
            e_v = v_act - v_act_star

            # Full-state feedback law
            u = u_star - K_full @ (x_full - x_full_star)

            if t - last_log_time >= log_interval:
                max_eq = float(np.max(np.abs(e_q)))
                max_ev = float(np.max(np.abs(e_v)))
                max_u = float(np.max(np.abs(u)))

                print(f"\n[LQR FULL-STATE] t = {t:6.3f} s")
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

            target_time = min(sim_duration, t + dt)
            simulator.AdvanceTo(target_time)

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.", flush=True)


# =============================================================================
# MAIN: choose between joint-space and full-state LQR from CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Spot standing LQR (joint-space or full-state)."
    )
    parser.add_argument(
        "--mode",
        choices=["joint", "full"],
        default="full",
        help="LQR design mode: 'joint' (current joint-space LQR, default) or "
             "'full' (floating-base full-state LQR).",
    )
    parser.add_argument(
        "--t_final",
        type=float,
        default=10.0,
        help="Simulation duration in seconds.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.05,
        help="Simulation logging step.",
    )
    parser.add_argument(
        "--log_interval",
        type=float,
        default=0.10,
        help="Logging interval in seconds.",
    )
    args = parser.parse_args()

    if args.mode == "joint":
        # 1) Design joint-space LQR on the contact-inclusive model
        (K_joint,
         x_full_star,
         x_joint_star,
         u_star,
         n_x_full,
         n_u,
         idx_q_act,
         idx_v_act) = compute_joint_space_lqr_gain()

        # 2) Build closed-loop runtime diagram (Spot + ground + Meshcat + joint LQR)
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

        # 3) Run simulation with continuous logging
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
            t_final=args.t_final,
            dt=args.dt,
            log_interval=args.log_interval,
        )

    else:
        # 1) Design FULL-STATE (floating-base) LQR
        (K_full,
         x_full_star,
         u_star,
         n_x_full,
         n_u,
         idx_q_act,
         idx_v_act,
         joint_names_design) = compute_full_joint_space_lqr_gain()

        # 2) Build closed-loop runtime diagram (Spot + ground + Meshcat + full-state LQR)
        (root_diagram,
         plant,
         q_star,
         v_star,
         S,
         joint_names_runtime) = build_full_lqr_closed_loop_diagram(
            K_full,
            x_full_star,
            u_star,
            n_x_full,
            n_u,
            idx_q_act,
            idx_v_act,
        )

        # 3) Run simulation with logging based on full state
        run_full_lqr_sim_with_logging(
            root_diagram=root_diagram,
            plant=plant,
            x_full_star=x_full_star,
            K_full=K_full,
            u_star=u_star,
            S=S,
            joint_names=joint_names_runtime,
            t_final=args.t_final,
            dt=args.dt,
            log_interval=args.log_interval,
        )


if __name__ == "__main__":
    main()
