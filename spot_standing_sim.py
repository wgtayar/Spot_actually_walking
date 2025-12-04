# Spot standing simulation with joint-level PD on Spot’s actuated joints.
#########################################################################
# Usage:
#   - To import disturbance helper from another script:
#       from spot_standing_sim import apply_first_joint_position_disturbance
#   - Run with no disturbance (just stand there):
#       python3 spot_standing_sim.py
#   - Run with a disturbance at 15 s, magnitude 0.6 rad:
#       python3 spot_standing_sim.py --disturbance_time 15.0 --disturbance_magnitude 0.6
#
# If the robot twitches and recovers, the PD is working. If it doesn’t move
# at all, either the gains are zero or the robot has achieved true inner peace.


import argparse
import numpy as np
from pydrake.all import (
    JointActuatorIndex,
    Simulator,
    StartMeshcat,
)

from spot_model import SpotModel

# Controller configuration (used for both SpotModel and logging)
JOINT_KP = 200.0
JOINT_KD = 20.0


def _describe_actuated_joints_for_instance(plant, model_instance):
    """Return list of (local_actuator_index, joint_name) for the given model_instance."""
    descriptions = []
    local_index = 0
    for i in range(plant.num_actuators()):
        actuator = plant.get_joint_actuator(JointActuatorIndex(i))
        if actuator.model_instance() != model_instance:
            continue
        joint = actuator.joint()
        descriptions.append((local_index, joint.name()))
        local_index += 1
    return descriptions

def compute_pd_torques_for_instance(
    plant,
    plant_context,
    model_instance,
    desired_state_port,
    kp: float,
    kd: float,
):
    # Current state
    q_full = plant.GetPositions(plant_context)
    v_full = plant.GetVelocities(plant_context)

    # Desired state as fixed on the desired_state_port
    x_des = desired_state_port.Eval(plant_context)

    # Count how many position / velocity DOFs are actuated for this model_instance
    n_q_act = 0
    n_v_act = 0
    for i in range(plant.num_actuators()):
        actuator = plant.get_joint_actuator(JointActuatorIndex(i))
        if actuator.model_instance() != model_instance:
            continue
        joint = actuator.joint()
        n_q_act += joint.num_positions()
        n_v_act += joint.num_velocities()

    assert x_des.shape[0] == n_q_act + n_v_act, (
        f"Desired state length {x_des.shape[0]} does not match "
        f"actuated dims {n_q_act + n_v_act}"
    )

    q_des = x_des[:n_q_act]
    v_des = x_des[n_q_act:]

    joint_names = []
    tau_values = []

    kq = 0
    kv = 0
    for i in range(plant.num_actuators()):
        actuator = plant.get_joint_actuator(JointActuatorIndex(i))
        if actuator.model_instance() != model_instance:
            continue

        joint = actuator.joint()
        nq_j = joint.num_positions()
        nv_j = joint.num_velocities()

        # Desired joint state for this actuator (in "actuated" ordering)
        q_des_j = q_des[kq : kq + nq_j]
        v_des_j = v_des[kv : kv + nv_j]

        # Actual joint state in the full plant state
        q_idx = joint.position_start()
        v_idx = joint.velocity_start()
        q_j = q_full[q_idx : q_idx + nq_j]
        v_j = v_full[v_idx : v_idx + nv_j]

        # PD torque: τ = Kp (q* - q) + Kd (v* - v)
        tau_j = kp * (q_des_j - q_j) + kd * (v_des_j - v_j)

        # For Spot, these are 1-DOF hinge joints; we log a scalar.
        tau_values.append(float(np.squeeze(tau_j)))
        joint_names.append(joint.name())

        kq += nq_j
        kv += nv_j

    return joint_names, np.array(tau_values)


def build_joint_desired_state(plant, plant_context, model_instance, desired_state_port):
    port_dim = desired_state_port.size()

    q_full = plant.GetPositions(plant_context)
    v_full = plant.GetVelocities(plant_context)

    q_des_list = []
    v_des_list = []

    for i in range(plant.num_actuators()):
        actuator = plant.get_joint_actuator(JointActuatorIndex(i))
        if actuator.model_instance() != model_instance:
            continue

        joint = actuator.joint()
        nq_j = joint.num_positions()
        nv_j = joint.num_velocities()

        q_start = joint.position_start()
        v_start = joint.velocity_start()

        q_des_list.extend(q_full[q_start : q_start + nq_j])
        v_des_list.extend(v_full[v_start : v_start + nv_j])

    q_des = np.array(q_des_list)
    v_des = np.array(v_des_list)

    x_des = np.concatenate([q_des, v_des])

    assert x_des.shape[0] == port_dim, (
        f"Desired state length {x_des.shape[0]} does not match port size {port_dim}"
    )

    return x_des


def apply_first_joint_position_disturbance(
    plant,
    plant_context,
    model_instance,
    magnitude: float = 0.2,
) -> str | None:

    q_full = plant.GetPositions(plant_context)
    disturbed_joint_name = None

    for i in range(plant.num_actuators()):
        actuator = plant.get_joint_actuator(JointActuatorIndex(i))
        if actuator.model_instance() != model_instance:
            continue
        joint = actuator.joint()
        q_start = joint.position_start()
        q_full[q_start] += magnitude
        disturbed_joint_name = joint.name()
        break  # only perturb the first actuated joint

    plant.SetPositions(plant_context, q_full)
    return disturbed_joint_name


def run_sim(disturbance_time: float | None, disturbance_magnitude: float):
    meshcat = StartMeshcat()

    # Enable the built-in joint PD controllers.
    model = SpotModel(
        time_step=1e-3,
        enable_joint_pd=True,
        joint_kp=JOINT_KP,
        joint_kd=JOINT_KD,
    )
    diagram = model.build_robot_diagram(meshcat=meshcat)

    simulator = Simulator(diagram)
    root_context = simulator.get_mutable_context()

    plant = model.plant
    model_instance = model.model_instance
    plant_context = plant.GetMyMutableContextFromRoot(root_context)

    # For pretty listing of actuated joints
    actuated_joint_descriptions = _describe_actuated_joints_for_instance(
        plant, model_instance
    )

    print("=" * 80)
    print("SPOT STANDING SIMULATION")
    print("=" * 80)
    print("  Controller:")
    print("    Type : Joint-level PD on Spot's actuated joints (MultibodyPlant internal)")
    print(f"    Gains: kp = {JOINT_KP:.1f}, kd = {JOINT_KD:.1f}")
    print("")
    print("  Actuated joints (for this Spot model instance):")
    for local_index, joint_name in actuated_joint_descriptions:
        print(f"    [{local_index:2d}] joint '{joint_name}'")
    print("=" * 80)

    desired_state_port = plant.get_desired_state_input_port(model_instance)

    x_des = build_joint_desired_state(
        plant, plant_context, model_instance, desired_state_port
    )

    # Fix the desired state input
    desired_state_port.FixValue(plant_context, x_des)

    print("\n  Desired joint state initialized from current configuration.")
    print("  PD controller will hold this pose unless disturbed.\n")

    simulator.Initialize()
    simulator.set_target_realtime_rate(1.0)

    dt = 0.02  # simulation step
    log_interval = 1.0  # seconds between torque logs
    last_log_time = None

    # python3 spot_standing_sim.py --disturbance_time 5.0 --disturbance_magnitude 0.1
    if disturbance_time is None:
        disturbance_applied = True  # never apply
        print("Running simulation with NO disturbance. Press Ctrl+C to stop.")
    else:
        disturbance_applied = False
        print(
            f"Running simulation. Disturbance will be applied at t = "
            f"{disturbance_time:.2f} s (magnitude = {disturbance_magnitude} rad)."
        )
        print("Press Ctrl+C to stop.")

    try:
        while True:
            # Always work with up-to-date contexts
            root_context = simulator.get_mutable_context()
            plant_context = plant.GetMyMutableContextFromRoot(root_context)

            current_time = root_context.get_time()
            target_time = current_time + dt

            if (not disturbance_applied) and (target_time >= disturbance_time):
                simulator.AdvanceTo(disturbance_time)

                # Refresh contexts after advancing
                root_context = simulator.get_mutable_context()
                plant_context = plant.GetMyMutableContextFromRoot(root_context)

                joint_name = apply_first_joint_position_disturbance(
                    plant,
                    plant_context,
                    model_instance,
                    magnitude=disturbance_magnitude,
                )

                if joint_name is not None:
                    print(
                        f"Applied disturbance at t = {disturbance_time:.2f} s "
                        f"to joint '{joint_name}' (+{disturbance_magnitude} rad)."
                    )
                else:
                    print("Warning: could not find an actuated joint to disturb.")

                # Log PD torques immediately after the disturbance
                joint_names, tau = compute_pd_torques_for_instance(
                    plant,
                    plant_context,
                    model_instance,
                    desired_state_port,
                    JOINT_KP,
                    JOINT_KD,
                )
                if tau.size > 0:
                    max_tau = float(np.max(np.abs(tau)))
                else:
                    max_tau = 0.0

                print("  Controller torques immediately after disturbance:")
                print(f"    Max |tau|: {max_tau:.3f} N·m")
                for name, tau_i in zip(joint_names, tau):
                    print(f"    {name:>24s}: tau = {tau_i:+.3f} N·m")

                disturbance_applied = True

            else:
                simulator.AdvanceTo(target_time)

                # Refresh contexts after advancing
                root_context = simulator.get_mutable_context()
                plant_context = plant.GetMyMutableContextFromRoot(root_context)

            # Periodic logging of PD torques during simulation
            t = root_context.get_time()
            if (last_log_time is None) or (t - last_log_time >= log_interval):
                joint_names, tau = compute_pd_torques_for_instance(
                    plant,
                    plant_context,
                    model_instance,
                    desired_state_port,
                    JOINT_KP,
                    JOINT_KD,
                )
                if tau.size > 0:
                    max_tau = float(np.max(np.abs(tau)))
                else:
                    max_tau = 0.0

                print(f"\n[Controller: Joint PD] t = {t:6.3f} s")
                print(f"  Max |tau| over actuated joints: {max_tau:.3f} N·m")
                for name, tau_i in zip(joint_names, tau):
                    print(f"    {name:>24s}: tau = {tau_i:+.3f} N·m")

                last_log_time = t

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")


def main():
    parser = argparse.ArgumentParser(
        description="Spot standing simulation with optional PD disturbance."
    )
    parser.add_argument(
        "--disturbance_time",
        type=float,
        default=None,
        help="Time [s] at which to apply a joint disturbance. "
             "If omitted, no disturbance is applied.",
    )
    parser.add_argument(
        "--disturbance_magnitude",
        type=float,
        default=0.2,
        help="Disturbance magnitude in radians (default: 0.2).",
    )
    args = parser.parse_args()

    run_sim(
        disturbance_time=args.disturbance_time,
        disturbance_magnitude=args.disturbance_magnitude,
    )


if __name__ == "__main__":
    main()
