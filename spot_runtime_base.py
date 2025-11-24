# external/spot/spot_runtime_base.py
#
# Minimal script to run Spot in a standing pose with zero actuation
# and print the positions of all frames containing "foot" in their name.

from __future__ import annotations

import numpy as np

from pydrake.all import (
    Simulator,
    FrameIndex,
)

from spot_lqr_standing import (
    build_spot_runtime_diagram,
    get_default_standing_state,
)
from spot_footstep_plan import (
    generate_straight_footstep_plan,
    print_footstep_plan,
)


def debug_print_all_frames(plant, plant_context):
    """
    Print the world positions of all frames in the plant.
    
    Args:
        plant: MultibodyPlant instance
        plant_context: Context for the plant
    """
    world_frame = plant.world_frame()
    
    print("\nListing all frames and their world positions:")
    print("=" * 80)
    
    # Iterate through all frames
    for i in range(plant.num_frames()):
        frame = plant.get_frame(FrameIndex(i))
        frame_name = frame.name()
        
        # Compute position of frame origin in world frame
        p_WF = plant.CalcPointsPositions(
            plant_context,
            frame,
            np.array([[0.0], [0.0], [0.0]]),  # Origin of the frame
            world_frame
        )
        
        # p_WF is a 3x1 matrix, extract as 1D array
        position = p_WF.flatten()
        
        print(f"  [{i:3d}] {frame_name:35s}: [{position[0]:8.4f}, {position[1]:8.4f}, {position[2]:8.4f}]")
    
    print("=" * 80)


def get_foot_positions(plant, plant_context):
    """
    Get the world positions of the four foot frames (lower leg frames).
    
    Args:
        plant: MultibodyPlant instance
        plant_context: Context for the plant
        
    Returns:
        foot_positions: numpy array of shape (4, 3) with positions in order:
                       [front_left, front_right, rear_left, rear_right]
        foot_dict: dict mapping frame name to 3D position
    """
    world_frame = plant.world_frame()
    
    # Define the four foot frame names in order
    foot_frame_names = [
        "front_left_lower_leg",
        "front_right_lower_leg",
        "rear_left_lower_leg",
        "rear_right_lower_leg"
    ]
    
    foot_positions = np.zeros((4, 3))
    foot_dict = {}
    
    for i, frame_name in enumerate(foot_frame_names):
        # Get the frame by name
        frame = plant.GetFrameByName(frame_name)
        
        # Compute position of frame origin in world frame
        p_WF = plant.CalcPointsPositions(
            plant_context,
            frame,
            np.array([[0.0], [0.0], [0.0]]),  # Origin of the frame
            world_frame
        )
        
        # p_WF is a 3x1 matrix, extract as 1D array
        position = p_WF.flatten()
        
        foot_positions[i, :] = position
        foot_dict[frame_name] = position
    
    return foot_positions, foot_dict


def print_foot_positions(plant, plant_context):
    """
    Print the world positions of the four foot frames (lower leg frames).
    
    Args:
        plant: MultibodyPlant instance
        plant_context: Context for the plant
    """
    foot_positions, foot_dict = get_foot_positions(plant, plant_context)
    
    print("\nFoot positions at standing pose:")
    print("-" * 60)
    
    # Define the four foot frame names in the same order
    foot_frame_names = [
        "front_left_lower_leg",
        "front_right_lower_leg",
        "rear_left_lower_leg",
        "rear_right_lower_leg"
    ]
    
    for i, frame_name in enumerate(foot_frame_names):
        pos = foot_positions[i, :]
        print(f"  {frame_name:25s}: [{pos[0]:8.4f}, {pos[1]:8.4f}, {pos[2]:8.4f}]")
    
    print("-" * 60)


def main():
    print("Building Spot runtime diagram...")
    diagram, plant = build_spot_runtime_diagram(time_step=0.0)
    print(f"  Plant has {plant.num_positions()} positions, "
          f"{plant.num_velocities()} velocities, "
          f"{plant.num_actuators()} actuators")
    
    # Create simulator and contexts
    simulator = Simulator(diagram)
    root_context = simulator.get_mutable_context()
    plant_context = plant.GetMyMutableContextFromRoot(root_context)
    
    # Set standing state
    print("\nSetting nominal standing state...")
    q_star, v_star = get_default_standing_state(plant)
    plant.SetPositions(plant_context, q_star)
    plant.SetVelocities(plant_context, v_star)
    
    # Fix actuation to zero
    print("Fixing actuation input to zero...")
    n_u = plant.num_actuators()
    actuation_port = diagram.get_input_port(0)
    
    # Verify port size matches number of actuators
    assert actuation_port.size() == n_u, \
        f"Actuation port size {actuation_port.size()} != num_actuators {n_u}"
    
    u_zero = np.zeros(n_u)
    actuation_port.FixValue(root_context, u_zero)
    
    # Debug: print all frames and their positions
    # debug_print_all_frames(plant, plant_context)
    
    # Print foot positions at the standing pose
    print_foot_positions(plant, plant_context)
    
    # Verify foot positions array
    foot_pos_array, foot_dict = get_foot_positions(plant, plant_context)
    print(f"\nFoot positions array shape: {foot_pos_array.shape}")
    
    # Generate and print a sample footstep plan
    print("\n" + "=" * 80)
    print("Generating footstep plan for straight-line walking...")
    print("=" * 80)
    
    total_forward_distance = 0.6  # meters
    step_length = 0.15  # meters
    
    footsteps = generate_straight_footstep_plan(
        foot_pos_array,
        total_forward_distance,
        step_length
    )
    
    print(f"\nGenerated {len(footsteps)} footsteps")
    print(f"  Total forward distance: {total_forward_distance} m")
    print(f"  Step length: {step_length} m")
    
    # Print first 6 steps for inspection
    print_footstep_plan(footsteps, max_steps=6)
    
    # Run simulation with visualization
    print("\nRunning simulation for 2.0 seconds...")
    simulator.set_target_realtime_rate(1.0)
    simulator.Initialize()
    simulator.AdvanceTo(2.0)
    
    print("\nSimulation complete!")


if __name__ == "__main__":
    main()
