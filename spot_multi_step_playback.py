# external/spot/spot_multi_step_playback.py
#
# Script to compute a multi-step walking trajectory using gait_optimization
# and play it back in Meshcat as a kinematic animation.

from __future__ import annotations

import numpy as np
import time

from pydrake.all import (
    Simulator,
    Parser,
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    StartMeshcat,
    MeshcatVisualizer,
    PiecewisePolynomial,
)

from spot_lqr_standing import (
    get_default_standing_state,
)
from spot_footstep_plan import (
    generate_straight_footstep_plan,
    get_leg_names,
)
from gait_optimization import gait_optimization

from underactuated import ConfigureParser


def build_optimization_plant():
    """
    Build a plant specifically for gait optimization.
    This needs to capture the model instance index.
    
    Returns:
        diagram, plant, spot_model_instance
    """
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant)
    
    ConfigureParser(parser)
    
    # Load Spot and capture the model instance
    (spot,) = parser.AddModelsFromUrl(
        "package://underactuated/models/spot/spot.dmd.yaml"
    )
    parser.AddModelsFromUrl(
        "package://underactuated/models/littledog/ground.urdf"
    )
    
    plant.Finalize()
    diagram = builder.Build()
    
    return diagram, plant, spot


def get_foot_positions(plant, plant_context):
    """
    Get the world positions of the four foot frames (lower leg frames).
    
    Args:
        plant: MultibodyPlant instance
        plant_context: Context for the plant
        
    Returns:
        foot_positions: numpy array of shape (4, 3) with positions in order:
                       [front_left, front_right, rear_left, rear_right]
    """
    world_frame = plant.world_frame()
    
    # Define the four foot frame names in order (using foot_center frames)
    foot_frame_names = [
        "front_left_foot_center",
        "front_right_foot_center",
        "rear_left_foot_center",
        "rear_right_foot_center"
    ]
    
    foot_positions = np.zeros((4, 3))
    
    for i, frame_name in enumerate(foot_frame_names):
        frame = plant.GetFrameByName(frame_name)
        p_WF = plant.CalcPointsPositions(
            plant_context,
            frame,
            np.array([[0.0], [0.0], [0.0]]),
            world_frame
        )
        foot_positions[i, :] = p_WF.flatten()
    
    return foot_positions


def compute_multi_step_trajectory(num_steps: int = 4):
    """
    Compute a multi-step walking trajectory using gait_optimization.
    
    Args:
        num_steps: Number of footsteps to execute
    
    Returns:
        t_sol: Time samples (numpy array)
        q_sol: Joint trajectory (PiecewisePolynomial)
        v_sol: Velocity trajectory (PiecewisePolynomial)
        q0: Initial joint configuration (numpy array)
    """
    print("=" * 80)
    print(f"Computing Multi-Step Trajectory ({num_steps} steps)")
    print("=" * 80)
    
    # Build plant for optimization (need model instance)
    print("\nBuilding optimization plant...")
    diagram, plant, spot_model = build_optimization_plant()
    
    # Create simulator and contexts
    simulator = Simulator(diagram)
    root_context = simulator.get_mutable_context()
    plant_context = plant.GetMyMutableContextFromRoot(root_context)
    
    # Set standing state
    print("Setting nominal standing state...")
    q_star, v_star = get_default_standing_state(plant)
    plant.SetPositions(plant_context, q_star)
    plant.SetVelocities(plant_context, v_star)
    
    # Save the initial configuration
    q0 = plant.GetPositions(plant_context).copy()
    
    # Get initial foot positions
    print("Getting initial foot positions...")
    foot_pos_array = get_foot_positions(plant, plant_context)
    
    # Compute ground height from mean foot z position (for information)
    ground_height = np.mean(foot_pos_array[:, 2])
    print(f"\nGround height (mean foot z): {ground_height:.4f}")
    
    print("\nInitial foot positions (our ordering: FL, FR, RL, RR):")
    leg_names = get_leg_names()
    for i, name in enumerate(leg_names):
        pos = foot_pos_array[i, :]
        print(f"  {name:25s}: [{pos[0]:8.4f}, {pos[1]:8.4f}, {pos[2]:8.4f}]")
    
    # Map our leg_index to gait_optimization index
    # Our ordering: FL (0), FR (1), RL (2), RR (3)
    # gait_optimization ordering: RB (0), RF (1), LF (2), LB (3)
    our_to_opt_index = {
        0: 2,  # front_left -> LF (index 2)
        1: 1,  # front_right -> RF (index 1)
        2: 3,  # rear_left -> LB (index 3)
        3: 0,  # rear_right -> RB (index 0)
    }
    
    # Storage for trajectory segments
    segment_times = []
    segment_q_traj = []
    segment_v_traj = []
    t_offset = 0.0
    
    # Set box_height (ground level)
    box_height = 0.0
    
    # Step parameters
    step_length = 0.15  # meters per step
    
    # Diagonal gait sequence: FL -> RR -> FR -> RL
    # Our leg indices: 0=FL, 1=FR, 2=RL, 3=RR
    gait_sequence = [0, 3, 1, 2]  # front_left, rear_right, front_right, rear_left
    
    print("\n" + "=" * 80)
    print("Optimizing trajectory for each step...")
    print("=" * 80)
    
    # Loop over each step, using the gait sequence
    for step_num in range(num_steps):
        print(f"\n--- Step {step_num + 1}/{num_steps} ---")
        
        # Print current foot positions before planning
        print(f"  Current foot positions:")
        for i, name in enumerate(leg_names):
            pos = foot_pos_array[i, :]
            print(f"    {name:15s}: x={pos[0]:7.4f}, y={pos[1]:7.4f}, z={pos[2]:7.4f}")
        
        # Select which leg moves based on gait sequence
        leg_index = gait_sequence[step_num % len(gait_sequence)]
        leg_name = leg_names[leg_index]
        
        # Compute target position for this leg
        target_position = foot_pos_array[leg_index].copy()
        target_position[0] += step_length  # Move forward in x
        
        # Create footstep manually
        from spot_footstep_plan import Footstep
        footstep = Footstep(
            leg_index=leg_index,
            leg_name=leg_name,
            target_position=target_position
        )
        
        print(f"  Selected leg from gait sequence:")
        print(f"    Leg index: {leg_index} ({leg_name})")
        print(f"    Target: x={footstep.target_position[0]:.4f}, "
              f"y={footstep.target_position[1]:.4f}, z={footstep.target_position[2]:.4f}")
        
        # Build next_foot array for gait_optimization
        # gait_optimization expects shape (4, 2) with ordering: RB, RF, LF, LB
        # Start with current positions (x, y only - ground plane coordinates)
        next_foot = np.zeros((4, 2))
        next_foot[0, :] = [foot_pos_array[3, 0], foot_pos_array[3, 1]]  # RB = rear_right
        next_foot[1, :] = [foot_pos_array[1, 0], foot_pos_array[1, 1]]  # RF = front_right
        next_foot[2, :] = [foot_pos_array[0, 0], foot_pos_array[0, 1]]  # LF = front_left
        next_foot[3, :] = [foot_pos_array[2, 0], foot_pos_array[2, 1]]  # LB = rear_left
        
        # Map footstep leg_index to gait_optimization index
        opt_foot_idx = our_to_opt_index[footstep.leg_index]
        
        # Update the stepping foot with target position
        next_foot[opt_foot_idx, 0] = footstep.target_position[0]  # Update x
        next_foot[opt_foot_idx, 1] = footstep.target_position[1]  # Update y
        
        opt_leg_names = ["rear_right (RB)", "front_right (RF)", 
                        "front_left (LF)", "rear_left (LB)"]
        print(f"  Stepping foot: {opt_leg_names[opt_foot_idx]}")
        
        # Debug: print body height and foot positions
        current_q = plant.GetPositions(plant_context)
        print(f"  Debug - Body height (q[6]): {current_q[6]:.4f}")
        print(f"  Debug - next_foot array:")
        for i, name in enumerate(opt_leg_names):
            print(f"    {name}: x={next_foot[i, 0]:.4f}, y={next_foot[i, 1]:.4f}")
        
        # Call gait_optimization for this step
        t_step, q_step, v_step, q_end = gait_optimization(
            plant,
            plant_context,
            spot_model,
            next_foot,
            opt_foot_idx,
            box_height
        )
        
        print(f"  Optimization complete: duration = {t_step[-1] - t_step[0]:.4f} s")
        
        # Store segment trajectories and time arrays
        segment_times.append(t_step.copy())
        segment_q_traj.append(q_step)
        segment_v_traj.append(v_step)
        
        # Update time offset for next segment
        t_offset += (t_step[-1] - t_step[0])
        
        # Update plant state to end of this step
        plant.SetPositions(plant_context, q_end)
        plant.SetVelocities(plant_context, v_step.value(t_step[-1]).flatten())
        
        # Recompute foot positions for next iteration
        foot_pos_array = get_foot_positions(plant, plant_context)
        
        print(f"  Updated foot positions after step:")
    
    print("\n" + "=" * 80)
    print("All steps optimized! Stitching trajectory segments...")
    print("=" * 80)
    
    # Stitch trajectories: concatenate times with proper offsets, skip duplicate endpoints
    t_sol_list = []
    q_samples_list = []
    v_samples_list = []
    cumulative_time = 0.0
    
    for i, (t_seg, q_traj, v_traj) in enumerate(zip(segment_times, segment_q_traj, segment_v_traj)):
        if i == 0:
            # First segment: include all samples
            t_sol_list.append(t_seg)
            q_samples = np.column_stack([q_traj.value(t) for t in t_seg])
            v_samples = np.column_stack([v_traj.value(t) for t in t_seg])
        else:
            # Subsequent segments: skip first sample (duplicates last of previous segment)
            t_sol_list.append(t_seg[1:] + cumulative_time)
            q_samples = np.column_stack([q_traj.value(t) for t in t_seg[1:]])
            v_samples = np.column_stack([v_traj.value(t) for t in t_seg[1:]])
        
        q_samples_list.append(q_samples)
        v_samples_list.append(v_samples)
        
        # Update cumulative time for next segment
        cumulative_time += (t_seg[-1] - t_seg[0])
    
    # Concatenate all time samples and trajectory samples
    t_sol = np.concatenate(t_sol_list)
    q_all = np.concatenate(q_samples_list, axis=1)
    v_all = np.concatenate(v_samples_list, axis=1)
    
    # Create new piecewise polynomials over the full time horizon
    q_sol = PiecewisePolynomial.FirstOrderHold(t_sol, q_all)
    v_sol = PiecewisePolynomial.FirstOrderHold(t_sol, v_all)
    
    print(f"  Total trajectory duration: {t_sol[-1]:.4f} s")
    print(f"  Total time samples: {len(t_sol)}")
    
    return t_sol, q_sol, v_sol, q0


def build_visualization_diagram():
    """
    Build a diagram with Meshcat visualization for kinematic playback.
    
    Returns:
        diagram: Built diagram
        plant: MultibodyPlant instance
        meshcat: Meshcat instance
        visualizer: MeshcatVisualizer instance
    """
    print("\n" + "=" * 80)
    print("Building visualization diagram...")
    print("=" * 80)
    
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant)
    
    ConfigureParser(parser)
    
    # Load Spot and ground
    parser.AddModelsFromUrl(
        "package://underactuated/models/spot/spot.dmd.yaml"
    )
    parser.AddModelsFromUrl(
        "package://underactuated/models/littledog/ground.urdf"
    )
    
    plant.Finalize()
    
    # Start Meshcat and add visualizer
    meshcat = StartMeshcat()
    visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat=meshcat)
    
    diagram = builder.Build()
    
    print(f"  Visualization diagram built")
    print(f"  Meshcat URL: {meshcat.web_url()}")
    
    return diagram, plant, meshcat, visualizer


def playback_trajectory(t_sol, q_sol, diagram, plant, meshcat, visualizer):
    """
    Play back the trajectory in Meshcat as a kinematic animation.
    
    Args:
        t_sol: Time samples (numpy array)
        q_sol: Joint trajectory (PiecewisePolynomial)
        diagram: Diagram with visualization
        plant: MultibodyPlant instance
        meshcat: Meshcat instance
        visualizer: MeshcatVisualizer instance
    """
    print("\n" + "=" * 80)
    print("Playing back trajectory in Meshcat...")
    print("=" * 80)
    
    # Create contexts ONCE at the start (reuse in the loop)
    root_context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(root_context)
    
    # Start recording using the visualizer
    visualizer.StartRecording()
    
    # Set up playback parameters
    t_start = t_sol[0]
    t_end = t_sol[-1]
    num_frames = 200  # More frames for longer trajectory
    time_samples = np.linspace(t_start, t_end, num_frames)
    playback_rate = 1.0  # Real-time playback
    frame_delay = (t_end - t_start) / num_frames * playback_rate
    
    print(f"  Trajectory duration: {t_end - t_start:.4f} s")
    print(f"  Number of frames: {num_frames}")
    print(f"  Frame delay: {frame_delay:.4f} s")
    print(f"  Playback rate: {playback_rate}x")
    print(f"\nPlaying trajectory...")
    
    # Get a foot frame for debugging
    front_left_foot = plant.GetFrameByName("front_left_foot_center")
    world_frame = plant.world_frame()
    
    # Playback loop
    for i, t in enumerate(time_samples):
        # Set the diagram time for this frame
        root_context.SetTime(float(t))
        
        # Evaluate trajectory at this time
        q = q_sol.value(t).flatten()
        
        # Set plant positions on the VISUALIZATION plant
        plant.SetPositions(plant_context, q)
        
        # Publish to Meshcat (this sends the pose update)
        diagram.ForcedPublish(root_context)
        
        # Delay for visualization
        time.sleep(frame_delay)
        
        # Progress indicator with debug info
        if (i + 1) % 40 == 0 or i == num_frames - 1 or i == 0:
            progress = (i + 1) / num_frames * 100
            # Debug: print configuration changes
            foot_pos = plant.CalcPointsPositions(
                plant_context,
                front_left_foot,
                np.array([[0.0], [0.0], [0.0]]),
                world_frame
            ).flatten()
            print(f"  Progress: {progress:.1f}% (frame {i + 1}/{num_frames}) | "
                  f"t = {t:.3f} s | "
                  f"body x = {q[4]:.3f} | "
                  f"FL foot z = {foot_pos[2]:.3f}")
    
    # Stop and publish recording using the visualizer
    visualizer.StopRecording()
    visualizer.PublishRecording()
    
    print("\n" + "=" * 80)
    print("Playback complete!")
    print("=" * 80)
    print(f"Recording published to Meshcat")
    print(f"You can replay it using the Meshcat controls")
    print(f"Meshcat URL: {meshcat.web_url()}")


def main():
    """
    Main function: compute multi-step trajectory and play it back in Meshcat.
    """
    # Compute the multi-step trajectory
    t_sol, q_sol, v_sol, q0 = compute_multi_step_trajectory(num_steps=10)
    
    # Build visualization diagram
    diagram, plant, meshcat, visualizer = build_visualization_diagram()
    
    # Set initial configuration in visualization
    print("\nSetting initial configuration in visualization...")
    root_context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(root_context)
    
    # Set initial configuration
    plant.SetPositions(plant_context, q0)
    diagram.ForcedPublish(root_context)
    
    print("\nWaiting 2 seconds before starting playback...")
    time.sleep(2.0)
    
    # Play back the trajectory
    playback_trajectory(t_sol, q_sol, diagram, plant, meshcat, visualizer)
    
    print("\n" + "=" * 80)
    print("Script complete!")
    print("=" * 80)
    print("The Meshcat window will remain open.")
    print("Press Ctrl+C to exit.")
    
    # Keep the script running so Meshcat stays open
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    main()
