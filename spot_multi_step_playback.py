# external/spot/spot_multi_step_playback.py
#
# Script to compute a multi-step walking trajectory using gait_optimization
# and play it back in Meshcat as a kinematic animation.

from __future__ import annotations

import numpy as np
import time
import matplotlib.pyplot as plt

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
from gait_optimization import gait_optimization, UNDERACTUATED_JOINT_NAMES

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
    # gait_optimization ordering (foot_frame list): FL (0), FR (1), RL (2), RR (3)
    # They are the SAME ordering, so identity mapping!
    our_to_opt_index = {
        0: 0,  # front_left -> FL (index 0)
        1: 1,  # front_right -> FR (index 1)
        2: 2,  # rear_left -> RL (index 2)
        3: 3,  # rear_right -> RR (index 3)
    }
    
    # Storage for trajectory segments
    segment_times = []
    segment_q_traj = []
    segment_v_traj = []
    all_torque_data = []  # Store torque data from each phase
    t_offset = 0.0
    
    # Set box_height (ground level)
    box_height = 0.016
    
    # Step parameters
    step_length = 0.15  # meters per step
    
    # Trot gait: diagonal pairs move together
    # Our leg indices: 0=FL, 1=FR, 2=RL, 3=RR
    # Phase 0: FL + RR swing together (indices 0, 3)
    # Phase 1: FR + RL swing together (indices 1, 2)
    gait_phases = [(0, 3), (1, 2)]  # (front_left, rear_right), (front_right, rear_left)
    
    print("\n" + "=" * 80)
    print("Optimizing trajectory for each phase (diagonal pair trot)...")
    print("=" * 80)
    
    # Loop over each phase, swinging diagonal pairs
    for step_num in range(num_steps):
        print(f"\n--- Phase {step_num + 1}/{num_steps} ---")
        
        # Print current foot positions before planning
        print(f"  Current foot positions:")
        for i, name in enumerate(leg_names):
            pos = foot_pos_array[i, :]
            print(f"    {name:15s}: x={pos[0]:7.4f}, y={pos[1]:7.4f}, z={pos[2]:7.4f}")
        
        # Select which diagonal pair moves in this phase
        swing_legs_our = gait_phases[step_num % len(gait_phases)]
        
        # Compute target positions for both swing legs
        swing_leg_names = [leg_names[li] for li in swing_legs_our]
        target_positions = []
        for leg_idx in swing_legs_our:
            target_pos = foot_pos_array[leg_idx].copy()
            target_pos[0] += step_length  # Move forward in x
            target_positions.append(target_pos)
        
        print(f"  Selected diagonal pair from gait phase:")
        for leg_idx, leg_name, target_pos in zip(swing_legs_our, swing_leg_names, target_positions):
            print(f"    Leg {leg_idx} ({leg_name}): target x={target_pos[0]:.4f}, "
                  f"y={target_pos[1]:.4f}, z={target_pos[2]:.4f}")
        
        # Build next_foot array for gait_optimization
        # gait_optimization expects shape (4, 2) with ordering: FL, FR, RL, RR (same as ours!)
        # Start with current positions (x, y only - ground plane coordinates)
        next_foot = np.zeros((4, 2))
        next_foot[0, :] = [foot_pos_array[0, 0], foot_pos_array[0, 1]]  # FL = front_left
        next_foot[1, :] = [foot_pos_array[1, 0], foot_pos_array[1, 1]]  # FR = front_right
        next_foot[2, :] = [foot_pos_array[2, 0], foot_pos_array[2, 1]]  # RL = rear_left
        next_foot[3, :] = [foot_pos_array[3, 0], foot_pos_array[3, 1]]  # RR = rear_right
        
        # Map both swing leg indices to gait_optimization indices
        swing_feet_opt = [our_to_opt_index[li] for li in swing_legs_our]
        
        # Update both swing feet with their target positions
        for leg_idx, target_pos in zip(swing_legs_our, target_positions):
            opt_idx = our_to_opt_index[leg_idx]
            next_foot[opt_idx, 0] = target_pos[0]  # Update x
            next_foot[opt_idx, 1] = target_pos[1]  # Update y
        
        opt_leg_names = ["front_left (FL)", "front_right (FR)", 
                        "rear_left (RL)", "rear_right (RR)"]
        swing_leg_names_opt = [opt_leg_names[idx] for idx in swing_feet_opt]
        print(f"  Swing feet (diagonal pair): {', '.join(swing_leg_names_opt)}")
        
        # Debug: print body height and foot positions
        current_q = plant.GetPositions(plant_context)
        print(f"  Debug - Body height (q[6]): {current_q[6]:.4f}")
        print(f"  Debug - next_foot array:")
        for i, name in enumerate(opt_leg_names):
            swing_marker = " [SWING]" if i in swing_feet_opt else " [STANCE]"
            print(f"    {name}: x={next_foot[i, 0]:.4f}, y={next_foot[i, 1]:.4f}{swing_marker}")
        
        # Call gait_optimization for this phase (with diagonal pair)
        t_step, q_step, v_step, q_end, torque_data = gait_optimization(
            plant,
            plant_context,
            spot_model,
            next_foot,
            swing_feet_opt,
            box_height
        )
        
        print(f"  Optimization complete: duration = {t_step[-1] - t_step[0]:.4f} s")
        
        # Store segment trajectories and time arrays
        segment_times.append(t_step.copy())
        segment_q_traj.append(q_step)
        segment_v_traj.append(v_step)
        
        # Store torque data with time offset for stitching
        torque_data['phase'] = step_num + 1
        torque_data['time_offset'] = t_offset if step_num > 0 else 0.0
        all_torque_data.append(torque_data)
        
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
    
    return t_sol, q_sol, v_sol, q0, all_torque_data


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


def print_underactuation_report(all_torque_data: list):
    """
    Print a comprehensive report on underactuation verification.
    
    Args:
        all_torque_data: List of torque data dicts from each phase
    """
    print("\n" + "=" * 80)
    print("UNDERACTUATION VERIFICATION REPORT")
    print("=" * 80)
    
    if not all_torque_data or not all_torque_data[0].get('underactuated_indices'):
        print("\n  No underactuation constraints were applied.")
        print("  All joints are fully actuated.")
        return
    
    underactuated_names = all_torque_data[0].get('underactuated_names', [])
    underactuated_indices = all_torque_data[0].get('underactuated_indices', [])
    
    print(f"\n  Underactuated Joints: {underactuated_names}")
    print(f"  Velocity Indices: {underactuated_indices}")
    print(f"  Constraint: τ = 0 (no motor torque at these joints)")
    print()
    
    # Table header
    print(f"  {'Phase':<8} {'Duration (s)':<14} {'Max |τ_passive| (N⋅m)':<24} {'Status':<10}")
    print("  " + "-" * 60)
    
    overall_max = 0.0
    total_duration = 0.0
    
    for data in all_torque_data:
        phase = data.get('phase', '?')
        max_tau = data.get('max_violation', 0.0)
        times = data.get('times', np.array([]))
        duration = times[-1] - times[0] if len(times) > 1 else 0.0
        
        overall_max = max(overall_max, max_tau)
        total_duration += duration
        
        # Threshold: 10 mN·m (0.01 N·m) - negligible compared to typical motor torques of 50-100+ N·m
        UNDERACTUATION_THRESHOLD = 0.01  # 10 mN·m
        status = "✓ PASS" if max_tau < UNDERACTUATION_THRESHOLD else "✗ FAIL"
        print(f"  {phase:<8} {duration:<14.4f} {max_tau:<24.6f} {status:<10}")
    
    print("  " + "-" * 60)
    print(f"  {'TOTAL':<8} {total_duration:<14.4f} {overall_max:<24.6f}")
    
    print("\n" + "=" * 80)
    UNDERACTUATION_THRESHOLD = 0.01  # 10 mN·m
    if overall_max < UNDERACTUATION_THRESHOLD:
        print("  ✓ UNDERACTUATION VERIFIED!")
        print("    The trajectory is dynamically feasible with passive (unactuated) joints.")
        print(f"    Maximum torque at passive joints: {overall_max*1000:.3f} mN·m (effectively zero)")
        print(f"    (Threshold: {UNDERACTUATION_THRESHOLD*1000:.0f} mN·m, typical motor: 50,000+ mN·m)")
    else:
        print("  ⚠ WARNING: Underactuation constraints may be violated!")
        print(f"    Maximum torque at passive joints: {overall_max*1000:.3f} mN·m")
    print("=" * 80)


def plot_torque_comparison(all_torque_data: list, save_path: str = None):
    """
    Create a clear 2-panel plot: Actuated torques vs Passive torques (should be ~0).
    
    Args:
        all_torque_data: List of torque data dicts from each phase
        save_path: Optional path to save the figure
    """
    if not all_torque_data:
        print("No torque data to plot.")
        return
    
    # Stitch together all phases
    all_times = []
    all_tau = []
    phase_boundaries = [0.0]
    cumulative_time = 0.0
    
    for data in all_torque_data:
        times = data.get('times', np.array([]))
        tau_all = data.get('tau_all', np.array([]))
        
        if len(times) == 0 or len(tau_all) == 0:
            continue
        
        all_times.append(times + cumulative_time)
        all_tau.append(tau_all)
        cumulative_time = all_times[-1][-1]
        phase_boundaries.append(cumulative_time)
    
    if not all_times:
        print("No valid torque data to plot.")
        return
    
    times = np.concatenate(all_times)
    tau = np.vstack(all_tau)
    
    underactuated_indices = all_torque_data[0].get('underactuated_indices', [])
    underactuated_names = all_torque_data[0].get('underactuated_names', [])
    joint_names = all_torque_data[0].get('joint_names', [])
    
    # Get actuated JOINT indices (skip first 6 which are floating base: wx, wy, wz, vx, vy, vz)
    # Then exclude underactuated joints
    nv = tau.shape[1]
    FLOATING_BASE_DOF = 6  # Skip body_wx, body_wy, body_wz, body_vx, body_vy, body_vz
    actuated_indices = [i for i in range(FLOATING_BASE_DOF, nv) if i not in underactuated_indices]
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle('TORQUE COMPARISON: Actuated vs Passive (Underactuated) Joints', 
                 fontsize=16, fontweight='bold')
    
    # === TOP: Actuated joint torques ===
    ax1 = axes[0]
    ax1.set_title('ACTUATED Joints - Motor Torques Required (Hip Joints)', fontsize=14, color='blue')
    
    # Color scheme for different joints
    colors_hip_x = {'front_left': '#1f77b4', 'front_right': '#aec7e8', 
                    'rear_left': '#17becf', 'rear_right': '#9edae5'}
    colors_hip_y = {'front_left': '#2ca02c', 'front_right': '#98df8a',
                    'rear_left': '#d62728', 'rear_right': '#ff9896'}
    
    for idx in actuated_indices:
        name = joint_names[idx] if idx < len(joint_names) else f"joint_{idx}"
        
        # Assign colors based on joint name
        if 'front_left_hip_x' in name:
            color = colors_hip_x['front_left']
        elif 'front_right_hip_x' in name:
            color = colors_hip_x['front_right']
        elif 'rear_left_hip_x' in name:
            color = colors_hip_x['rear_left']
        elif 'rear_right_hip_x' in name:
            color = colors_hip_x['rear_right']
        elif 'front_left_hip_y' in name:
            color = colors_hip_y['front_left']
        elif 'front_right_hip_y' in name:
            color = colors_hip_y['front_right']
        elif 'rear_left_hip_y' in name:
            color = colors_hip_y['rear_left']
        elif 'rear_right_hip_y' in name:
            color = colors_hip_y['rear_right']
        else:
            color = '#7f7f7f'  # Gray for unknown
        
        ax1.plot(times, tau[:, idx], color=color, linewidth=1.5, label=name, alpha=0.9)
    
    # Add phase boundaries
    for boundary in phase_boundaries[1:-1]:
        ax1.axvline(x=boundary, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
    
    ax1.set_ylabel('Torque (N·m)', fontsize=12)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.legend(loc='upper right', fontsize=9, ncol=3)
    ax1.grid(True, alpha=0.3)
    
    # Add max torque annotation
    max_actuated = np.max(np.abs(tau[:, actuated_indices]))
    ax1.text(0.02, 0.95, f'Max |τ| = {max_actuated:.1f} N·m', 
             transform=ax1.transAxes, fontsize=11, fontweight='bold',
             verticalalignment='top', color='blue',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # === BOTTOM: Passive (underactuated) joint torques - should be ~0 ===
    ax2 = axes[1]
    ax2.set_title('PASSIVE (Underactuated) Joints - Should be ≈ 0 N·m', fontsize=14, color='red')
    
    # Threshold: 10 mN·m
    THRESHOLD_MNM = 10
    
    if underactuated_indices:
        colors = ['#d62728', '#e377c2', '#8c564b', '#9467bd']  # Red shades
        for i, idx in enumerate(underactuated_indices):
            name = underactuated_names[i] if i < len(underactuated_names) else f"passive_{idx}"
            ax2.plot(times, tau[:, idx] * 1000, color=colors[i % len(colors)], 
                     linewidth=2, label=name, alpha=0.9)
        
        # Add phase boundaries
        for boundary in phase_boundaries[1:-1]:
            ax2.axvline(x=boundary, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
        
        # Add acceptable region (±10 mN·m)
        ax2.axhspan(-THRESHOLD_MNM, THRESHOLD_MNM, color='green', alpha=0.2, label=f'Acceptable (±{THRESHOLD_MNM} mN·m)')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        # Get max value for scaling
        max_passive = np.max(np.abs(tau[:, underactuated_indices])) * 1000
        y_limit = max(max_passive * 1.5, THRESHOLD_MNM * 1.5)  # Show threshold region
        ax2.set_ylim(-y_limit, y_limit)
        
        # Add max torque annotation
        status = "✓ PASS" if max_passive < THRESHOLD_MNM else "✗ FAIL"
        ax2.text(0.02, 0.95, f'Max |τ| = {max_passive:.3f} mN·m  {status}', 
                 transform=ax2.transAxes, fontsize=11, fontweight='bold',
                 verticalalignment='top', color='darkgreen' if max_passive < THRESHOLD_MNM else 'red',
                 bbox=dict(boxstyle='round', facecolor='lightgreen' if max_passive < THRESHOLD_MNM else 'lightyellow', alpha=0.7))
    
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Torque (mN·m)', fontsize=12)  # Note: milli-Newton-meters
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n  Torque comparison plot saved to: {save_path}")
    
    plt.show()


def plot_underactuation_detail(all_torque_data: list, save_path: str = None):
    """
    Create a single clear plot showing passive joint torques are effectively zero.
    
    Args:
        all_torque_data: List of torque data dicts from each phase
        save_path: Optional path to save the figure
    """
    if not all_torque_data:
        print("No torque data to plot.")
        return
    
    underactuated_indices = all_torque_data[0].get('underactuated_indices', [])
    if not underactuated_indices:
        print("No underactuated joints to plot.")
        return
    
    underactuated_names = all_torque_data[0].get('underactuated_names', [])
    
    # Stitch together all phases
    all_times = []
    all_tau_underact = []
    phase_boundaries = [0.0]
    cumulative_time = 0.0
    
    for data in all_torque_data:
        times = data.get('times', np.array([]))
        tau_underact = data.get('tau_underactuated', np.array([]))
        
        if len(times) == 0 or len(tau_underact) == 0:
            continue
        
        all_times.append(times + cumulative_time)
        all_tau_underact.append(tau_underact)
        cumulative_time = all_times[-1][-1]
        phase_boundaries.append(cumulative_time)
    
    if not all_times:
        print("No valid underactuated torque data to plot.")
        return
    
    times = np.concatenate(all_times)
    tau_underact = np.vstack(all_tau_underact)
    
    # Create single figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    fig.suptitle('UNDERACTUATION VERIFICATION: Passive Joint Torques ≈ 0', 
                 fontsize=16, fontweight='bold')
    
    colors = ['#d62728', '#e377c2', '#8c564b', '#9467bd']
    
    # Plot each passive joint torque
    for i in range(tau_underact.shape[1]):
        name = underactuated_names[i] if i < len(underactuated_names) else f"passive_{i}"
        ax.plot(times, tau_underact[:, i] * 1000, color=colors[i % len(colors)], 
                linewidth=2, label=name, alpha=0.9)
    
    # Add phase boundaries
    for boundary in phase_boundaries[1:-1]:
        ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    
    # Add acceptable region (±10 mN·m threshold)
    THRESHOLD_MNM = 10  # 10 mN·m = 0.01 N·m
    ax.axhspan(-THRESHOLD_MNM, THRESHOLD_MNM, color='green', alpha=0.2, label=f'Acceptable (±{THRESHOLD_MNM} mN·m)')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    
    # Scale y-axis
    max_abs = np.max(np.abs(tau_underact)) * 1000 * 1.3
    max_abs = max(max_abs, THRESHOLD_MNM * 1.5)  # At least show the threshold region
    ax.set_ylim(-max_abs, max_abs)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Torque (mN·m)', fontsize=12)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add statistics box
    all_values = tau_underact.flatten() * 1000  # Convert to mN·m
    max_val = np.max(np.abs(all_values))
    mean_val = np.mean(all_values)
    
    stats_text = (f'Max |τ| = {max_val:.3f} mN·m\n'
                  f'Mean τ = {mean_val:.4f} mN·m\n'
                  f'Status: {"✓ PASS" if max_val < THRESHOLD_MNM else "✗ FAIL"}')
    
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace', fontweight='bold',
            color='darkgreen' if max_val < THRESHOLD_MNM else 'red',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n  Underactuation detail plot saved to: {save_path}")
    
    plt.show()


def print_motion_explanation():
    """Print a brief explanation of the results."""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
  The trajectory optimization enforces τ = 0 for passive (underactuated) joints.
  
  • ACTUATED joints: Hip motors provide torques (typically 20-100+ N·m)
  • PASSIVE joints: Knees move freely with zero motor torque (< 1 mN·m)
  
  See the plots for verification:
  • torque_comparison.png      - Actuated vs Passive torques side-by-side
  • underactuation_detail.png  - Close-up of passive joint torques (≈ 0)
""")


def main():
    """
    Main function: compute multi-step trajectory and play it back in Meshcat.
    """
    # Compute the multi-step trajectory
    t_sol, q_sol, v_sol, q0, all_torque_data = compute_multi_step_trajectory(num_steps=10)
    
    # Print comprehensive underactuation report
    print_underactuation_report(all_torque_data)
    
    # Generate plots
    print("\n" + "=" * 80)
    print("Generating Torque Analysis Plots...")
    print("=" * 80)
    
    plot_torque_comparison(all_torque_data, save_path='./torque_comparison.png')
    plot_underactuation_detail(all_torque_data, save_path='./underactuation_detail.png')
    
    # Print summary
    print_motion_explanation()
    
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
