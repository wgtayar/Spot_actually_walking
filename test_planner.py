#!/usr/bin/env python3
"""Quick test of trajectory planner with new cost weights."""

from mpc_trajectory_planner import build_joint_only_mpc_model, plan_joint_trajectory, visualize_trajectory_plan
import numpy as np

# Build model
print("Building MPC model...")
model = build_joint_only_mpc_model(dt=0.01)

# Define goal
y_start = model.C_y @ model.x_star
y_goal = y_start.copy()

# Front legs more bent, rear legs less bent
y_goal[1] += 0.2   # front_left_hip_y
y_goal[4] += 0.2   # front_right_hip_y
y_goal[7] -= 0.2   # rear_left_hip_y
y_goal[10] -= 0.2  # rear_right_hip_y

print(f'\n=== GOAL CONFIGURATION ===')
print(f'Front legs: hip_y {y_start[1]:.3f} → {y_goal[1]:.3f} rad')
print(f'Rear legs:  hip_y {y_start[7]:.3f} → {y_goal[7]:.3f} rad')
print(f'Change magnitude: {np.linalg.norm(y_goal - y_start):.3f} rad')

# Plan trajectory
plan = plan_joint_trajectory(model, y_start, y_goal, N_horizon=200, duration=2.0, u_max=100.0)

if plan.success:
    print(f'\n✓ Planning successful!')
    print(f'  Motion magnitude: {np.linalg.norm(plan.y_traj[-1] - plan.y_traj[0]):.3f} rad')
    print(f'  Max joint motion: {np.max(np.abs(plan.y_traj[-1] - plan.y_traj[0])):.3f} rad')
    
    # Check if there's actual motion
    joint_changes = np.abs(plan.y_traj[-1] - plan.y_traj[0])
    print(f'\n  Per-joint changes:')
    for i, change in enumerate(joint_changes):
        if change > 0.01:  # Only show significant changes
            print(f'    Joint {i}: {change:.3f} rad ({np.rad2deg(change):.1f}°)')
    
    visualize_trajectory_plan(plan, model)
else:
    print(f'\n✗ Planning failed!')
