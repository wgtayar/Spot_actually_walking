# external/spot/spot_footstep_plan.py
#
# Simple footstep planner for Spot on flat ground.
# Generates sequences of footsteps for straight-line walking.

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Footstep:
    """
    Data structure for a single footstep.
    
    Attributes:
        leg_index: Integer in {0, 1, 2, 3} identifying which leg moves
                  0 = front_left, 1 = front_right, 2 = rear_left, 3 = rear_right
        leg_name: String name of the leg frame
        target_position: numpy array of shape (3,) with target [x, y, z] in world frame
    """
    leg_index: int
    leg_name: str
    target_position: np.ndarray


def get_leg_names():
    """
    Get the ordered list of leg frame names.
    
    Returns:
        List of four leg frame names in the standard order:
        [front_left, front_right, rear_left, rear_right]
    """
    return [
        "front_left_lower_leg",
        "front_right_lower_leg",
        "rear_left_lower_leg",
        "rear_right_lower_leg"
    ]


def generate_straight_footstep_plan(
    initial_feet_positions: np.ndarray,
    total_forward_distance: float,
    step_length: float
) -> list[Footstep]:
    """
    Generate a footstep plan for straight-line walking in the +x direction.
    
    Uses a diagonal gait pattern: front_left -> rear_right -> front_right -> rear_left
    
    Args:
        initial_feet_positions: numpy array of shape (4, 3) with initial foot positions
                               in order [front_left, front_right, rear_left, rear_right]
        total_forward_distance: Total distance to walk forward in x direction (meters)
        step_length: Length of each step in x direction (meters)
        
    Returns:
        List of Footstep objects in time order
    """
    # Validate input
    assert initial_feet_positions.shape == (4, 3), \
        f"Expected shape (4, 3), got {initial_feet_positions.shape}"
    assert total_forward_distance > 0, "total_forward_distance must be positive"
    assert step_length > 0, "step_length must be positive"
    
    leg_names = get_leg_names()
    
    # Gait pattern: diagonal pairs
    # 0: front_left, 3: rear_right, 1: front_right, 2: rear_left
    gait_sequence = [0, 3, 1, 2]
    
    # Track current positions of all feet
    current_positions = initial_feet_positions.copy()
    
    # Initial x positions of front feet
    initial_front_left_x = initial_feet_positions[0, 0]
    initial_front_right_x = initial_feet_positions[1, 0]
    initial_front_x = max(initial_front_left_x, initial_front_right_x)
    
    # Target x for termination
    target_x = initial_front_x + total_forward_distance
    
    footsteps = []
    gait_index = 0
    
    while True:
        # Select which leg moves in this step
        leg_index = gait_sequence[gait_index % len(gait_sequence)]
        leg_name = leg_names[leg_index]
        
        # Compute target position: move forward by step_length in x
        target_pos = current_positions[leg_index].copy()
        target_pos[0] += step_length  # Advance in x direction
        
        # Create and store the footstep
        footstep = Footstep(
            leg_index=leg_index,
            leg_name=leg_name,
            target_position=target_pos
        )
        footsteps.append(footstep)
        
        # Update the current position of this leg
        current_positions[leg_index] = target_pos
        
        # Check termination condition: front feet have advanced enough
        current_front_left_x = current_positions[0, 0]
        current_front_right_x = current_positions[1, 0]
        current_front_max_x = max(current_front_left_x, current_front_right_x)
        
        if current_front_max_x >= target_x:
            break
        
        # Move to next leg in gait pattern
        gait_index += 1
    
    return footsteps


def print_footstep_plan(footsteps: list[Footstep], max_steps: int = None):
    """
    Print a footstep plan in a readable format.
    
    Args:
        footsteps: List of Footstep objects
        max_steps: Maximum number of steps to print (None = print all)
    """
    if not footsteps:
        print("Empty footstep plan")
        return
    
    print("\nFootstep plan:")
    print("=" * 80)
    
    num_to_print = len(footsteps) if max_steps is None else min(max_steps, len(footsteps))
    
    for i, step in enumerate(footsteps[:num_to_print]):
        pos = step.target_position
        print(f"  Step {i:2d}: leg {step.leg_name:25s} -> target "
              f"[{pos[0]:7.4f}, {pos[1]:7.4f}, {pos[2]:7.4f}]")
    
    if max_steps is not None and len(footsteps) > max_steps:
        print(f"  ... ({len(footsteps) - max_steps} more steps)")
    
    print("=" * 80)
    print(f"Total steps: {len(footsteps)}")
