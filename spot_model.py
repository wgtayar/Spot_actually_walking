# external/spot/spot_model.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from pydrake.all import (
    Diagram,
    DiscreteContactApproximation,
    Meshcat,
    MeshcatVisualizer,
    ModelInstanceIndex,
    MultibodyPlant,
    RobotDiagramBuilder,
    SceneGraph,
    StartMeshcat,
)

from underactuated import ConfigureParser


# Same contact point and initial foot positions as in spot_jumping.py
FOOT_IN_LEG = np.array([0.0, 0.0, -0.3365 - 0.036])
INITIAL_FOOT_POSITIONS = np.array(
    [
        [2.20252120e-01, 1.65945000e-01, 0.0],   # front left
        [2.20252120e-01, -1.65945000e-01, 0.0],  # front right
        [-3.75447880e-01, 1.65945000e-01, 0.0],  # rear left
        [-3.75447880e-01, -1.65945000e-01, 0.0], # rear right
    ]
)


@dataclass
class SpotModel:
    """Container for all useful handles related to the Spot model."""
    diagram: Diagram
    plant: MultibodyPlant
    scene_graph: SceneGraph
    meshcat: Optional[Meshcat]
    visualizer: Optional[MeshcatVisualizer]
    model_instance: ModelInstanceIndex

    default_context: Diagram  # actually a Context, kept for convenience
    default_plant_context: Diagram  # plant sub-context

    default_q: np.ndarray

    body_frame: object
    foot_frames: List[object]
    foot_in_leg: np.ndarray
    initial_foot_positions: np.ndarray


def build_spot_robot_diagram(
    time_step: float = 1e-4,
    meshcat: Optional[Meshcat] = None,
    with_visualizer: bool = True,
    use_lagged_contact: bool = True,
) -> SpotModel:
    """
    Build a MultibodyPlant + SceneGraph diagram for Spot standing on flat ground.

    This follows the same setup as spot_jumping.py (RobotDiagramBuilder,
    ConfigureParser, Spot + ground models, discrete contact, etc.), but wraps it
    into a reusable function and returns a SpotModel with all the key handles.

    Parameters
    ----------
    time_step : float
        Discrete time step for the plant (passed to RobotDiagramBuilder).
    meshcat : Meshcat, optional
        Existing Meshcat instance. If None and with_visualizer=True, a new one
        is created via StartMeshcat().
    with_visualizer : bool
        If True, a MeshcatVisualizer is added to the diagram builder.
    use_lagged_contact : bool
        If True, set discrete contact approximation to kLagged.

    Returns
    -------
    SpotModel
        Dataclass containing diagram, plant, contexts, default pose, and frames.
    """

    # Start Meshcat if we want visualization and none was provided.
    if with_visualizer and meshcat is None:
        meshcat = StartMeshcat()
    elif not with_visualizer:
        meshcat = None

    # Build the robot + scene_graph just like in spot_jumping.py.
    robot_builder = RobotDiagramBuilder(time_step=time_step)
    plant = robot_builder.plant()
    scene_graph = robot_builder.scene_graph()
    parser = robot_builder.parser()

    # Configure package paths so "package://underactuated/..." works.
    ConfigureParser(parser)

    # Spot robot model and a simple ground plane.
    (spot_instance,) = parser.AddModelsFromUrl(
        "package://underactuated/models/spot/spot.dmd.yaml"
    )
    parser.AddModelsFromUrl(
        "package://underactuated/models/littledog/ground.urdf"
    )

    if use_lagged_contact:
        plant.set_discrete_contact_approximation(
            DiscreteContactApproximation.kLagged
        )

    # No more topology changes allowed after Finalize.
    plant.Finalize()

    # Optionally attach a Meshcat visualizer.
    visualizer = None
    if with_visualizer and meshcat is not None:
        builder = robot_builder.builder()
        visualizer = MeshcatVisualizer.AddToBuilder(
            builder, scene_graph, meshcat=meshcat
        )

    # Build the final diagram and create a default context.
    diagram = robot_builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(diagram_context)

    # Default configuration: use plant defaults, then apply same small z-offset
    # as in spot_jumping.py so the feet sit nicely on the ground.
    q0 = plant.GetDefaultPositions().copy()
    # Index 6 is the base z position for this model.
    q0[6] -= 0.02889683
    plant.SetPositions(plant_context, q0)

    # Publish once so that, if Meshcat is attached, we see the initial pose.
    diagram.ForcedPublish(diagram_context)

    # Useful frames.
    body_frame = plant.GetFrameByName("body", spot_instance)
    foot_frames = [
        plant.GetFrameByName("front_left_lower_leg", spot_instance),
        plant.GetFrameByName("front_right_lower_leg", spot_instance),
        plant.GetFrameByName("rear_left_lower_leg", spot_instance),
        plant.GetFrameByName("rear_right_lower_leg", spot_instance),
    ]

    return SpotModel(
        diagram=diagram,
        plant=plant,
        scene_graph=scene_graph,
        meshcat=meshcat,
        visualizer=visualizer,
        model_instance=spot_instance,
        default_context=diagram_context,
        default_plant_context=plant_context,
        default_q=q0,
        body_frame=body_frame,
        foot_frames=foot_frames,
        foot_in_leg=FOOT_IN_LEG.copy(),
        initial_foot_positions=INITIAL_FOOT_POSITIONS.copy(),
    )
