# Library for constructing the Spot + ground model as a RobotDiagram.
#
# This is a light wrapper around Antonio Avila's `spot_jumping.py` setup,
# but packaged as a reusable class so we can share it across scripts
# (standing, MPC, etc.).

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from pydrake.all import (
    DiscreteContactApproximation,
    JointActuatorIndex,
    Meshcat,
    MeshcatVisualizer,
    MultibodyPlant,
    PdControllerGains,
    RobotDiagram,
    RobotDiagramBuilder,
    SceneGraph,
    StartMeshcat,
    Parser,
)


@dataclass
class SpotHandles:
    diagram: RobotDiagram
    plant: MultibodyPlant
    scene_graph: SceneGraph
    model_instance: object
    visualizer: MeshcatVisualizer


class SpotModel:
    def __init__(
        self,
        time_step: float = 1e-4,
        enable_joint_pd: bool = False,
        joint_kp: float = 200.0,
        joint_kd: float = 20.0,
    ) -> None:
        self.time_step = float(time_step)
        self.enable_joint_pd = bool(enable_joint_pd)
        self.joint_kp = float(joint_kp)
        self.joint_kd = float(joint_kd)

        self._diagram: Optional[RobotDiagram] = None
        self._plant: Optional[MultibodyPlant] = None
        self._scene_graph: Optional[SceneGraph] = None
        self._model_instance: Optional[object] = None
        self._visualizer: Optional[MeshcatVisualizer] = None

    @property
    def diagram(self) -> RobotDiagram:
        assert self._diagram is not None, "Call build_robot_diagram() first."
        return self._diagram

    @property
    def plant(self) -> MultibodyPlant:
        assert self._plant is not None, "Call build_robot_diagram() first."
        return self._plant

    @property
    def scene_graph(self) -> SceneGraph:
        assert self._scene_graph is not None, "Call build_robot_diagram() first."
        return self._scene_graph

    @property
    def model_instance(self):
        assert self._model_instance is not None, "Call build_robot_diagram() first."
        return self._model_instance

    @property
    def visualizer(self) -> MeshcatVisualizer:
        assert self._visualizer is not None, "Call build_robot_diagram() first."
        return self._visualizer

    def build_robot_diagram(self, meshcat: Optional[Meshcat] = None) -> RobotDiagram:
        # Builds the Spot & ground diagram and returns it.
        if meshcat is None:
            meshcat = StartMeshcat()

        robot_builder = RobotDiagramBuilder(time_step=self.time_step)
        plant = robot_builder.plant()
        scene_graph = robot_builder.scene_graph()
        parser: Parser = robot_builder.parser()

        from underactuated import ConfigureParser

        ConfigureParser(parser)
        (spot_instance,) = parser.AddModelsFromUrl(
            "package://underactuated/models/spot/spot.dmd.yaml"
        )
        parser.AddModelsFromUrl(
            "package://underactuated/models/littledog/ground.urdf")

        if self.time_step > 0.0:
            plant.set_discrete_contact_approximation(
                DiscreteContactApproximation.kLagged
            )

        # Optional low-level PD on each actuated joint of Spot.
        if self.enable_joint_pd:
            self._configure_joint_pd_gains(plant, spot_instance)

        plant.Finalize()

        builder = robot_builder.builder()
        visualizer = MeshcatVisualizer.AddToBuilder(
            builder, scene_graph, meshcat=meshcat
        )

        diagram = robot_builder.Build()

        self._diagram = diagram
        self._plant = plant
        self._scene_graph = scene_graph
        self._model_instance = spot_instance
        self._visualizer = visualizer

        return diagram

    def _configure_joint_pd_gains(self, plant: MultibodyPlant, model_instance) -> None:
        # Enable simple low-level PD on every joint actuator in model_instance.
        num_actuators_total = plant.num_actuators()

        for i in range(num_actuators_total):
            actuator = plant.get_joint_actuator(JointActuatorIndex(i))
            if actuator.model_instance() != model_instance:
                # Skip actuators that belong to other models (for example spot's floating base's actuators).
                continue
            gains = PdControllerGains(p=self.joint_kp, d=self.joint_kd)
            actuator.set_controller_gains(gains)

    def get_default_standing_state(self):
        assert self._plant is not None, "Call build_robot_diagram() first."
        plant = self._plant

        q = plant.GetDefaultPositions().copy()
        # Index 6 is base z; subtract small offset so feet are on ground.
        q[6] -= 0.02889683
        v = np.zeros(plant.num_velocities())
        return q, v
