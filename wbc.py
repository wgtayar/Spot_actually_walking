##### Whole Body Controller given foot forces #####
import numpy as np
from pydrake.all import (
    IpoptSolver,
    DiscreteContactApproximation,
    RobotDiagramBuilder,
    StartMeshcat,
    MathematicalProgram,
    SnoptSolver,
    AddUnitQuaternionConstraintOnPlant,
    MeshcatVisualizer,
    OrientationConstraint,
    RotationMatrix,
    AutoDiffXd,
    ExtractGradient,
    ExtractValue,
    JacobianWrtVariable,
    InitializeAutoDiff,
    PositionConstraint,
    PiecewisePolynomial,
    eq,
    namedview,
)
PARENT_FOLDER = "jump_in_place_forces" # where forces got backed out


