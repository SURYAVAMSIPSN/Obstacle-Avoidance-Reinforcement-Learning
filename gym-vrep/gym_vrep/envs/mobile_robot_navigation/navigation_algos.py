import abc
from typing import Any
from typing import NoReturn
from typing import Tuple

import numpy as np

from gym_vrep.envs.mobile_robot_navigation.robots.smartbot import SmartBot


class Base(metaclass=abc.ABCMeta):
    """A superclass for navigation tasks.
    """

    def __init__(self, robot: SmartBot, dt):
        """A constructor of superclass

        Args:
            robot: An object of mobile robot.
            dt: A delta time of simulation.
        """
        self._robot = robot
        self._target_position = None
        self._dt = dt

        self._pose = np.zeros(3)

    @abc.abstractmethod
    def compute_position(self, goal: np.ndarray) -> Any:
        """An abstract method for computing position.

        Args:
            goal: Desired goal of mobile robot.
        """
        return NotImplementedError

    @abc.abstractmethod
    def reset(self, start_pose: np.ndarray) -> Any:
        """An abstract method for object reset functionality.

        Args:
            start_pose: Start pose of mobile robot.
        """
        return NotImplementedError

    @staticmethod
    def _angle_correction(angle: float or np.ndarray) -> np.ndarray:
        """Static method that correct given angles into range -pi, pi

        Args:
            angle: An array of angles or single angle

        Returns:
            new_angle: Angles or angle after correction.
        """
        new_angle = None
        if angle >= 0:
            new_angle = np.fmod((angle + np.pi), (2 * np.pi)) - np.pi

        if angle < 0:
            new_angle = np.fmod((angle - np.pi), (2 * np.pi)) + np.pi

        new_angle = np.round(angle, 3)
        return new_angle


class Ideal(Base):
    """A class that computes robot polar coordinates of mobile robot based on
    absolute pose received from simulation engine and kinematic model.

    """

    def __init__(self, robot: SmartBot, dt: float):
        """A constructor of class.

        Args:
            robot: An object of mobile robot.
            dt: A delta time of simulation.
        """
        super(Ideal, self).__init__(robot, dt)

    def compute_position(self, goal) -> np.ndarray:
        """A method that computer polar coordinates of mobile robot.

        Args:
            goal: Desired goal of mobile robot.

        Returns:
            polar_coordinates: A polar coordinates of mobile robot.
        """
        pose = np.round(self._robot.get_2d_pose(), 3)
        pose[2] = self._angle_correction(pose[2])

        distance = np.linalg.norm(pose[0:2] - goal)

        theta = np.arctan2(goal[1] - pose[1], goal[0] - pose[0])

        theta = self._angle_correction(theta)

        heading_angle = self._angle_correction(theta - pose[2])

        polar_coordinates = np.round(np.array([distance, heading_angle]), 3)
        return polar_coordinates

    def reset(self, start_pose: np.ndarray) -> NoReturn:
        """A method that reset member variable.

        Args:
            start_pose: Start pose of mobile robot.
        """
        self._pose = start_pose


