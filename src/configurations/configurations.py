from src.configurations.geometries import Geometry

import taichi as ti


@ti.data_oriented
class Configuration:
    """This class represents a starting configuration for the MLS-MPM algorithm."""

    def __init__(self, name: str, geometries: list[Geometry]):
        self.name = name
        self.initial_geometries = []
        self.subsequent_geometries = []
        for geometry in geometries:
            if geometry.frame_threshold == 0:
                self.initial_geometries.append(geometry)
            else:
                self.subsequent_geometries.append(geometry)

        # Sort this by frame_threshold, so only the first element has to be checked against.
        self.subsequent_geometries.sort(key=(lambda g: g.frame_threshold))
