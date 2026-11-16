"""Measured/synthetic field polygons, separate from ordinary navigation permits.

All coordinates are metres in ``frame_id``. These checks never imply that an
unknown region is empty or that a motion/placement has physically succeeded.
"""
from dataclasses import dataclass
from types import MappingProxyType
from treasure_auto.contracts import core as c, models as m
from treasure_auto.mapping.geometry import (
    validate_polygon, finite, contained, covered, collides)


@dataclass(frozen=True)
class FieldRegions:
    frame_id: str
    boundary: tuple
    central: tuple
    full_areas: object
    collection_areas: object
    board_polygons: tuple = ()
    wall_polygons: tuple = ()
    error_m: float = 0.

    def __post_init__(self):
        c.require(type(self.frame_id) is str and bool(self.frame_id), 'frame_id', 'explicit frame required')
        finite(self.error_m, 'field_error_m')
        full = {getattr(k, 'value', k): v for k, v in self.full_areas.items()}
        collection = {getattr(k, 'value', k): v for k, v in self.collection_areas.items()}
        c.require(set(full) == set(collection) == set('ABCD'), 'areas', 'all four full and collection regions required')
        for polygon in (self.boundary, self.central) + tuple(full.values()) + tuple(collection.values()) + self.board_polygons + self.wall_polygons:
            validate_polygon(polygon)
        c.require(covered(self.central, self.boundary), 'central', 'central region outside field')
        for area in full:
            c.require(covered(full[area], self.boundary) and covered(collection[area], full[area]),
                      'areas', 'collection must lie within its full region and field')
        object.__setattr__(self, 'full_areas', MappingProxyType(full))
        object.__setattr__(self, 'collection_areas', MappingProxyType(collection))

    def _margin(self, footprint, error_m, frame_id):
        validate_polygon(footprint)
        finite(error_m, 'footprint_error_m')
        c.require(frame_id is None or frame_id == self.frame_id, 'frame_id', 'field frame mismatch')
        return error_m + self.error_m

    def area_of(self, footprint, error_m=0., frame_id=None):
        """Whole footprint plus uncertainty; a straddling object is not inside."""
        margin = self._margin(footprint, error_m, frame_id)
        if not contained(footprint, self.boundary, margin):
            return 'boundary'
        for name, polygon in self.full_areas.items():
            if contained(footprint, polygon, margin):
                return name
        if contained(footprint, self.central, margin):
            return 'central'
        return 'outside'

    def push_prohibited(self, footprint, error_m=0., frame_id=None):
        """Contact with *any* full scoring region, board or wall denies pushing."""
        try:
            margin = self._margin(footprint, error_m, frame_id)
            return (not contained(footprint, self.boundary, margin) or
                    any(collides(footprint, p, margin) for p in
                        tuple(self.full_areas.values()) + self.board_polygons + self.wall_polygons))
        except (c.ContractError, TypeError, ValueError, AttributeError):
            return True

    def temporary_ground_ok(self, area, footprint, error_m=0., free_observed=False,
                            approach_clear=False, regrasp_clear=False, obstacles=(), frame_id=None):
        """Caller supplies current clear-ground and access evidence, including A."""
        try:
            if not all(value is True for value in (free_observed, approach_clear, regrasp_clear)):
                return False
            name = getattr(area, 'value', area)
            margin = self._margin(footprint, error_m, frame_id)
            if name not in self.full_areas or not contained(footprint, self.full_areas[name], margin):
                return False
            blockers = (self.collection_areas[name],) + self.board_polygons + self.wall_polygons + tuple(obstacles)
            for polygon in blockers:
                validate_polygon(polygon)
                if collides(footprint, polygon, margin):
                    return False
            return True
        except (c.ContractError, TypeError, ValueError, AttributeError):
            return False

    def recovery_priority(self, footprint, error_m=0., normal_known_layout=False,
                          reliable_grasp=False, abnormal=False, push_safe=False, frame_id=None):
        try:
            margin = self._margin(footprint, error_m, frame_id)
            if any(collides(footprint, p, margin) for p in self.full_areas.values()):
                return 'arm_recovery'
            if self.area_of(footprint, error_m, frame_id) == 'boundary':
                return 'wait'
            if normal_known_layout is True:
                return 'grasp'
            safe = push_safe is True and not self.push_prohibited(footprint, error_m, frame_id)
            if self.area_of(footprint, error_m, frame_id) == 'central' and abnormal is True:
                return 'push' if safe else ('grasp' if reliable_grasp is True else 'wait')
            if reliable_grasp is True:
                return 'grasp'
            return 'push' if safe and self.area_of(footprint, error_m, frame_id) == 'outside' else 'wait'
        except (c.ContractError, TypeError, ValueError, AttributeError):
            return 'wait'
