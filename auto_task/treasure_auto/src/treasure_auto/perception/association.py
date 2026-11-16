"""Conservative image-view association. No inventory, occupancy or target-lock authority."""
from dataclasses import dataclass, replace
import uuid
from treasure_auto.contracts import core as c, models as m
from .geometry import distance


@dataclass
class _Track:
    last: m.TargetObservation
    view: object
    confirmations: int


class TargetTracker:
    def __init__(self, config):
        self.config = config
        self.tracks = []

    def break_continuity(self):
        for track in self.tracks:
            track.confirmations = 0

    def invisible(self):
        return tuple(replace(t.last, top=None, visibility=m.Visibility.OCCLUDED)
                     for t in self.tracks if t.last.target_id is not None)

    @staticmethod
    def view_key(image, scene):
        e = image.envelope
        # Scene changes the semantic filter, not physical identity within an unchanged view.
        return (image.camera_frame, image.capture_pose, image.width, image.height, image.distortion,
                e.base_motion_epoch, e.layout_recovery_round, e.rebuild_generation,
                e.versions.calibration, e.versions.map, e.versions.config, e.versions.model)

    def update(self, candidates, image, scene, previous_targets):
        known = {t.last.target_id for t in self.tracks if t.last.target_id}
        for old in previous_targets:
            if old.target_id and old.target_id not in known:
                # C0 previous_targets has no capture pose/epoch: keep history, never infer alignment.
                self.tracks.append(_Track(old, None, 0)); known.add(old.target_id)
        view = self.view_key(image, scene)
        eligible = lambda t: t.top is not None and t.category is not None
        links = {}
        for i, (_, candidate) in enumerate(candidates):
            links[i] = []
            if eligible(candidate):
                for j, track in enumerate(self.tracks):
                    old = track.last
                    if track.view == view and eligible(old) and old.category is candidate.category:
                        if not old.evidence or image.image.captured_at - old.evidence[-1].captured_at >= self.config.max_age_s:
                            continue
                        # Corner ordering starts at min-v, which can switch vertices under tiny tilts.
                        # Match cyclic clockwise order; never assume physical corner identity by index.
                        corners = candidate.top.corners
                        displacement = min(max(distance(a, b) for a, b in zip(old.top.corners,
                                           corners[shift:] + corners[:shift])) for shift in range(4))
                        if displacement <= self.config.association_distance_px:
                            links[i].append(j)
        inverse = {j: [i for i in links if j in links[i]] for j in range(len(self.tracks))}
        original_tracks = tuple(self.tracks)
        seen, result, by_entity = set(), [], {}
        for i, (entity_id, candidate) in enumerate(candidates):
            matches = links[i]
            chosen = None
            if len(matches) == 1 and len(inverse[matches[0]]) == 1:
                chosen = matches[0]
                track = self.tracks[chosen]
                track.confirmations += 1
                target_id = track.last.target_id
                if target_id is None and track.confirmations >= self.config.min_confirmations:
                    target_id = image.envelope.run_id + ':p04:' + uuid.uuid4().hex
                candidate = replace(candidate, target_id=target_id,
                                    identity_confirmed=c.Fact.CONFIRMED if target_id else c.Fact.UNKNOWN)
                track.last = candidate
            elif eligible(candidate) and not matches:
                # A changed view or unmatched same-class history may be the same physical brick.
                uncertain_history = any(t.view != view or
                                        (t.last.category is candidate.category and not inverse[j])
                                        for j, t in enumerate(original_tracks))
                if not uncertain_history:
                    chosen = len(self.tracks)
                    self.tracks.append(_Track(candidate, view, 1))
            if chosen is not None:
                seen.add(chosen)
            result.append(candidate)
            by_entity[entity_id] = candidate
        for j, track in enumerate(self.tracks):
            if j not in seen:
                track.confirmations = 0
                if track.last.target_id:
                    result.append(replace(track.last, top=None, visibility=m.Visibility.OCCLUDED))
        return tuple(result), by_entity

