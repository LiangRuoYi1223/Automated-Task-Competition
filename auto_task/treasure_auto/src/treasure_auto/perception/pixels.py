"""Image provenance, acquisition gate and exactly-once inverse pixel transforms."""
import hashlib
from dataclasses import dataclass
from typing import Tuple
from treasure_auto.contracts import core as c, models as m, rules


@dataclass(frozen=True)
class RawImage:
    width: int
    height: int
    encoding: str
    data: bytes

    def validate(self, image):
        c.require((self.width, self.height, self.encoding) == (image.width, image.height, image.encoding),
                  'image', 'payload resolution/encoding mismatch')
        c.require(self.encoding in {'mono8', 'rgb8', 'bgr8'}, 'encoding', 'adapter must decode to supported raw pixels')
        channels = 1 if self.encoding == 'mono8' else 3
        c.require(type(self.data) is bytes and len(self.data) == self.width * self.height * channels,
                  'image.data', 'empty, damaged or incomplete raw pixels')
        c.require(image.image.sha256 is not None and hashlib.sha256(self.data).hexdigest() == image.image.sha256,
                  'image.sha256', 'payload digest missing or mismatched', c.ErrorCode.ID_CONFLICT)

    def array(self):
        import numpy as np
        shape = (self.height, self.width) if self.encoding == 'mono8' else (self.height, self.width, 3)
        return np.frombuffer(self.data, dtype=np.uint8).reshape(shape)


@dataclass(frozen=True)
class PixelSet:
    points: Tuple[m.PixelPoint, ...]
    space: str
    transform_id: str


@dataclass(frozen=True)
class TransformPlan:
    transform_id: str
    full_width: int
    full_height: int
    crop_width: int
    crop_height: int
    input_width: int
    input_height: int
    transform: m.PixelTransform
    distortion: m.Distortion

    def __post_init__(self):
        c.require(type(self.transform_id) is str and bool(self.transform_id.strip()), 'transform_id', 'required')
        c.require(all(type(v) is int and v > 0 for v in (self.full_width, self.full_height,
                  self.crop_width, self.crop_height, self.input_width, self.input_height)), 'dimensions', 'positive integers required')
        t = self.transform
        c.require(type(t) is m.PixelTransform, 'transform', 'C0 PixelTransform required')
        c.require(self.distortion in {m.Distortion.RAW, m.Distortion.CORRECTED}, 'distortion', 'unknown pixel distortion')
        c.require(min(t.crop_u, t.crop_v, t.pad_u, t.pad_v) >= 0, 'transform', 'negative crop/padding')
        c.require(t.crop_u + self.crop_width <= self.full_width and t.crop_v + self.crop_height <= self.full_height,
                  'crop', 'crop exceeds original image')
        c.require(t.pad_u + t.scale_x * self.crop_width <= self.input_width and
                  t.pad_v + t.scale_y * self.crop_height <= self.input_height,
                  'input', 'scaled crop exceeds backend image')

    def restore(self, pixels):
        c.require(type(pixels) is PixelSet and pixels.space == 'backend', 'pixel_space',
                  'expected backend coordinates; refusing repeated restoration')
        c.require(pixels.transform_id == self.transform_id, 'transform_id', 'mixed preprocessing chains')
        c.require(type(pixels.points) is tuple, 'points', 'immutable pixel tuple required')
        t, points = self.transform, []
        for p in pixels.points:
            c.require(type(p) is m.PixelPoint, 'points', 'C0 finite pixel coordinates required')
            x, y = (p.u - t.pad_u) / t.scale_x, (p.v - t.pad_v) / t.scale_y
            c.require(0 <= x < self.crop_width and 0 <= y < self.crop_height,
                      'points', 'point is in padding or outside actual crop')
            points.append(m.PixelPoint(x + t.crop_u, y + t.crop_v))
        return PixelSet(tuple(points), 'full', self.transform_id)


class FrameGate:
    """Instance/run-scoped. Admission alone is not a static-scene certificate."""
    def __init__(self, expected_size, max_age_s):
        self.expected_size, self.max_age_s = expected_size, max_age_s
        self.accepted_count = 0
        self._ids, self._last_capture = {}, {}

    def validate(self, image, raw, context, capture_floor, sensor_now):
        rules.validate_envelope(image.envelope, context)
        c.require(image.envelope.request_id is not None and image.envelope.base_motion_epoch is not None,
                  'image.envelope', 'request and base movement epoch required')
        rules.require_physical_evidence(image.image, image.envelope, context.now, context.origin)
        c.require(image.capture_pose is not None, 'capture_pose', 'capture-aligned pose required')
        c.require(bool(image.camera_frame.strip()), 'camera_frame', 'camera source required')
        c.require((image.width, image.height) == self.expected_size, 'resolution', 'unexpected image size')
        raw.validate(image)
        c.require(context.now - image.image.received_at < self.max_age_s, 'received_at', 'old receipt', c.ErrorCode.EXPIRED)
        capture = image.image.captured_at
        c.require(capture - capture_floor > 0, 'captured_at', 'capture predates current acquisition barrier', c.ErrorCode.STALE_GEOMETRY)
        age = sensor_now - capture  # Same sensor domain ONLY. Never subtract receipt from capture.
        c.require(0 <= age < self.max_age_s, 'captured_at', 'future or stale acquisition', c.ErrorCode.EXPIRED)

    def accept(self, image, raw, context, capture_floor, sensor_now):
        self.validate(image, raw, context, capture_floor, sensor_now)
        domain = (image.envelope.run_id, image.camera_frame)
        capture = image.image.captured_at
        previous = self._last_capture.get(domain)
        if previous is not None:
            c.require(capture - previous > 0, 'captured_at', 'duplicate or out-of-order capture', c.ErrorCode.DUPLICATE)
        keys = ((domain, 'observation', image.observation_id), (domain, 'evidence', image.image.evidence_id),
                (domain, 'event', image.envelope.event_id), (domain, 'uri', image.image.uri))
        for key in keys:
            if key in self._ids:
                c.require(self._ids[key] == image.image.sha256, key[1], 'same frame identifier with different pixels', c.ErrorCode.ID_CONFLICT)
                c.require(False, key[1], 'frame identifier already consumed', c.ErrorCode.DUPLICATE)
        for key in keys:
            self._ids[key] = image.image.sha256
        self._last_capture[domain] = capture
        self.accepted_count += 1

