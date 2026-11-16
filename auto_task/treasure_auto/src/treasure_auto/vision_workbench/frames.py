"""Pixel inputs for an offline workbench, with no ROS or control imports."""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterator

FRAME_RECORD_TYPE = 'treasure_auto.vision_workbench.frame'
FRAME_RECORD_SCHEMA_VERSION = 1


@dataclass
class Frame:
    bgr: Any
    metadata: Dict[str, Any]

    def __post_init__(self):
        import numpy as np
        if (not isinstance(self.bgr, np.ndarray) or self.bgr.dtype != np.uint8 or
                self.bgr.ndim != 3 or self.bgr.shape[2] != 3 or
                self.bgr.shape[0] <= 0 or self.bgr.shape[1] <= 0):
            raise ValueError('Frame requires a nonempty uint8 HWC BGR image')
        if not isinstance(self.metadata, dict):
            raise ValueError('Frame metadata must be a dictionary')
        # Own the pixels: a ROS callback or video decoder may reuse its buffer.
        self.bgr = np.array(self.bgr, dtype=np.uint8, order='C', copy=True)
        self.metadata = deepcopy(self.metadata)
        self.metadata.update(pixel_metadata(self.bgr))
        self.metadata['actionable'] = False
        self.metadata['trusted_c0_identity'] = False


def pixel_metadata(bgr):
    return dict(width=int(bgr.shape[1]), height=int(bgr.shape[0]),
        canonical_encoding='bgr8',
        pixel_sha256=hashlib.sha256(bgr.tobytes(order='C')).hexdigest(),
        pixel_hash_basis='decoded canonical bgr8, tightly packed row-major bytes')


def positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(name + ' must be a positive integer')
    return value


def _file(path):
    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError('Input must be a file: ' + str(source))
    return source


def _capture_record(source, encoded, source_sha256, bgr):
    """Bind our sidecar to these bytes; retain its claims without promoting trust."""
    sidecar = source.with_suffix('.json')
    if not sidecar.exists():
        return None
    sidecar = sidecar.resolve(strict=True)
    raw = sidecar.read_bytes()
    try:
        record = json.loads(raw.decode('utf-8'))
    except (UnicodeError, ValueError) as error:
        raise ValueError('Cannot read image sidecar JSON: ' + str(sidecar)) from error
    if not isinstance(record, dict) or record.get('record_type') != FRAME_RECORD_TYPE:
        return None
    if (type(record.get('schema_version')) is not int or
            record['schema_version'] != FRAME_RECORD_SCHEMA_VERSION):
        raise ValueError('Unsupported workbench image sidecar schema: ' + str(sidecar))
    if record.get('stored_encoding') != 'png' or not encoded.startswith(b'\x89PNG\r\n\x1a\n'):
        raise ValueError('Workbench image record does not describe a PNG: ' + str(sidecar))
    if record.get('png_sha256') != source_sha256:
        raise ValueError('Workbench sidecar PNG file hash mismatch: ' + str(sidecar))
    actual = pixel_metadata(bgr)
    if any(record.get(key) != actual[key] for key in ('pixel_sha256', 'canonical_encoding', 'width', 'height')):
        raise ValueError('Workbench sidecar canonical pixel hash or description mismatch: ' + str(sidecar))
    return dict(record=record, sidecar_path=str(sidecar),
                sidecar_sha256=hashlib.sha256(raw).hexdigest(),
                verification='PNG bytes and canonical BGR pixels match this sidecar; capture claims remain unverified')


def load_image(path) -> Frame:
    import cv2
    import numpy as np
    source = _file(path)
    encoded = source.read_bytes()
    if not encoded:
        raise ValueError('Image file is empty: ' + str(source))
    try:
        bgr = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error as error:
        raise ValueError('Cannot decode image: ' + str(source)) from error
    if bgr is None:
        raise ValueError('Cannot decode image: ' + str(source))
    source_sha256 = hashlib.sha256(encoded).hexdigest()
    metadata = dict(source_kind='image_file', origin='offline_replay',
        source_path=str(source), source_sha256=source_sha256,
        source_hash_basis='original encoded file bytes',
        decoder='opencv.imdecode(IMREAD_COLOR)', capture_timestamp_verified=False)
    capture_record = _capture_record(source, encoded, source_sha256, bgr)
    if capture_record is not None:
        metadata['capture_record'] = capture_record
    return Frame(bgr, metadata)


def iter_video(path, max_frames=100, stride=1) -> Iterator[Frame]:
    import cv2
    positive_integer(max_frames, 'max_frames')
    positive_integer(stride, 'stride')
    source = _file(path)
    digest = hashlib.sha256()
    with source.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    video = cv2.VideoCapture(str(source))
    if not video.isOpened():
        video.release()
        raise ValueError('Cannot open video: ' + str(source))
    reported = float(video.get(cv2.CAP_PROP_FRAME_COUNT))
    reported_count = reported if math.isfinite(reported) and reported > 0 else None
    integrity_note = ('Container frame count is reported metadata, not proof of a complete camera sequence; '
                      'when count is unknown, decoder EOF cannot reliably distinguish damage from normal EOF')
    index, yielded = 0, 0
    try:
        while yielded < max_frames:
            ok, bgr = video.read()
            if not ok:
                if reported_count is not None and index < reported_count:
                    raise ValueError('Truncated or undecodable video: decoded {} frames, container reports {}: {}'.format(
                        index, reported_count, source))
                if index == 0:
                    raise ValueError('Video contains no decodable frames: ' + str(source))
                break
            if index % stride == 0:
                position = float(video.get(cv2.CAP_PROP_POS_MSEC))
                yield Frame(bgr, dict(source_kind='video_file', origin='offline_replay',
                    source_path=str(source), source_sha256=digest.hexdigest(),
                    source_hash_basis='complete original encoded video file bytes',
                    frame_index=index, video_position_ms=position if math.isfinite(position) else None,
                    reported_frame_count=reported_count, video_integrity_note=integrity_note,
                    capture_sequence_completeness_verified=False,
                    video_position_basis='decoder media position; not a camera capture clock',
                    capture_timestamp_verified=False))
                yielded += 1
            index += 1
    finally:
        video.release()
