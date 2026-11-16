"""Read-only ROS image capture and append-only evidence files for diagnostics.

These records are not C0 LIVE observations. No driver or actuator is started.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import queue
import threading
import time
import uuid

from .frames import (FRAME_RECORD_SCHEMA_VERSION, FRAME_RECORD_TYPE,
                     Frame, pixel_metadata, positive_integer)


def _nonnegative_integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(name + ' must be a nonnegative integer')
    return value


def _finite_nonnegative(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(name + ' must be a finite nonnegative number')
    return value


def decode_ros_image(message, session_id, received_monotonic_s) -> Frame:
    """Decode uint8 ROS images, including row padding, without importing ROS."""
    import numpy as np
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError('session_id must be a nonempty string')
    _finite_nonnegative(received_monotonic_s, 'received_monotonic_s')
    try:
        width = positive_integer(message.width, 'width')
        height = positive_integer(message.height, 'height')
        step = positive_integer(message.step, 'step')
        encoding = message.encoding
        if encoding not in ('rgb8', 'bgr8', 'mono8'):
            raise ValueError('Supported ROS image encodings: rgb8, bgr8, mono8')
        channels = 1 if encoding == 'mono8' else 3
        if step < width * channels or height * step > 256 * 1024 * 1024:
            raise ValueError('Invalid row step or excessive image payload')
        payload = message.data
        if isinstance(payload, (str, int, float, bool)):
            raise ValueError('ROS image data must be a byte buffer or uint8 sequence')
        if isinstance(payload, (list, tuple)):
            if any(type(v) is not int or not 0 <= v <= 255 for v in payload):
                raise ValueError('ROS image data contains non-uint8 values')
        else:
            try:
                view = memoryview(payload)
            except TypeError as error:
                raise ValueError('ROS image data must expose a uint8 byte buffer') from error
            if view.ndim != 1 or view.itemsize != 1 or view.format != 'B' or not view.c_contiguous:
                raise ValueError('ROS image data must be a contiguous one-dimensional uint8 buffer')
        try:
            raw = bytes(payload)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError('Invalid ROS image data') from error
        if len(raw) != height * step:
            raise ValueError('ROS image data length does not equal height * step')
        seq = _nonnegative_integer(message.header.seq, 'header.seq')
        secs = _nonnegative_integer(message.header.stamp.secs, 'stamp.secs')
        nsecs = _nonnegative_integer(message.header.stamp.nsecs, 'stamp.nsecs')
        if seq > 4294967295 or nsecs >= 1000000000:
            raise ValueError('Invalid ROS header sequence or timestamp')
        frame_id = message.header.frame_id
        if not isinstance(frame_id, str) or not frame_id.strip():
            raise ValueError('ROS image frame_id must be nonempty')
    except AttributeError as error:
        raise ValueError('Incomplete sensor_msgs/Image fields') from error
    rows = np.frombuffer(raw, dtype=np.uint8).reshape(height, step)
    packed = rows[:, :width * channels].reshape(height, width, channels)
    if encoding == 'mono8':
        bgr = np.repeat(packed, 3, axis=2)
    elif encoding == 'rgb8':
        bgr = packed[:, :, ::-1]
    else:
        bgr = packed
    return Frame(bgr, dict(source_kind='ros_image', origin='ros_capture_unverified',
        session_id=session_id, frame_id=frame_id, header_seq=seq,
        camera_stamp=dict(secs=secs, nsecs=nsecs),
        camera_stamp_basis='sensor_msgs/Image.header.stamp; separate from receipt clock',
        capture_timestamp_verified=False, received_monotonic_s=received_monotonic_s,
        source_encoding=encoding, source_step=step,
        source_is_bigendian=getattr(message, 'is_bigendian', None),
        endianness_note='8-bit channel values require no byte swapping',
        raw_ros_data_sha256=hashlib.sha256(raw).hexdigest(),
        raw_ros_hash_basis='complete Image.data buffer, including row padding'))


class SessionWriter:
    """Create a new directory and save lossless BGR PNGs plus explicit metadata."""
    def __init__(self, output_path):
        self.output_path = Path(output_path).expanduser().resolve()
        self.output_path.mkdir(parents=True, exist_ok=False)
        self.session_id = uuid.uuid4().hex
        self._next_index = 1
        self._lock = threading.Lock()

    def write(self, frame) -> dict:
        import cv2
        if not isinstance(frame, Frame):
            raise ValueError('SessionWriter expects a Frame')
        with self._lock:
            name = 'frame_{:06d}'.format(self._next_index)
            image_path = self.output_path / (name + '.png')
            metadata_path = self.output_path / (name + '.json')
            if image_path.exists() or metadata_path.exists():
                raise FileExistsError('Refusing to overwrite session evidence: ' + name)
            ok, encoded = cv2.imencode('.png', frame.bgr)
            if not ok:
                raise ValueError('Cannot encode canonical image as PNG')
            png = encoded.tobytes()
            record = deepcopy(frame.metadata)
            source_session = record.get('session_id')
            if source_session is not None and source_session != self.session_id:
                record['source_session_id'] = source_session
            record.update(pixel_metadata(frame.bgr))
            record.update(record_type=FRAME_RECORD_TYPE, schema_version=FRAME_RECORD_SCHEMA_VERSION,
                session_id=self.session_id, session_directory_name=self.output_path.name,
                image_path=str(image_path), metadata_path=str(metadata_path),
                stored_encoding='png', png_sha256=hashlib.sha256(png).hexdigest(),
                png_hash_basis='saved PNG file bytes', actionable=False,
                trusted_c0_identity=False)
            serialized = (json.dumps(record, ensure_ascii=False, allow_nan=False, indent=2) + '\n').encode('utf-8')
            created = []
            try:
                # Exclusive creation also protects against another process racing us.
                with image_path.open('xb') as image_file:
                    created.append(image_path)
                    with metadata_path.open('xb') as metadata_file:
                        created.append(metadata_path)
                        image_file.write(png)
                        metadata_file.write(serialized)
            except Exception:
                for path in created:
                    path.unlink(missing_ok=True)
                raise
            self._next_index += 1
            return record


def capture_ros(topic, output, max_frames=100, timeout_s=10.0) -> dict:
    """Subscribe for a bounded session; never publish or launch device drivers.

    Timestamp and sequence continuity are checked within each frame_id. Sequence
    wrap, camera resets and clock resets require starting a new capture session.
    """
    if not isinstance(topic, str) or not topic.strip() or any(c.isspace() for c in topic):
        raise ValueError('A nonempty ROS image topic without whitespace is required')
    positive_integer(max_frames, 'max_frames')
    _finite_nonnegative(timeout_s, 'timeout_s')
    if timeout_s == 0:
        raise ValueError('timeout_s must be positive')
    try:
        import rospy
        from sensor_msgs.msg import Image
    except ImportError as error:
        raise RuntimeError('ROS1 capture needs rospy and sensor_msgs in the robot ROS environment; offline input does not') from error
    writer = SessionWriter(output)
    incoming = queue.Queue(maxsize=8)
    finished = threading.Event()
    count_lock = threading.Lock()
    dropped = [0]

    def receive(message):
        if finished.is_set():
            return
        try:
            incoming.put_nowait((message, time.monotonic()))
        except queue.Full:
            with count_lock:
                dropped[0] += 1

    rows, rejections, seen_ids, seen_stamps, latest = [], {}, {}, {}, {}
    started = time.monotonic()
    subscriber = None
    reason, status, error_record = 'timeout', 'finished', None
    summary = None
    try:
        if not rospy.core.is_initialized():
            rospy.init_node('treasure_vision_capture', anonymous=True, disable_signals=True)
        # The observation timeout starts after ROS initialization. An OS-level
        # blocked initialization or disk write is not a hard real-time bound.
        deadline = time.monotonic() + timeout_s
        subscriber = rospy.Subscriber(topic, Image, receive, queue_size=1, buff_size=2**24)
        while len(rows) < max_frames:
            if rospy.is_shutdown():
                reason = 'ros_shutdown'
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                message, received = incoming.get(timeout=min(0.05, remaining))
            except queue.Empty:
                continue
            rejection = None
            try:
                frame = decode_ros_image(message, writer.session_id, received)
                meta = frame.metadata
                stamp = (meta['camera_stamp']['secs'], meta['camera_stamp']['nsecs'])
                camera, seq = meta['frame_id'], meta['header_seq']
                identity = (camera, seq)
                stamp_identity = (camera,) + stamp
                digest = meta['raw_ros_data_sha256']
                if stamp == (0, 0):
                    rejection = 'missing_capture_stamp'
                elif identity in seen_ids:
                    rejection = 'identity_conflict' if seen_ids[identity] != digest else 'duplicate_sequence'
                elif stamp_identity in seen_stamps:
                    rejection = 'identity_conflict' if seen_stamps[stamp_identity] != digest else 'duplicate_stamp'
                elif camera in latest and seq <= latest[camera][0]:
                    rejection = 'sequence_regression'
                elif camera in latest and stamp <= latest[camera][1]:
                    rejection = 'timestamp_regression'
            except (ValueError, TypeError, OverflowError):
                rejection = 'invalid_image'
            if rejection is not None:
                rejections[rejection] = rejections.get(rejection, 0) + 1
                continue
            meta['topic'] = topic
            rows.append(writer.write(frame))
            seen_ids[identity], seen_stamps[stamp_identity] = digest, digest
            latest[camera] = (seq, stamp)
        if len(rows) >= max_frames:
            reason = 'max_frames'
    except BaseException as error:
        status = 'interrupted' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'failed'
        reason = 'interrupted' if status == 'interrupted' else 'error'
        error_record = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        finished.set()
        cleanup_error = None
        try:
            if subscriber is not None:
                subscriber.unregister()
        except Exception as error:
            cleanup_error = error
            if error_record is None:
                status, reason = 'failed', 'error'
                error_record = dict(type=type(error).__name__, message=str(error))
        summary_path = writer.output_path / 'summary.json'
        summary = dict(source_kind='ros_image', origin='ros_capture_unverified', topic=topic,
            output_dir=str(writer.output_path), session_id=writer.session_id,
            summary_path=str(summary_path), status=status, error=error_record,
            requested_max_frames=max_frames, timeout_s=timeout_s,
            max_frames_reached=len(rows) >= max_frames,
            accepted_frames=len(rows), rejected_frames=sum(rejections.values()),
            dropped_frames=dropped[0], rejections=rejections, frames=rows,
            duration_s=time.monotonic() - started, stopped_reason=reason,
            timed_out=reason == 'timeout', actionable=False, trusted_c0_identity=False)
        with summary_path.open('x', encoding='utf-8') as stream:
            json.dump(summary, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write('\n')
        if cleanup_error is not None and status == 'failed':
            raise cleanup_error
    return summary
