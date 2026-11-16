"""Synthetic pixels only; these tests do not prove camera or robot readiness."""
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def api(test, name):
    try:
        return importlib.import_module('treasure_auto.vision_workbench.' + name)
    except ModuleNotFoundError as error:
        test.fail('Vision input implementation missing: ' + str(error))


def message(data=None, encoding='rgb8', width=2, height=2, step=8,
            seq=1, secs=10, nsecs=1, frame_id='synthetic_optical'):
    if data is None:
        data = bytes([255, 0, 0, 0, 255, 0, 90, 91,
                      0, 0, 255, 12, 34, 56, 92, 93])
    return types.SimpleNamespace(width=width, height=height, step=step,
        encoding=encoding, data=data, is_bigendian=0,
        header=types.SimpleNamespace(seq=seq, frame_id=frame_id,
            stamp=types.SimpleNamespace(secs=secs, nsecs=nsecs)))


class FrameInputTests(unittest.TestCase):
    def test_modules_import_without_numpy_opencv_or_ros(self):
        package_path = str(Path(__file__).resolve().parents[1] / 'src')
        script = """import importlib.abc, sys
class BlockImageAndRos(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('numpy', 'cv2', 'rospy', 'sensor_msgs'):
            raise ImportError('optional dependency blocked: ' + fullname)
sys.meta_path.insert(0, BlockImageAndRos())
sys.path.insert(0, sys.argv[1])
import treasure_auto.vision_workbench.frames
import treasure_auto.vision_workbench.capture
"""
        result = subprocess.run([sys.executable, '-B', '-c', script, package_path],
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_image_load_preserves_source_and_distinguishes_file_from_pixel_hash(self):
        f = api(self, 'frames')
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / '原图.png'
            pixels = np.array([[[1, 2, 3], [250, 0, 20]]], np.uint8)
            encoded = cv2.imencode('.png', pixels)[1].tobytes()
            source.write_bytes(encoded)
            frame = f.load_image(source)
            np.testing.assert_array_equal(frame.bgr, pixels)
            self.assertEqual(source.read_bytes(), encoded)
            self.assertEqual(frame.metadata['source_sha256'], hashlib.sha256(encoded).hexdigest())
            self.assertEqual(frame.metadata['pixel_sha256'], hashlib.sha256(bytes([1, 2, 3, 250, 0, 20])).hexdigest())
            self.assertEqual(frame.metadata['canonical_encoding'], 'bgr8')
            self.assertEqual(frame.metadata['origin'], 'offline_replay')
            self.assertFalse(frame.metadata['actionable'])

    def test_image_load_rejects_missing_or_undecodable_source(self):
        f = api(self, 'frames')
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / 'bad.jpg'
            bad.write_bytes(b'not an image')
            for path in (bad, Path(tmp) / 'missing.png', Path(tmp)):
                with self.subTest(path=path), self.assertRaises((ValueError, OSError)):
                    f.load_image(path)

    def test_frame_rejects_non_uint8_or_non_bgr_pixels(self):
        f = api(self, 'frames')
        for pixels in (np.zeros((3, 3), np.uint8), np.zeros((3, 3, 4), np.uint8),
                       np.zeros((3, 3, 3), np.float32), np.zeros((0, 3, 3), np.uint8)):
            with self.subTest(shape=pixels.shape), self.assertRaises(ValueError):
                f.Frame(pixels, {})

    def test_frame_retains_explicit_synthetic_origin_and_owns_pixel_buffer(self):
        f = api(self, 'frames')
        pixels = np.full((2, 2, 3), 7, np.uint8)
        frame = f.Frame(pixels, {'origin': 'synthetic_demo', 'actionable': True})
        pixels[:] = 99
        self.assertEqual(frame.metadata['origin'], 'synthetic_demo')
        self.assertFalse(frame.metadata['actionable'])
        self.assertEqual(int(frame.bgr[0, 0, 0]), 7)

    def test_video_is_bounded_stride_replay_with_source_frame_indices(self):
        f = api(self, 'frames')
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / 'synthetic.avi'
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 5, (32, 24))
            self.assertTrue(writer.isOpened(), 'Local OpenCV cannot write synthetic MJPEG fixture')
            try:
                for gray in (10, 30, 50, 70, 90):
                    writer.write(np.full((24, 32, 3), gray, np.uint8))
            finally:
                writer.release()
            frames = list(f.iter_video(video, max_frames=2, stride=2))
            self.assertEqual([x.metadata['frame_index'] for x in frames], [0, 2])
            self.assertTrue(all(x.metadata['origin'] == 'offline_replay' for x in frames))
            self.assertTrue(all(x.metadata['source_kind'] == 'video_file' for x in frames))
            self.assertTrue(all(not x.metadata['actionable'] for x in frames))
            self.assertTrue(all(x.metadata['reported_frame_count'] == 5 for x in frames))
            self.assertTrue(all(x.metadata['capture_sequence_completeness_verified'] is False for x in frames))
            self.assertLess(abs(float(frames[1].bgr.mean()) - 50), 3)

    def test_truncated_video_is_not_reported_as_clean_eof_but_frame_limit_is_allowed(self):
        f = api(self, 'frames')
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / 'truncated.avi'
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 5, (64, 64))
            self.assertTrue(writer.isOpened())
            rng = np.random.default_rng(42)
            try:
                for _ in range(10):
                    writer.write(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8))
            finally:
                writer.release()
            raw = video.read_bytes()
            self.assertGreater(len(raw), 10000)
            video.write_bytes(raw[:10000])
            probe = cv2.VideoCapture(str(video))
            try:
                self.assertEqual(int(probe.get(cv2.CAP_PROP_FRAME_COUNT)), 10)
            finally:
                probe.release()
            # Intentional bounded previews can stop before examining the damage.
            self.assertEqual(len(list(f.iter_video(video, max_frames=1))), 1)
            with self.assertRaises(ValueError):
                list(f.iter_video(video, max_frames=100))

    def test_video_rejects_invalid_limits_and_unreadable_file(self):
        f = api(self, 'frames')
        for maximum, stride in ((0, 1), (True, 1), (2, 0), (2, 1.5)):
            with self.subTest(maximum=maximum, stride=stride), self.assertRaises(ValueError):
                list(f.iter_video('unused.avi', max_frames=maximum, stride=stride))
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / 'bad.avi'
            bad.write_bytes(b'not a video')
            with self.assertRaises((ValueError, OSError)):
                list(f.iter_video(bad))


class RosImageDecodeTests(unittest.TestCase):
    def test_non_byte_buffers_cannot_masquerade_as_valid_uint8_image(self):
        cap = api(self, 'capture')
        for data in ({i: i for i in range(16)}, np.zeros(8, np.uint16),
                     np.zeros((2, 8), np.uint8)):
            with self.subTest(kind=type(data)), self.assertRaises(ValueError):
                cap.decode_ros_image(message(data=data), 's', 1)

    def test_rgb_padding_is_removed_and_channels_are_reordered(self):
        cap = api(self, 'capture')
        frame = cap.decode_ros_image(message(), 'synthetic-session', 100.25)
        np.testing.assert_array_equal(frame.bgr,
            np.array([[[0, 0, 255], [0, 255, 0]], [[255, 0, 0], [56, 34, 12]]], np.uint8))
        self.assertEqual(frame.metadata['camera_stamp'], {'secs': 10, 'nsecs': 1})
        self.assertEqual(frame.metadata['received_monotonic_s'], 100.25)
        self.assertEqual(frame.metadata['header_seq'], 1)
        self.assertEqual(frame.metadata['frame_id'], 'synthetic_optical')
        self.assertEqual(frame.metadata['session_id'], 'synthetic-session')
        self.assertEqual(frame.metadata['origin'], 'ros_capture_unverified')
        self.assertFalse(frame.metadata['trusted_c0_identity'])
        self.assertFalse(frame.metadata['actionable'])

    def test_mono_padding_and_bgr_are_decoded_without_color_drift(self):
        cap = api(self, 'capture')
        mono = cap.decode_ros_image(message(bytes([4, 7, 99, 8, 9, 88]),
            'mono8', step=3), 's', 1)
        np.testing.assert_array_equal(mono.bgr,
            np.array([[[4, 4, 4], [7, 7, 7]], [[8, 8, 8], [9, 9, 9]]], np.uint8))
        bgr = cap.decode_ros_image(message(bytes([1, 2, 3, 4, 5, 6]),
            'bgr8', height=1, step=6), 's', 1)
        np.testing.assert_array_equal(bgr.bgr, np.array([[[1, 2, 3], [4, 5, 6]]], np.uint8))

    def test_invalid_dimensions_bytes_encoding_headers_and_clocks_are_rejected(self):
        cap = api(self, 'capture')
        bad_messages = [message(width=0), message(width=True), message(step=5),
            message(data=b'short'), message(data=b'\0' * 17), message(encoding='16UC1'),
            message(data=[-1] * 16), message(data=16), message(seq=-1),
            message(secs=-1), message(nsecs=1000000000), message(frame_id='')]
        for item in bad_messages:
            with self.subTest(item=item), self.assertRaises(ValueError):
                cap.decode_ros_image(item, 's', 1)
        for session, received in (('', 1), ('s', float('nan')), ('s', -1), ('s', True)):
            with self.subTest(session=session, received=received), self.assertRaises(ValueError):
                cap.decode_ros_image(message(), session, received)


class SessionStorageTests(unittest.TestCase):
    def test_same_directory_names_get_distinct_session_ids_and_resave_owns_new_id(self):
        f, cap = api(self, 'frames'), api(self, 'capture')
        with tempfile.TemporaryDirectory() as tmp:
            first = cap.SessionWriter(Path(tmp) / 'a' / 'frames')
            second = cap.SessionWriter(Path(tmp) / 'b' / 'frames')
            self.assertNotEqual(first.session_id, second.session_id)
            self.assertEqual(len(first.session_id), 32)
            frame = f.Frame(np.zeros((2, 2, 3), np.uint8),
                            {'origin': 'offline_replay', 'session_id': 'prior-source-session'})
            row = first.write(frame)
            self.assertEqual(row['session_id'], first.session_id)
            self.assertEqual(row['source_session_id'], 'prior-source-session')
            self.assertEqual(row['session_directory_name'], 'frames')
            replay = f.load_image(row['image_path'])
            resaved = second.write(replay)
            self.assertEqual(resaved['session_id'], second.session_id)
            self.assertEqual(resaved['capture_record']['record']['session_id'], first.session_id)

    def test_saved_ros_image_replay_keeps_verified_sidecar_provenance_offline(self):
        f, cap = api(self, 'frames'), api(self, 'capture')
        with tempfile.TemporaryDirectory() as tmp:
            writer = cap.SessionWriter(Path(tmp) / 'ros_session')
            frame = cap.decode_ros_image(message(), 'synthetic-ros-session', 105.5)
            frame.metadata['topic'] = '/synthetic/rgb'
            row = writer.write(frame)
            self.assertEqual(row['record_type'], 'treasure_auto.vision_workbench.frame')
            self.assertEqual(row['schema_version'], 1)
            loaded = f.load_image(row['image_path'])
            provenance = loaded.metadata['capture_record']
            self.assertEqual(provenance['record'], row)
            self.assertEqual(provenance['record']['camera_stamp'], {'secs': 10, 'nsecs': 1})
            self.assertEqual(provenance['record']['header_seq'], 1)
            self.assertEqual(provenance['record']['frame_id'], 'synthetic_optical')
            self.assertEqual(provenance['record']['topic'], '/synthetic/rgb')
            self.assertEqual(provenance['record']['origin'], 'ros_capture_unverified')
            self.assertEqual(provenance['sidecar_path'], str(Path(row['metadata_path']).resolve()))
            self.assertEqual(provenance['sidecar_sha256'],
                hashlib.sha256(Path(row['metadata_path']).read_bytes()).hexdigest())
            self.assertEqual(loaded.metadata['source_sha256'], row['png_sha256'])
            self.assertEqual(loaded.metadata['origin'], 'offline_replay')
            self.assertFalse(loaded.metadata['actionable'])
            self.assertFalse(loaded.metadata['trusted_c0_identity'])

    def test_saved_synthetic_source_keeps_original_origin_in_nested_record(self):
        f, cap = api(self, 'frames'), api(self, 'capture')
        with tempfile.TemporaryDirectory() as tmp:
            writer = cap.SessionWriter(Path(tmp) / 'synthetic')
            saved = writer.write(f.Frame(np.full((2, 2, 3), 17, np.uint8), {'origin': 'synthetic_demo'}))
            loaded = f.load_image(saved['image_path'])
            self.assertEqual(loaded.metadata['capture_record']['record']['origin'], 'synthetic_demo')
            self.assertEqual(loaded.metadata['origin'], 'offline_replay')

    def test_record_hash_schema_and_pixel_description_mismatches_are_rejected(self):
        f, cap = api(self, 'frames'), api(self, 'capture')
        mutations = ({'png_sha256': '0' * 64}, {'pixel_sha256': '0' * 64},
                     {'schema_version': 99}, {'canonical_encoding': 'rgb8'}, {'width': 55})
        for changed in mutations:
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as tmp:
                writer = cap.SessionWriter(Path(tmp) / 'session')
                saved = writer.write(cap.decode_ros_image(message(), 's', 1))
                saved.update(changed)
                Path(saved['metadata_path']).write_text(json.dumps(saved), encoding='utf-8')
                with self.assertRaises(ValueError):
                    f.load_image(saved['image_path'])

    def test_encoded_image_modification_is_rejected_even_if_pixels_stay_identical(self):
        f, cap = api(self, 'frames'), api(self, 'capture')
        with tempfile.TemporaryDirectory() as tmp:
            writer = cap.SessionWriter(Path(tmp) / 'session')
            saved = writer.write(cap.decode_ros_image(message(), 's', 1))
            image = Path(saved['image_path'])
            image.write_bytes(image.read_bytes() + b'changed encoded file')
            with self.assertRaises(ValueError):
                f.load_image(image)

    def test_unmarked_json_does_not_claim_capture_provenance(self):
        f = api(self, 'frames')
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / 'ordinary.png'
            image.write_bytes(cv2.imencode('.png', np.zeros((2, 2, 3), np.uint8))[1].tobytes())
            image.with_suffix('.json').write_text(json.dumps({'png_sha256': 'wrong',
                'origin': 'ros_capture_unverified', 'camera_stamp': {'secs': 10, 'nsecs': 0}}), encoding='utf-8')
            loaded = f.load_image(image)
            self.assertNotIn('capture_record', loaded.metadata)
            self.assertEqual(loaded.metadata['origin'], 'offline_replay')

    def test_session_stores_lossless_png_and_independent_digests(self):
        f, cap = api(self, 'frames'), api(self, 'capture')
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'parent' / 'session'
            writer = cap.SessionWriter(output)
            frame = f.Frame(np.array([[[1, 2, 3], [4, 5, 6]]], np.uint8),
                            {'origin': 'offline_replay', 'source_sha256': 'source-file-hash'})
            row = writer.write(frame)
            image_path, json_path = Path(row['image_path']), Path(row['metadata_path'])
            self.assertTrue(image_path.is_file())
            self.assertEqual(json.loads(json_path.read_text(encoding='utf-8')), row)
            self.assertEqual(row['png_sha256'], hashlib.sha256(image_path.read_bytes()).hexdigest())
            self.assertEqual(row['pixel_sha256'], hashlib.sha256(bytes([1, 2, 3, 4, 5, 6])).hexdigest())
            self.assertEqual(row['source_sha256'], 'source-file-hash')
            np.testing.assert_array_equal(f.load_image(image_path).bgr, frame.bgr)
            second = writer.write(frame)
            self.assertNotEqual(row['image_path'], second['image_path'])
            with self.assertRaises(FileExistsError):
                cap.SessionWriter(output)

    def test_session_rejects_sidecar_collision_without_touching_existing_file(self):
        f, cap = api(self, 'frames'), api(self, 'capture')
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'session'
            writer = cap.SessionWriter(output)
            collision = output / 'frame_000001.json'
            collision.write_bytes(b'keep this original')
            with self.assertRaises(FileExistsError):
                writer.write(f.Frame(np.zeros((2, 2, 3), np.uint8), {}))
            self.assertEqual(collision.read_bytes(), b'keep this original')
            self.assertFalse((output / 'frame_000001.png').exists())


class BoundedCaptureTests(unittest.TestCase):
    def test_no_messages_timeout_is_bounded_and_subscription_is_released(self):
        cap = api(self, 'capture')
        subscriptions = []
        class EmptySubscriber:
            def __init__(self, *args, **kwargs):
                del args, kwargs
                self.unregistered = False
                subscriptions.append(self)
            def unregister(self):
                self.unregistered = True
        ros = types.ModuleType('rospy')
        ros.core = types.SimpleNamespace(is_initialized=lambda: True)
        ros.Subscriber, ros.is_shutdown = EmptySubscriber, lambda: False
        sensors, sensor_messages = types.ModuleType('sensor_msgs'), types.ModuleType('sensor_msgs.msg')
        sensor_messages.Image, sensors.msg = object, sensor_messages
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules,
                {'rospy': ros, 'sensor_msgs': sensors, 'sensor_msgs.msg': sensor_messages}):
            started = time.monotonic()
            summary = cap.capture_ros('/synthetic/image', Path(tmp) / 'empty', 3, 0.08)
            self.assertLess(time.monotonic() - started, 0.8)
            self.assertEqual(summary['accepted_frames'], 0)
            self.assertTrue(summary['timed_out'])
            self.assertTrue(subscriptions[0].unregistered)
            saved = json.loads((Path(tmp) / 'empty' / 'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(saved, summary)
            self.assertEqual(saved['status'], 'finished')

    def test_failed_or_interrupted_subscription_leaves_honest_session_summary(self):
        cap = api(self, 'capture')
        for failure in (RuntimeError('synthetic subscription failure'), KeyboardInterrupt()):
            with self.subTest(failure=type(failure)), tempfile.TemporaryDirectory() as tmp:
                ros = types.ModuleType('rospy')
                ros.core = types.SimpleNamespace(is_initialized=lambda: True)
                def fail_subscribe(*args, **kwargs):
                    del args, kwargs
                    raise failure
                ros.Subscriber, ros.is_shutdown = fail_subscribe, lambda: False
                sensors, sensor_messages = types.ModuleType('sensor_msgs'), types.ModuleType('sensor_msgs.msg')
                sensor_messages.Image, sensors.msg = object, sensor_messages
                output = Path(tmp) / 'session'
                with patch.dict(sys.modules, {'rospy': ros, 'sensor_msgs': sensors,
                        'sensor_msgs.msg': sensor_messages}), self.assertRaises(type(failure)):
                    cap.capture_ros('/synthetic/image', output, 1, 0.1)
                saved = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
                self.assertEqual(saved['accepted_frames'], 0)
                self.assertEqual(saved['status'], 'interrupted' if isinstance(failure, KeyboardInterrupt) else 'failed')
                self.assertFalse(saved['max_frames_reached'])

    def test_capture_rejects_replays_regressions_and_identity_conflicts(self):
        cap = api(self, 'capture')
        class FakeSubscriber:
            def __init__(self, topic, kind, callback, **kwargs):
                del topic, kind, kwargs
                self.unregistered = False
                streams = [message(), message(), message(data=bytes([0] * 16)),
                    message(seq=0, secs=11), message(seq=2, secs=9),
                    message(seq=2, secs=11), message(seq=3, secs=12)]
                for item in streams:
                    callback(item)
            def unregister(self):
                self.unregistered = True
        ros = types.ModuleType('rospy')
        ros.core = types.SimpleNamespace(is_initialized=lambda: True)
        ros.Subscriber = FakeSubscriber
        ros.is_shutdown = lambda: False
        sensors = types.ModuleType('sensor_msgs')
        sensor_messages = types.ModuleType('sensor_msgs.msg')
        sensor_messages.Image = object
        sensors.msg = sensor_messages
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules,
                {'rospy': ros, 'sensor_msgs': sensors, 'sensor_msgs.msg': sensor_messages}):
            summary = cap.capture_ros('/synthetic/image', Path(tmp) / 'session', 3, 1)
            self.assertEqual(summary['accepted_frames'], 3)
            self.assertEqual(summary['rejected_frames'], 4)
            self.assertEqual(summary['stopped_reason'], 'max_frames')
            self.assertEqual(summary['rejections']['identity_conflict'], 1)
            self.assertEqual(len(list((Path(tmp) / 'session').glob('*.png'))), 3)
            self.assertEqual(json.loads((Path(tmp) / 'session' / 'summary.json').read_text(encoding='utf-8')), summary)

    def test_invalid_capture_parameters_fail_before_ros_import_or_output_creation(self):
        cap = api(self, 'capture')
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'session'
            for topic, maximum, timeout in (('', 1, 1), ('/x', 0, 1),
                    ('/x', True, 1), ('/x', 1, 0), ('/x', 1, float('inf'))):
                with self.subTest(topic=topic, maximum=maximum, timeout=timeout), self.assertRaises(ValueError):
                    cap.capture_ros(topic, output, maximum, timeout)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
