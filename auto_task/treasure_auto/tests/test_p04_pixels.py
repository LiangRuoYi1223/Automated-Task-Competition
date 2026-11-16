"""P04-T01/T04/T07: adversarial inputs, with SIMULATED values only."""
import hashlib
import importlib
import json
from pathlib import Path
import sys
import unittest
from dataclasses import replace

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / 'src'))
from treasure_auto.contracts import core as c, models as m
from contract_fixtures.samples import envelope, stamp, sensor_stamp, evidence


def module(test, name):
    try:
        return importlib.import_module('treasure_auto.perception.' + name)
    except ModuleNotFoundError as error:
        test.fail('P04 implementation missing: ' + str(error))


def image_frame(n=1, seconds=20, width=320, height=200, **changes):
    data = bytes([80]) * (width * height)
    ev = evidence(evidence_id='frame-' + str(n), target_id=None, action_id=None,
                  captured_at=sensor_stamp(seconds), uri='sim://frame/' + str(n),
                  sha256=hashlib.sha256(data).hexdigest())
    env = envelope(c.Owner.P01, target_id=None, action_id=None,
                   event_id='image-' + str(n), sequence=n)
    fields = dict(envelope=env, observation_id='observation-' + str(n), image=ev,
                  width=width, height=height, encoding='mono8', camera_frame='camera',
                  distortion=m.Distortion.RAW,
                  capture_pose=c.MapPose('map', 0, 0, 0, 'm', 'rad', .01, .01),
                  preprocessing=m.PixelTransform(0, 0, 1, 1, 0, 0))
    fields.update(changes)
    return m.ImageObservation(**fields), data


def context_for(image, now=111):
    e = image.envelope
    return m.ConsumerContext(e.run_id, e.instance_id, e.control_epoch, stamp(now),
                             e.versions, e.request_id, e.action_id, e.target_id,
                             e.layout_recovery_round, e.rebuild_generation,
                             e.base_motion_epoch, e.origin)


def config_dict():
    return dict(config_version='sim-config-v1', measurement='simulated',
                model_version='sim-model-v1', class_map_version='sim-classes-v1',
                classes={'7': 'doll', '9': 'clock', '11': 'banana', '13': 'cake'},
                frame_size=[320, 200], max_age_s=2.0, min_confidence=.7,
                min_edge_points=3, min_edge_span_px=5., max_line_error_px=1.5,
                max_extrapolation_px=6., min_area_px2=30., min_corner_sine=.15,
                association_distance_px=5., min_confirmations=2,
                tool_min_axis_px=3., tool_max_axis_px=30., tool_max_error_px=1.,
                min_contrast=0., max_saturation_fraction=1., min_sharpness=0.)


class PixelTests(unittest.TestCase):
    def test_T01_non_square_crop_scale_and_padding_inverse(self):
        p = module(self, 'pixels')
        plan = p.TransformPlan('crop-v1', 640, 360, 200, 100, 320, 240,
                               m.PixelTransform(100, 50, 1.5, 2, 10, 20), m.Distortion.RAW)
        restored = plan.restore(p.PixelSet((m.PixelPoint(40, 60), m.PixelPoint(280, 200)), 'backend', 'crop-v1'))
        self.assertEqual(restored.points, (m.PixelPoint(120, 70), m.PixelPoint(280, 140)))
        self.assertEqual(restored.space, 'full')
        with self.assertRaises(c.ContractError):
            plan.restore(restored)

    def test_T01_padding_out_of_bounds_and_wrong_chain_rejected(self):
        p = module(self, 'pixels')
        plan = p.TransformPlan('v', 320, 200, 100, 80, 140, 120,
                               m.PixelTransform(20, 10, 1, 1, 20, 20), m.Distortion.RAW)
        for points, chain in [((m.PixelPoint(2, 20),), 'v'), ((m.PixelPoint(130, 100),), 'v'),
                              ((m.PixelPoint(40, 40),), 'wrong')]:
            with self.subTest(points=points, chain=chain), self.assertRaises(c.ContractError):
                plan.restore(p.PixelSet(points, 'backend', chain))
        with self.assertRaises(c.ContractError):
            p.TransformPlan('v', 320, 200, 100, 80, 90, 120,
                            m.PixelTransform(20, 10, 1, 1, 20, 20), m.Distortion.RAW)

    def test_T01_missing_distortion_or_nonfinite_transform_rejected(self):
        p = module(self, 'pixels')
        with self.assertRaises(c.ContractError):
            p.TransformPlan('v', 320, 200, 320, 200, 320, 200,
                            m.PixelTransform(0, 0, 1, 1, 0, 0), m.Distortion.UNKNOWN)
        with self.assertRaises(c.ContractError):
            m.PixelTransform(0, 0, float('nan'), 1, 0, 0)

    def test_R01_valid_payload_digest_and_current_request(self):
        p = module(self, 'pixels')
        im, data = image_frame()
        gate = p.FrameGate((320, 200), 2.)
        gate.accept(im, p.RawImage(320, 200, 'mono8', data), context_for(im), sensor_stamp(19), sensor_stamp(21))
        self.assertEqual(gate.accepted_count, 1)
        for raw in [p.RawImage(320, 200, 'mono8', data[:-1]), p.RawImage(320, 200, 'mono8', bytes(len(data)))]:
            with self.assertRaises(c.ContractError):
                p.FrameGate((320, 200), 2.).accept(im, raw, context_for(im), sensor_stamp(19), sensor_stamp(21))

    def test_T04_duplicate_capture_and_renamed_evidence_do_not_add_frames(self):
        p = module(self, 'pixels')
        im, data = image_frame()
        gate = p.FrameGate((320, 200), 2.)
        raw = p.RawImage(320, 200, 'mono8', data)
        gate.accept(im, raw, context_for(im), sensor_stamp(19), sensor_stamp(21))
        later, _ = image_frame(2, 20)
        for duplicate in (im, later, replace(later, image=replace(im.image, captured_at=sensor_stamp(21)))):
            with self.subTest(duplicate=duplicate.observation_id), self.assertRaises(c.ContractError):
                gate.accept(duplicate, raw, context_for(im), sensor_stamp(19), sensor_stamp(21))
        self.assertEqual(gate.accepted_count, 1)

    def test_R01_old_future_bad_pose_resolution_origin_and_clock_rejected(self):
        p = module(self, 'pixels')
        im, data = image_frame()
        variants = [replace(im, capture_pose=None), replace(im, width=321),
                    replace(im, envelope=replace(im.envelope, base_motion_epoch=9)),
                    replace(im, envelope=replace(im.envelope, request_id='old')),
                    replace(im, image=replace(im.image, origin=c.Origin.REPLAY)),
                    replace(im, image=replace(im.image, kind=c.EvidenceKind.ECHO)),
                    replace(im, image=replace(im.image, captured_at=sensor_stamp(18))),
                    replace(im, image=replace(im.image, captured_at=sensor_stamp(25))),
                    replace(im, image=replace(im.image, captured_at=stamp(20))),
                    replace(im, image=replace(im.image, received_at=stamp(108)))]
        for item in variants:
            with self.subTest(item=item), self.assertRaises(c.ContractError):
                p.FrameGate((320, 200), 2.).accept(item, p.RawImage(320, 200, 'mono8', data),
                                                context_for(im), sensor_stamp(19), sensor_stamp(21))

    def test_R01_identical_pixels_with_new_capture_are_valid_stationary_observations(self):
        p = module(self, 'pixels')
        gate = p.FrameGate((320, 200), 2.)
        for n, t in [(1, 20), (2, 20.6)]:
            im, data = image_frame(n, t)
            gate.accept(im, p.RawImage(320, 200, 'mono8', data), context_for(im), sensor_stamp(19), sensor_stamp(21))
        self.assertEqual(gate.accepted_count, 2)


class ConfigTests(unittest.TestCase):
    def test_R07_unmeasured_defaults_do_not_enable_live(self):
        cfg = module(self, 'config')
        path = PACKAGE / 'config/perception/unmeasured.json'
        self.assertTrue(path.is_file(), 'explicit unmeasured configuration required')
        parsed = cfg.PerceptionConfig.from_dict(json.loads(path.read_text(encoding='utf-8')))
        with self.assertRaises(c.ContractError) as caught:
            parsed.require_ready(c.Origin.LIVE)
        self.assertEqual(caught.exception.code, c.ErrorCode.UNCONFIGURED)

    def test_R02_exact_mapping_and_simulated_parameters_do_not_enable_live(self):
        cfg = module(self, 'config')
        parsed = cfg.PerceptionConfig.from_dict(config_dict())
        parsed.require_ready(c.Origin.SIMULATED)
        self.assertEqual(parsed.category('7'), m.Category.DOLL)
        self.assertIsNone(parsed.category('teddy bear'))
        with self.assertRaises(c.ContractError):
            parsed.require_ready(c.Origin.LIVE)

    def test_R07_invalid_unknown_boolean_and_nonfinite_config_rejected(self):
        cfg = module(self, 'config')
        changes = [dict(max_age_s=-1), dict(min_confirmations=True), dict(min_confidence=float('nan')),
                   dict(frame_size=[0, 200]), dict(classes={'0': 'teddy bear'}), dict(typo=1),
                   dict(tool_min_axis_px=50., tool_max_axis_px=10.), dict(min_corner_sine=2.)]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(c.ContractError):
                cfg.PerceptionConfig.from_dict(dict(config_dict(), **change))


if __name__ == '__main__':
    unittest.main()
