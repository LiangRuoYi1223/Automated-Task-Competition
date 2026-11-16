"""P00-T01/T04/T05/T06: public boundaries, no ROS or hardware."""
import dataclasses
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
try:
    from treasure_auto.contracts import core as c
except ImportError:
    c = None


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(c, 'P00 core is not implemented')

    def stamp(self, value=10.0, **kw):
        data = dict(seconds=value, basis=c.ClockBasis.MONOTONIC,
                    clock_id='sim-clock', boot_id='sim-boot')
        data.update(kw)
        return c.ClockStamp(**data)

    def test_T01_missing_frame_rejected_with_field(self):
        with self.assertRaises(c.ContractError) as e:
            c.ArmPoint('', 1, 2, 3, 'mm', 0.5)
        self.assertEqual(e.exception.field, 'frame_id')

    def test_T01_nan_and_infinity_rejected(self):
        for value in (float('nan'), float('inf'), -float('inf'), True):
            with self.subTest(value=value), self.assertRaises(c.ContractError) as e:
                c.ArmPoint('arm', value, 2, 3, 'mm', 0.5)
            self.assertEqual(e.exception.field, 'x')

    def test_T01_no_implicit_unit_conversion(self):
        with self.assertRaises(c.ContractError) as e:
            c.ArmPoint('arm', 1, 2, 3, 'm', 0.5)
        self.assertEqual(e.exception.field, 'unit')

    def test_T01_missing_fields_not_zero_filled(self):
        with self.assertRaises(c.ContractError) as e:
            c.from_data(c.ArmPoint, {'frame_id': 'arm', 'y': 2, 'z': 3,
                                     'unit': 'mm', 'error_mm': 0.5})
        self.assertIn('x', e.exception.field)

    def test_T01_map_units_and_error_nonnegative(self):
        for unit, error in (('mm', 0.1), ('m', -0.1)):
            with self.subTest(unit=unit, error=error), self.assertRaises(c.ContractError):
                c.MapPose('map', 1, 2, 0, unit, 'rad', error, 0.01)

    def test_T04_timeout_and_unknown_holding_are_independent(self):
        self.assertNotEqual(c.Outcome.UNKNOWN, c.Outcome.FAILED)
        self.assertNotEqual(c.Outcome.TIMED_OUT, c.Outcome.FAILED)
        result = c.ActionResult(c.Outcome.TIMED_OUT, 'trial_lift',
                                c.Reason(c.ErrorCode.TIMEOUT, 'physical', 'no new observation'),
                                c.Fact.CONFIRMED, c.Fact.CONFIRMED,
                                c.Fact.CONFIRMED, c.Fact.UNKNOWN, (), 2.0)
        holding = c.HoldingSnapshot(c.Holding.UNKNOWN, 'brick-1', None, ())
        self.assertEqual(result.physical, c.Fact.UNKNOWN)
        self.assertEqual(holding.state, c.Holding.UNKNOWN)

    def test_T04_failure_requires_reason(self):
        with self.assertRaises(c.ContractError):
            c.ActionResult(c.Outcome.FAILED, 'grasp', None, c.Fact.UNKNOWN,
                           c.Fact.UNKNOWN, c.Fact.UNKNOWN, c.Fact.UNKNOWN, (), 0)

    def test_T04_held_requires_target_empty_forbids_target(self):
        for state, target in ((c.Holding.HELD, None), (c.Holding.EMPTY, 'brick-1')):
            with self.subTest(state=state), self.assertRaises(c.ContractError):
                c.HoldingSnapshot(state, target, None, ())

    def test_T05_same_clock_subtraction(self):
        self.assertEqual(self.stamp(12) - self.stamp(10), 2)

    def test_T05_cross_clock_or_boot_subtraction_rejected(self):
        for kw in (dict(basis=c.ClockBasis.SENSOR, boot_id=None),
                   dict(clock_id='another'), dict(boot_id='new-boot')):
            with self.subTest(kw=kw), self.assertRaises(c.ContractError) as e:
                self.stamp(12, **kw) - self.stamp(10)
            self.assertEqual(e.exception.code, c.ErrorCode.CLOCK_MISMATCH)

    def test_T05_capture_and_receipt_remain_separate(self):
        evidence = c.EvidenceRef('e-1', 'run-1', 'brick-1', None,
                                 c.EvidenceKind.PHYSICAL, c.Origin.SIMULATED,
                                 self.stamp(100, basis=c.ClockBasis.SENSOR,
                                            clock_id='sim-camera', boot_id=None),
                                 self.stamp(10), self.stamp(11),
                                 'sim://frame/1', None)
        self.assertEqual(evidence.captured_at.seconds, 100)
        self.assertEqual(evidence.received_at.seconds, 10)
        self.assertEqual(c.loads(c.dumps(evidence)), evidence)

    def test_T05_monotonic_requires_boot(self):
        with self.assertRaises(c.ContractError):
            self.stamp(boot_id=None)

    def test_T06_roundtrip_is_immutable(self):
        point = c.ArmPoint('arm', 1, 2, 3, 'mm', 0.1)
        self.assertEqual(c.loads(c.dumps(point)), point)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            point.x = 100

    def test_T06_unknown_version_or_fields_rejected(self):
        data = json.loads(c.dumps(self.stamp()))
        for mutated in (dict(data, contract_version='2.0.0'),
                        dict(data, payload=dict(data['payload'], generation=1)),
                        dict(data, extra=True)):
            with self.subTest(mutated=mutated), self.assertRaises(c.ContractError):
                c.loads(json.dumps(mutated))

    def test_T06_duplicate_json_keys_and_nan_rejected(self):
        for data in ('{"type":"ClockStamp","type":"ArmPoint"}', '{"x":NaN}'):
            with self.subTest(data=data), self.assertRaises(c.ContractError):
                c.loads(data)

    def test_T06_raw_enum_or_mutable_list_rejected(self):
        with self.assertRaises(c.ContractError):
            c.HoldingSnapshot('unknown', None, None, ())
        with self.assertRaises(c.ContractError):
            c.HoldingSnapshot(c.Holding.UNKNOWN, None, None, [])

    def test_T06_nested_error_has_complete_field_path(self):
        data = c.to_data(c.HoldingSnapshot(c.Holding.UNKNOWN, None, None, ()))
        data['tool_offset'] = dict(frame_id='tool', x=float('nan'), y=0, z=0,
                                   unit='mm', error_mm=0)
        with self.assertRaises(c.ContractError) as e:
            c.from_data(c.HoldingSnapshot, data)
        self.assertEqual(e.exception.field, 'tool_offset.x')


if __name__ == '__main__':
    unittest.main()
