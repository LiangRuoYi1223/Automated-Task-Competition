"""P01-T02/T06: real health logic with synthetic observations, never a robot."""
import dataclasses
import json
from pathlib import Path
import sys
import unittest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / 'src'))
from contract_fixtures import samples as s
from contract_fixtures.fakes import FakeClock
from treasure_auto.contracts import core as c
try:
    from treasure_auto.devices.health import DeviceRule, DeviceObservation, HealthMonitor
    from treasure_auto.devices.config import load_control_config
except ImportError:
    DeviceRule = DeviceObservation = HealthMonitor = load_control_config = None


class HealthTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(HealthMonitor, 'P01 health implementation required')
        self.clock = FakeClock(s.stamp(110))
        self.rule = DeviceRule('camera', 'sensor_msgs/Image', '/camera', 'camera',
                               'sim-driver-v1', 0.5, ('base->camera',), 2, 0.1)
        self.monitor = HealthMonitor(self.clock, s.envelope(c.Owner.P01),
                                     (self.rule,), {'vision': ('camera',)})

    def observation(self, **changes):
        values = dict(sample_id='f1', received_at=self.clock.now(), captured_at=s.sensor_stamp(),
                      process_running=True, driver='/camera', actual_type='sensor_msgs/Image',
                      subscriber_type='sensor_msgs/Image', message_count=1, frame_id='camera',
                      driver_version='sim-driver-v1', transforms=('base->camera',),
                      valid_content=True)
        values.update(changes)
        return DeviceObservation(**values)

    def ready(self):
        self.monitor.update('camera', self.observation())
        self.clock.advance(.11)
        self.monitor.update('camera', self.observation(sample_id='f2', message_count=2,
                                                     captured_at=s.sensor_stamp(20.11)))

    def test_T02_node_without_message_is_unavailable(self):
        self.monitor.update('camera', self.observation(message_count=0))
        self.assertEqual(self.monitor.health().capabilities[0].status, c.Fact.REJECTED)
        self.assertIn('message', self.monitor.health().capabilities[0].reason.field)

    def test_T02_wrong_actual_subscription_and_missing_tf_reject(self):
        for fields in ({'actual_type': 'std_msgs/Empty'}, {'subscriber_type': 'std_msgs/Empty'},
                       {'transforms': ()}, {'driver_version': 'wrong'},
                       {'driver': '/imposter'}, {'frame_id': 'wrong'}, {'valid_content': False},
                       {'received_at': s.stamp(112)}, {'received_at': s.stamp(110, 'other-boot')}):
            with self.subTest(fields=fields):
                self.monitor.update('camera', self.observation(**fields))
                self.assertNotEqual(self.monitor.health().capabilities[0].status, c.Fact.CONFIRMED)

    def test_T02_stale_observation_revokes(self):
        self.ready()
        self.assertEqual(self.monitor.health().capabilities[0].status, c.Fact.CONFIRMED)
        self.clock.advance(.501)
        self.assertEqual(self.monitor.health().capabilities[0].status, c.Fact.REJECTED)

    def test_T02_snapshot_expires_with_last_message_not_last_poll(self):
        self.ready()
        self.clock.advance(.4)
        snapshot = self.monitor.health()
        self.assertLessEqual(snapshot.envelope.expires_at.seconds, 110.61)

    def test_T02_confirmed_capability_carries_interface_not_physical_evidence(self):
        self.ready()
        snapshot = self.monitor.health()
        self.assertTrue(snapshot.capabilities[0].evidence)
        for ev in snapshot.capabilities[0].evidence:
            self.assertEqual(ev.kind, c.EvidenceKind.INTERFACE)
            self.assertEqual(ev.origin, c.Origin.SIMULATED)
            self.assertEqual(ev.received_at, self.clock.now())
            self.assertLessEqual(ev.valid_until.seconds, 110.61)

    def test_T06_duplicate_frame_cannot_keep_ready_capability_fresh(self):
        self.ready()
        self.clock.advance(.4)
        self.monitor.update('camera', self.observation(sample_id='f2', message_count=2,
                                                     captured_at=s.sensor_stamp(20.11)))
        self.clock.advance(.11)
        self.assertEqual(self.monitor.health().capabilities[0].status, c.Fact.REJECTED)

    def test_T06_repeated_frame_cannot_accumulate_recovery(self):
        original = self.observation()
        self.monitor.update('camera', original)
        self.clock.advance(.2)
        self.monitor.update('camera', dataclasses.replace(original, received_at=self.clock.now()))
        self.assertEqual(self.monitor.health().capabilities[0].status, c.Fact.UNKNOWN)

    def test_T06_stale_gap_resets_recovery_even_without_polling(self):
        self.monitor.update('camera', self.observation())
        self.clock.advance(.6)
        self.monitor.update('camera', self.observation(sample_id='f2', message_count=2,
                                                     captured_at=s.sensor_stamp(20.6)))
        self.assertEqual(self.monitor.health().capabilities[0].status, c.Fact.UNKNOWN)

    def test_T06_camera_failure_does_not_revoke_unrelated_base(self):
        self.monitor = HealthMonitor(self.clock, s.envelope(c.Owner.P01),
                                    (self.rule, dataclasses.replace(self.rule, name='odom')),
                                    {'vision': ('camera',), 'base': ('odom',)})
        for name in ('camera', 'odom'):
            self.monitor.update(name, self.observation())
        self.clock.advance(.11)
        for name in ('camera', 'odom'):
            self.monitor.update(name, self.observation(sample_id='f2', message_count=2,
                                                     captured_at=s.sensor_stamp(20.11)))
        self.monitor.update('camera', self.observation(valid_content=False))
        statuses = {cap.name: cap.status for cap in self.monitor.health().capabilities}
        self.assertEqual(statuses, {'vision': c.Fact.REJECTED, 'base': c.Fact.CONFIRMED})

    def test_T02_missing_unmeasured_policy_cannot_enable_monitor(self):
        for value in (None, 0, float('nan'), True):
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                dataclasses.replace(self.rule, max_age_s=value)


class ConfigTests(unittest.TestCase):
    def test_T05_unmeasured_default_has_no_motion_limits(self):
        self.assertIsNotNone(load_control_config, 'P01 config validator required')
        config = load_control_config(PACKAGE / 'config/devices/device_control.json')
        self.assertIsNone(config.max_linear_m_s)
        self.assertIsNone(config.arm_bounds_mm)
        self.assertFalse(config.real_output_enabled)

    def test_T05_config_rejects_unsafe_and_unknown_values(self):
        self.assertIsNotNone(load_control_config, 'P01 config validator required')
        from treasure_auto.devices.config import validate_control_config
        data = json.loads((PACKAGE / 'config/devices/device_control.json').read_text(encoding='utf-8'))
        for field, value in (('max_linear_m_s', float('nan')), ('real_output_enabled', True),
                             ('arm_adapter', 'fixed_F10000'), ('contract_version', '2.0.0'),
                             ('extra', 1), ('max_angular_rad_s', True)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_control_config(dict(data, **{field: value}))


if __name__ == '__main__':
    unittest.main()
