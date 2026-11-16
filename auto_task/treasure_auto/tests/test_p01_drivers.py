"""P01-R01/R02/R06 bounded polling and single owner, no subprocess/device starts."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from contract_fixtures import samples as s
from contract_fixtures.fakes import FakeClock
try:
    from treasure_auto.devices.drivers import DriverSession, SerialLease
except ImportError:
    DriverSession = SerialLease = None


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(DriverSession, 'P01 bounded connection state machine required')
        self.clock = FakeClock(s.stamp(110))

    def session(self, **changes):
        options = dict(clock=self.clock, timeout_s=1, managed=True, recovering=False,
                       startup_effects_verified=False, origin='simulated')
        options.update(changes)
        return DriverSession(**options)

    def test_T01_existing_qualified_driver_reused_without_launch(self):
        session = self.session()
        result = session.poll(process=True, interface=True, messages=True, transforms=True)
        self.assertEqual(result.state, 'ready')
        self.assertEqual(result.operation, 'reuse')
        self.assertFalse(result.physical_verified)

    def test_T02_process_and_interface_do_not_prove_messages(self):
        session = self.session()
        self.assertEqual(session.poll(True, True, False, False).state, 'waiting_messages')
        self.clock.advance(1)
        self.assertEqual(session.poll(True, True, True, True).state, 'timed_out')

    def test_T02_all_connection_stages_share_one_deadline(self):
        session = self.session(startup_effects_verified=True)
        self.assertEqual(session.poll(False, False, False, False).operation, 'start_once')
        self.clock.advance(.6)
        self.assertEqual(session.poll(True, False, False, False).state, 'waiting_interface')
        self.clock.advance(.4)
        self.assertEqual(session.poll(True, True, False, False).state, 'timed_out')

    def test_T06_recovery_or_unknown_side_effects_never_request_restart(self):
        for session in (self.session(), self.session(recovering=True, startup_effects_verified=True),
                        self.session(origin='live', startup_effects_verified=True)):
            with self.subTest(session=session):
                self.assertEqual(session.poll(False, False, False, False).operation, 'none')

    def test_T02_launch_failure_is_terminal_and_not_retried(self):
        session = self.session(startup_effects_verified=True)
        session.poll(False, False, False, False)
        result = session.poll(False, False, False, False, launch_error='synthetic launch error')
        self.assertEqual(result.state, 'failed')
        self.assertEqual(session.poll(True, True, True, True).state, 'failed')

    def test_T06_serial_owner_excludes_second_adapter(self):
        self.assertIsNotNone(SerialLease, 'P01 serial ownership required')
        first, second = SerialLease('sim://arm-1'), SerialLease('sim://arm-1')
        with first:
            with self.assertRaises(RuntimeError):
                second.acquire()
        with second:
            self.assertTrue(second.owned)

    def test_T02_external_driver_is_never_started(self):
        session = self.session(managed=False, startup_effects_verified=True)
        self.assertEqual(session.poll(False, False, False, False).operation, 'none')


if __name__ == '__main__':
    unittest.main()
