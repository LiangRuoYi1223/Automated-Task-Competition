"""P01-T01: preserve read-only CLI even after the execution package is importable."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / 'src'))
from treasure_auto.cli import main
from treasure_auto.devices.gateway import DeviceGateway


class ReadOnlyTests(unittest.TestCase):
    def test_T01_offline_cli_calls_no_fake_actuator(self):
        with patch.object(DeviceGateway, 'submit_base') as base, \
                patch.object(DeviceGateway, 'submit_arm') as arm, \
                patch.object(DeviceGateway, 'submit_pump') as pump, \
                patch.object(DeviceGateway, 'stop') as stop, redirect_stdout(io.StringIO()):
            code = main(['--profile', str(PACKAGE / 'config/hardware.json'),
                         '--snapshot', str(PACKAGE / 'examples/healthy_interfaces.json')])
        self.assertEqual(code, 0)
        self.assertEqual([base.call_count, arm.call_count, pump.call_count, stop.call_count], [0, 0, 0, 0])


if __name__ == '__main__':
    unittest.main()
