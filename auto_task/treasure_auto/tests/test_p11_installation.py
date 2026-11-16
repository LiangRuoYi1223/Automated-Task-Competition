"""P11-R08/T01/T08: exercise a real wheel in an isolated installation.

Set P11_BUILD_PYTHON to an offline Python with setuptools, wheel and pip when
the interpreter running unittest does not provide those build tools. All
builds, virtual environments, probes and logs remain under packaging/artifacts.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile


PACKAGE = Path(__file__).resolve().parents[1]
PACKAGING = PACKAGE / 'tests' / 'integration' / 'packaging'
BUILDER = os.environ.get('P11_BUILD_PYTHON', sys.executable)
EXPECTED_PACKAGES = (
    'treasure_auto', 'treasure_auto.classification', 'treasure_auto.contracts',
    'treasure_auto.devices', 'treasure_auto.geometry', 'treasure_auto.integration',
    'treasure_auto.manipulation', 'treasure_auto.mapping', 'treasure_auto.mission',
    'treasure_auto.navigation', 'treasure_auto.perception', 'treasure_auto.persistence',
    'treasure_auto.placement', 'treasure_auto.runtime', 'treasure_auto.vision_workbench',
)


class InstalledWheelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        artifacts = PACKAGING / 'artifacts'
        artifacts.mkdir(parents=True, exist_ok=True)
        cls.work = Path(tempfile.mkdtemp(prefix='run-', dir=str(artifacts)))
        cls.source = cls.work / 'source'
        cls.source.mkdir()
        for name in ('src', 'config', 'examples', 'launch', 'scripts'):
            shutil.copytree(str(PACKAGE / name), str(cls.source / name),
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        for name in ('setup.py', 'package.xml', 'CMakeLists.txt'):
            shutil.copy2(str(PACKAGE / name), str(cls.source / name))
        cls.outside = cls.work / 'outside-source'
        cls.outside.mkdir()
        temporary = cls.work / 'tmp'
        temporary.mkdir()
        cls.environment = os.environ.copy()
        for key in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP', 'VIRTUAL_ENV'):
            cls.environment.pop(key, None)
        cls.environment.update(PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1',
                               PIP_NO_INDEX='1', PIP_DISABLE_PIP_VERSION_CHECK='1',
                               TEMP=str(temporary), TMP=str(temporary), TMPDIR=str(temporary))
        cls.build = cls.run_command('build', [BUILDER, '-I', '-m', 'pip', 'wheel',
            '--no-index', '--no-deps', '--no-build-isolation', '--no-cache-dir',
            '--wheel-dir', str(cls.work / 'wheel'), str(cls.source)])
        cls.wheel = next((cls.work / 'wheel').glob('*.whl'), None)
        cls.install = None
        if cls.build.returncode == 0 and cls.wheel is not None:
            cls.venv = cls.work / 'venv'
            cls.venv_result = cls.run_command('venv', [BUILDER, '-I', '-m', 'venv',
                                                     '--without-pip', str(cls.venv)])
            cls.python = cls.venv / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
            if cls.venv_result.returncode == 0:
                cls.install = cls.run_command('install', [BUILDER, '-I', '-m', 'pip',
                    '--python', str(cls.python), 'install', '--no-index', '--no-deps',
                    '--no-compile', str(cls.wheel)])
        cls.resource_hashes = {}
        for directory in ('config', 'examples', 'launch'):
            for path in (cls.source / directory).rglob('*'):
                if path.is_file():
                    cls.resource_hashes[path.relative_to(cls.source).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        for relative in ('contracts/schema_v1.json', 'integration/dependencies.json'):
            path = cls.source / 'src' / 'treasure_auto' / relative
            if path.is_file():
                cls.resource_hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        cls.manifest = cls.work / 'expected_resources.json'
        cls.manifest.write_text(json.dumps(cls.resource_hashes, indent=2), encoding='utf-8')
        (artifacts / 'latest.json').write_text(json.dumps({
            'run_directory': str(cls.work), 'builder': str(BUILDER),
            'build_returncode': cls.build.returncode,
            'install_returncode': None if cls.install is None else cls.install.returncode,
            'wheel': None if cls.wheel is None else str(cls.wheel),
            'wheel_sha256': None if cls.wheel is None else hashlib.sha256(cls.wheel.read_bytes()).hexdigest(),
        }, indent=2), encoding='utf-8')

    @classmethod
    def run_command(cls, label, command):
        result = subprocess.run(command, cwd=str(cls.outside), env=cls.environment,
                                capture_output=True, text=True, encoding='utf-8',
                                errors='replace', timeout=120)
        (cls.work / (label + '.log')).write_text(
            json.dumps(command, ensure_ascii=False) + '\nexit=' + str(result.returncode) +
            '\n' + result.stdout + '\n' + result.stderr, encoding='utf-8')
        return result

    def require_installation(self):
        self.assertEqual(self.build.returncode, 0, self.build.stdout + self.build.stderr)
        self.assertIsNotNone(self.wheel, 'Build did not produce a wheel')
        self.assertEqual(self.venv_result.returncode, 0,
                         self.venv_result.stdout + self.venv_result.stderr)
        self.assertIsNotNone(self.install, 'Installation was not attempted')
        self.assertEqual(self.install.returncode, 0, self.install.stdout + self.install.stderr)

    def probe(self, mode):
        self.require_installation()
        result = self.run_command(mode, [str(self.python), '-I', '-B',
            str(PACKAGING / 'scripts' / 'installed_probe.py'), mode, str(self.manifest)])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return json.loads(result.stdout)

    def test_T08_real_wheel_builds_and_installs_without_catkin(self):
        self.require_installation()

    def test_T08_installed_subpackages_import_without_devices_or_source_paths(self):
        report = self.probe('imports')
        self.assertEqual(report['packages'], list(EXPECTED_PACKAGES))
        self.assertGreaterEqual(len(report['modules']), 90)
        self.assertEqual(report['hardware_attempts'], [])
        self.assertEqual(report['threads_started'], 0)
        self.assertTrue(report['all_modules_installed'])

    def test_T08_installed_resources_are_complete_and_match_sources(self):
        report = self.probe('resources')
        self.assertEqual(report['sha256'], self.resource_hashes)
        self.assertIn('contracts/schema_v1.json', report['sha256'])
        self.assertIn('integration/dependencies.json', report['sha256'])
        self.assertIn('launch/match_offline.launch', report['sha256'])
        self.assertTrue(report['all_resources_installed'])

    def test_T08_resource_lookup_rejects_escape_and_missing_paths(self):
        report = self.probe('resource_errors')
        self.assertEqual(report['rejected'], 10)
        self.assertTrue(report['explicit_root'])
        self.assertTrue(report['missing_root_does_not_fallback'])

    def test_T01_original_installed_preflight_remains_offline(self):
        self.probe('preflight')

    def test_T08_installed_offline_entry_completes_simulated_match(self):
        report = self.probe('offline_entry')
        self.assertEqual(report['arrivals'], 10)
        self.assertEqual(report['origin'], 'simulated')
        self.assertEqual(report['hardware_outputs'], 0)

    def test_T08_wheel_excludes_tests_and_verification_logs(self):
        self.require_installation()
        with zipfile.ZipFile(str(self.wheel)) as archive:
            names = archive.namelist()
        self.assertFalse([name for name in names if any(
            part in ('tests', 'verification', '__pycache__') for part in name.split('/'))], names)
        self.assertFalse([name for name in names if name.endswith(('.log', '.pyc'))], names)
        self.assertTrue(any(name.endswith('/scripts/device_preflight.py') for name in names))
        self.assertTrue(any(name.endswith('/scripts/run_match_offline.py') for name in names))
        self.assertTrue(any(name.endswith('/scripts/vision_workbench.py') for name in names))

    def test_installed_vision_help_needs_no_image_or_ros_dependencies(self):
        report = self.probe('vision_help')
        self.assertTrue(report['help_available'])
        self.assertEqual(report['hardware_attempts'], [])


class PackagingSyntaxTests(unittest.TestCase):
    def test_packaging_sources_use_python38_grammar(self):
        for path in (PACKAGE / 'setup.py',
                     PACKAGE / 'src' / 'treasure_auto' / '__init__.py',
                     PACKAGE / 'src' / 'treasure_auto' / 'integration' / 'resources.py'):
            with self.subTest(path=str(path)):
                self.assertTrue(path.is_file(), 'Missing packaging source: ' + str(path))
                ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path), feature_version=(3, 8))


if __name__ == '__main__':
    unittest.main()
