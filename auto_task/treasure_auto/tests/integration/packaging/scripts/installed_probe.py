"""Run with the installed interpreter and -I; never add source/test paths."""
import contextlib
import hashlib
import importlib
import importlib.abc
import io
import json
from pathlib import Path
import pkgutil
import runpy
import sys
import threading


BLOCKED = {'rospy', 'rosgraph', 'roslib', 'actionlib', 'sensor_msgs', 'geometry_msgs',
           'nav_msgs', 'tf', 'serial', 'cv2', 'contract_fixtures'}
attempts = []


class NoDevices(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in BLOCKED:
            attempts.append('import:' + fullname)
            raise AssertionError('Installed offline code imported device/test dependency: ' + fullname)


def audit(event, args):
    if event in ('socket.connect', 'socket.connect_ex', 'socket.bind',
                 'socket.getaddrinfo', 'subprocess.Popen', 'os.system'):
        attempts.append(event)
        raise AssertionError('Offline operation attempted external access: ' + event)


sys.meta_path.insert(0, NoDevices())
sys.addaudithook(audit)
initial_threads = {thread.ident for thread in threading.enumerate()}
mode, manifest_path = sys.argv[1:]
prefix = Path(sys.prefix).resolve()


def inside_install(path):
    try:
        Path(path).resolve().relative_to(prefix)
        return True
    except ValueError:
        return False


def execute_script(name, arguments):
    bindir = prefix / ('Scripts' if sys.platform == 'win32' else 'bin')
    script = bindir / name
    assert script.is_file(), 'Installed script is missing: ' + str(script)
    sys.argv = [str(script)] + arguments
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        try:
            runpy.run_path(str(script), run_name='__main__')
        except SystemExit as exit_status:
            assert exit_status.code == 0, captured.getvalue()
    return captured.getvalue()


if mode == 'imports':
    import treasure_auto
    names = ['treasure_auto'] + sorted(
        found.name for found in pkgutil.walk_packages(treasure_auto.__path__, 'treasure_auto.'))
    modules = [importlib.import_module(name) for name in names]
    assert all(inside_install(module.__file__) for module in modules)
    assert all(not (Path(path) / 'tests' / 'contract_fixtures').is_dir() for path in sys.path)
    output = {'packages': [module.__name__ for module in modules if hasattr(module, '__path__')],
              'modules': names, 'hardware_attempts': attempts,
              'threads_started': len({thread.ident for thread in threading.enumerate()} - initial_threads),
              'all_modules_installed': True}
elif mode == 'resources':
    from treasure_auto.integration.resources import get_resource_path
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    resources = {name: get_resource_path(name) for name in manifest}
    assert all(path.is_file() and inside_install(path) for path in resources.values())
    output = {'sha256': {name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for name, path in resources.items()}, 'all_resources_installed': True}
elif mode == 'resource_errors':
    from treasure_auto.integration.resources import get_resource_path
    invalid = ('', '.', '..', '../package.xml', '/etc/passwd', 'C:\\Windows\\win.ini',
               'config/../../package.xml', 'config\\..\\hardware.json', 'config/no-such-file.json', 'config')
    rejected = 0
    for value in invalid:
        try:
            get_resource_path(value)
        except (ValueError, FileNotFoundError):
            rejected += 1
        else:
            raise AssertionError('Accepted invalid/missing resource: ' + value)
    actual = get_resource_path('config/mission/offline.example.json')
    share = actual.parents[2]
    assert get_resource_path('config/mission/offline.example.json', root=share) == actual
    try:
        get_resource_path('config/mission/offline.example.json', root=Path.cwd() / 'absent')
    except FileNotFoundError:
        pass
    else:
        raise AssertionError('Explicit missing root incorrectly fell back to another root')
    output = {'rejected': rejected, 'explicit_root': True, 'missing_root_does_not_fallback': True}
elif mode == 'preflight':
    from treasure_auto.integration.resources import get_resource_path
    destination = Path.cwd() / 'installed_preflight_report.json'
    execute_script('device_preflight.py', [
        '--profile', str(get_resource_path('config/hardware.json')),
        '--snapshot', str(get_resource_path('examples/healthy_interfaces.json')),
        '--output', str(destination)])
    report = json.loads(destination.read_text(encoding='utf-8'))
    assert report['passed'] is True and report['mode'] == 'offline'
    assert not attempts
    output = {'passed': True, 'hardware_attempts': attempts}
elif mode == 'vision_help':
    help_text = execute_script('vision_workbench.py', ['--help'])
    assert 'inspect' in help_text and 'capture-ros' in help_text
    assert not attempts
    output = {'help_available': True, 'hardware_attempts': attempts}
elif mode == 'offline_entry':
    destination = Path.cwd() / 'installed_match_report.json'
    execute_script('run_match_offline.py', ['--mode', 'offline', '--layout', 'V1',
        '--scenario', 'nominal', '--output', str(destination)])
    report = json.loads(destination.read_text(encoding='utf-8'))
    assert not attempts
    assert report['status'] == 'passed' and report['mode'] == 'offline'
    output = report['results']['nominal']
else:
    raise AssertionError('Unknown probe mode: ' + mode)
print(json.dumps(output, ensure_ascii=True, sort_keys=True))
