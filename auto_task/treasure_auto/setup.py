"""Catkin-compatible registration and standalone, offline wheel packaging."""
from pathlib import Path
from xml.etree import ElementTree

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent
packages = find_packages(str(ROOT / 'src'), exclude=(
    '*.verification', '*.verification.*', '*.tests', '*.tests.*'))
registration = {'packages': packages, 'package_dir': {'': 'src'}}

try:
    from catkin_pkg.python_setup import generate_distutils_setup
except ModuleNotFoundError as error:
    if error.name not in ('catkin_pkg', 'catkin_pkg.python_setup'):
        raise
    # The offline package requires no ROS installation. Keep its metadata in
    # package.xml so the standalone and catkin distributions cannot drift.
    metadata = ElementTree.parse(str(ROOT / 'package.xml')).getroot()
    registration.update(
        name=metadata.findtext('name'), version=metadata.findtext('version'),
        description=metadata.findtext('description'),
        license=metadata.findtext('license'),
        maintainer=metadata.findtext('maintainer'),
        maintainer_email=metadata.find('maintainer').get('email'))
else:
    registration = generate_distutils_setup(**registration)

shared_files = []
for directory in ('config', 'examples', 'launch'):
    for folder in sorted(path for path in (ROOT / directory).rglob('*') if path.is_dir()):
        files = sorted(path for path in folder.iterdir() if path.is_file())
        if files:
            shared_files.append((
                'share/treasure_auto/' + folder.relative_to(ROOT).as_posix(),
                [path.relative_to(ROOT).as_posix() for path in files]))
    files = sorted(path for path in (ROOT / directory).iterdir() if path.is_file())
    if files:
        shared_files.append(('share/treasure_auto/' + directory,
                             [path.relative_to(ROOT).as_posix() for path in files]))

setup(**dict(registration,
    python_requires='>=3.8',
    include_package_data=False,
    package_data={
        'treasure_auto.contracts': ['schema_v1.json'],
        'treasure_auto.integration': ['*.json'],
    },
    data_files=shared_files,
    scripts=['scripts/device_preflight.py', 'scripts/run_match_offline.py',
             'scripts/vision_workbench.py'],
    entry_points={'console_scripts': [
        'treasure-auto-offline=treasure_auto.integration.cli:main',
        'treasure-auto-vision=treasure_auto.vision_workbench.cli:main',
    ]},
    zip_safe=False,
))
