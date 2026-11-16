"""Locate installed package resources without importing ROS or changing paths."""
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import sys


def get_resource_path(relative, root=None):
    """Return an existing resource file as an absolute :class:`Path`.

    Names are relative to the package share directory, e.g.
    ``config/mission/offline.example.json``. Internal data such as
    ``contracts/schema_v1.json`` and ``integration/dependencies.json`` reside
    next to the installed modules. Wheels install shared data under
    ``<prefix>/share/treasure_auto``; catkin uses the same share layout.

    An explicit ``root`` is authoritative and disables automatic fallback.
    Absolute names, traversal, directories and missing files are rejected.
    """
    name = os.fspath(relative)
    if not isinstance(name, str) or not name or '\x00' in name:
        raise ValueError('A non-empty relative resource name is required')
    name = name.replace('\\', '/')
    resource = PurePosixPath(name)
    if (resource.is_absolute() or PureWindowsPath(name).drive or
            not resource.parts or '..' in resource.parts):
        raise ValueError('Resource names must remain inside their resource root')

    package = Path(__file__).resolve().parents[1]
    if root is not None:
        roots = [Path(root)]
    else:
        roots = [package]
        # A checkout/devel-space relay keeps resources in the package root.
        if package.parent.name == 'src':
            roots.append(package.parent.parent)
        # The module location also finds non-default catkin/pip --prefix installs.
        roots.extend(parent / 'share' / 'treasure_auto' for parent in package.parents)
        roots.append(Path(sys.prefix) / 'share' / 'treasure_auto')

    visited = set()
    for base in roots:
        base = base.resolve()
        if base in visited:
            continue
        visited.add(base)
        path = base.joinpath(*resource.parts).resolve()
        try:
            path.relative_to(base)
        except ValueError:
            raise ValueError('Resource resolves outside its resource root')
        if path.is_file():
            return path
    raise FileNotFoundError('Resource is not installed at the selected root: ' + name)
