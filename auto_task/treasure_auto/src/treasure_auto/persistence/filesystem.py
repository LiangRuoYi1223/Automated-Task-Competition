"""Bounded local paths. Root must be dedicated and not attacker-writable.

No links/reparse points/hardlinks are traversed. This is not a hostile-user
filesystem sandbox: administrators changing the tree concurrently are unsupported.
"""
import hashlib
import json
import os
import re
import stat
import tempfile
from pathlib import Path
from treasure_auto.contracts.core import require, ErrorCode, ContractError


def digest(data):
    return hashlib.sha256(data).hexdigest()


def run_name(value):
    require(isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', value) is not None,
            'run_id', 'portable single directory name required')
    require(value.upper() not in {'CON', 'PRN', 'AUX', 'NUL'} |
            {'COM' + str(i) for i in range(10)} | {'LPT' + str(i) for i in range(10)},
            'run_id', 'reserved device name')
    return value


def is_link(path):
    info = os.lstat(str(path))
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, 'st_file_attributes', 0) & 0x400)


def checked(path, root=None):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts, 'path', 'absolute path without parent traversal required')
    if root is not None:
        root = Path(root)
        require(path != root and root in path.parents, 'path', 'outside dedicated root', ErrorCode.NOT_AUTHORIZED)
    for part in reversed((path,) + tuple(path.parents)):
        if os.path.lexists(str(part)):
            require(not is_link(part), 'path', 'link or reparse point refused', ErrorCode.NOT_AUTHORIZED)
            info = os.lstat(str(part))
            require(not stat.S_ISREG(info.st_mode) or info.st_nlink == 1,
                    'path', 'hardlinked file refused', ErrorCode.NOT_AUTHORIZED)
    return path


def sync_directory(directory):
    # Windows has no stdlib directory FlushFileBuffers. Do not claim power-loss
    # durability there. Linux fsync failures propagate rather than being ignored.
    if os.name != 'nt':
        fd = os.open(str(directory), os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def atomic_write(path, content):
    path = checked(path)
    require(path.parent.is_dir(), 'path', 'parent must already exist')
    fd, name = tempfile.mkstemp(prefix='.pending-', dir=str(path.parent))
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        require(digest(temporary.read_bytes()) == digest(content), 'file', 'write verification failed', ErrorCode.STORAGE_FAILURE)
        checked(path)
        os.replace(str(temporary), str(path))
        sync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json(path, value):
    atomic_write(path, json.dumps(value, ensure_ascii=False, sort_keys=True,
                                  separators=(',', ':'), allow_nan=False).encode('utf-8'))


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'json', 'duplicate key')
            result[key] = value
        return result
    return json.loads(checked(path).read_text(encoding='utf-8'), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def scan_tree(path, root):
    path = checked(path, root)
    if not path.exists():
        return
    for child in path.iterdir():
        checked(child, root)
        info = child.lstat()
        require(stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode), 'path', 'special file refused')
        if child.is_dir():
            scan_tree(child, root)


def remove_run(path, root):
    path, root = Path(path), Path(root)
    require(path.parent == root / 'runs', 'path', 'only direct run directory may be removed')
    run_name(path.name)
    scan_tree(path, root)  # Complete preflight before deleting even one file.
    def remove(directory):
        checked(directory, root)
        if not directory.exists():
            return
        for child in directory.iterdir():
            checked(child, root)
            if child.is_dir():
                remove(child)
            else:
                child.unlink()
        directory.rmdir()
    remove(path)
    sync_directory(path.parent)


class RootLease:
    """Nonblocking OS process lock, held until the writer really exits."""
    def __init__(self, path):
        self.stream = None
        path = checked(path)
        stream = open(str(path), 'a+b')
        try:
            if os.name == 'nt':
                import msvcrt
                if path.stat().st_size == 0:
                    stream.write(b'0')
                    stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.stream = stream
        except OSError as error:
            stream.close()
            raise ContractError(ErrorCode.CONFLICT, 'runtime_root', 'another writer owns the root: ' + str(error))

    def close(self):
        if self.stream is not None:
            if os.name == 'nt':
                import msvcrt
                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
            self.stream.close()
            self.stream = None
