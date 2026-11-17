"""Run the comparison's offline checks without changing the global environment."""
import ast
from pathlib import Path
import sys
import unittest


def main():
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    local = root / '.local_deps'
    if local.is_dir():
        sys.path.insert(0, str(local))
    files = [p for p in root.rglob('*.py') if '.local_deps' not in p.parts]
    for path in files:
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path), feature_version=(3, 8))
    print('Python 3.8 syntax: {} own files passed (not a Python 3.8 runtime test)'.format(len(files)))
    tests = unittest.defaultTestLoader.discover(str(root / 'tests'))
    result = unittest.TextTestRunner(verbosity=2).run(tests)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
