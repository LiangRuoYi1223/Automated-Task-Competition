"""Conservative fixed-art template baseline, with no trained model or COCO map.

The caller supplies an already perspective-rectified BGR top-face *candidate*.
Actual user templates belong in doll/, clock/, banana/ and cake/ subdirectories;
this module does not ship synthetic art as usable competition templates.

Comparison uses 128 x 128 grayscale images and zero-mean, unit-norm normalized
cross-correlation (NCC). Each template is searched over in-plane rotations
0, 5, ..., 355 degrees. Scores in [-1, 1] are similarities, not probabilities.
This simple baseline assumes comparable centering, scale and crop boundaries.
It does not compensate for residual perspective, occlusion, reflections,
uneven lighting, or substantial translation. Rotation interpolation and square
clipping can also affect its score; a 5-degree search is not a guarantee for
every angle. Synthetic unit tests establish program behavior, not field accuracy.

Importing this module, or constructing its empty preparation state, does not
import OpenCV or NumPy. No source images are modified or written.
"""
import math
import hashlib
from copy import deepcopy
from numbers import Real
from pathlib import Path


CATEGORIES = ('doll', 'clock', 'banana', 'cake')
METHOD = 'grayscale_ncc_rotation'
_IMAGE_SUFFIXES = frozenset(('.png', '.jpg', '.jpeg'))
_SIZE = 128
_MIN_SIDE = 24
_MIN_GRAY_STD = 3.0
_ROTATION_STEP = 5


def _threshold(name, value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(name + ' must be a finite number in [0, 1]')
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(name + ' must be a finite number in [0, 1]')
    return value


def _validate_bgr(bgr):
    import numpy as np

    if not isinstance(bgr, np.ndarray):
        raise TypeError('image must be a NumPy uint8 BGR array')
    if bgr.dtype != np.uint8:
        raise ValueError('image dtype must be uint8')
    if bgr.ndim != 3 or bgr.shape[2] != 3 or min(bgr.shape[:2]) <= 0:
        raise ValueError('image must have positive height and width and 3 BGR channels')


def _gray(bgr):
    import cv2

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (_SIZE, _SIZE), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(gray, (3, 3), 0).astype('float32')


def _unit_vector(gray):
    import numpy as np

    vector = gray.reshape(-1).copy()
    vector -= float(vector.mean())
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-6:
        return None
    return vector / norm


def _rotation_bank(gray):
    import cv2
    import numpy as np

    border = np.concatenate((gray[0], gray[-1], gray[1:-1, 0], gray[1:-1, -1]))
    background = float(np.median(border))
    center = ((_SIZE - 1) / 2.0, (_SIZE - 1) / 2.0)
    vectors = []
    for angle in range(0, 360, _ROTATION_STEP):
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(gray, matrix, (_SIZE, _SIZE),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=background)
        vector = _unit_vector(rotated)
        if vector is not None:
            vectors.append(vector)
    return vectors


class TemplateClassifier:
    """Load actual fixed-art crops and conservatively classify candidate pixels.

    Args:
        directory: Existing template root, or None for empty preparation mode.
            Each class subdirectory may contain multiple PNG/JPEG files directly.
            Missing or empty classes keep ``ready`` false. Corrupt, too-small or
            blank supported files are configuration errors, never silently loaded.
        min_score: Minimum NCC similarity, finite number in [0, 1]. Default .75.
        min_margin: Required best-minus-runner-up class score, in [0, 1].
            Default .08. Exact cross-class ties always reject, even if set to zero.

    All four classes must be present before any known category is returned.
    Templates are read once at construction; reconstruct after changing files.
    """

    def __init__(self, directory=None, min_score=0.75, min_margin=0.08):
        self.min_score = _threshold('min_score', min_score)
        self.min_margin = _threshold('min_margin', min_margin)
        self._banks = {category: None for category in CATEGORIES}
        self._manifest = dict(method=METHOD, template_root=None, files=[],
                              comparison_size_px=_SIZE, rotation_step_degrees=_ROTATION_STEP,
                              min_score=self.min_score, min_margin=self.min_margin,
                              score_basis='zero-mean unit-norm grayscale NCC; not probability')
        if directory is None:
            return
        if isinstance(directory, str) and not directory.strip():
            raise ValueError('template directory must not be an empty string')
        root = Path(directory).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError('template directory does not exist: ' + str(root))
        if not root.is_dir():
            raise NotADirectoryError('template root is not a directory: ' + str(root))
        self._manifest['template_root'] = str(root)

        import cv2
        import numpy as np

        for category in CATEGORIES:
            class_directory = root / category
            if not class_directory.exists():
                continue
            if not class_directory.is_dir():
                raise NotADirectoryError('template class is not a directory: ' + str(class_directory))
            vectors = []
            paths = sorted(path for path in class_directory.iterdir()
                           if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES)
            for path in paths:
                # Reading bytes also handles non-ASCII Windows paths reliably.
                encoded = path.read_bytes()
                data = np.frombuffer(encoded, dtype=np.uint8)
                try:
                    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
                except cv2.error as error:
                    raise ValueError('cannot decode template: ' + str(path)) from error
                if bgr is None:
                    raise ValueError('cannot decode template: ' + str(path))
                if min(bgr.shape[:2]) < _MIN_SIDE:
                    raise ValueError('template is too small (minimum side 24 px): ' + str(path))
                gray = _gray(bgr)
                if float(gray.std()) < _MIN_GRAY_STD:
                    raise ValueError('template is blank or low contrast: ' + str(path))
                vectors.extend(_rotation_bank(gray))
                self._manifest['files'].append(dict(category=category, path=str(path.resolve()),
                                                     sha256=hashlib.sha256(encoded).hexdigest()))
            if vectors:
                self._banks[category] = np.stack(vectors)

    @property
    def manifest(self):
        """A defensive copy identifying exactly the loaded template bytes."""
        return deepcopy(self._manifest)

    @property
    def ready(self):
        """Whether usable template files were loaded for all four categories."""
        return all(self._banks[category] is not None for category in CATEGORIES)

    def _result(self, category='unknown', score=None, scores=None, reason='templates_missing'):
        return dict(category=category, score=score,
                    scores={} if scores is None else scores,
                    reason=reason, method=METHOD, templates_ready=self.ready)

    def classify(self, bgr):
        """Return category/score/scores/reason/method/templates_ready.

        Malformed arrays raise TypeError/ValueError, including when templates
        are missing. Well-formed uint8 BGR images may return unknown because
        templates are missing, the image is too small/blank, the score is low,
        or two categories are too similar. Such refusals are normal results.
        The input array is never modified.
        """
        _validate_bgr(bgr)
        if not self.ready:
            return self._result()
        if min(bgr.shape[:2]) < _MIN_SIDE:
            return self._result(reason='image_too_small')
        gray = _gray(bgr)
        if float(gray.std()) < _MIN_GRAY_STD:
            return self._result(reason='image_blank')
        vector = _unit_vector(gray)
        scores = {}
        for category in CATEGORIES:
            similarity = float((self._banks[category] @ vector).max())
            scores[category] = max(-1.0, min(1.0, similarity))
        ordered = sorted(scores, key=scores.get, reverse=True)
        best = ordered[0]
        score = scores[best]
        if score < self.min_score:
            return self._result(score=score, scores=scores, reason='low_score')
        margin = score - scores[ordered[1]]
        if margin <= 1e-7 or margin < self.min_margin:
            return self._result(score=score, scores=scores, reason='ambiguous')
        return self._result(category=best, score=score, scores=scores, reason='matched')
