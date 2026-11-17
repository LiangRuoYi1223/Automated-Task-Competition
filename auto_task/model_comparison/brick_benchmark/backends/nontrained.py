"""Read-only adapter around the existing nontrained workbench."""
import hashlib
from pathlib import Path
import sys
from ..contracts import Instance


def source_hashes():
    root = Path(__file__).resolve().parents[3] / 'treasure_auto' / 'src'
    paths = [root / 'treasure_auto' / 'vision_workbench' / (name + '.py')
             for name in ('contours', 'lines', 'templates', 'cli')]
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


class Backend:
    def __init__(self, config):
        self.config = config
        self.method = config.get('method', 'lsd')
        if self.method not in ('contour', 'lsd', 'edgedrawing'):
            raise ValueError('nontrained method must be contour/lsd/edgedrawing')
        root = Path(__file__).resolve().parents[3] / 'treasure_auto' / 'src'
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from treasure_auto.vision_workbench.templates import TemplateClassifier
        self.classifier = TemplateClassifier(config.get('templates'), **config.get('classification', {}))
        self.hashes = source_hashes()
        self.artifact_manifests = {'classification_templates': self.classifier.manifest}

    def predict(self, bgr):
        import cv2
        import numpy as np
        from treasure_auto.vision_workbench.cli import warp_candidate
        from treasure_auto.vision_workbench.contours import detect_candidates
        from treasure_auto.vision_workbench.lines import detect_candidates as detect_lines
        result = (detect_candidates(bgr, self.config.get('detection', {})) if self.method == 'contour'
                  else detect_lines(bgr, self.config.get('detection', {}),
                                    self.config.get('line_detection', {}), method=self.method))
        if result.get('budget_truncated'):
            raise RuntimeError('nontrained search truncated; cannot count as a complete prediction')
        instances = []
        for candidate in result['candidates']:
            mask = np.zeros(bgr.shape[:2], np.uint8)
            corners = candidate['corners']
            cv2.fillConvexPoly(mask, np.rint(corners).astype(np.int32), 1)
            classification = self.classifier.classify(warp_candidate(bgr, corners, 128))
            instances.append(Instance(classification['category'], 1., mask.astype(bool), corners,
                {'method': self.method, 'templates_ready': self.classifier.ready,
                 'classification': classification, 'score_basis': 'unranked geometric candidate',
                 'unresolved_count': len(result.get('unresolved', [])),
                 'geometry_basis': 'original_nontrained_edges', 'actionable': False}))
        return instances


def create(config):
    return Backend(config)
