"""P05 0.1.0: offline board interpretation. No ROS, device or storage side effects."""
from .config import ClassificationConfig, BoardTemplate, SlotRegion
from .observations import FrameInput, BoardCandidate, DirectionHint, PatternInput, FrameAnalysis
from .geometry import analyze_frame
from .service import ClassificationService
from .recovery import ReadProgress

__version__ = '0.1.0'
