"""Small standard-library projective geometry core; no image detector is implied."""
import math
from treasure_auto.contracts import core as c, models as m
from .config import number
from .observations import FrameInput, BoardCandidate, PatternInput, DirectionHint, FrameAnalysis


def area(points):
    return sum(a.u*b.v-b.u*a.v for a,b in zip(points,points[1:]+points[:1]))/2


def cross(a,b,d):
    return (b.u-a.u)*(d.v-b.v)-(b.v-a.v)*(d.u-b.u)


def polygon(points, width, height, field):
    c.require(type(points) is tuple and len(points)>=3 and all(type(p) is m.PixelPoint for p in points),
              field,'immutable polygon with full-image points required')
    c.require(all(0<=p.u<width and 0<=p.v<height for p in points),field,'outside full image')
    signs=[cross(a,b,d) for a,b,d in zip(points,points[1:]+points[:1],points[2:]+points[:2])]
    c.require(all(x>1e-9 for x in signs) or all(x<-1e-9 for x in signs),field,'nonconvex or degenerate polygon')


def homography(corners):
    """Map semantic board corners to the unit square, using pivoted elimination."""
    matrix=[]
    for p,(x,y) in zip(corners,((0,0),(1,0),(1,1),(0,1))):
        u,v=p.u,p.v
        matrix.extend(([u,v,1,0,0,0,-x*u,-x*v,x],[0,0,0,u,v,1,-y*u,-y*v,y]))
    for col in range(8):
        pivot=max(range(col,8),key=lambda row:abs(matrix[row][col]))
        c.require(abs(matrix[pivot][col])>1e-12,'board.homography','singular transform')
        matrix[col],matrix[pivot]=matrix[pivot],matrix[col]
        divisor=matrix[col][col]
        matrix[col]=[x/divisor for x in matrix[col]]
        for row in range(8):
            if row!=col:
                factor=matrix[row][col]
                matrix[row]=[a-factor*b for a,b in zip(matrix[row],matrix[col])]
    return tuple(matrix[i][8] for i in range(8))+(1.0,)


def transform(point, h):
    divisor=h[6]*point.u+h[7]*point.v+h[8]
    c.require(abs(divisor)>1e-12,'board.homography','point at infinity')
    return m.PixelPoint((h[0]*point.u+h[1]*point.v+h[2])/divisor,
                        (h[3]*point.u+h[4]*point.v+h[5])/divisor)


def analyze_frame(frame, template, config):
    """All-or-nothing board interpretation. Partial assignments are diagnostics only."""
    assignments=[]; raw=(); h=(); rectified=[]; sources=()
    try:
        c.require(type(frame) is FrameInput,'frame','P05 offline FrameInput required')
        config.require_ready(frame.image.envelope.origin); template.require_ready(frame.image.envelope.origin)
        c.require(frame.image.distortion is m.Distortion.CORRECTED,'image.distortion','rectified full-image coordinates required')
        c.require(type(frame.boards) is tuple and len(frame.boards)==1 and type(frame.boards[0]) is BoardCandidate,
                  'board.count','exactly one complete board required')
        board=frame.boards[0]
        c.require(board.complete is c.Fact.CONFIRMED,'board.complete','partial/unknown board')
        c.require(len(board.corners)==4,'board.corners','four corners required')
        polygon(board.corners,frame.image.width,frame.image.height,'board.corners')
        raw=board.corners
        c.require(abs(area(raw))>=config.min_board_area_px2,'board.area','board too small')
        c.require(type(board.directions) is tuple and bool(board.directions),'board.direction','direction missing')
        for hint in board.directions:
            c.require(type(hint) is DirectionHint and hint.marker_key in template.marker_keys,
                      'board.direction','unverified direction source')
            c.require(type(hint.corner_order) is tuple and len(hint.corner_order)==4 and
                      all(type(i) is int for i in hint.corner_order) and set(hint.corner_order)=={0,1,2,3},
                      'board.direction','semantic corner order required')
            c.require(type(hint.evidence) is tuple and bool(hint.evidence),'board.direction.evidence','direction evidence missing')
        c.require(len({hint.corner_order for hint in board.directions})==1,
                  'board.direction','conflicting direction sources',c.ErrorCode.CONFLICT)
        sources=tuple(hint.marker_key for hint in board.directions)
        ordered=tuple(raw[i] for i in board.directions[0].corner_order)
        polygon(ordered,frame.image.width,frame.image.height,'board.direction')
        c.require(area(ordered)>0,'board.direction','mirror conflicts with verified template',c.ErrorCode.CONFLICT)
        for a,b,d in zip(ordered,ordered[1:]+ordered[:1],ordered[2:]+ordered[:2]):
            sine=abs(cross(a,b,d))/(math.hypot(b.u-a.u,b.v-a.v)*math.hypot(d.u-b.u,d.v-b.v))
            c.require(sine>=config.min_corner_sine,'board.corners','near-collinear corner')
        h=homography(ordered)
        c.require(type(frame.patterns) is tuple,'patterns','immutable pattern sequence required')
        used=set()
        for pattern in frame.patterns:
            c.require(type(pattern) is PatternInput,'pattern','typed pattern input required')
            number(pattern.confidence,'pattern.confidence',0,1)
            c.require(pattern.confidence>=config.min_pattern_confidence,'pattern.confidence','ambiguous/low confidence')
            c.require(type(pattern.category) is m.Category and pattern.category is not m.Category.CAKE,
                      'pattern.category','unknown or cake on ordinary board',c.ErrorCode.INVALID_MAPPING)
            c.require(type(pattern.evidence) is tuple and bool(pattern.evidence),'pattern.evidence','pattern evidence missing')
            polygon(pattern.polygon,frame.image.width,frame.image.height,'pattern.polygon')
            pts=tuple(transform(p,h) for p in pattern.polygon)
            rectified.append(pts)
            # Full containment with an inward uncertainty margin is deliberately
            # stricter than majority-area assignment: a cross-slot polygon is rejected.
            matches=[slot for slot in template.slots if all(
                slot.left+config.slot_margin < p.u < slot.right-config.slot_margin and
                slot.top+config.slot_margin < p.v < slot.bottom-config.slot_margin for p in pts)]
            c.require(len(matches)==1,'slot.assignment','cross-slot/outside/ambiguous pattern',c.ErrorCode.INVALID_MAPPING)
            slot=matches[0]
            c.require(slot.area not in used,'slot.'+slot.area.value,'more than one pattern in slot',c.ErrorCode.INVALID_MAPPING)
            used.add(slot.area)
            assignments.append(m.SlotAssignment(slot.area,pattern.category))
        c.require(used=={m.Area.B,m.Area.C,m.Area.D},'slots','missing slot; never infer remainder',c.ErrorCode.INVALID_MAPPING)
        c.require({a.category for a in assignments}=={m.Category.DOLL,m.Category.CLOCK,m.Category.BANANA},
                  'slots.category','duplicate category',c.ErrorCode.INVALID_MAPPING)
        return FrameAnalysis(True,tuple(sorted(assignments,key=lambda a:a.area.value)),None,False,raw,h,tuple(rectified),sources)
    except c.ContractError as error:
        return FrameAnalysis(False,tuple(assignments),c.Reason(error.code,error.field,error.detail),
                             error.code is c.ErrorCode.CONFLICT,raw,h,tuple(rectified),sources)
