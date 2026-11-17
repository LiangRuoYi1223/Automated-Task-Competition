"""Bounded line-segment diagnostics with observed original-image edge support.

No result establishes object identity, an upward plane, or permission to move.
Thresholds are versioned engineering defaults, not measured field calibration.
"""
from copy import deepcopy
import math
import time

from .contours import _configuration, _number

METHOD_VERSION = 'observed-lines-v1'
DEFAULTS = {
    'parameter_version': 'line-diagnostic-v1',
    'channels': ['gray'], 'denoise': False, 'enhance': False,
    'lsd_refine': 'standard', 'lsd_scale': .8,
    'min_segment_length_px': 12., 'support_band_px': 2,
    'max_support_extension_px': 0.,
    'gradient_threshold': 20., 'gradient_alignment': .65,
    'merge_angle_deg': 5., 'merge_distance_px': 2.,
    'merge_gap_fraction': .12, 'max_merge_gap_px': 12.,
    'min_support_rate': .8, 'max_gap_fraction': .08, 'max_gap_px': 10.,
    'max_extrapolation_px': 6., 'dedup_iou': .8,
    'contrast_offset_px': 4., 'min_contrast': 3.,
    'max_segments': 160, 'max_groups': 100, 'max_hypotheses': 1200,
    'frame_budget_ms': 200.,
}


def capabilities():
    import cv2
    return {'opencv_version': cv2.__version__,
            'lsd': callable(getattr(cv2, 'createLineSegmentDetector', None)),
            'edgedrawing': callable(getattr(getattr(cv2, 'ximgproc', None),
                                          'createEdgeDrawing', None))}


def validate_config(settings=None):
    if settings is None:
        settings = {}
    if type(settings) is not dict or set(settings) - set(DEFAULTS):
        raise ValueError('unknown or invalid line_detection configuration')
    cfg = deepcopy(DEFAULTS)
    cfg.update(settings)
    if not isinstance(cfg['parameter_version'], str) or not cfg['parameter_version'].strip():
        raise ValueError('parameter_version must be a nonempty string')
    channels = cfg['channels']
    if (not isinstance(channels, list) or not channels or len(channels) > 4 or
            any(type(ch) is not str or ch not in ('gray', 'b', 'g', 'r') for ch in channels) or
            len(set(channels)) != len(channels)):
        raise ValueError('channels must be distinct gray/b/g/r channel names')
    for key in ('denoise', 'enhance'):
        if type(cfg[key]) is not bool:
            raise ValueError(key + ' must be boolean')
    if cfg['lsd_refine'] not in ('none', 'standard', 'advanced'):
        raise ValueError('lsd_refine must be none/standard/advanced')
    cfg['lsd_scale'] = _number(cfg['lsd_scale'], 'lsd_scale', maximum=1., positive=True)
    for key in ('support_band_px', 'max_segments', 'max_groups', 'max_hypotheses'):
        maximum = 8 if key == 'support_band_px' else 10000
        if type(cfg[key]) is not int or not 1 <= cfg[key] <= maximum:
            raise ValueError(key + ' must be a bounded positive integer')
    for key in ('min_segment_length_px', 'gradient_threshold', 'merge_distance_px',
                'max_merge_gap_px', 'max_gap_px', 'max_extrapolation_px',
                'contrast_offset_px', 'frame_budget_ms'):
        cfg[key] = _number(cfg[key], key, positive=True)
    for key in ('gradient_alignment', 'min_support_rate', 'max_gap_fraction',
                'merge_gap_fraction', 'dedup_iou'):
        cfg[key] = _number(cfg[key], key, maximum=1., positive=True)
    cfg['merge_angle_deg'] = _number(cfg['merge_angle_deg'], 'merge_angle_deg',
                                    maximum=30., positive=True)
    cfg['min_contrast'] = _number(cfg['min_contrast'], 'min_contrast', maximum=255.)
    cfg['max_support_extension_px'] = _number(cfg['max_support_extension_px'],
                                             'max_support_extension_px',maximum=128.)
    return cfg


def _fit(points, ids, channels, cv2, np, max_error):
    points = np.unique(np.asarray(points, np.float32).reshape(-1, 2), axis=0)
    if len(points) < 3:
        return None
    fitted = cv2.fitLine(points, cv2.DIST_HUBER, 0, .01, .01).ravel()
    direction, center = fitted[:2].astype(float), fitted[2:].astype(float)
    residual = np.abs((points-center)[:, 0]*direction[1] -
                      (points-center)[:, 1]*direction[0])
    # A small number of observed pixels may be discarded; none are fabricated.
    keep = residual <= max_error
    if keep.sum() < 3 or keep.mean() < .85:
        return None
    points = points[keep]
    fitted = cv2.fitLine(points, cv2.DIST_HUBER, 0, .01, .01).ravel()
    direction, center = fitted[:2].astype(float), fitted[2:].astype(float)
    if direction[0] < 0 or (abs(direction[0]) < 1e-8 and direction[1] < 0):
        direction = -direction
    projection = (points-center) @ direction
    residual = np.abs((points-center)[:, 0]*direction[1] -
                      (points-center)[:, 1]*direction[0])
    return dict(points=points, direction=direction, center=center,
                lo=float(projection.min()), hi=float(projection.max()),
                residual=float(residual.max()), ids=sorted(set(ids)),
                channels=sorted(set(channels)))


def _observed_support(segment, gx, gy, cfg, np):
    """Select integer image pixels from the raw gradient band, not a drawn line."""
    a, b = np.asarray(segment[:2], float), np.asarray(segment[2:], float)
    direction = b-a
    length = float(np.linalg.norm(direction))
    if length < cfg['min_segment_length_px']:
        return np.empty((0, 2), np.float32)
    direction /= length
    normal = np.array([-direction[1], direction[0]])
    extension=cfg['max_support_extension_px']
    parameters=np.arange(-extension,length+extension+1.)
    positions = a + parameters[:, None]*direction
    offsets = np.arange(-cfg['support_band_px'], cfg['support_band_px']+1.)
    pixels = np.rint(positions[:, None, :] + offsets[None, :, None]*normal).astype(int)
    h, w = gx.shape
    valid = ((pixels[:, :, 0] > 0) & (pixels[:, :, 0] < w-1) &
             (pixels[:, :, 1] > 0) & (pixels[:, :, 1] < h-1))
    x, y = np.clip(pixels[:, :, 0], 0, w-1), np.clip(pixels[:, :, 1], 0, h-1)
    xg, yg = gx[y,x], gy[y,x]
    magnitude = np.hypot(xg, yg)
    alignment = np.abs(xg*normal[0]+yg*normal[1])/np.maximum(magnitude,1e-12)
    valid &= ((magnitude >= cfg['gradient_threshold']) &
              (alignment >= cfg['gradient_alignment']))
    # Prefer the nearest equally strong pixel so ties do not systematically
    # push evidence to the outside edge of the sampling band.
    score = np.where(valid, magnitude-.01*np.abs(offsets)[None,:], -1.)
    which = np.argmax(score, axis=1)
    rows = np.arange(len(positions))
    supported=score[rows,which]>=0
    if extension:
        # Grow from each detector endpoint only while actually observed support
        # continues. A long missing run terminates growth; it is never bridged.
        for indices in (range(max(0,int(math.ceil(extension)))-1,-1,-1),
                        range(min(len(parameters),int(math.floor(extension+length))+1),len(parameters))):
            missing=0
            stopped=False
            for index in indices:
                if stopped:
                    supported[index]=False
                    continue
                missing=0 if supported[index] else missing+1
                if missing>cfg['max_gap_px']:
                    stopped=True
    return np.unique(pixels[rows,which][supported], axis=0).astype(np.float32)


def _mergeable(a, b, cfg, np):
    dot = abs(float(a['direction'] @ b['direction']))
    if dot < math.cos(math.radians(cfg['merge_angle_deg'])):
        return False
    d = a['direction']
    delta = b['center']-a['center']
    if abs(float(delta[0]*d[1]-delta[1]*d[0])) > cfg['merge_distance_px']:
        return False
    bp = (b['points']-a['center']) @ d
    gap = max(a['lo']-float(bp.max()),float(bp.min())-a['hi'],0.)
    allowance = min(cfg['max_merge_gap_px'], cfg['merge_gap_fraction']*
                    max(a['hi']-a['lo'], b['hi']-b['lo']))
    return gap <= allowance


def _intersection(a, b, min_sine):
    ax, ay = a['direction']; bx, by = b['direction']
    determinant = ax*by-ay*bx
    if abs(determinant) < min_sine:
        return None
    qx, qy = b['center']-a['center']
    t = (qx*by-qy*bx)/determinant
    return a['center']+t*a['direction']


def _extra(line, point):
    t = float((point-line['center']) @ line['direction'])
    return max(line['lo']-t,t-line['hi'],0.)


def _edge_metrics(a, b, line, cfg, shared, np):
    d = b-a
    length = float(np.linalg.norm(d))
    if length < shared.min_edge_span_px:
        raise ValueError('edge support too short')
    d /= length
    projection = (line['points']-a) @ d
    mask = (projection >= -.5) & (projection <= length+.5)
    points, projection = line['points'][mask], projection[mask]
    if len(points) < shared.min_edge_points:
        raise ValueError('insufficient distinct observed edge points')
    bins = np.unique(np.clip(np.floor(projection+.5).astype(int),0,max(0,int(length))))
    intervals = []
    start = previous = int(bins[0])
    for value in bins[1:]:
        value = int(value)
        if value > previous+1:
            intervals.append([max(0.,start-.5), min(length,previous+.5)])
            start = value
        previous = value
    intervals.append([max(0.,start-.5), min(length,previous+.5)])
    gaps = [intervals[0][0],length-intervals[-1][1]]
    gaps += [right[0]-left[1] for left,right in zip(intervals, intervals[1:])]
    support = sum(hi-lo for lo,hi in intervals)/length
    max_gap = max(gaps)
    residual = np.abs((points-a)[:,0]*d[1]-(points-a)[:,1]*d[0])
    if support < cfg['min_support_rate']:
        raise ValueError('edge support rate below configured minimum')
    if max_gap > min(cfg['max_gap_px'], cfg['max_gap_fraction']*length):
        raise ValueError('edge has a long unsupported gap')
    if float(residual.max()) > shared.max_line_error_px:
        raise ValueError('observed edge residual exceeds configured maximum')
    return points.tolist(), dict(support_intervals=intervals,
        support_interval_basis='distance in pixels from this edge first corner',
        support_rate=float(support), max_gap_px=float(max_gap),
        residual_px=float(residual.max()), segment_ids=line['ids'],
        channels=line['channels'], span_px=float(projection.max()-projection.min()),
        length_px=length)


def _group_record(index, line, np):
    """Keep observed intervals; the enclosing span is not declared visible."""
    projection=(line['points']-line['center'])@line['direction']-line['lo']
    values=np.unique(np.floor(projection+.5).astype(int))
    intervals=[]
    start=last=int(values[0])
    for value in values[1:]:
        value=int(value)
        if value>last+1:
            intervals.append([max(0.,start-.5),min(line['hi']-line['lo'],last+.5)])
            start=value
        last=value
    intervals.append([max(0.,start-.5),min(line['hi']-line['lo'],last+.5)])
    ends=[line['center']+line['lo']*line['direction'],
          line['center']+line['hi']*line['direction']]
    return dict(id=index,segment_ids=line['ids'],channels=line['channels'],
        observed_point_count=len(line['points']),support_span_px=line['hi']-line['lo'],
        residual_px=line['residual'],endpoints=[p.tolist() for p in ends],
        support_intervals=intervals,support_interval_basis='distance in pixels from first group endpoint',
        observed_points=line['points'].tolist())


def _candidate(order, corners, groups, raw_bgr, shared, cfg, cv2, np):
    corners = np.asarray(corners, float)
    turns=[]
    for i in range(4):
        first=corners[(i+1)%4]-corners[i]
        second=corners[(i+2)%4]-corners[(i+1)%4]
        turns.append(float(first[0]*second[1]-first[1]*second[0]))
    if not (all(t>1e-6 for t in turns) or all(t < -1e-6 for t in turns)):
        raise ValueError('crossed or nonconvex quadrilateral')
    area = float(cv2.contourArea(corners.astype(np.float32)))
    h,w = raw_bgr.shape[:2]
    if not shared.min_area_px2 <= area <= shared.max_area_fraction*w*h:
        raise ValueError('quadrilateral area outside configured bounds')
    if ((corners[:,0] <= 1).any() or (corners[:,0] >= w-2).any() or
            (corners[:,1] <= 1).any() or (corners[:,1] >= h-2).any()):
        raise ValueError('frame-truncated or boundary-adjacent quadrilateral')
    if shared.roi is not None:
        x,y,rw,rh = shared.roi
        if not ((corners[:,0]>x+1).all() and (corners[:,0]<x+rw-1).all() and
                (corners[:,1]>y+1).all() and (corners[:,1]<y+rh-1).all()):
            raise ValueError('quadrilateral crosses or touches manual roi boundary')
    for x,y,rw,rh in shared.board_regions:
        rect = np.array([[x,y],[x+rw,y],[x+rw,y+rh],[x,y+rh]],np.float32)
        if cv2.intersectConvexConvex(corners.astype(np.float32),rect)[0] > 0:
            raise ValueError('candidate overlaps manually excluded board region')
    # Canonical edge i joins corner i to corner i+1.
    signed = sum(corners[i,0]*corners[(i+1)%4,1]-corners[(i+1)%4,0]*corners[i,1] for i in range(4))
    if signed < 0:
        corners = corners[::-1]
        order = [order[2],order[1],order[0],order[3]]
    start = min(range(4),key=lambda i:(round(float(corners[i,1]),8),float(corners[i,0])))
    corners = np.roll(corners,-start,axis=0)
    order = order[start:]+order[:start]
    edges,metrics,corner_metrics,contrasts = [],[],[],[]
    for i,index in enumerate(order):
        line = groups[index]
        points,metric = _edge_metrics(corners[i],corners[(i+1)%4],line,cfg,shared,np)
        edges.append(points); metrics.append(metric)
        before = groups[order[(i-1)%4]]
        extra = max(_extra(before,corners[i]),_extra(line,corners[i]))
        if extra > cfg['max_extrapolation_px']:
            raise ValueError('corner requires excessive extrapolation')
        determinant = abs(float(before['direction'][0]*line['direction'][1]-before['direction'][1]*line['direction'][0]))
        error = (1.+before['residual']+line['residual'])/determinant
        corner_metrics.append(dict(extrapolation_px=extra,error_px=error,
            basis='intersection_of_observed_edges',
            visibility='within_observed_span' if extra <= 1e-6 else 'extended_intersection'))
        # Raw-image side-band contrast is diagnostic. It cannot prove a top plane.
        a,b = corners[i],corners[(i+1)%4]
        d = (b-a)/np.linalg.norm(b-a); normal=np.array([-d[1],d[0]])
        t = np.linspace(.1,.9,24)[:,None]
        center=a+(b-a)*t
        pairs=[]
        for sign in (-1,1):
            xy=np.rint(center+sign*cfg['contrast_offset_px']*normal).astype(int)
            xy[:,0]=np.clip(xy[:,0],0,w-1);xy[:,1]=np.clip(xy[:,1],0,h-1)
            pairs.append(raw_bgr[xy[:,1],xy[:,0]].astype(float))
        contrasts.append(float(np.median(np.max(np.abs(pairs[0]-pairs[1]),axis=1))))
    bbox = cv2.boundingRect(corners.astype(np.float32))
    return dict(corners=corners.tolist(),edges=edges,
        contour=[point for edge in edges for point in edge],bbox=list(map(int,bbox)),
        edge_metrics=metrics,corner_metrics=corner_metrics,area_px2=area,
        contrast_by_edge=contrasts,contrast_basis='median maximum BGR channel side-band difference',group_ids=order,
        quality_score=float(sum(m['support_rate'] for m in metrics)/4.),
        quality_score_basis='mean observed edge support for ranking only; not probability',
        body_confirmed=False,plane_basis='unknown',actionable=False,
        internal_boundaries=0,occlusion_unresolved=False,ambiguity_reasons=[])


def _iou(a,b,cv2,np):
    aa=np.asarray(a['corners'],np.float32);bb=np.asarray(b['corners'],np.float32)
    intersection=float(cv2.intersectConvexConvex(aa,bb)[0])
    return intersection/max(a['area_px2']+b['area_px2']-intersection,1e-9)


def _ambiguities(candidates, groups, cfg, cv2, np, expired, internal_edges):
    """Keep explicit frame/nesting/side and cross-frame ambiguity out of qualified set."""
    enclosing_contours,_=cv2.findContours(internal_edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    clearance=max(3.,cfg['merge_distance_px'])
    for candidate in candidates:
        if expired():
            return
        corners=np.asarray(candidate['corners'],np.float32)
        # The outer rim of a printed frame may fail the four-line geometric
        # gates while its inner rim succeeds. Retain surrounding raw contours
        # as ambiguity evidence independently of accepted quadrilaterals.
        # A raster-scale paired edge around a solid face is ignored.
        probes=list(corners)+[(corners[i]+corners[(i+1)%4])/2 for i in range(4)]
        candidate['enclosing_boundary_count']=0
        for contour in enclosing_contours:
            if expired():return
            if abs(cv2.contourArea(contour)) <= candidate['area_px2']:
                continue
            if all(cv2.pointPolygonTest(contour,tuple(map(float,p)),True)>clearance for p in probes):
                candidate['enclosing_boundary_count']+=1
        if candidate['enclosing_boundary_count']:
            candidate['ambiguity_reasons'].append('enclosing_observed_boundary_may_be_internal_frame')
        interior=np.zeros(internal_edges.shape,np.uint8)
        cv2.fillConvexPoly(interior,np.rint(corners).astype(np.int32),255)
        interior=cv2.erode(interior,np.ones((9,9),np.uint8))
        observed_internal=cv2.bitwise_and(internal_edges,interior)
        internal_contours,_=cv2.findContours(observed_internal,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        internal_count=sum(len(c)>=8 for c in internal_contours)
        candidate['internal_boundary_pixel_count']=int(np.count_nonzero(observed_internal))
        candidate['internal_boundaries']=internal_count
        if internal_count:
            candidate['ambiguity_reasons'].append('internal_boundary_print_or_occlusion_unresolved')
        for other in candidates:
            if expired():
                return
            if other is candidate:
                continue
            inside=[cv2.pointPolygonTest(corners,tuple(map(float,p)),True)>cfg['merge_distance_px'] for p in other['corners']]
            if all(inside):
                candidate['internal_boundaries'] += 1
                candidate['ambiguity_reasons'].append('internal_quadrilateral_print_or_occlusion_unresolved')
            outside=np.asarray(other['corners'],np.float32)
            if all(cv2.pointPolygonTest(outside,tuple(map(float,p)),True)>cfg['merge_distance_px'] for p in corners):
                candidate['ambiguity_reasons'].append('nested_frame_may_be_internal_print')
            common=set(candidate['group_ids']) & set(other['group_ids'])
            if common and _iou(candidate,other,cv2,np)<cfg['dedup_iou']:
                candidate['ambiguity_reasons'].append('shared_edge_neighbor_or_side_face_unresolved')
        # An observed line spanning the interior and meeting distinct outer
        # sides may be a join between bricks or a top/side boundary. No
        # semantic interpretation of that line is available at this stage.
        for index,line in enumerate(groups):
            if index in candidate['group_ids']:
                continue
            ends=[line['center']+line['lo']*line['direction'],
                  line['center']+line['hi']*line['direction']]
            distances=[cv2.pointPolygonTest(corners,tuple(map(float,p)),True) for p in ends]
            center_distance=cv2.pointPolygonTest(corners,
                tuple(map(float,(ends[0]+ends[1])/2)),True)
            if (all(abs(d)<4. for d in distances) and center_distance>5. and
                    line['hi']-line['lo']>.5*math.sqrt(candidate['area_px2'])):
                candidate['ambiguity_reasons'].append('interior_seam_neighbor_or_side_face_unresolved')
        if min(candidate['contrast_by_edge']) < cfg['min_contrast']:
            candidate['ambiguity_reasons'].append('weak_or_inconsistent_boundary_contrast')
        candidate['ambiguity_reasons']=sorted(set(candidate['ambiguity_reasons']))
        candidate['occlusion_unresolved']=bool(candidate['ambiguity_reasons'])


def detect_candidates(bgr, config, line_config=None, method='lsd'):
    import cv2
    import numpy as np
    if (not isinstance(bgr,np.ndarray) or bgr.dtype!=np.uint8 or bgr.ndim!=3 or
            bgr.shape[2]!=3 or min(bgr.shape[:2])<3):
        raise ValueError('image must be a nonempty uint8 BGR array with three channels')
    cfg=validate_config(line_config)
    shared=_configuration(config,bgr.shape[1],bgr.shape[0])
    available=capabilities()
    if method not in ('lsd','edgedrawing'):
        raise ValueError('unknown line detection method')
    if not available.get(method):
        raise ValueError(method+' frontend unavailable in this OpenCV build')
    start=time.perf_counter()
    result=dict(candidates=[],unresolved=[],rejected=[],method=method,
        method_version=METHOD_VERSION,parameter_version=cfg['parameter_version'],
        line_configuration=cfg,capabilities=available,budget_truncated=False,
        truncation_reasons=[],region_basis='manual' if shared.roi is not None or shared.board_regions else 'unconfigured',
        search_region=dict(roi=shared.roi,coordinate_transform='identity; full-image extraction',
                          source='manual' if shared.roi is not None else 'full_image'),
        segment_count=0,group_count=0,hypotheses_tested=0,
        visibility_limitations=['top plane, identity and height unknown',
            'side faces and internal frames can remain geometrically indistinguishable'])
    def truncated(reason):
        result['budget_truncated']=True
        if reason not in result['truncation_reasons']:
            result['truncation_reasons'].append(reason)
    def expired():
        if (time.perf_counter()-start)*1000. > cfg['frame_budget_ms']:
            truncated('frame_budget_ms'); return True
        return False
    raw_gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
    channels={'gray':raw_gray,'b':bgr[:,:,0],'g':bgr[:,:,1],'r':bgr[:,:,2]}
    raw=[]
    gradient={}
    for name in cfg['channels']:
        if expired(): break
        original=channels[name]
        gradient[name]=(cv2.Sobel(original,cv2.CV_32F,1,0,ksize=3),
                        cv2.Sobel(original,cv2.CV_32F,0,1,ksize=3))
        image=original.copy()
        if cfg['denoise']: image=cv2.GaussianBlur(image,(3,3),.6)
        if cfg['enhance']: image=cv2.createCLAHE(2.,(8,8)).apply(image)
        try:
            if method=='lsd':
                refinement={'none':cv2.LSD_REFINE_NONE,'standard':cv2.LSD_REFINE_STD,
                            'advanced':cv2.LSD_REFINE_ADV}[cfg['lsd_refine']]
                detected=cv2.createLineSegmentDetector(refinement,cfg['lsd_scale']).detect(image)[0]
            else:
                frontend=cv2.ximgproc.createEdgeDrawing()
                frontend.detectEdges(image)
                detected=frontend.detectLines()
        except (cv2.error,AttributeError) as error:
            raise ValueError(method+' frontend unavailable or failed: '+str(error)) from error
        if detected is not None:
            for segment in np.asarray(detected).reshape(-1,4):
                length=float(np.linalg.norm(segment[2:]-segment[:2]))
                if length>=cfg['min_segment_length_px']:
                    raw.append((length,name,segment.copy()))
    raw.sort(key=lambda item:(-item[0],item[1],tuple(map(float,item[2]))))
    result['raw_segment_count']=len(raw)
    if len(raw)>cfg['max_segments']: truncated('max_segments')
    raw=raw[:cfg['max_segments']]
    result['segment_count']=len(raw)
    groups=[]
    segments=[]
    for index,(_,name,segment) in enumerate(raw):
        if expired(): break
        points=_observed_support(segment,*gradient[name],cfg,np)
        delta=segment[2:].astype(float)-segment[:2].astype(float)
        length=float(np.linalg.norm(delta))
        segments.append(dict(id=index,channel=name,endpoints=segment.reshape(2,2).astype(float).tolist(),
                             length_px=length,direction=(delta/length).tolist(),
                             observed_point_count=len(points)))
        fitted=_fit(points,[index],[name],cv2,np,shared.max_line_error_px)
        if fitted is None: continue
        merged=False
        for gi,group in enumerate(groups):
            if _mergeable(group,fitted,cfg,np):
                combined=_fit(np.concatenate([group['points'],fitted['points']]),
                    group['ids']+fitted['ids'],group['channels']+fitted['channels'],cv2,np,shared.max_line_error_px)
                if combined is not None:
                    groups[gi]=combined;merged=True;break
        if not merged:
            if len(groups)>=cfg['max_groups']:
                truncated('max_groups');continue
            groups.append(fitted)
    # Revisit groups after a bridge fragment joins an earlier group. A single
    # insertion pass otherwise leaves two genuinely collinear groups separate.
    changed=True
    while changed and not expired():
        changed=False
        for a in range(len(groups)):
            if expired():break
            for b in range(a+1,len(groups)):
                if expired():break
                if not _mergeable(groups[a],groups[b],cfg,np):continue
                combined=_fit(np.concatenate([groups[a]['points'],groups[b]['points']]),
                    groups[a]['ids']+groups[b]['ids'],groups[a]['channels']+groups[b]['channels'],
                    cv2,np,shared.max_line_error_px)
                if combined is not None:
                    groups[a]=combined;groups.pop(b);changed=True;break
            if changed:break
    result['segments']=segments
    result['group_count']=len(groups)
    result['groups']=[_group_record(i,g,np) for i,g in enumerate(groups)]
    adjacent=[{} for _ in groups]
    for a in range(len(groups)):
        if expired(): break
        for b in range(a+1,len(groups)):
            if expired():break
            point=_intersection(groups[a],groups[b],shared.min_corner_sine)
            if point is not None and max(_extra(groups[a],point),_extra(groups[b],point))<=cfg['max_extrapolation_px']:
                adjacent[a][b]=point;adjacent[b][a]=point
    for group,neighbors in zip(result['groups'],adjacent):
        group['adjacent_group_ids']=list(neighbors)
    proposed=[];seen=set();stop=False
    for a in range(len(groups)):
        if stop or expired(): break
        for b in adjacent[a]:
            if b<=a: continue
            for cc in adjacent[b]:
                if cc<=a: continue
                for d in adjacent[cc]:
                    if d<=b or d<=a or d not in adjacent[a]: continue
                    key=(a,b,cc,d)
                    if len(set(key))!=4 or key in seen:continue
                    seen.add(key)
                    if result['hypotheses_tested']>=cfg['max_hypotheses']:
                        truncated('max_hypotheses');stop=True;break
                    if expired():stop=True;break
                    result['hypotheses_tested']+=1
                    corners=[adjacent[d][a],adjacent[a][b],adjacent[b][cc],adjacent[cc][d]]
                    try:
                        candidate=_candidate(list(key),corners,groups,bgr,shared,cfg,cv2,np)
                        proposed.append(candidate)
                    except ValueError as error:
                        result['rejected'].append(dict(reason=str(error),group_ids=list(key),
                            corners=[p.tolist() for p in corners]))
                if stop:break
            if stop:break
    internal_edges=np.zeros(raw_gray.shape,np.uint8)
    if proposed:
        for name in set(cfg['channels']+['gray']):
            if expired():break
            internal_edges=cv2.bitwise_or(internal_edges,
                cv2.Canny(channels[name],shared.canny_low,shared.canny_high,L2gradient=True))
    # Establish ambiguity before suppressing near-identical quadrilaterals;
    # otherwise a printed frame's outer rim can disappear before it explains
    # why its inner rim is not a confirmed outer body boundary.
    _ambiguities(proposed,groups,cfg,cv2,np,expired,internal_edges)
    proposed.sort(key=lambda candidate:(-candidate['quality_score'],-candidate['area_px2']))
    unique=[]
    for candidate in proposed:
        if expired():break
        if not any(_iou(candidate,other,cv2,np)>=cfg['dedup_iou'] for other in unique):
            unique.append(candidate)
    result['timing_ms']=(time.perf_counter()-start)*1000.
    if result['timing_ms']>cfg['frame_budget_ms']:truncated('frame_budget_ms')
    for candidate in unique:
        if result['budget_truncated']:
            candidate['ambiguity_reasons'].append('incomplete_search_ambiguity_not_fully_checked')
            candidate['occlusion_unresolved']=True
        result['unresolved' if candidate['ambiguity_reasons'] else 'candidates'].append(candidate)
    result['search_complete']=not result['budget_truncated']
    result['no_candidate_reason']=None
    if not result['candidates']:
        if result['budget_truncated']:
            result['no_candidate_reason']='incomplete_search_budget_truncated'
        elif result['unresolved']:
            result['no_candidate_reason']='observed_quadrilaterals_have_unresolved_ambiguity'
        elif not result['hypotheses_tested']:
            result['no_candidate_reason']='no_observed_four_edge_cycle'
        else:
            result['no_candidate_reason']='all_four_edge_hypotheses_failed_quality_gates'
    return result
