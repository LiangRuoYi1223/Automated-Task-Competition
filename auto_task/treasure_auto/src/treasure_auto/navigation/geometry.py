"""Conservative continuous route sweep; no sampling holes between waypoints."""
import math
from treasure_auto.contracts import core as c, models as m
from treasure_auto.mapping.geometry import (inside,edges,intersects,point_segment,validate_polygon)
from .observations import reason


def segment_distance(a,b,p,q):
    if intersects(a,b,p,q):return 0.
    return min(point_segment(a,p,q),point_segment(b,p,q),point_segment(p,a,b),point_segment(q,a,b))


def check_route(bundle,route,profile,holding,space):
    try:
        shapes=profile.shapes(holding)
        c.require(type(route) is tuple and len(route)>=2 and all(type(p) is c.MapPose for p in route),
                  'route','immutable route with start/end required')
        c.require(space.envelope.versions.map==bundle.metadata.map_version,'space.map','different map',c.ErrorCode.VERSION_MISMATCH)
        radius=max(math.hypot(p.x,p.y) for shape in shapes for p in shape)+profile.error_m
        radius+=max(p.error_m for p in route)+bundle.metadata.origin.error_m
        grid_radius=math.hypot(bundle.raster.width,bundle.raster.height)*bundle.metadata.resolution_m_per_cell
        radius+=2*grid_radius*math.sin(min(math.pi,bundle.metadata.origin.error_rad)/2)
        frame=bundle.metadata.origin.frame_id
        c.require(all(p.frame_id==frame for p in route),'route.frame','all poses must be in map')
        blockers=[]
        for item in space.occupancies:
            c.require(item.frame_id==frame,'occupancy.frame','wrong frame')
            validate_polygon(item.footprint_m)
            blockers.append((item.footprint_m,item.error_m,'occupancy.'+item.target_id))
        for row in range(bundle.raster.height):
            for col in range(bundle.raster.width):
                if (255-bundle.raster.pixels[row*bundle.raster.width+col])/255>=bundle.free_threshold:
                    blockers.append((tuple(bundle.grid_corner(x,y) for x,y in
                        ((col,row),(col+1,row),(col+1,row+1),(col,row+1))),0.,'grid'))
        grid=tuple(bundle.grid_corner(x,y) for x,y in ((0,0),(bundle.raster.width,0),
                           (bundle.raster.width,bundle.raster.height),(0,bundle.raster.height)))
        for first,second in zip(route,route[1:]):
            a,b=m.Point2(first.x,first.y),m.Point2(second.x,second.y)
            for boundary in (bundle.boundary,grid):
                c.require(inside(a,boundary) and inside(b,boundary) and
                          all(segment_distance(a,b,p,q)>radius+1e-10 for p,q in edges(boundary)),
                          'boundary','swept complete envelope crosses field/map boundary',c.ErrorCode.CONFLICT)
            for shape,error,label in blockers:
                c.require(not inside(a,shape) and not inside(b,shape) and
                          all(segment_distance(a,b,p,q)>radius+error+1e-10 for p,q in edges(shape)),
                          label,'swept envelope intersects occupied or unknown space',c.ErrorCode.OCCUPANCY_UNRESOLVED)
        return m.ContractCheck(True,None)
    except c.ContractError as error:return m.ContractCheck(False,reason(error))
