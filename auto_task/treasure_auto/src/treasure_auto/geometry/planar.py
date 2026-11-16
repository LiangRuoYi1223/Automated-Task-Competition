"""Convex polygon calculations in the caller's explicit single unit/frame."""
import itertools
import math
from treasure_auto.contracts import core as c


def cross(a,b,p):return (b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0])


def area(polygon):
    if len(polygon)<3:return 0.
    # Local origin avoids cancellation of large absolute map-coordinate products.
    x,y=polygon[0];p=tuple((a-x,b-y) for a,b in polygon)
    return abs(math.fsum(a[0]*b[1]-a[1]*b[0] for a,b in zip(p,p[1:]+p[:1])))/2


def convex(points):
    p=tuple(tuple(x) for x in points)
    c.require(len(p)>=3 and len(set(p))==len(p) and all(len(x)==2 and all(type(v) in (int,float) and math.isfinite(v) for v in x) for x in p),
              'polygon','finite distinct vertices required')
    turns=[cross(a,b,x) for a,b in zip(p,p[1:]+p[:1]) for x in p if x!=a and x!=b]
    c.require(all(x>1e-14 for x in turns) or all(x < -1e-14 for x in turns),'polygon','ordered strictly convex polygon required')
    return p if turns[0]>0 else tuple(reversed(p))


def halfplane(poly,a,b,margin=0.):
    if not poly:return ()
    length=math.hypot(b[0]-a[0],b[1]-a[1]);out=[]
    def side(p):return cross(a,b,p)/length-margin
    for s,e in zip(poly,poly[1:]+poly[:1]):
        ds,de=side(s),side(e)
        if ds>=0:out.append(s)
        if (ds>=0)!=(de>=0):
            t=ds/(ds-de);out.append((s[0]+t*(e[0]-s[0]),s[1]+t*(e[1]-s[1])))
    return tuple(out)


def intersection(a,b):
    out=a
    for start,end in zip(b,b[1:]+b[:1]):out=halfplane(out,start,end)
    return out


def erode(poly,margin):
    out=poly
    for a,b in zip(poly,poly[1:]+poly[:1]):out=halfplane(out,a,b,margin)
    return out


def union_area(polygons):
    # At most three physical lower bricks in these layouts. Exact inclusion/exclusion.
    c.require(len(polygons)<=3,'supports','finite first-version support set limited to three')
    total=0.
    for n in range(1,len(polygons)+1):
        for group in itertools.combinations(polygons,n):
            p=group[0]
            for other in group[1:]:p=intersection(p,other)
            total+=area(p)*(1 if n%2 else -1)
    return max(0.,total)


def clearance(poly,point):
    return min(cross(a,b,point)/math.hypot(b[0]-a[0],b[1]-a[1]) for a,b in zip(poly,poly[1:]+poly[:1]))
