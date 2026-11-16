"""P02 private static-bundle format; embedded public records remain C0 1.0.0.

    Images are 8-bit P2/P5 PGM, top row first; zero is occupied and 255 free.
    A bundle version covers map, regions, boundary, and all named waypoints.
"""
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from treasure_auto.contracts import core as c, models as m
from .geometry import finite, validate_polygon, inside, covered

MAX_CELLS = 16000000


@dataclass(frozen=True)
class Raster:
    width: int
    height: int
    pixels: tuple

    def __post_init__(self):
        c.require(type(self.width) is int and type(self.height) is int and
                  self.width > 0 and self.height > 0 and self.width*self.height <= MAX_CELLS,
                  'dimensions', 'positive bounded grid dimensions required')
        c.require(type(self.pixels) is tuple and len(self.pixels) == self.width*self.height and
                  all(type(p) is int and 0 <= p <= 255 for p in self.pixels),
                  'pixels', 'exactly width*height bytes required')


def read_pgm(data):
    c.require(type(data) is bytes and len(data) <= MAX_CELLS*4 + 4096, 'image', 'invalid/oversize image')
    tokens = re.finditer(rb'#[^\r\n]*|[^\s#]+', data)
    header = []
    end = 0
    for match in tokens:
        if match.group().startswith(b'#'):
            continue
        header.append(match.group())
        end = match.end()
        if len(header) == 4:
            break
    try:
        magic, width, height, maximum = header
        width, height, maximum = int(width), int(height), int(maximum)
        c.require(magic in (b'P2', b'P5') and maximum == 255, 'image', '8-bit P2/P5 required')
        c.require(0 < width*height <= MAX_CELLS and width > 0 and height > 0, 'image', 'invalid dimensions')
        if magic == b'P5':
            c.require(end < len(data) and data[end:end+1].isspace(), 'image', 'missing raster separator')
            end += 2 if data[end:end+2] == b'\r\n' else 1
            pixels = tuple(data[end:])
        else:
            pixels = tuple(int(t.group()) for t in re.finditer(rb'#[^\r\n]*|[^\s#]+', data[end:])
                           if not t.group().startswith(b'#'))
        return Raster(width, height, pixels)
    except (ValueError, TypeError) as error:
        if isinstance(error, c.ContractError):
            raise
        raise c.ContractError(c.ErrorCode.INVALID_FIELD, 'image', 'malformed PGM') from None


@dataclass(frozen=True)
class MapBundle:
    metadata: m.MapMetadata
    raster: Raster
    boundary: tuple
    central: tuple
    waypoints: tuple
    free_threshold: float
    occupied_threshold: float
    waypoint_evidence: tuple = ()

    def __post_init__(self):
        c.require(type(self.metadata) is m.MapMetadata and type(self.raster) is Raster,
                  'bundle', 'C0 metadata and Raster required')
        e, version, frame = self.metadata.envelope, self.metadata.map_version, self.metadata.origin.frame_id
        c.require(e.producer is c.Owner.P02 and e.versions.map == version, 'metadata.version',
                  'P02 metadata envelope must bind map version', c.ErrorCode.VERSION_MISMATCH)
        finite(self.free_threshold, 'free_threshold')
        finite(self.occupied_threshold, 'occupied_threshold')
        c.require(0 < self.free_threshold < self.occupied_threshold < 1,
                  'thresholds', '0 < free < occupied < 1 required')
        validate_polygon(self.boundary, 'boundary')
        validate_polygon(self.central, 'central')
        c.require(covered(self.central,self.boundary),'central','central area extends outside field')
        c.require(type(self.waypoints) is tuple, 'waypoints', 'immutable tuple required')
        c.require({r.area for r in self.metadata.regions} == set(m.Area) and len(self.metadata.regions) == 4,
                  'regions', 'exactly A/B/C/D required')
        for r in self.metadata.regions:
            validate_polygon(r.polygon_m, 'region')
            c.require(covered(r.polygon_m,self.boundary),'region','collection region extends outside field')
            c.require(r.frame_id == frame and inside(m.Point2(r.placement_center.x, r.placement_center.y), r.polygon_m),
                      'placement_center', 'center must lie inside same-frame region')
        c.require(len({p.name for p in self.waypoints}) == len(self.waypoints), 'waypoints', 'duplicate name')
        for p in self.waypoints:
            c.require(type(p) is m.WaypointResult and p.envelope.versions.map == version and
                      p.envelope.versions.config == e.versions.config and p.envelope.origin is e.origin,
                      'waypoints.version', 'mixed version or provenance', c.ErrorCode.VERSION_MISMATCH)
            c.require(p.pose is None or p.pose.frame_id == frame, 'waypoints.frame_id', 'wrong frame')
        c.require(type(self.waypoint_evidence) is tuple and
                  len({name for name, refs in self.waypoint_evidence}) == len(self.waypoint_evidence),
                  'waypoint_evidence', 'unique immutable evidence groups required')
        for name, refs in self.waypoint_evidence:
            c.require(name in {p.name for p in self.waypoints} and type(refs) is tuple and
                      all(type(ref) is c.EvidenceRef and ref.origin is e.origin for ref in refs),
                      'waypoint_evidence', 'unknown point or mismatched provenance')

    def grid_corner(self, col, row):
        r, o = self.metadata.resolution_m_per_cell, self.metadata.origin
        x, y = col*r, (self.raster.height-row)*r
        return m.Point2(o.x + math.cos(o.yaw)*x - math.sin(o.yaw)*y,
                        o.y + math.sin(o.yaw)*x + math.cos(o.yaw)*y)

    def cell_center(self, col, row):
        c.require(type(col) is int and type(row) is int and
                  0 <= col < self.raster.width and 0 <= row < self.raster.height,
                  'cell', 'cell outside raster')
        p = self.grid_corner(col + .5, row + .5)
        return c.MapPose(self.metadata.origin.frame_id, p.x, p.y, 0, 'm', 'rad',
                         self.metadata.origin.error_m, self.metadata.origin.error_rad)

    def world_cell(self, x, y):
        c.require(type(x) in (int,float) and type(y) in (int,float) and math.isfinite(x) and math.isfinite(y),
                  'point', 'finite map coordinates required')
        o, r = self.metadata.origin, self.metadata.resolution_m_per_cell
        dx, dy = x-o.x, y-o.y
        gx, gy = (math.cos(o.yaw)*dx+math.sin(o.yaw)*dy)/r, (-math.sin(o.yaw)*dx+math.cos(o.yaw)*dy)/r
        c.require(0 <= gx < self.raster.width and 0 <= gy < self.raster.height, 'point', 'outside raster')
        return math.floor(gx), self.raster.height-1-math.floor(gy)


def _pairs(items):
    result = {}
    for k, v in items:
        c.require(k not in result, k, 'duplicate JSON key', c.ErrorCode.ID_CONFLICT)
        result[k] = v
    return result


def load_bundle(path, expected_version):
    try:
        path = Path(path)
        data = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=_pairs,
                          parse_constant=lambda s: c.require(False, s, 'non-finite JSON'))
        fields = {'format','contract_version','width','height','row_order','image_sha256','metadata',
                  'boundary','central','waypoints_uri','waypoints_sha256','free_threshold','occupied_threshold'}
        c.require(type(data) is dict and set(data) == fields, 'manifest', 'exact P02 manifest fields required')
        c.require(type(data['format']) is int and data['format'] == 1 and data['contract_version'] == c.CONTRACT_VERSION,
                  'format', 'unsupported file or C0 version', c.ErrorCode.VERSION_MISMATCH)
        c.require(data['row_order'] == 'top_to_bottom', 'row_order', 'unsupported row convention')
        meta = c.from_data(m.MapMetadata, data['metadata'])
        c.require(meta.map_version == expected_version, 'map_version', 'wrong selected version', c.ErrorCode.VERSION_MISMATCH)
        c.require(meta.image_uri == 'grid.pgm', 'image_uri', 'bundle-local grid.pgm required')
        image = path.parent / meta.image_uri
        c.require(not image.is_symlink(), 'image_uri', 'linked image not allowed')
        raw = image.read_bytes()
        c.require(hashlib.sha256(raw).hexdigest() == data['image_sha256'], 'image_sha256', 'image checksum mismatch')
        raster = read_pgm(raw)
        c.require(type(data['width']) is int and type(data['height']) is int and
                  (data['width'],data['height']) == (raster.width,raster.height), 'dimensions', 'image size mismatch')
        c.require(data['waypoints_uri'] == 'waypoints.json','waypoints_uri','bundle-local waypoints.json required')
        points_path=path.parent/'waypoints.json'
        c.require(not points_path.is_symlink(),'waypoints_uri','linked points file not allowed')
        point_bytes=points_path.read_bytes()
        c.require(hashlib.sha256(point_bytes).hexdigest() == data['waypoints_sha256'],
                  'waypoints_sha256','point file checksum mismatch')
        point_data=json.loads(point_bytes.decode('utf-8'),object_pairs_hook=_pairs)
        c.require(type(point_data) is dict and set(point_data) ==
                  {'format','contract_version','map_version','waypoints','waypoint_evidence'},
                  'waypoints','exact point-file fields required')
        c.require(type(point_data['format']) is int and point_data['format'] == 1 and
                  point_data['contract_version'] == c.CONTRACT_VERSION and point_data['map_version'] == expected_version,
                  'waypoints.version','mixed map/point file versions',c.ErrorCode.VERSION_MISMATCH)
        return MapBundle(meta, raster, tuple(c.from_data(m.Point2, p) for p in data['boundary']),
                         tuple(c.from_data(m.Point2, p) for p in data['central']),
                         tuple(c.from_data(m.WaypointResult, p) for p in point_data['waypoints']),
                         data['free_threshold'], data['occupied_threshold'],
                         tuple((item['name'],tuple(c.from_data(c.EvidenceRef, e) for e in item['evidence']))
                               for item in point_data['waypoint_evidence']))
    except OSError as error:
        raise c.ContractError(c.ErrorCode.UNAVAILABLE, 'map_file', str(error)) from None
    except (ValueError, TypeError, KeyError) as error:
        if isinstance(error, c.ContractError):
            raise
        raise c.ContractError(c.ErrorCode.INVALID_FIELD, 'manifest', 'malformed map bundle') from None


def with_version(bundle, version):
    def update(e):
        return replace(e, versions=replace(e.versions, map=version))
    return replace(bundle, metadata=replace(bundle.metadata, map_version=version,
                                           envelope=update(bundle.metadata.envelope), image_uri='grid.pgm'),
                   waypoints=tuple(replace(p, envelope=update(p.envelope)) for p in bundle.waypoints))


def save_bundle(bundle, root, version):
    """Write, fsync, reload/compare, then publish a new immutable directory.

    Existing versions are never overwritten. Failed pending directories are kept
    for diagnosis; publishing does not change an active competition session.
    """
    c.require(type(version) is str and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', version),
              'version', 'safe nonempty version token required')
    try:
        root = Path(root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        target = root / version
        c.require(not target.exists(), 'version', 'version already exists', c.ErrorCode.CONFLICT)
        pending = Path(tempfile.mkdtemp(prefix='.pending-', dir=str(root)))
        updated = with_version(bundle, version)
        raw = ('P5\n{} {}\n255\n'.format(updated.raster.width, updated.raster.height).encode('ascii') +
               bytes(updated.raster.pixels))
        point_data=dict(format=1,contract_version=c.CONTRACT_VERSION,map_version=version,
                        waypoints=c.to_data(updated.waypoints),
                        waypoint_evidence=[dict(name=name,evidence=c.to_data(refs)) for name,refs in updated.waypoint_evidence])
        point_bytes=json.dumps(point_data,ensure_ascii=False,allow_nan=False,indent=2).encode('utf-8')
        data = dict(format=1, contract_version=c.CONTRACT_VERSION, width=updated.raster.width,
                    height=updated.raster.height, row_order='top_to_bottom', image_sha256=hashlib.sha256(raw).hexdigest(),
                    metadata=c.to_data(updated.metadata), boundary=c.to_data(updated.boundary),
                    central=c.to_data(updated.central), waypoints_uri='waypoints.json',
                    waypoints_sha256=hashlib.sha256(point_bytes).hexdigest(),
                    free_threshold=updated.free_threshold, occupied_threshold=updated.occupied_threshold,
                    )
        for name, content in (('grid.pgm', raw), ('waypoints.json',point_bytes), ('map.json', json.dumps(data, ensure_ascii=False,
                              allow_nan=False, indent=2).encode('utf-8'))):
            with (pending/name).open('xb') as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        loaded = load_bundle(pending/'map.json', version)
        c.require(loaded == updated, 'reload', 'saved map differs from source', c.ErrorCode.STORAGE_FAILURE)
        # rename to a nonempty existing version fails; no replacement of old data.
        os.rename(str(pending), str(target))
        return load_bundle(target/'map.json', version)
    except OSError as error:
        raise c.ContractError(c.ErrorCode.STORAGE_FAILURE, 'save_bundle', str(error)) from None
