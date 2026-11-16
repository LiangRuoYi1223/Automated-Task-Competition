"""Cross-record guards for offline observation adapters; no source is inferred."""
from dataclasses import asdict, fields, is_dataclass, replace
from enum import Enum
from treasure_auto.contracts import core as c, models as m, rules
from .config import text
from .observations import FrameInput, frame_evidence


def data(value):
    if isinstance(value,Enum): return value.value
    if is_dataclass(value): return {f.name:data(getattr(value,f.name)) for f in fields(value)}
    if isinstance(value,tuple): return [data(v) for v in value]
    return value


def validate_frame(frame, observation, context, config, template, now):
    c.require(type(frame) is FrameInput,'adapter','missing typed frame details',c.ErrorCode.UNAVAILABLE)
    c.require(frame.image==observation.image,'adapter.image','details belong to another image',c.ErrorCode.ID_CONFLICT)
    c.require(observation.template_version==template.version,'template_version','wrong template',c.ErrorCode.VERSION_MISMATCH)
    rules.validate_envelope(frame.image.envelope,replace(context,now=now))
    c.require(frame.image.envelope.producer is c.Owner.P04,'image.producer','P04 observation required')
    c.require(frame.image.capture_pose is not None and frame.image.capture_pose.frame_id=='map',
              'image.capture_pose','aligned map capture pose missing',c.ErrorCode.UNAVAILABLE)
    c.require(frame.waypoint_version==config.waypoint_version,'waypoint_version','unverified observation pose',c.ErrorCode.VERSION_MISMATCH)
    for name in ('acquisition_id','segment_id'): text(getattr(frame,name),name)
    c.require(frame.stationary is c.Fact.CONFIRMED and bool(frame.stationary_evidence),
              'stationary','actual stable observation condition missing',c.ErrorCode.EVIDENCE_MISSING)
    c.require(frame.image.image.sha256 is not None,'image.sha256','pixel identity missing',c.ErrorCode.EVIDENCE_MISSING)
    captured=frame.image.image.captured_at
    c.require(captured is not None,'captured_at','capture clock missing',c.ErrorCode.EVIDENCE_MISSING)
    c.require(0<=frame.sensor_now-captured<=config.max_frame_age_s,'captured_at','stale/future sensor frame',c.ErrorCode.EXPIRED)
    c.require(captured-frame.capture_not_before>=0,'capture_not_before','capture predates observation barrier',c.ErrorCode.EXPIRED)
    c.require(now-frame.image.image.received_at<=config.max_frame_age_s,'received_at','stale received frame',c.ErrorCode.EXPIRED)
    for ref in frame_evidence(frame):
        rules.require_physical_evidence(ref,frame.image.envelope,now,context.origin)
        c.require(0<=captured-ref.captured_at<=config.max_frame_age_s,
                  'evidence.captured_at','future/stale/unrelated source evidence',c.ErrorCode.EXPIRED)
        c.require(ref.captured_at-frame.capture_not_before>=0,'evidence.captured_at','evidence predates barrier',c.ErrorCode.EXPIRED)
    return frame_evidence(frame)


def validate_saved(snapshot, run, context, prefix, now):
    c.require(type(snapshot) is m.StateSnapshot,'snapshot','no committed state',c.ErrorCode.STORAGE_FAILURE)
    c.require(snapshot.run.envelope.run_id==run.envelope.run_id and
              snapshot.run.started_at==run.started_at and snapshot.run.deadline==run.deadline,
              'snapshot.run','wrong run/original deadline',c.ErrorCode.STALE_RUN)
    c.require(snapshot.run.envelope.instance_id==context.instance_id and
              snapshot.run.envelope.control_epoch==context.control_epoch,
              'snapshot.control','current runtime recovery registration required',c.ErrorCode.STALE_CONTROL)
    mapping=snapshot.mapping
    c.require(type(mapping) is m.MappingSnapshot,'snapshot.mapping','no committed mapping',c.ErrorCode.STORAGE_FAILURE)
    c.require(mapping.envelope.run_id==run.envelope.run_id and mapping.envelope.producer is c.Owner.P05,
              'mapping.run','wrong mapping owner or run',c.ErrorCode.STALE_RUN)
    rules.require_provenance(mapping,context.origin)
    c.require(mapping.mapping_version.startswith(prefix) and
              snapshot.space.envelope.versions.mapping==mapping.mapping_version,
              'mapping.version','unverified template/configuration or state version',c.ErrorCode.VERSION_MISMATCH)
    for name in ('config','map','calibration','model'):
        c.require(getattr(mapping.envelope.versions,name)==getattr(context.versions,name),
                  'mapping.versions.'+name,'historical configuration changed',c.ErrorCode.VERSION_MISMATCH)
    c.require(mapping.envelope.versions.mapping==mapping.mapping_version,'mapping.version','inconsistent mapping envelope')
    c.require(0<=now-mapping.frozen_at and mapping.frozen_at-run.started_at>=0 and
              run.deadline-mapping.frozen_at>0,'mapping.frozen_at','invalid original freeze time')
    # Historical evidence may have expired: restoration is context, never a fresh
    # camera observation, a motion permission, or a recovered confirmation window.
    c.require(all(e.run_id==context.run_id and e.kind is c.EvidenceKind.PHYSICAL and
                  e.captured_at is not None and e.sha256 is not None for e in mapping.evidence),
              'mapping.evidence','incomplete historical proof',c.ErrorCode.EVIDENCE_MISSING)
    return mapping
