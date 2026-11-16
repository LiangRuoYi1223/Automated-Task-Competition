"""P10A conjunctions over C0 facts. They do not generate physical observations."""
from dataclasses import fields, is_dataclass
from treasure_auto.contracts import core as c, models as m, rules as r


def context(envelope, now, versions=None):
    return m.ConsumerContext(envelope.run_id, envelope.instance_id, envelope.control_epoch,
                             now, versions or envelope.versions, None, None, None,
                             None, None, None, envelope.origin)


def physical(evidence, envelope, now, target=None):
    c.require(bool(evidence), 'evidence', 'new physical evidence required', c.ErrorCode.EVIDENCE_MISSING)
    for item in evidence:
        r.require_physical_evidence(item, envelope, now, envelope.origin)
        if target is not None:
            c.require(item.target_id == target, 'evidence.target_id', 'wrong held object', c.ErrorCode.STALE_REQUEST)


def health_check(health, run, now, config):
    r.validate_envelope(health.envelope, context(run.envelope, now))
    c.require(health.envelope.producer is c.Owner.P01, 'health.producer', 'P01 facts required')
    names = [x.name for x in health.capabilities]
    c.require(len(names) == len(set(names)), 'health', 'duplicate capability')
    caps = {x.name: x for x in health.capabilities}
    for name in config.required_capabilities:
        cap = caps.get(name)
        c.require(cap is not None and cap.status is c.Fact.CONFIRMED and bool(cap.evidence),
                  name, 'capability not confirmed', c.ErrorCode.UNAVAILABLE)
        for ev in cap.evidence:
            c.require(ev.run_id == run.envelope.run_id and ev.origin is run.envelope.origin,
                      'health.evidence', 'wrong evidence provenance')
            c.require(now - ev.received_at >= 0 and ev.valid_until - now > 0,
                      'health.evidence', 'stale or future capability', c.ErrorCode.EXPIRED)


def permission_check(request, run, now, config, intent, receipt, scene_epoch, targets):
    r.require_provenance(request, run.envelope.origin)
    current = context(run.envelope, now)
    r.validate_envelope(request.envelope, current)
    c.require(request.envelope.base_motion_epoch == run.envelope.base_motion_epoch,
              'base_motion_epoch', 'old target after base motion', c.ErrorCode.STALE_GEOMETRY)
    c.require(request.envelope.action_id is not None, 'action_id', 'saved action required')
    c.require(intent is not None and receipt == request.intent_receipt,
              'intent_receipt', 'exact registered intent/receipt required', c.ErrorCode.STORAGE_FAILURE)
    r.validate_intent_receipt(intent, receipt)
    c.require(intent.envelope == request.envelope, 'intent', 'request must bind original intent', c.ErrorCode.ID_CONFLICT)
    base_sources = (m.MotionSource.NAVIGATION, m.MotionSource.PRECISION, m.MotionSource.RECOVERY, m.MotionSource.DEBUG)
    if request.source in base_sources:
        c.require(intent.operation in (m.IntentOperation.BASE_GOAL, m.IntentOperation.BASE_VELOCITY),
                  'intent.operation', 'wrong base operation')
    else:
        expected = m.IntentOperation.ARM if request.source is m.MotionSource.ARM else m.IntentOperation.PUMP
        c.require(intent.operation is expected, 'intent.operation', 'wrong actuator operation')
    health_check(request.health, run, now, config)
    for value, owners in ((request.localization, (c.Owner.P02,)), (request.readiness, (c.Owner.P07,)),
                          (request.space, (c.Owner.P08,)), (request.old_stop, (c.Owner.P01, c.Owner.P03))):
        r.validate_envelope(value.envelope, current)
        c.require(value.envelope.producer in owners, 'producer', 'wrong fact owner')
        c.require(value is request.space or value.envelope.base_motion_epoch == request.envelope.base_motion_epoch,
                  'base_motion_epoch', 'motion invalidated previous evidence', c.ErrorCode.STALE_GEOMETRY)
    c.require(request.localization.trusted is c.Fact.CONFIRMED
              and request.localization.base_motion_epoch == request.envelope.base_motion_epoch,
              'localization', 'trusted current location required')
    physical(request.localization.evidence, request.localization.envelope, now)
    r.require_space_version(request.envelope, request.space)
    c.require(request.readiness.posture_confirmed is c.Fact.CONFIRMED, 'posture', 'posture unknown')
    holding = request.readiness.holding
    c.require(holding.state in (c.Holding.EMPTY, c.Holding.HELD), 'holding', 'unknown/dropped holding blocks output')
    physical(holding.evidence, request.readiness.envelope, now, holding.target_id)
    physical(request.readiness.evidence, request.readiness.envelope, now)
    stop = request.old_stop
    c.require(stop.physically_stopped is c.Fact.CONFIRMED and stop.old_queue_empty is c.Fact.CONFIRMED,
              'old_stop', 'old actions and base must actually stop', c.ErrorCode.NOT_AUTHORIZED)
    physical(stop.evidence, stop.envelope, now)
    if run.envelope.versions.mapping is None:
        c.require(request.source in base_sources and request.purpose == 'read_board',
                  'mapping', 'mapping absent: only board observation navigation allowed', c.ErrorCode.INVALID_MAPPING)
    if request.source is m.MotionSource.DEBUG:
        c.require(False, 'debug', 'manual relocation is by staff, not an unreviewed motor command', c.ErrorCode.NOT_AUTHORIZED)
    if request.source is m.MotionSource.ARM:
        phases = tuple(segment.phase for segment in intent.arm_plan.segments)
        if any(p in (m.Phase.APPROACH, m.Phase.DESCEND, m.Phase.CONTACT) for p in phases):
            if request.purpose == 'place':
                c.require(request.envelope.producer is c.Owner.P08,
                          'producer', 'only P08 may request placement continuation', c.ErrorCode.NOT_AUTHORIZED)
                c.require(holding.state is c.Holding.HELD and holding.target_id == request.envelope.target_id,
                          'holding', 'placement must carry the same confirmed object')
                c.require(m.Phase.CONTACT not in phases, 'phase', 'grasp contact is not a placement phase')
            else:
                c.require(holding.state is c.Holding.EMPTY, 'holding', 'second grasp while held is forbidden')
    if targets:
        c.require(request.stillness is not None, 'stillness', 'collapse requires a new certificate', c.ErrorCode.EVIDENCE_MISSING)
    if request.stillness is not None:
        c.require(request.stillness.envelope.producer is c.Owner.P08, 'stillness.producer', 'P08 certificate required')
        r.validate_stillness(request.stillness, request.envelope, now, targets, scene_epoch)
    return tuple(collect_evidence(request))


def collect_evidence(value):
    if isinstance(value, c.EvidenceRef):
        yield value
    elif is_dataclass(value):
        for field in fields(value):
            yield from collect_evidence(getattr(value, field.name))
    elif isinstance(value, tuple):
        for item in value:
            yield from collect_evidence(item)


def expiration_bounds(value):
    if isinstance(value, c.Envelope):
        yield value.expires_at.seconds
    elif isinstance(value, c.EvidenceRef):
        yield value.valid_until.seconds
    elif is_dataclass(value):
        for field in fields(value):
            yield from expiration_bounds(getattr(value, field.name))
    elif isinstance(value, tuple):
        for item in value:
            yield from expiration_bounds(item)
