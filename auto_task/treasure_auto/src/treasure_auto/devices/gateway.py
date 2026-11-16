"""C0 DevicePort implementation for offline assembly, AFTER any velocity smoother.

Only bounded, nonblocking simulated transports are accepted. There is no ROS
publisher or serial connection here. Real transport/firmware cancellation and
process exclusivity require a separate reviewed integration; setting a config
flag cannot enable them. Disk IO and long business tasks never run under the
output lock. P10A owns permits and P10B owns durable intention commits.
"""
from dataclasses import replace
import threading
from uuid import uuid4
from treasure_auto.contracts import core as c, models as m, rules as r
from .adapters import encode_arm
from .config import positive


BASE_SOURCES = (m.MotionSource.NAVIGATION, m.MotionSource.PRECISION,
                m.MotionSource.RECOVERY, m.MotionSource.DEBUG)
BASE_OWNERS = {m.MotionSource.NAVIGATION: (c.Owner.P03,), m.MotionSource.PRECISION: (c.Owner.P03,),
               m.MotionSource.RECOVERY: (c.Owner.P03,), m.MotionSource.DEBUG: (c.Owner.P10A, c.Owner.P11)}


def channel(source):
    return 'base' if source in BASE_SOURCES else source.value


class DeviceGateway:
    def __init__(self, clock, run, config, transport, health_provider, outbox, serial_lease,
                 physical_validator=None):
        if run.envelope.origin is not c.Origin.SIMULATED or getattr(transport, 'origin', None) is not c.Origin.SIMULATED:
            raise ValueError('only explicit offline simulated transport is delivered; live output blocked')
        if config.config_version != run.envelope.versions.config or config.real_output_enabled:
            raise ValueError('configuration version mismatch or unverified live control')
        self.clock, self.run, self.config = clock, run, config
        self.transport, self._health_provider, self.outbox = transport, health_provider, outbox
        self.serial_lease, self._physical_validator = serial_lease, physical_validator
        self._lock = threading.RLock()
        self._permits, self._retired, self._intents = {}, set(), {}
        self._seen, self._events, self._sequences = {}, {}, {}
        self._active, self._latest = {}, {}
        self._last_now, self._closed, self._sequence = clock.now(), False, 0
        self._last_stop = None
        self._stops = {}
        self._pending_stops = set()
        if serial_lease is not None:
            serial_lease.bind(self)

    def _context(self, now):
        e = self.run.envelope
        return m.ConsumerContext(e.run_id, e.instance_id, e.control_epoch, now, e.versions,
                                 None, None, None, e.layout_recovery_round, e.rebuild_generation,
                                 e.base_motion_epoch, e.origin)

    def _now(self):
        now = self.clock.now()
        c.monotonic(now, 'now')
        c.require(now - self._last_now >= 0, 'now', 'clock moved backwards', c.ErrorCode.CLOCK_MISMATCH)
        c.require(now - self.run.started_at >= 0, 'now', 'clock predates run', c.ErrorCode.CLOCK_MISMATCH)
        self._last_now = now
        return now

    def _open(self, now):
        c.require(not self._closed, 'gateway', 'gateway latched stopped', c.ErrorCode.NOT_AUTHORIZED)
        c.require(self.run.deadline - now > 0, 'deadline', 'original run deadline reached', c.ErrorCode.DEADLINE)
        c.require(self.outbox.failure is None, 'outbox', 'persistence failure', c.ErrorCode.STORAGE_FAILURE)
        c.require(not self._pending_stops, 'stop', 'a stop send remains unresolved', c.ErrorCode.UNAVAILABLE)

    def _envelope(self, original, now):
        self._sequence += 1
        return replace(original, producer=c.Owner.P01, event_id=str(uuid4()), sequence=self._sequence,
                       emitted_at=now, expires_at=replace(now, seconds=now.seconds + 1.0))

    def _event(self, cmd, status, reason, accepted, submitted, echoed=c.Fact.UNKNOWN,
               physical=c.Fact.UNKNOWN, evidence=(), now=None, elapsed=0.):
        now = now or self._last_now
        phase = cmd.phase.value if hasattr(cmd, 'phase') else (cmd.source.value if hasattr(cmd, 'source') else 'stop')
        event = m.ActionEvent(self._envelope(cmd.envelope, now), c.ActionResult(
            status, reason=reason, phase=phase, accepted=accepted, submitted=submitted,
            echoed=echoed, physical=physical, evidence=evidence, elapsed_s=max(0., elapsed)))
        self.outbox.append(event)
        return event

    def health(self):
        return self._health_provider()

    def _health(self, now):
        snapshot = self.health()
        now = self._now()  # health() emits after its own sample, never validate against an earlier instant.
        r.validate_envelope(snapshot.envelope, self._context(now))
        c.require(snapshot.envelope.producer is c.Owner.P01, 'health.producer', 'wrong health producer')
        caps = {cap.name: cap.status for cap in snapshot.capabilities}
        c.require(len(caps) == len(snapshot.capabilities), 'capabilities', 'duplicate capability')
        return caps

    def install_permit(self, permit):
        """Explicit P10A/P11 handoff, never called implicitly by submit_*()."""
        with self._lock:
            now = self._now(); self._open(now)
            r.validate_envelope(permit.envelope, self._context(now))
            c.require(permit.envelope.expires_at - self.run.deadline <= 0,
                      'permit.expires_at', 'permit extends past original deadline', c.ErrorCode.DEADLINE)
            c.require(permit.permit_id not in self._retired, 'permit_id', 'revoked permit cannot be reused', c.ErrorCode.NOT_AUTHORIZED)
            key = channel(permit.source)
            current = self._permits.get(key)
            c.require(current is None or current == permit, 'permit', 'source handoff requires explicit revocation', c.ErrorCode.CONFLICT)
            self._permits[key] = permit

    def register_intent(self, command, intent, receipt):
        """Consume a confirmed C0 receipt; no disk IO, no receipt-id invention."""
        with self._lock:
            now = self._now(); self._open(now)
            r.validate_envelope(command.envelope, self._context(now))
            r.validate_intent_receipt(intent, receipt)
            r.require_provenance(intent, self.run.envelope.origin)
            r.require_provenance(command, self.run.envelope.origin)
            c.require(receipt.commit.transaction_id == command.intent_receipt_id,
                      'intent_receipt_id', 'receipt transaction does not match command', c.ErrorCode.ID_CONFLICT)
            c.require(intent.envelope == command.envelope, 'intent.envelope', 'command and intent identities differ', c.ErrorCode.ID_CONFLICT)
            if type(command) is m.BaseCommand:
                matches = intent.operation is m.IntentOperation.BASE_VELOCITY and intent.base_velocity_m_s_rad_s == (command.linear_m_s, command.angular_rad_s)
            elif type(command) is m.ArmCommand:
                plan = intent.arm_plan
                matches = (intent.operation is m.IntentOperation.ARM and plan is not None and
                           plan.reachable is c.Fact.CONFIRMED and plan.collision_free is c.Fact.CONFIRMED and
                           plan.target == command.target and plan.segments == (m.PathSegment(command.phase, command.target, command.speed_mm_s, command.timeout_s),))
                if plan is not None:
                    r.validate_envelope(plan.envelope, self._context(now))
                    c.require(plan.envelope.producer is c.Owner.P06, 'arm_plan.producer', 'geometry must come from P06')
                    for name in ('request_id', 'action_id', 'target_id'):
                        c.require(getattr(plan.envelope, name) == getattr(command.envelope, name),
                                  'arm_plan.' + name, 'plan belongs to another action', c.ErrorCode.ID_CONFLICT)
            elif type(command) is m.PumpCommand:
                matches = intent.operation is m.IntentOperation.PUMP and intent.pump_enabled == command.enabled
            else:
                matches = False
            c.require(matches, 'intent', 'committed payload must bind exact command including stage/speed/timeout', c.ErrorCode.ID_CONFLICT)
            payload = c.dumps(command)
            old = self._intents.get(command.intent_receipt_id)
            c.require(old is None or old == (payload, intent), 'intent_receipt_id', 'receipt reused for another command', c.ErrorCode.ID_CONFLICT)
            self._intents[command.intent_receipt_id] = (payload, intent)

    def _prerequisites(self, cmd, now):
        saved = self._intents.get(cmd.intent_receipt_id)
        c.require(saved is not None and saved[0] == c.dumps(cmd),
                  'intent_receipt_id', 'exact durable intent not registered', c.ErrorCode.STORAGE_FAILURE)
        intent = saved[1]
        evidence = cmd.permit.prerequisite_evidence + getattr(cmd, 'preconditions', ())
        if intent.arm_plan is not None:
            r.validate_envelope(intent.arm_plan.envelope, self._context(now))
            evidence += intent.arm_plan.evidence
        if type(cmd) is m.PumpCommand:
            evidence += cmd.holding.evidence
        for item in evidence:
            c.require(item.run_id == cmd.envelope.run_id and item.origin is cmd.envelope.origin,
                      'preconditions', 'evidence provenance mismatch', c.ErrorCode.SIMULATED_EVIDENCE)
            c.require(now - item.received_at >= 0 and item.valid_until - now > 0,
                      'preconditions', 'prerequisite expired or future', c.ErrorCode.EXPIRED)

    def _check(self, cmd, source, now):
        self._open(now)
        r.validate_envelope(cmd.envelope, self._context(now))
        c.require(cmd.envelope.action_id is not None, 'action_id', 'stable action identity required')
        key = channel(source)
        expected_owners = BASE_OWNERS[source] if key == 'base' else (c.Owner.P07, c.Owner.P08)
        c.require(cmd.envelope.producer in expected_owners, 'producer', 'source owner mismatch', c.ErrorCode.NOT_AUTHORIZED)
        permit = self._permits.get(key)
        c.require(permit is not None and permit == cmd.permit and permit.source is source,
                  'permit', 'not the installed active permit', c.ErrorCode.NOT_AUTHORIZED)
        r.validate_envelope(permit.envelope, self._context(now))
        # purpose is P10A's business scope (e.g. read_board), not a local phase
        # alias. Full installed-permit equality and the saved command bind it.
        # Permits may bind a work order or target; a bound value is never a wildcard.
        for name in ('request_id', 'action_id', 'target_id'):
            value = getattr(permit.envelope, name)
            c.require(value is None or value == getattr(cmd.envelope, name), name, 'permit scope mismatch', c.ErrorCode.NOT_AUTHORIZED)
        c.require(self._health(now).get(key) is c.Fact.CONFIRMED,
                  'health.' + key, 'dependent capability unavailable', c.ErrorCode.UNAVAILABLE)
        now = self._now()
        self._open(now)
        r.validate_envelope(cmd.envelope, self._context(now))
        r.validate_envelope(permit.envelope, self._context(now))
        self._prerequisites(cmd, now)
        if key != 'base':
            c.require(self.serial_lease is not None and self.serial_lease.owns(self),
                      'serial', 'unique serial owner missing', c.ErrorCode.UNAVAILABLE)
        return key, now

    def _submit(self, cmd, source):
        with self._lock:
            now = self._last_now
            try:
                now = self._now()
                self.tick()
                key, now = self._check(cmd, source, now)
                payload = c.dumps(cmd)
                action = cmd.envelope.action_id
                if action in self._seen:
                    c.require(self._seen[action] == payload, 'action_id', 'action reused with different payload', c.ErrorCode.ID_CONFLICT)
                    return self._latest[action]
                event_key = (cmd.envelope.producer, cmd.envelope.event_id)
                c.require(event_key not in self._events, 'event_id', 'command event id already consumed', c.ErrorCode.ID_CONFLICT)
                stream = (cmd.envelope.producer, source)
                c.require(cmd.envelope.sequence > self._sequences.get(stream, -1),
                          'sequence', 'old or repeated sequence', c.ErrorCode.STALE_REQUEST)
                if key != 'base':
                    c.require(key not in self._active, 'action', 'prior action not yet resolved', c.ErrorCode.CONFLICT)
                dispatch = None
                if key == 'base':
                    c.require(self.config.max_linear_m_s is not None and self.config.max_angular_rad_s is not None,
                              'base_limits', 'base limits unmeasured', c.ErrorCode.UNCONFIGURED)
                    c.require(abs(cmd.linear_m_s) <= self.config.max_linear_m_s and abs(cmd.angular_rad_s) <= self.config.max_angular_rad_s,
                              'velocity', 'base limit exceeded', c.ErrorCode.LIMIT_EXCEEDED)
                elif key == 'arm':
                    dispatch = encode_arm(cmd, self.config)
                c.require(self.outbox.check_dispatch_capacity(), 'outbox', 'cannot record dispatch result', c.ErrorCode.STORAGE_FAILURE)
                # Reserve identities BEFORE calling a transport whose failure can be ambiguous.
                self._seen[action], self._events[event_key], self._sequences[stream] = payload, payload, cmd.envelope.sequence
                self._event(cmd, c.Outcome.UNKNOWN, c.Reason(c.ErrorCode.EVIDENCE_MISSING, 'accepted', 'validated; not yet submitted'),
                            c.Fact.CONFIRMED, c.Fact.UNKNOWN, now=now)
                self._open(now)  # An outbox overflow must not send a new action.
                self._active[key] = (cmd, now)
                try:
                    if key == 'base':
                        self.transport.send_base(cmd.linear_m_s, cmd.angular_rad_s)
                    elif key == 'arm':
                        self.transport.send_arm(dispatch)
                    else:
                        self.transport.send_pump(cmd.enabled)
                except Exception as error:
                    result = self._event(cmd, c.Outcome.UNKNOWN, c.Reason(c.ErrorCode.UNAVAILABLE, 'transport', str(error) or type(error).__name__),
                                         c.Fact.CONFIRMED, c.Fact.UNKNOWN, now=now)
                    self._latest[action] = result
                    self.revoke(source)
                    return result
                details = ('submitted; physical confirmation pending' if dispatch is None else
                           'x_mm={} y_mm={} z_mm={} speed_mm_s={} feed_mm_min={} timeout_s={}; physical confirmation pending'.format(
                               dispatch.x_mm, dispatch.y_mm, dispatch.z_mm, cmd.speed_mm_s, dispatch.feed_mm_min, dispatch.timeout_s))
                result = self._event(cmd, c.Outcome.UNKNOWN, c.Reason(c.ErrorCode.EVIDENCE_MISSING, 'physical', details),
                                     c.Fact.CONFIRMED, c.Fact.CONFIRMED, now=now)
                self._latest[action] = result
                if self.outbox.failure is not None:
                    self._closed = True
                    self._halt(('base', 'arm', 'pump'))
                return result
            except c.ContractError as error:
                result = self._event(cmd, c.Outcome.REJECTED, c.Reason(error.code, error.field, error.detail),
                                     c.Fact.REJECTED, c.Fact.REJECTED, now=now)
                if error.code in (c.ErrorCode.CLOCK_MISMATCH, c.ErrorCode.DEADLINE) or self.outbox.failure is not None:
                    self._closed = True
                    self._halt(('base', 'arm', 'pump'))
                return result

    def submit_base(self, command):
        if type(command) is not m.BaseCommand or command.source not in BASE_SOURCES:
            raise ValueError('expected base command and base motion source')
        return self._submit(command, command.source)

    def submit_arm(self, command):
        if type(command) is not m.ArmCommand:
            raise ValueError('expected C0 ArmCommand with explicit stage speed')
        return self._submit(command, m.MotionSource.ARM)

    def submit_pump(self, command):
        if type(command) is not m.PumpCommand:
            raise ValueError('expected C0 PumpCommand')
        return self._submit(command, m.MotionSource.PUMP)

    def _halt(self, keys):
        success = True
        for key in keys:
            permit = self._permits.pop(key, None)
            if permit:
                self._retired.add(permit.permit_id)
            self._active.pop(key, None)
            try:
                if key == 'base':
                    self.transport.send_base(0., 0.)
                elif key == 'arm':
                    self.transport.cancel_arm()
                # A stop NEVER infers holding state or toggles the pump.
                self._pending_stops.discard(key)
            except Exception:
                success = False
                self._pending_stops.add(key)
        return success

    def revoke(self, source=None):
        with self._lock:
            keys = ('base', 'arm', 'pump') if source is None else (channel(source),)
            for key in keys:
                if key in self._active:
                    cmd, began = self._active[key]
                    self._latest[cmd.envelope.action_id] = self._event(cmd, c.Outcome.CANCELLED,
                        c.Reason(c.ErrorCode.CANCELLED, 'permit', 'permission revoked; physical state unknown'),
                        c.Fact.CONFIRMED, self._latest[cmd.envelope.action_id].result.submitted,
                        now=self._last_now, elapsed=self._last_now - began)
            return self._halt(keys)

    def tick(self):
        """Independent safety tick; never reads disk or waits for business work."""
        with self._lock:
            if self._pending_stops:
                self._halt(tuple(self._pending_stops))
            if self._closed:
                return
            try:
                now = self._now(); self._open(now)
                caps = self._health(now)
                now = self._now(); self._open(now)
            except Exception:
                self._closed = True
                self._halt(('base', 'arm', 'pump'))
                return
            for key, permit in tuple(self._permits.items()):
                try:
                    r.validate_envelope(permit.envelope, self._context(now))
                    c.require(caps.get(key) is c.Fact.CONFIRMED, 'health', 'dependent capability lost', c.ErrorCode.UNAVAILABLE)
                    active = self._active.get(key)
                    if active:
                        cmd, began = active
                        timeout = getattr(cmd, 'timeout_s', None)
                        if timeout is not None and now - began >= timeout:
                            self._latest[cmd.envelope.action_id] = self._event(cmd, c.Outcome.TIMED_OUT,
                                c.Reason(c.ErrorCode.TIMEOUT, 'action', 'completion not confirmed within stage timeout'),
                                c.Fact.CONFIRMED, self._latest[cmd.envelope.action_id].result.submitted,
                                now=now, elapsed=now - began)
                            self._halt((key,))
                            continue
                        r.validate_envelope(cmd.envelope, self._context(now))
                        self._prerequisites(cmd, now)
                except c.ContractError:
                    self.revoke(permit.source)

    def stop(self, request):
        with self._lock:
            try:
                now = self._now()
            except c.ContractError:
                self._closed = True
                self._halt(('base', 'arm', 'pump'))
                raise
            r.validate_envelope(request.envelope, self._context(now))
            c.require(request.envelope.producer is c.Owner.P10A, 'stop.producer', 'only P10A issues stop requests', c.ErrorCode.NOT_AUTHORIZED)
            identity = request.envelope.event_id
            payload = c.dumps(request)
            if identity in self._stops:
                old_payload, report = self._stops[identity]
                c.require(payload == old_payload, 'stop.event_id', 'stop id reused', c.ErrorCode.ID_CONFLICT)
                if report.submitted is c.Fact.UNKNOWN:
                    self._halt(tuple(self._pending_stops))
                    if not self._pending_stops:
                        report = replace(report, envelope=self._envelope(request.envelope, now), submitted=c.Fact.CONFIRMED)
                        self._stops[identity] = (payload, report)
                        self._last_stop = report
                        self._event(request, c.Outcome.UNKNOWN, c.Reason(c.ErrorCode.EVIDENCE_MISSING, 'stop', 'retry submitted; physical stop still unknown'),
                                    c.Fact.CONFIRMED, c.Fact.CONFIRMED, now=now)
                return report
            submitted = self.revoke()
            self._last_stop = m.StopReport(self._envelope(request.envelope, now), c.Fact.CONFIRMED,
                                           c.Fact.CONFIRMED if submitted else c.Fact.UNKNOWN,
                                           c.Fact.UNKNOWN, c.Fact.UNKNOWN, c.Fact.UNKNOWN, ())
            self._stops[identity] = (payload, self._last_stop)
            self._event(request, c.Outcome.UNKNOWN, c.Reason(c.ErrorCode.EVIDENCE_MISSING, 'stop', 'stop requested; physical stop and old queue require independent evidence'),
                        c.Fact.CONFIRMED, self._last_stop.submitted, now=now)
            return self._last_stop

    def record_feedback(self, action_id, evidence):
        # Potentially long domain verification must not hold the output lock.
        # Validate the active action again afterwards to discard cancellation races.
        with self._lock:
            now = self._now(); self.tick()
            candidate = next((cmd for cmd, _ in self._active.values() if cmd.envelope.action_id == action_id), None)
            c.require(candidate is not None, 'action_id', 'no active action', c.ErrorCode.STALE_REQUEST)
        validated = False
        if evidence.kind is c.EvidenceKind.PHYSICAL and self._physical_validator is not None:
            r.require_physical_evidence(evidence, candidate.envelope, now, candidate.envelope.origin)
            validated = self._physical_validator(candidate, evidence) is True
        with self._lock:
            now = self._now(); self.tick()
            active = next(((key, cmd, began) for key, (cmd, began) in self._active.items()
                           if cmd.envelope.action_id == action_id), None)
            c.require(active is not None, 'action_id', 'no active action; late feedback rejected', c.ErrorCode.STALE_REQUEST)
            key, cmd, began = active
            c.require(evidence.run_id == cmd.envelope.run_id and evidence.action_id == action_id and
                      evidence.target_id == cmd.envelope.target_id and evidence.origin is cmd.envelope.origin,
                      'evidence', 'feedback identity/provenance mismatch', c.ErrorCode.STALE_REQUEST)
            c.require(0 <= now - evidence.received_at and evidence.valid_until - now > 0,
                      'evidence', 'feedback expired or future', c.ErrorCode.EXPIRED)
            c.require(evidence.received_at - began >= 0, 'evidence.received_at',
                      'feedback predates command submission', c.ErrorCode.STALE_REQUEST)
            last = self._latest[action_id]
            if evidence in last.result.evidence:
                return last
            physical, echoed = c.Fact.UNKNOWN, last.result.echoed
            if evidence.kind is c.EvidenceKind.ECHO:
                echoed = c.Fact.CONFIRMED
            elif evidence.kind is c.EvidenceKind.PHYSICAL:
                r.require_physical_evidence(evidence, cmd.envelope, now, cmd.envelope.origin)
                if validated:
                    physical = c.Fact.CONFIRMED
            success = physical is c.Fact.CONFIRMED
            result = self._event(cmd, c.Outcome.SUCCESS if success else c.Outcome.UNKNOWN,
                                 None if success else c.Reason(c.ErrorCode.EVIDENCE_MISSING, 'physical', 'echo/reference alone cannot prove physical completion'),
                                 c.Fact.CONFIRMED, last.result.submitted, echoed, physical,
                                 last.result.evidence + (evidence,), now, now - began)
            self._latest[action_id] = result
            if success and key != 'base':
                del self._active[key]
            return result


class SafetyWorker:
    """Explicitly started independent monitor. The period is supplied, not measured here."""
    def __init__(self, gateway, period_s):
        positive(period_s, 'period_s')
        self.gateway, self.period_s = gateway, period_s
        self._stop, self._thread = threading.Event(), None

    def start(self):
        if self._thread is not None:
            raise RuntimeError('safety worker already started')
        def run():
            while not self._stop.is_set():
                self.gateway.tick()
                self._stop.wait(self.period_s)
        self._thread = threading.Thread(target=run, name='p01-safety', daemon=True)
        self._thread.start()
        return self

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1., self.period_s * 2))
        self.gateway.revoke()

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()
