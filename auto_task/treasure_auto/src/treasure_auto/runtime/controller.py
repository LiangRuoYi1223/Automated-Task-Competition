"""C0 runtime decisions, with no device or persistence implementation.

Business entry points have a single caller. The independent authority can be
revoked concurrently; external calls never run under its lock. Returned permits
are proposals until the final output consumer checks authorize_output again.
"""
from dataclasses import replace
from uuid import uuid4
from treasure_auto.contracts import core as c, models as m, rules as r
from .authority import PermitAuthority
from .guards import context, health_check, physical, permission_check, expiration_bounds, collect_evidence


def reason(error):
    return c.Reason(error.code, error.field, error.detail)


class RuntimeController:
    # P10A selects the C0 strategy explicitly for P10B assembly; no second clock.
    remaining = staticmethod(r.remaining_seconds)

    def __init__(self, config, clock, versions, instance_id, control_epoch, id_factory=None):
        c.require(type(instance_id) is str and bool(instance_id.strip()), 'instance_id', 'identity required')
        c.require(type(control_epoch) is int and control_epoch >= 1, 'control_epoch', 'positive epoch required')
        self.config, self.clock, self.versions = config, clock, versions
        self.instance_id, self.control_epoch = instance_id, control_epoch
        self._id = id_factory or (lambda: str(uuid4()))
        self.authority = PermitAuthority(config, lambda: self.clock.now())
        self._run = None
        self.admission = None
        self._state = m.RunState.PREPARATION
        self._initialized = False
        self._initializing = False
        self._initialization_attempted = False
        self._prestart_stopped = False
        self._events, self._sequences, self._intents = {}, {}, {}
        self._domain_stops, self._domain_stop_sequences = {}, {}
        self._sequence = 0
        self._holding = c.HoldingSnapshot(c.Holding.UNKNOWN, None, None, ())
        self._phase = None
        self.stop_report = None
        self._last_stop = None
        self._last_stop_ticket = None
        self._stop_requests = []
        self._stop_reports = []
        self._facts_after = None
        self._capture_floors = {}
        self._health = None
        self._scene_epoch, self._targets = 0, ()
        self._manual_count = 0
        self._first_lift = False
        self._history_known = True
        self._debug_pending = False
        self._recovering = False
        self._snapshot = None
        self._storage_fault = None
        self._manual_proposal = None
        self._debug_started_at = None

    @property
    def state(self):
        if self._run is not None and self.authority.reason is not None and self._state not in (m.RunState.RECOVERING, m.RunState.ENDED):
            return m.RunState.STOPPING
        return self._state

    @property
    def run(self):
        return None if self._run is None else replace(self._run, state=self.state)

    @property
    def remaining_s(self):
        return None if self._run is None else self.remaining(self._run, self.clock.now())

    @property
    def manual_debug_count(self):
        return self._manual_count if self._history_known else None

    @property
    def cached_health(self):
        return self._health

    @property
    def stop_requests(self):
        return tuple(self._stop_requests)

    @property
    def stop_reports(self):
        return tuple(self._stop_reports)

    def _invalidate_facts(self, cause):
        # Publish barriers before the authority linearization point. A racing
        # resume either sees this barrier or has its old ticket/permit revoked.
        self._facts_after = self.clock.now()
        self.authority.revoke(cause)

    def _remember_captures(self, evidence):
        for item in evidence:
            stamp = item.captured_at
            if stamp is not None:
                key = (stamp.basis,stamp.clock_id,stamp.boot_id)
                self._capture_floors[key] = max(stamp.seconds,self._capture_floors.get(key,-1))

    def _after_invalidation(self, evidence):
        if self._facts_after is None:
            return
        barrier = self._facts_after
        for item in evidence:
            c.require(item.received_at-barrier > 0, 'revalidation', 'new reception after invalidation required', c.ErrorCode.EVIDENCE_MISSING)
            stamp = item.captured_at
            if item.kind is not c.EvidenceKind.PHYSICAL:
                continue
            key = (stamp.basis,stamp.clock_id,stamp.boot_id)
            own = (barrier.basis,barrier.clock_id,barrier.boot_id)
            if key == own:
                c.require(stamp-barrier > 0, 'revalidation.capture', 'late old capture rejected', c.ErrorCode.EVIDENCE_MISSING)
            else:
                c.require(key in self._capture_floors and stamp.seconds > self._capture_floors[key],
                          'revalidation.capture', 'new capture in known sensor clock required; unknown clock needs adapter proof', c.ErrorCode.EVIDENCE_MISSING)

    def _envelope(self, base, now=None, **changes):
        now = now or self.clock.now()
        self._sequence += 1
        return replace(base, producer=c.Owner.P10A, event_id=self._id(), sequence=self._sequence,
                       emitted_at=now, expires_at=replace(now, seconds=now.seconds+self.config.heartbeat_timeout_s), **changes)

    def arm(self):
        c.require(self._run is None and self.admission is None, 'arm', 'explicit new-run admission is single use', c.ErrorCode.NOT_AUTHORIZED)
        now = self.clock.now()
        c.monotonic(now, 'now')
        self.admission = c.Envelope(self._id(), self.instance_id, self.control_epoch,
                                    c.Owner.P10A, self._id(), 0, None, None, None, now,
                                    replace(now, seconds=now.seconds+self.config.heartbeat_timeout_s),
                                    self.versions, c.Origin.SIMULATED, 0, 0, 0)
        self._state = m.RunState.OFFLINE if self.config.mode == 'offline' else m.RunState.ARMED
        return self.admission

    def _decision(self, denial=None, permit=None):
        c.require(self._run is not None, 'run', 'C0 1.0.0 has no pre-start RuntimeDecision', c.ErrorCode.NOT_AUTHORIZED)
        stopped = self.authority.reason
        stop = None
        if stopped is not None:
            ticket = self.authority.ticket
            phase = self._phase.value if self._phase is not None else 'unknown'
            detail = '{}; phase={}; pump=preserve (no generic off command)'.format(stopped.detail, phase)
            stop_reason = replace(stopped, detail=detail)
            if (self._last_stop is None or self._last_stop_ticket != ticket or self._last_stop.reason != stop_reason
                    or self._last_stop.holding != self._holding
                    or self._last_stop.envelope.expires_at - self.clock.now() <= 0):
                stop_id = self._id()
                self._last_stop = m.StopRequest(self._envelope(self._run.envelope,request_id=stop_id,action_id=stop_id,target_id=None), stop_reason, self._holding)
                self._last_stop_ticket = ticket
                self._stop_requests.append(self._last_stop)
            stop = self._last_stop
        return m.RuntimeDecision(self.run, permit, stop, denial or stopped)

    def _check_event(self, event, now):
        base = self._run.envelope if self._run is not None else self.admission
        c.require(base is not None, 'state', 'not armed', c.ErrorCode.NOT_AUTHORIZED)
        key = (event.envelope.run_id, event.envelope.producer, event.envelope.event_id)
        payload = c.dumps(event)
        if key in self._events:
            c.require(self._events[key] == payload, 'event_id', 'duplicate with different payload', c.ErrorCode.ID_CONFLICT)
            return False
        r.validate_envelope(event.envelope, context(base, now))
        stream = (event.envelope.producer, event.envelope.request_id, event.envelope.action_id)
        c.require(event.envelope.sequence > self._sequences.get(stream, -1), 'sequence', 'old signal', c.ErrorCode.STALE_REQUEST)
        # Remember even rejected signals: they must not become authorized on replay.
        self._events[key] = payload
        self._sequences[stream] = event.envelope.sequence
        return True

    def request_domain_stop(self, request):
        """Revoke first, then mint a genuine P10A stop for the output gateway.

        A fresh stop from this controller generation remains valid when its
        geometry has expired. It cannot authorize motion or adopt held facts.
        No external I/O is performed under the authority lock.
        """
        c.require(type(request) is m.StopRequest and self._run is not None,
                  'stop', 'domain StopRequest and current run required', c.ErrorCode.NOT_AUTHORIZED)
        c.require(request.envelope.producer in (c.Owner.P03,c.Owner.P07,c.Owner.P08,c.Owner.P09),
                  'stop.producer', 'motion domain required', c.ErrorCode.NOT_AUTHORIZED)
        now=self.clock.now()
        r.validate_envelope(request.envelope, context(self._run.envelope,now,request.envelope.versions))
        key=(request.envelope.producer,request.envelope.event_id)
        payload=c.dumps(request)
        if key in self._domain_stops:
            c.require(self._domain_stops[key]==payload,'stop.event_id','changed duplicate stop',c.ErrorCode.ID_CONFLICT)
        else:
            stream=(request.envelope.producer,request.envelope.request_id,request.envelope.action_id)
            c.require(request.envelope.sequence>self._domain_stop_sequences.get(stream,-1),
                      'stop.sequence','old domain stop',c.ErrorCode.STALE_REQUEST)
            self._domain_stops[key]=payload
            self._domain_stop_sequences[stream]=request.envelope.sequence
            self._invalidate_facts(request.reason)
            if self._state is not m.RunState.ENDED:
                self._state=m.RunState.STOPPING
        return self._decision()

    def signal(self, signal):
        now = self.clock.now()
        if self._run is not None:
            self.authority.watch(now)
        try:
            if not self._check_event(signal, now):
                return self._decision()
            c.require(signal.envelope.producer is c.Owner.P10A, 'signal.producer', 'configured signal adapter must normalize to P10A')
            if self._run is None and signal.kind is m.SignalKind.EMERGENCY_STOP:
                self._prestart_stopped = True
                c.require(False, 'stop_lock', 'pre-start emergency latched; no run created', c.ErrorCode.NOT_AUTHORIZED)
            if signal.kind is m.SignalKind.START:
                c.require(signal.referee_authorized is c.Fact.CONFIRMED, 'referee', 'legal start required', c.ErrorCode.NOT_AUTHORIZED)
                if self._run is None:
                    c.require(not self._prestart_stopped, 'stop_lock', 'pre-start emergency remains latched', c.ErrorCode.NOT_AUTHORIZED)
                    self._run = m.RunContext(self._envelope(self.admission, now), m.RunState.RECOVERING,
                                             signal.envelope.event_id, now, replace(now,seconds=now.seconds+600), False)
                    self._start_signal = signal
                    self._state = m.RunState.RECOVERING
                    self.authority.bind(self._run)
                return self._decision()
            c.require(self._run is not None, 'run', 'signal needs current run', c.ErrorCode.NOT_AUTHORIZED)
            if signal.kind is m.SignalKind.MANUAL_DEBUG:
                c.require(signal.referee_authorized is c.Fact.CONFIRMED, 'referee', 'referee permission required', c.ErrorCode.NOT_AUTHORIZED)
                c.require(self.remaining_s > 0, 'deadline', 'debug does not stop clock', c.ErrorCode.DEADLINE)
                c.require(self._history_known and not self._first_lift, 'first_lift', 'lift/history prohibits manual debug', c.ErrorCode.NOT_AUTHORIZED)
                c.require(self._manual_count < 3, 'manual_debug_count', 'maximum three', c.ErrorCode.LIMIT_EXCEEDED)
                self._manual_count += 1
                self._debug_pending = True
                self._manual_proposal = None
                self._debug_started_at = now
                self._intents.clear()
                self.authority.revoke(c.Reason(c.ErrorCode.NOT_AUTHORIZED, 'manual_debug',
                                             'staff return to A; new epoch, relocalization and mapping reread required'))
            elif signal.kind is m.SignalKind.DEADLINE:
                c.require(self.remaining_s == 0, 'deadline', 'early deadline signal rejected')
                self.authority.watch(now)
            elif signal.kind in (m.SignalKind.EMERGENCY_STOP, m.SignalKind.FINISH):
                self.authority.revoke(c.Reason(c.ErrorCode.CANCELLED, signal.kind.value,
                                             'stop requested; physical completion remains unknown'), True)
            if self._state is not m.RunState.ENDED:
                self._state = m.RunState.STOPPING
            return self._decision()
        except c.ContractError as error:
            if self._run is None:
                raise
            return self._decision(reason(error))

    def initialize(self, store, initial_snapshot, previous_end_marker=None):
        """Call on business worker; P10B I/O is deliberately outside authority lock."""
        c.require(self._run is not None and not self._recovering, 'run', 'start first')
        if self._initialized:
            return self._decision()
        if self._initialization_attempted:
            return self._decision(c.Reason(c.ErrorCode.STORAGE_FAILURE, 'initialize',
                                         'prior submission unresolved; explicit P10B reconciliation required'))
        c.require(not self._initializing, 'initialize', 'initialization already pending')
        c.require(initial_snapshot.run == self.run, 'snapshot.run', 'initialize exact original run')
        ticket = self.authority.ticket
        self._initializing = True
        self._initialization_attempted = True
        try:
            if self._run.envelope.versions.mapping is not None:
                mapping = initial_snapshot.mapping
                c.require(mapping is not None and mapping.mapping_version == self._run.envelope.versions.mapping,
                          'mapping', 'version alone is not a frozen classification map', c.ErrorCode.INVALID_MAPPING)
                r.validate_envelope(mapping.envelope,context(self._run.envelope,self.clock.now()))
                c.require(mapping.envelope.producer is c.Owner.P05
                          and mapping.frozen_at-self._run.started_at >= 0,
                          'mapping', 'P05 current-run freeze required')
                physical(mapping.evidence,mapping.envelope,self.clock.now())
            # P10B persists the proposed active state. Actual output remains
            # inhibited until the commit/readback and authority ticket succeed.
            proposed_run = replace(self._run,state=m.RunState.OFFLINE if self.config.mode == 'offline' else m.RunState.RUNNING)
            proposed_snapshot = replace(initial_snapshot,run=proposed_run)
            request = m.InitializeRunRequest(proposed_run, proposed_snapshot, self._start_signal, previous_end_marker)
            receipt = store.initialize_run(request)
            c.require(receipt.status is m.CommitStatus.COMMITTED, 'initialize', 'commit rejected or unknown; read before retry', c.ErrorCode.STORAGE_FAILURE)
            read = store.read(self._run.envelope.run_id)
            c.require(read.status is c.Outcome.SUCCESS and read.snapshot is not None
                      and read.snapshot == proposed_snapshot and receipt.revision == read.snapshot.revision
                      and receipt.space_version == read.snapshot.space.version,
                      'initialize.read', 'readback must match committed original snapshot', c.ErrorCode.STORAGE_FAILURE)
            self.authority.release(ticket, self.clock.now())
            self._initialized = True
            self._snapshot = read.snapshot
            self._state = m.RunState.OFFLINE if self.config.mode == 'offline' else m.RunState.RUNNING
            return self._decision()
        except c.ContractError as error:
            self.authority.revoke(reason(error))
            self._state = m.RunState.STOPPING
            return self._decision(reason(error))
        except Exception as error:
            failure = c.Reason(c.ErrorCode.STORAGE_FAILURE, 'initialize', type(error).__name__)
            self.authority.revoke(failure)
            self._state = m.RunState.STOPPING
            return self._decision(failure)
        finally:
            self._initializing = False

    def heartbeat(self, envelope):
        now = self.clock.now()
        r.validate_envelope(envelope, context(self._run.envelope, now))
        c.require(envelope.producer is c.Owner.P09, 'heartbeat.producer', 'mission heartbeat only')
        key = ('heartbeat', envelope.instance_id)
        previous = self._sequences.get(key, -1)
        c.require(envelope.sequence > previous, 'heartbeat.sequence', 'duplicate or old heartbeat', c.ErrorCode.STALE_REQUEST)
        self._sequences[key] = envelope.sequence
        self.authority.heartbeat(now)

    def remember_intent(self, intent, receipt, source):
        """Bind C0 intent plus local source policy; never writes or dispatches it."""
        r.validate_envelope(intent.envelope, context(self._run.envelope, self.clock.now()))
        r.validate_intent_receipt(intent, receipt)
        r.require_provenance(intent, self._run.envelope.origin)
        key = intent.envelope.action_id
        value = (intent, receipt, source)
        c.require(key not in self._intents or self._intents[key] == value, 'action_id', 'saved intent cannot change', c.ErrorCode.ID_CONFLICT)
        self._intents[key] = value

    def _permission(self, request, resume):
        now = self.clock.now()
        self.authority.watch(now)
        ticket = self.authority.ticket
        try:
            c.require(self._initialized, 'persistence', 'original run/recovery not committed', c.ErrorCode.STORAGE_FAILURE)
            c.require(self._storage_fault is None, 'persistence', 'storage fault latched; durable recovery required', c.ErrorCode.STORAGE_FAILURE)
            c.require(not self._debug_pending, 'manual_debug',
                      'C0 lacks durable debug epoch/reread protocol; remain stopped', c.ErrorCode.NOT_AUTHORIZED)
            c.require(self._state is not m.RunState.ENDED, 'state', 'ended', c.ErrorCode.NOT_AUTHORIZED)
            if not resume:
                c.require(self.authority.reason is None, 'stop_lock', 'explicit revalidation required', c.ErrorCode.NOT_AUTHORIZED)
            saved = self._intents.get(request.envelope.action_id)
            c.require(saved is not None and saved[2] is request.source, 'intent', 'source/intent not registered', c.ErrorCode.STORAGE_FAILURE)
            evidence = permission_check(request, self.run, now, self.config, saved[0], saved[1], self._scene_epoch, self._targets)
            self._after_invalidation(evidence)
            self._health = request.health
            if resume:
                self.authority.release(ticket, now)
            expiry = min([now.seconds+self.config.lease_s, self._run.deadline.seconds] + list(expiration_bounds(request)))
            env = self._envelope(request.envelope, now)
            env = replace(env, expires_at=replace(now, seconds=expiry))
            permit = m.ControlPermit(env, self._id(), request.source, request.purpose, evidence)
            self.authority.install(permit, ticket, self.clock.now())
            self._holding = request.readiness.holding
            self._remember_captures(evidence)
            self._health = request.health
            self._state = m.RunState.OFFLINE if self.config.mode == 'offline' else m.RunState.RUNNING
            return self._decision(permit=permit)
        except c.ContractError as error:
            factual_fields = {'holding','posture','old_stop','localization','continuous','evidence'} | set(self.config.required_capabilities)
            if error.field in factual_fields or error.field.startswith('evidence.'):
                self._remember_captures(collect_evidence(request))
                self._invalidate_facts(reason(error))
            return self._decision(reason(error))

    def request_permission(self, request):
        return self._permission(request, False)

    def resume(self, request):
        """Explicit same-run revalidation, never caused by a healthy heartbeat alone."""
        return self._permission(request, True)

    def authorize_output(self, permit, envelope, source, real_output=True):
        return self.authority.allows(permit, envelope, source, self.clock.now(), real_output)

    def notify_storage_fault(self, failure):
        """Consume an already sampled P10B fault; never polls a DB or waits on I/O."""
        c.require(type(failure) is c.Reason and failure.code is c.ErrorCode.STORAGE_FAILURE,
                  'storage_fault','P10B storage failure reason required')
        self._storage_fault = failure
        self.authority.revoke(failure)
        return self._decision()

    def supervise(self, now, health):
        # Port samples can arrive out of order across threads. Re-read the local
        # clock; PermitAuthority also samples under its own short lock.
        now = self.clock.now()
        self.authority.watch(now)
        if self._run is None:
            return None  # No fictitious RunContext before a legal start.
        if health is not None:
            try:
                self._health = health
                health_check(health, self.run, now, self.config)
                self._health = health
            except c.ContractError as error:
                self._invalidate_facts(reason(error))
        if self.authority.reason is not None and self._state is not m.RunState.ENDED:
            self._state = m.RunState.STOPPING
        return self._decision()

    def observe_phase(self, phase, readiness):
        r.validate_envelope(readiness.envelope, context(self._run.envelope, self.clock.now()))
        c.require(readiness.envelope.producer is c.Owner.P07, 'holding.producer', 'P07 only')
        # UNKNOWN is retained; physical proof is only required for positive claims.
        if readiness.holding.state in (c.Holding.HELD, c.Holding.EMPTY):
            physical(readiness.holding.evidence, readiness.envelope, self.clock.now(), readiness.holding.target_id)
        self._remember_captures(collect_evidence(readiness))
        changed = (self._holding.state, self._holding.target_id) != (readiness.holding.state, readiness.holding.target_id)
        self._holding, self._phase = readiness.holding, phase
        if changed:
            self._invalidate_facts(c.Reason(c.ErrorCode.NOT_AUTHORIZED, 'holding', 'held object/state changed; revalidate drive posture'))
        if readiness.holding.state in (c.Holding.UNKNOWN, c.Holding.DROPPED) or readiness.posture_confirmed is not c.Fact.CONFIRMED:
            self._invalidate_facts(c.Reason(c.ErrorCode.NOT_AUTHORIZED, 'readiness', 'holding/posture invalidated'))

    def record_stop_report(self, report):
        r.validate_envelope(report.envelope, context(self._run.envelope, self.clock.now()))
        c.require(report.envelope.producer in (c.Owner.P01, c.Owner.P03), 'stop.producer', 'device/navigation report required')
        if report.physically_stopped is c.Fact.CONFIRMED:
            physical(report.evidence, report.envelope, self.clock.now())
        self.stop_report = report
        if not self._stop_reports or self._stop_reports[-1] != report:
            self._stop_reports.append(report)

    def scene_changed(self, space, scene_epoch, affected_targets):
        c.require(type(scene_epoch) is int and scene_epoch > self._scene_epoch, 'scene_epoch', 'must increase')
        r.validate_envelope(space.envelope, context(self._run.envelope, self.clock.now(), space.envelope.versions))
        c.require(space.envelope.producer is c.Owner.P08 and bool(affected_targets), 'space', 'P08 affected objects required')
        for name in ('config','map','calibration','model','mapping','layout'):
            c.require(getattr(space.envelope.versions,name) == getattr(self._run.envelope.versions,name),
                      'versions.'+name, 'scene event cannot change frozen configuration', c.ErrorCode.VERSION_MISMATCH)
        self.authority.revoke(c.Reason(c.ErrorCode.STALE_GEOMETRY, 'scene', 'new brick movement invalidates all motion'))
        self._scene_epoch, self._targets = scene_epoch, tuple(affected_targets)
        # Notification only. Adoption requires P10B's committed snapshot below.

    def observe_localization(self, localization):
        r.validate_envelope(localization.envelope, context(self._run.envelope, self.clock.now()))
        c.require(localization.envelope.producer is c.Owner.P02, 'localization.producer', 'P02 only')
        current = self._run.envelope.base_motion_epoch
        c.require(localization.base_motion_epoch >= current
                  and localization.envelope.base_motion_epoch == localization.base_motion_epoch,
                  'base_motion_epoch', 'old or inconsistent movement epoch', c.ErrorCode.STALE_GEOMETRY)
        if localization.trusted is not c.Fact.CONFIRMED or localization.base_motion_epoch > current:
            self._remember_captures(localization.evidence)
            self._invalidate_facts(c.Reason(c.ErrorCode.STALE_GEOMETRY, 'localization', 'movement or lost localization invalidates old paths'))
            self._run = replace(self._run, envelope=replace(self._run.envelope, base_motion_epoch=localization.base_motion_epoch))

    def propose_space_update(self, previous, space):
        """P10A-owned run proposal for P09's atomic spatial transaction.

        No publication or permission is adopted before the P10B commit/readback.
        Only the spatial version may change through this narrow release path.
        """
        c.require(self._initialized and self._run is not None, 'run_update', 'initialize/register first')
        current=self._run
        names=('run_id','instance_id','control_epoch','origin','versions')
        c.require(all(getattr(previous.envelope,n)==getattr(current.envelope,n) for n in names) and
                  (previous.started_at,previous.deadline,previous.start_signal_id,previous.ever_restarted)==
                  (current.started_at,current.deadline,current.start_signal_id,current.ever_restarted),
                  'run_update','stored run is not the current original run',c.ErrorCode.STALE_RUN)
        now=self.clock.now()
        c.require(now-current.started_at>=0,'clock','clock predates original start',c.ErrorCode.CLOCK_MISMATCH)
        versions=space.envelope.versions
        r.validate_envelope(space.envelope,context(current.envelope,now,versions))
        c.require(space.envelope.producer is c.Owner.P08 and space.version==versions.space and
                  space.version==current.envelope.versions.space+1,
                  'space.version','exact next P08 spatial version required',c.ErrorCode.VERSION_MISMATCH)
        for name in ('config','map','calibration','model','mapping','layout'):
            c.require(getattr(versions,name)==getattr(current.envelope.versions,name),
                      'versions.'+name,'release cannot change frozen configuration',c.ErrorCode.VERSION_MISMATCH)
        return replace(previous,envelope=self._envelope(current.envelope,now,versions=versions))

    def accept_snapshot(self, snapshot, receipt):
        """Adopt an immutable P10B publication; never edits occupancy or inventories."""
        now = self.clock.now()
        c.require(self._initialized, 'snapshot', 'initialize/register first')
        c.require(receipt.status is m.CommitStatus.COMMITTED and receipt.revision == snapshot.revision
                  and receipt.space_version == snapshot.space.version,
                  'snapshot.commit', 'matching durable publication required', c.ErrorCode.STORAGE_FAILURE)
        c.require(snapshot.run.started_at == self._run.started_at and snapshot.run.deadline == self._run.deadline
                  and snapshot.run.start_signal_id == self._run.start_signal_id
                  and snapshot.run.ever_restarted == self._run.ever_restarted, 'run', 'original run fields changed')
        r.validate_envelope(snapshot.run.envelope, context(self._run.envelope, now, snapshot.run.envelope.versions))
        r.require_provenance(snapshot, self._run.envelope.origin)
        previous = self._run.envelope.versions
        versions = snapshot.run.envelope.versions
        for name in ('config','map','calibration','model','layout'):
            c.require(getattr(versions,name) == getattr(previous,name), 'versions.'+name, 'frozen configuration', c.ErrorCode.VERSION_MISMATCH)
        c.require(snapshot.space.envelope.producer is c.Owner.P08 and snapshot.space.envelope.versions == versions
                  and snapshot.space.version == versions.space, 'space', 'one committed spatial version required')
        r.validate_envelope(snapshot.space.envelope, context(snapshot.run.envelope,now))
        if self._snapshot is not None and snapshot == self._snapshot:
            return self._decision()
        c.require(self._snapshot is None or snapshot.revision > self._snapshot.revision,
                  'revision', 'old or conflicting publication', c.ErrorCode.CONFLICT)
        c.require(versions.space >= previous.space, 'space.version', 'cannot roll back space', c.ErrorCode.VERSION_MISMATCH)
        if snapshot.run.state in (m.RunState.ENDED,m.RunState.STOPPING):
            self.authority.revoke(c.Reason(c.ErrorCode.NOT_AUTHORIZED,'snapshot.state','authoritative terminal/stopped publication'),True)
            self._state = m.RunState.STOPPING
            return self._decision()
        if previous.mapping is not None:
            c.require(versions.mapping == previous.mapping and
                      (self._snapshot is None or self._snapshot.mapping is None or snapshot.mapping == self._snapshot.mapping),
                      'mapping', 'mapping is frozen', c.ErrorCode.INVALID_MAPPING)
        if versions.mapping != previous.mapping:
            mapping = snapshot.mapping
            c.require(mapping is not None and mapping.mapping_version == versions.mapping
                      and mapping.envelope.producer is c.Owner.P05, 'mapping', 'complete P05 mapping required')
            r.validate_envelope(mapping.envelope, context(snapshot.run.envelope,now))
            physical(mapping.evidence,mapping.envelope,now)
        self.authority.revoke(c.Reason(c.ErrorCode.VERSION_MISMATCH,'snapshot','committed state changed; revalidate'))
        self._snapshot = snapshot
        self._run = replace(self._run, envelope=replace(self._run.envelope,versions=versions))

    def manual_debug_proposal(self):
        """Offline handoff proposal. C0 needs a distinct durable debug registration cause."""
        c.require(self._debug_pending, 'manual_debug', 'no accepted referee debug')
        if self._manual_proposal is None:
            env = self._envelope(self._run.envelope, instance_id=self._id(), control_epoch=self.control_epoch+1)
            env = replace(env, versions=replace(env.versions,mapping=None))
            self._manual_proposal = replace(self._run,envelope=env,state=m.RunState.RECOVERING)
        return self._manual_proposal

    def accept_manual_registration(self, result, localization, returned_to_a):
        """Offline pure review; production registration/return-to-A evidence adapter is pending.

        Does not grant a permit. A full resume request is still mandatory; mapping
        stays absent until P05 rereads and P10B publishes it through accept_snapshot.
        """
        ticket = self.authority.ticket
        now = self.clock.now()
        try:
            c.require(self._debug_pending and self._manual_proposal is not None, 'manual_debug', 'proposal required')
            c.require(returned_to_a is c.Fact.CONFIRMED, 'return_to_A', 'staff return and P02 verification required')
            c.require(result.status is c.Outcome.SUCCESS and result.snapshot is not None
                      and result.snapshot.run == self._manual_proposal and result.snapshot.mapping is None,
                      'debug.registration', 'exact debug proposal must be committed', c.ErrorCode.STORAGE_FAILURE)
            snap = result.snapshot
            c.require(result.registration is not None and result.registration.status is m.CommitStatus.COMMITTED
                      and result.registration.revision == snap.revision and result.registration.space_version == snap.space.version,
                      'debug.registration', 'durable new instance registration required', c.ErrorCode.STORAGE_FAILURE)
            r.require_provenance(snap,self._run.envelope.origin)
            r.validate_envelope(localization.envelope,context(snap.run.envelope,now))
            c.require(localization.envelope.producer is c.Owner.P02 and localization.trusted is c.Fact.CONFIRMED
                      and localization.envelope.emitted_at - self._debug_started_at >= 0,
                      'debug.localization', 'new localization after staff relocation required')
            physical(localization.evidence,localization.envelope,now)
            c.require(all(e.received_at-self._debug_started_at >= 0 for e in localization.evidence),
                      'debug.localization', 'old localization cannot establish return to A')
            self.authority.rotate(snap.run,ticket,now)
            self._run, self._snapshot = snap.run, snap
            self.instance_id,self.control_epoch = snap.run.envelope.instance_id,snap.run.envelope.control_epoch
            self._intents.clear()
            self._events.clear()
            self._sequences.clear()
            self._debug_pending = False
            self._initialized = True
            self._state = m.RunState.RECOVERING
            return self._decision()
        except c.ContractError as error:
            return self._decision(reason(error))

    def confirm_first_lift(self, evidence):
        physical((evidence,), self._run.envelope, self.clock.now())
        self._first_lift = True

    def begin_recovery(self, original):
        c.require(self._run is None, 'recover', 'new controller required')
        c.require(original.envelope.instance_id != self.instance_id
                  and self.control_epoch > original.envelope.control_epoch,
                  'control_epoch', 'new instance and increasing epoch required', c.ErrorCode.STALE_CONTROL)
        # This is local isolation only. P01 exclusive process fencing is a live integration gate.
        now = self.clock.now()
        env = self._envelope(original.envelope, now, instance_id=self.instance_id, control_epoch=self.control_epoch)
        self._run = replace(original, envelope=env, state=m.RunState.RECOVERING, ever_restarted=True)
        self._state, self._recovering, self._history_known = m.RunState.RECOVERING, True, False
        self.authority.bind(self._run)
        self.authority.revoke(c.Reason(c.ErrorCode.NOT_AUTHORIZED, 'recovery', 'old queues isolated locally; durable registration and fresh scene required'))
        if original.state in (m.RunState.ENDED, m.RunState.STOPPING):
            self.authority.revoke(c.Reason(c.ErrorCode.NOT_AUTHORIZED, 'original.state',
                                         'ended/stopping state lacks durable nonterminal resume authority'), True)
        self.authority.watch(now)
        return self._decision()

    def recover(self, result):
        self.authority.watch(self.clock.now())
        self._initialized = False
        try:
            c.require(self._recovering and result.status is c.Outcome.SUCCESS and result.snapshot is not None,
                      'recovery', 'validated original snapshot required', c.ErrorCode.STORAGE_FAILURE)
            run = result.snapshot.run
            c.require(result.registration is not None and result.registration.status is m.CommitStatus.COMMITTED
                      and result.registration.revision == result.snapshot.revision
                      and result.registration.space_version == result.snapshot.space.version,
                      'registration', 'durable recovery registration required', c.ErrorCode.STORAGE_FAILURE)
            c.require((run.started_at, run.deadline, run.start_signal_id) ==
                      (self._run.started_at, self._run.deadline, self._run.start_signal_id),
                      'deadline', 'original time must be preserved', c.ErrorCode.CLOCK_MISMATCH)
            c.require((run.envelope.run_id, run.envelope.instance_id, run.envelope.control_epoch) ==
                      (self._run.envelope.run_id, self.instance_id, self.control_epoch)
                      and run.envelope.versions == self._run.envelope.versions and run.ever_restarted,
                      'instance', 'wrong recovery instance/version/restart flag', c.ErrorCode.STALE_CONTROL)
            r.require_provenance(result.snapshot, self._run.envelope.origin)
            if run.state in (m.RunState.ENDED,m.RunState.STOPPING):
                self.authority.revoke(c.Reason(c.ErrorCode.NOT_AUTHORIZED,'recovery.state','authoritative stopped/ended run'),True)
                return self._decision()
            c.require(r.remaining_seconds(run, self.clock.now()) > 0, 'deadline', 'expired during recovery', c.ErrorCode.DEADLINE)
            self._initialized = True
            self._snapshot = result.snapshot
            return self._decision()  # Still locked until resume with current facts and old_stop.
        except c.ContractError as error:
            self.authority.revoke(reason(error))
            return self._decision(reason(error))

    def accept_end(self, marker, report):
        """Consume P10B seal/closed-writer facts. Never deletes or writes files."""
        try:
            c.require(self.authority.reason is not None, 'stop_lock', 'stop must precede end')
            self.record_stop_report(report)
            c.require(self._last_stop is not None and report.envelope.request_id == self._last_stop.envelope.request_id
                      and report.envelope.action_id == self._last_stop.envelope.action_id
                      and report.envelope.emitted_at-self._last_stop.envelope.emitted_at >= 0,
                      'stop.identity', 'report must bind the current stop request', c.ErrorCode.STALE_REQUEST)
            c.require(all(e.received_at-self._last_stop.envelope.emitted_at >= 0 for e in report.evidence),
                      'stop.evidence', 'pre-stop observations cannot confirm this stop', c.ErrorCode.EVIDENCE_MISSING)
            for item in report.evidence:
                c.require(item.captured_at is not None
                          and (item.captured_at.basis,item.captured_at.clock_id,item.captured_at.boot_id) ==
                          (self._last_stop.envelope.emitted_at.basis,self._last_stop.envelope.emitted_at.clock_id,self._last_stop.envelope.emitted_at.boot_id),
                          'stop.capture', 'stop completion needs capture in runtime clock; sensor conversion proof not supplied', c.ErrorCode.CLOCK_MISMATCH)
                c.require(item.captured_at-self._last_stop.envelope.emitted_at >= 0,
                          'stop.capture','capture predates this stop',c.ErrorCode.EVIDENCE_MISSING)
            c.require(report.physically_stopped is c.Fact.CONFIRMED
                      and report.old_queue_empty is c.Fact.CONFIRMED,
                      'stop', 'echo alone cannot complete stopping', c.ErrorCode.EVIDENCE_MISSING)
            c.require(marker.run_id == self._run.envelope.run_id
                      and marker.original_deadline == self._run.deadline
                      and marker.ever_restarted == self._run.ever_restarted,
                      'end_marker', 'wrong original run or restart flag', c.ErrorCode.STALE_RUN)
            c.require(self.clock.now() - marker.ended_at >= 0 and marker.ended_at - self._run.started_at >= 0,
                      'ended_at', 'wrong/future end clock', c.ErrorCode.CLOCK_MISMATCH)
            c.require(marker.stop_confirmed is c.Fact.CONFIRMED
                      and marker.writers_closed is c.Fact.CONFIRMED and marker.sealed is c.Fact.CONFIRMED,
                      'end_marker', 'reliable stop, seal and closed writers required', c.ErrorCode.STORAGE_FAILURE)
            self.authority.revoke(c.Reason(c.ErrorCode.CANCELLED, 'ended', 'sealed run cannot restart'), True)
            self._state = m.RunState.ENDED
            return self._decision()
        except c.ContractError as error:
            return self._decision(reason(error))
