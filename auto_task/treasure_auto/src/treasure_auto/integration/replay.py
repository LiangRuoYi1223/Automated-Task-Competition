"""Deterministic fault injection around P09, never a production event adapter.

P03-P08 observations come from P09's explicitly simulated world. Domain
decisions remain in their owners. Real source-boundary tests live in tests/.
No device transport, ROS object, network client, or production store is created.
"""
from dataclasses import replace

from treasure_auto.contracts import core as c, models as m, rules as r
from treasure_auto.mission.checkpoint import SimulationJournal
from treasure_auto.mission.engine import MissionEngine
from treasure_auto.mission.inputs import Stage
from treasure_auto.mission.simulator import SimulatedWorld
from .field_replay import FIELD_SCENARIOS, run_field_replay, validate_field_replay


SCENARIOS = (
    'nominal', 'duplicate', 'late_event', 'old_instance',
    'unknown_grasp', 'unknown_transport', 'unknown_release', 'dropped_frames',
    'echo_only', 'failed_grasp', 'timed_out_grasp', 'conflicting_duplicate',
    'checkpoint_unknown', 'crash_before_grasp', 'crash_after_grasp',
    'crash_before_release', 'crash_after_release', 'restart_410', 'deadline',
) + FIELD_SCENARIOS
COMPLETING = frozenset(('nominal', 'duplicate', 'late_event', 'old_instance'))
UNKNOWN = frozenset(('unknown_grasp', 'unknown_transport', 'unknown_release', 'echo_only'))


def _event_frame(frame, event):
    return replace(frame, inputs=replace(frame.inputs, action_events=(event,)))


def run_replay(layout='V1', scenario='nominal', config=None):
    """Return a JSON-safe trace of one fake match, advanced to its original end.

All durations, image/physical observations and checkpoint storage are simulated.
The final deadline probe submits only an already-seen callback, never an action.
"""
    c.require(scenario in SCENARIOS, 'scenario', 'unsupported P11 replay')
    c.require(layout in ('V1', 'V2', 'V3'), 'layout', 'expected V1/V2/V3')
    if scenario in FIELD_SCENARIOS:
        return run_field_replay(layout,scenario,config)
    world = SimulatedWorld(layout)
    if config is not None:
        # P09 owns config validation. A mismatch cannot silently replace frozen
        # versions in the world, which would conceal an incompatible delivery.
        c.require(config.version == world.versions.config, 'config.version',
                  'simulation world uses p09-sim-config', c.ErrorCode.VERSION_MISMATCH)
        world.engine = MissionEngine(config, world.clock, SimulationJournal())
    stops = []
    world.engine.stop_sink = stops.append
    issued, events, injections = [], [], []
    restart = None
    last_event = None

    def advance(frame):
        nonlocal last_event
        for event in frame.inputs.action_events:
            last_event = event
            events.append(dict(event_id=event.envelope.event_id, stage=event.result.phase,
                               status=event.result.status.value, physical=event.result.physical.value,
                               at_s=world.clock.seconds, action_id=event.envelope.action_id))
        result = world.engine.advance(frame)
        d = result.directive
        if d is not None:
            issued.append(dict(stage=d.stage.value, owner=d.owner.value,
                               target_id=d.envelope.target_id, request_id=d.envelope.request_id,
                               action_id=d.envelope.action_id, at_s=world.clock.seconds,
                               category=d.order.category.value if d.order else None))
        return result

    out = advance(world.frame())

    def record(kind, **details):
        injections.append(dict(kind=kind, at_s=world.clock.seconds, **details))

    def restart_pending(directive, physical_after=False, saved_remaining=None):
        payload = world.engine.journal.load()
        pending_before = world.engine.pending.envelope.action_id
        if physical_after:
            # Physical change occurs in the fake environment before its result
            # reaches the scheduler. Missing success is never read as no action.
            world.complete(directive)
        world.clock.advance(10.)
        world.engine = MissionEngine(world.engine.config, world.clock,
                                     SimulationJournal(payload), stop_sink=stops.append)
        resumed = advance(world.frame())
        details = dict(pending_before=pending_before,
                       pending_after=world.engine.pending.envelope.action_id,
                       reissued=resumed.directive is not None,
                       remaining_s=r.remaining_seconds(world.run, world.clock.now()),
                       storage='simulated_in_memory_checkpoint')
        if saved_remaining is not None:
            details['saved_remaining_s'] = saved_remaining
        return resumed, details

    for _ in range(200):
        d = out.directive
        if d is None:
            break
        if scenario == 'restart_410':
            world.clock.advance(180.)
            advance(world.frame())
            out, restart = restart_pending(d, saved_remaining=420.)
            record(scenario)
            break
        crash_stage = Stage.GRASP if scenario.endswith('_grasp') else Stage.PLACE
        if scenario.startswith('crash_') and d.stage is crash_stage:
            out, restart = restart_pending(d, physical_after='_after_' in scenario)
            record(scenario)
            break
        if scenario == 'deadline' and d.stage is Stage.CARRY:
            world.clock.advance(world.run.deadline.seconds - world.clock.seconds)
            out = advance(world.frame())
            record(scenario)
            break
        if scenario == 'checkpoint_unknown':
            world.engine.journal.fail = True
            out = advance(world.frame())
            record(scenario)
            break

        world.clock.advance(.1)
        selected = ((scenario in ('unknown_grasp', 'echo_only', 'failed_grasp', 'timed_out_grasp') and d.stage is Stage.GRASP)
                    or (scenario in ('unknown_transport', 'dropped_frames') and d.stage is Stage.CARRY)
                    or (scenario == 'unknown_release' and d.stage is Stage.PLACE))
        if selected:
            if scenario in UNKNOWN:
                world.update(holding=c.HoldingSnapshot(c.Holding.UNKNOWN, d.envelope.target_id,
                                                       None, (world.proof(d.envelope.target_id),)))
            frame = world.complete(d, unknown=True)
            event = frame.inputs.action_events[0]
            if scenario == 'failed_grasp':
                event = replace(event, result=replace(event.result, status=c.Outcome.FAILED,
                                physical=c.Fact.REJECTED))
            elif scenario == 'timed_out_grasp':
                event = replace(event, result=replace(event.result, status=c.Outcome.TIMED_OUT,
                    reason=c.Reason(c.ErrorCode.TIMEOUT, 'grasp', 'simulated action timeout')))
            elif scenario == 'echo_only':
                event = replace(event, result=replace(event.result, status=c.Outcome.SUCCESS, reason=None,
                    evidence=tuple(replace(ev, kind=c.EvidenceKind.ECHO) for ev in event.result.evidence)))
            if scenario == 'dropped_frames':
                frame = replace(frame, readiness=None, inputs=replace(frame.inputs, action_events=()))
            else:
                frame = _event_frame(frame, event)
            record(scenario, stage=d.stage.value)
        else:
            if scenario in ('late_event', 'old_instance') and d.stage is Stage.GRASP and not injections:
                stale = world.complete(d, unknown=True)
                event = stale.inputs.action_events[0]
                if scenario == 'late_event':
                    old = replace(event.envelope,
                                  emitted_at=replace(world.clock.now(), seconds=world.clock.seconds-20),
                                  expires_at=replace(world.clock.now(), seconds=world.clock.seconds-.01))
                else:
                    old = replace(event.envelope, instance_id='obsolete-controller')
                rejected = advance(_event_frame(stale, replace(event, envelope=old)))
                record(scenario, issued=rejected.directive is not None, issues=list(rejected.issues))
                c.require(rejected.directive is None, 'callback', 'obsolete event dispatched new work')
            frame = world.complete(d)
            event = frame.inputs.action_events[0]

        out = advance(frame)
        if scenario == 'duplicate':
            repeated = advance(frame)
            record('duplicate', event_id=event.envelope.event_id,
                   issued=repeated.directive is not None)
            c.require(repeated.directive is None, 'callback', 'duplicate dispatched a new directive')
        elif scenario == 'conflicting_duplicate':
            conflict = replace(event, result=replace(event.result, physical=c.Fact.UNKNOWN))
            out = advance(_event_frame(frame, conflict))
            record(scenario)
            break
    else:
        raise RuntimeError('bounded P11 replay did not terminate')

    flow_phase, flow_stage = out.decision.phase.value, out.stage.value
    flow_reason = out.decision.end_reason.detail if out.decision.end_reason else None
    flow_stops = len(stops)
    flow_elapsed = world.clock.seconds - world.run.started_at.seconds
    counts = {category.value: 0 for category in m.Category}
    for delivery in world.snapshot.deliveries:
        counts[delivery.category.value] += 1
    points = world.score()
    holding = world.snapshot.holding.state.value
    layout_complete = out.decision.layout_complete.value
    # A fake clock can advance the rest of the match in one step. The request
    # deadline is never shifted. The old event cannot revive stopped work.
    world.clock.advance(world.run.deadline.seconds - world.clock.seconds)
    probe = advance(world.frame(events=(last_event,) if last_event else ()))
    return dict(schema_version=1, contract_version=c.CONTRACT_VERSION, origin='simulated',
                scenario=scenario, layout=layout, arrivals=len(world.snapshot.deliveries),
                counts=counts, delivery_keys=[d.key.run_id+':'+d.key.inventory_id for d in world.snapshot.deliveries],
                conditional_points=points, flow_phase=flow_phase, flow_stage=flow_stage,
                flow_end_reason=flow_reason, holding=holding, layout_complete=layout_complete,
                flow_elapsed_s=flow_elapsed, elapsed_s=world.clock.seconds-world.run.started_at.seconds,
                original_deadline_s=world.run.deadline.seconds, started_at_s=world.run.started_at.seconds,
                issued=issued, events=events, injections=injections, restart=restart,
                stop_requests=len(stops), flow_stop_requests=flow_stops,
                deadline_probe=dict(issued=probe.directive is not None, phase=probe.decision.phase.value,
                                    remaining_s=r.remaining_seconds(world.run, world.clock.now())),
                hardware_outputs=0,
                evidence_scope='P09 real scheduler; P03-P08 observations and checkpoint are SIMULATED',
                trace=world.engine.trace)


def validate_replay(result):
    """Acceptance expectations, separate from scheduling/domain algorithms."""
    if result.get('scenario') in FIELD_SCENARIOS:
        return validate_field_replay(result)
    failures = []
    def check(ok, message):
        if not ok:
            failures.append(message)
    scenario = result['scenario']
    check(scenario in SCENARIOS, 'unsupported scenario')
    check(result['origin'] == 'simulated' and result['hardware_outputs'] == 0, 'offline output boundary')
    check(result['elapsed_s'] == 600 and result['original_deadline_s'] == 700, 'original 600 second clock')
    check(not result['deadline_probe']['issued'] and result['deadline_probe']['phase'] == 'stopped', 'deadline locks output')
    keys = result['delivery_keys']
    check(len(keys) == len(set(keys)) == result['arrivals'], 'unique delivery keys')
    actions = [x['action_id'] for x in result['issued']]
    check(len(actions) == len(set(actions)), 'unique dispatch identities')
    if scenario != 'nominal':
        check(bool(result['injections']), 'fault was actually injected')
    if scenario in COMPLETING:
        check(result['arrivals'] == 10 and result['counts'] == {'cake': 1, 'banana': 3, 'doll': 3, 'clock': 3}, 'cake plus nine arrivals')
        check(result['layout_complete'] == 'confirmed', 'selected layouts complete')
        check(result['conditional_points'] == {'V1': 21, 'V2': 27, 'V3': 42}[result['layout']], 'conditional simulated score')
        check(len(actions) == 63, 'one directive for each stage')
        check(not any(x.get('issued', False) for x in result['injections']), 'rejected callback issued work')
    else:
        check(result['arrivals'] == (1 if scenario == 'crash_after_release' else 0), 'no inferred or repeated arrival')
        check(result['flow_stop_requests'] > 0, 'fault produces a stop request')
        if scenario in UNKNOWN:
            check(result['holding'] == 'unknown', 'unknown holding retained')
        if scenario == 'deadline':
            check(result['holding'] == 'held' and result['flow_phase'] == 'stopped', 'held deadline stop')
        else:
            check(result['flow_phase'] == 'recovery', 'fault holds progression for reconciliation')
        if scenario.startswith('crash_') or scenario == 'restart_410':
            restart = result['restart']
            check(result['flow_stage'] == 'revalidate_after_restart' and restart is not None,
                  'restart requires physical review')
            if restart:
                check(not restart['reissued'] and restart['pending_before'] == restart['pending_after'], 'no interrupted action replay')
                if scenario == 'restart_410':
                    check(restart['saved_remaining_s'] == 420 and restart['remaining_s'] == 410, 'downtime subtracted')
    return tuple(failures)
