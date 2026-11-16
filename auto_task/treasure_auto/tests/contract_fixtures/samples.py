"""SIMULATED examples only. Numbers are test vectors, not robot configuration."""
from dataclasses import replace
from itertools import permutations
from treasure_auto.contracts import core as c, models as m


def stamp(seconds=110.0, boot_id='sim-boot'):
    return c.ClockStamp(seconds, c.ClockBasis.MONOTONIC, 'sim-monotonic', boot_id)


def sensor_stamp(seconds=20.0):
    return c.ClockStamp(seconds, c.ClockBasis.SENSOR, 'sim-camera-clock', None)


def versions(space=0):
    return c.Versions('sim-config-v1', 'sim-map-v1', 'sim-calibration-v1',
                      'sim-model-v1', 'sim-mapping-v1', 'V1', space)


def envelope(producer=c.Owner.P07, **changes):
    fields = dict(run_id='sim-run-1', instance_id='sim-instance-1', control_epoch=1,
                  producer=producer, event_id='sim-event-1', sequence=1,
                  request_id='sim-work-1', action_id='sim-action-1', target_id='brick-1',
                  emitted_at=stamp(110), expires_at=stamp(120), versions=versions(),
                  origin=c.Origin.SIMULATED, layout_recovery_round=0,
                  rebuild_generation=0, base_motion_epoch=0)
    fields.update(changes)
    return c.Envelope(**fields)


def evidence(**changes):
    data = dict(evidence_id='sim-evidence-1', run_id='sim-run-1', target_id='brick-1',
                action_id='sim-action-1', kind=c.EvidenceKind.PHYSICAL, origin=c.Origin.SIMULATED,
                captured_at=sensor_stamp(), received_at=stamp(110), valid_until=stamp(120),
                uri='sim://frames/1', sha256=None)
    data.update(changes)
    return c.EvidenceRef(**data)


def consumer_context():
    e = envelope()
    return m.ConsumerContext(e.run_id, e.instance_id, e.control_epoch, stamp(111), e.versions,
                             e.request_id, e.action_id, e.target_id, 0, 0, 0, c.Origin.SIMULATED)


def timeout_result():
    return c.ActionResult(c.Outcome.TIMED_OUT, 'trial_lift',
                          c.Reason(c.ErrorCode.TIMEOUT, 'physical', 'no fresh observation'),
                          c.Fact.CONFIRMED, c.Fact.CONFIRMED, c.Fact.CONFIRMED,
                          c.Fact.UNKNOWN, (), 2.0)


def run_context():
    return m.RunContext(envelope(c.Owner.P10A), m.RunState.OFFLINE,
                        'sim-start-1', stamp(100), stamp(700), False)


def mapping_samples():
    return tuple(m.MappingSnapshot(envelope(c.Owner.P05), 'sim-mapping-v1',
                                    tuple(m.SlotAssignment(a, k) for a, k in zip((m.Area.B, m.Area.C, m.Area.D), order)),
                                    c.Fact.CONFIRMED, stamp(109), (evidence(),))
                 for order in permutations((m.Category.DOLL, m.Category.CLOCK, m.Category.BANANA)))


def layout_pair(kind):
    bricks = []
    for n in (1, 2, 3):
        layer = 1 if kind is m.LayoutKind.V1 else (1 if kind is m.LayoutKind.V2 and n < 3 else n)
        if kind is m.LayoutKind.V2 and n == 3:
            layer = 2
        supports = () if layer == 1 else (('brick-1', 'brick-2') if kind is m.LayoutKind.V2 else ('brick-{}'.format(n - 1),))
        bricks.append(m.LayoutBrick('brick-{}'.format(n), n, layer, supports,
                                    0.81 if layer > 1 else None,
                                    c.Fact.CONFIRMED, c.Fact.CONFIRMED, c.Fact.CONFIRMED,
                                    c.Fact.CONFIRMED if n == 1 else c.Fact.UNKNOWN, 0,
                                    (evidence(target_id='brick-{}'.format(n)),)))
    good = m.LayoutSnapshot(envelope(c.Owner.P08), m.Area.B, kind, 0, 0, 3,
                            tuple(bricks), c.Fact.CONFIRMED, c.Fact.CONFIRMED)
    if kind is m.LayoutKind.V1:
        bad = replace(good, bricks=(replace(bricks[0], layer=2),) + tuple(bricks[1:]))
    elif kind is m.LayoutKind.V2:
        bad = replace(good, bricks=tuple(bricks[:2]) + (replace(bricks[2], support_ids=('brick-1',)),))
    else:
        bad = replace(good, bricks=(bricks[0], replace(bricks[1], support_ratio_lower=0.8), bricks[2]))
    return good, bad


def footprint():
    return (m.Point2(0, 0), m.Point2(0.1, 0), m.Point2(0.1, 0.1), m.Point2(0, 0.1))


def space_snapshot(version=0):
    item = m.Occupancy('brick-1', m.OccupancyState.UNKNOWN, 'map', footprint(), 0, 0.1,
                       0.01, (), stamp(105), m.Visibility.OCCLUDED, None, (evidence(),))
    return m.SpaceSnapshot(envelope(c.Owner.P08, versions=versions(version)), version, (item,))


def top_surface():
    corners = (m.PixelPoint(100, 100), m.PixelPoint(200, 100),
               m.PixelPoint(200, 200), m.PixelPoint(100, 200))
    return m.TopSurface(corners, corners, 640, 480, 'sim-camera', m.Distortion.CORRECTED,
                        1.0, m.PlaneBasis.CALIBRATION, 100, m.Visibility.VISIBLE, (evidence(),))


def target_observation():
    return m.TargetObservation('sim-observation-1', 'brick-1', c.Fact.CONFIRMED,
                               m.Category.CAKE, top_surface(), m.Visibility.VISIBLE, (evidence(),))


def permit():
    return m.ControlPermit(envelope(c.Owner.P10A), 'sim-permit-1', m.MotionSource.ARM,
                           'synthetic test only', (evidence(),))


def arm_command():
    return m.ArmCommand(envelope(), permit(), 'sim-intent-receipt-1', m.Phase.TRIAL_LIFT,
                        c.ArmPoint('sim-arm', 100, 0, 100, 'mm', 1), 1, 'mm/s', 2, (evidence(),))


def arm_intent():
    command = arm_command()
    plan = m.GeometryPlan(envelope(c.Owner.P06), 'sim-path-1', command.target,
                           (m.PathSegment(command.phase, command.target, command.speed_mm_s, command.timeout_s),),
                           c.Fact.CONFIRMED, c.Fact.CONFIRMED, (evidence(),))
    return m.ActionIntent(envelope(), m.IntentOperation.ARM, None, plan, None, None, None)


def intent_receipt(intent):
    from treasure_auto.contracts.rules import intent_digest
    return m.IntentReceipt(intent.envelope.run_id, intent.envelope.action_id, intent_digest(intent),
                            m.CommitReceipt('sim-intent-commit', m.CommitStatus.COMMITTED, 1, 0, None))


def stillness_certificate():
    return m.StillnessCertificate(envelope(c.Owner.P08), ('brick-1',), footprint(), 'map',
                                  sensor_stamp(20), sensor_stamp(20.5), ('frame-1', 'frame-2'),
                                  c.Fact.CONFIRMED, c.Fact.CONFIRMED, 0, 0, 0.001, 0,
                                  (evidence(), evidence(evidence_id='sim-evidence-2', captured_at=sensor_stamp(20.5))))


def state_snapshot():
    inventory = m.InventoryItem('inventory-1', ('brick-1',), c.Fact.CONFIRMED, m.Category.CAKE,
                                 m.InventoryLocation.HELD, None)
    holding = c.HoldingSnapshot(c.Holding.HELD, 'brick-1',
                                c.ArmPoint('tool', 0, 0, 0, 'mm', 1), (evidence(),))
    return m.StateSnapshot(run_context(), 0, space_snapshot(), holding, (inventory,), (), (),
                            mapping_samples()[0], (m.Budget('sim-retry', 2, 4, 0, 0),))


def release_transaction():
    previous = state_snapshot()
    space = space_snapshot(1)
    space = replace(space, occupancies=(replace(space.occupancies[0], state=m.OccupancyState.ACTUAL,
                                               area=m.Area.A, visibility=m.Visibility.VISIBLE,
                                               observed_at=stamp(110)),))
    release = m.ReleaseEvent(envelope(c.Owner.P08, event_id='sim-release-1'),
                              m.DeliveryKey('sim-run-1', 'inventory-1'), m.Category.CAKE, m.Area.A,
                              'sim-mapping-v1', stamp(109.5), c.Fact.CONFIRMED, c.Fact.CONFIRMED, (evidence(),))
    return m.StateTransaction(envelope(c.Owner.P09, event_id='sim-transaction-1'), 'sim-tx-1', 0, 0,
                               space, c.HoldingSnapshot(c.Holding.EMPTY, None, None, (evidence(),)),
                               (replace(previous.inventory[0], location=m.InventoryLocation.REGION, area=m.Area.A),),
                               (), previous.mapping, (release,), (),
                               m.HoldingUpdate(envelope(c.Owner.P07), c.Holding.HELD,
                                  c.HoldingSnapshot(c.Holding.EMPTY, None, None, (evidence(),)), m.HoldingCause.RELEASE),
                               previous.budgets, None)


def replacement_transaction(previous, tx, transaction_id):
    # A rebuilt display slot/observation can change; canonical inventory identity never does.
    next_version = previous.space.version + 1
    header = replace(tx.envelope, event_id=transaction_id, versions=versions(previous.space.version), sequence=2,
                     layout_recovery_round=1, rebuild_generation=1)
    space = replace(previous.space, version=next_version,
                    envelope=replace(previous.space.envelope, versions=versions(next_version)))
    event = replace(tx.release_events[0], envelope=replace(tx.release_events[0].envelope,
                    event_id='sim-replacement-release', sequence=2, layout_recovery_round=1,
                    rebuild_generation=1, versions=versions(previous.space.version)))
    return replace(tx, envelope=header, transaction_id=transaction_id,
                   expected_revision=previous.revision, expected_space_version=previous.space.version,
                   space=space, release_events=(event,), holding_update=None)


def all_samples():
    image = m.ImageObservation(envelope(c.Owner.P04), 'sim-observation-1', evidence(), 640, 480,
                                'rgb8', 'sim-camera', m.Distortion.CORRECTED,
                                c.MapPose('map', 0, 0, 0, 'm', 'rad', 0.01, 0.01),
                                m.PixelTransform(0, 0, 1, 1, 0, 0))
    health = m.HealthSnapshot(envelope(c.Owner.P01), (m.Capability('sim-camera', c.Fact.UNKNOWN,
                              c.Reason(c.ErrorCode.UNAVAILABLE, 'camera', 'simulated missing camera'), ()),))
    plan = m.GeometryPlan(envelope(c.Owner.P06), 'sim-path-1', arm_command().target,
                           (m.PathSegment(m.Phase.TRIAL_LIFT, arm_command().target, 1, 2),),
                           c.Fact.CONFIRMED, c.Fact.UNKNOWN, (evidence(),))
    mission = m.MissionDecision(envelope(c.Owner.P09), m.MissionPhase.WAIT_MAPPING, None, c.Fact.UNKNOWN,
                                 c.Fact.UNKNOWN, m.ScoreEstimate(None, ('brick-1',), (), 'v2'), None, None)
    integration = m.IntegrationReport(c.CONTRACT_VERSION,
                     (m.VerificationResult('P00-T06', m.VerificationTier.OFFLINE, c.Fact.CONFIRMED, 'sim://log', None),),
                     (m.DependencyStatus(c.Owner.P01, None, None, c.Fact.UNKNOWN,
                       c.Reason(c.ErrorCode.UNAVAILABLE, 'devices', 'not delivered')),), c.Origin.SIMULATED)
    return dict(envelope=envelope(), health=health,
                localization=m.LocalizationSnapshot(envelope(c.Owner.P02), None, c.Fact.UNKNOWN, 0, (),
                              c.Reason(c.ErrorCode.UNAVAILABLE, 'pose', 'no field localization')),
                navigation=m.NavigationResult(envelope(c.Owner.P03), timeout_result(), None, None, c.Fact.UNKNOWN, None),
                perception=m.PerceptionResult(envelope(c.Owner.P04), m.ObservationState.EMPTY, (), None, None),
                image=image, mapping=mapping_samples()[0], geometry=plan,
                manipulation=m.ManipulationResult(envelope(), timeout_result(),
                             c.HoldingSnapshot(c.Holding.UNKNOWN, 'brick-1', None, ()), None, None),
                space=space_snapshot(), layout=layout_pair(m.LayoutKind.V3)[0],
                mission=mission, run=run_context(), transaction=release_transaction(),
                state=state_snapshot(), integration=integration, stillness=stillness_certificate())
