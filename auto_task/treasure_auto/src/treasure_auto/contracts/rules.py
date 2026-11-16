"""Pure C0 guards. They validate claims; they cannot establish physical truth."""
import hashlib
from dataclasses import fields, is_dataclass, replace
from .core import (ContractError, ErrorCode, Owner, Origin, Fact, Holding, EvidenceKind, Outcome,
                   Envelope, EvidenceRef, Record, require, monotonic, dumps)
from .models import (Area, Category, LayoutKind, Measurement, OccupancyState,
                     InventoryLocation, ConsumerContext, DeliveryReceipt, HoldingCause, CommitStatus, ReadResult)


def require_committed_space(space, readback):
    """Preserve P08 ownership and verify exact state through a trusted P10B read.

    readback(space) adapts PersistencePort.read(space.envelope.run_id). It must
    not construct a success result from the proposal it was asked to check.
    This is an assembly validator; no C0 wire type or version is changed.
    """
    require(space.envelope.producer is Owner.P08, 'space.producer', 'P08 owns spatial facts')
    require(callable(readback), 'space.commit', 'P10B readback adapter required', ErrorCode.UNCONFIGURED)
    try:
        read = readback(space)
    except Exception as error:
        raise ContractError(ErrorCode.STORAGE_FAILURE, 'space.commit', str(error)) from error
    require(type(read) is ReadResult and read.status is Outcome.SUCCESS and
            read.reason is None and read.snapshot is not None,
            'space.commit', 'committed snapshot is unavailable', ErrorCode.STORAGE_FAILURE)
    snapshot = read.snapshot
    require(snapshot.space == space, 'space.commit',
            'proposal differs from the exact committed spatial state', ErrorCode.STORAGE_FAILURE)
    run = snapshot.run.envelope
    env = space.envelope
    require((run.run_id, run.instance_id, run.control_epoch, run.origin, run.versions) ==
            (env.run_id, env.instance_id, env.control_epoch, env.origin, env.versions) and
            space.version == run.versions.space,
            'space.commit', 'published run and space must share identity and versions', ErrorCode.VERSION_MISMATCH)
    return snapshot


def intent_digest(intent):
    return hashlib.sha256(dumps(intent).encode('utf-8')).hexdigest()


def validate_intent_receipt(intent, receipt):
    require(receipt.commit.status is CommitStatus.COMMITTED,
            'intent_receipt', 'uncommitted intent cannot be dispatched', ErrorCode.STORAGE_FAILURE)
    require(receipt.run_id == intent.envelope.run_id and receipt.action_id == intent.envelope.action_id and
            receipt.intent_sha256 == intent_digest(intent), 'intent_receipt',
            'receipt must bind the exact immutable action including phase/speed/timeout', ErrorCode.ID_CONFLICT)


def validate_action_result(envelope, result, now):
    for evidence in result.evidence:
        require(evidence.run_id == envelope.run_id and evidence.origin is envelope.origin,
                'evidence', 'result provenance/run mismatch', ErrorCode.SIMULATED_EVIDENCE)
        if evidence.kind is EvidenceKind.PHYSICAL:
            require_physical_evidence(evidence, envelope, now, envelope.origin)


def validate_envelope(message, current):
    """Must run before a consumer changes current state; expiration is exclusive."""
    require(message.run_id == current.run_id, 'run_id', 'old/different run', ErrorCode.STALE_RUN)
    require((message.instance_id, message.control_epoch) == (current.instance_id, current.control_epoch),
            'control_epoch', 'old/different controller', ErrorCode.STALE_CONTROL)
    require(message.origin is current.origin, 'origin', 'provenance mismatch', ErrorCode.SIMULATED_EVIDENCE)
    for name in ('request_id', 'action_id', 'target_id'):
        expected = getattr(current, name)
        if expected is not None:
            require(getattr(message, name) == expected, name, 'wrong active identity', ErrorCode.STALE_REQUEST)
    for name in ('layout_recovery_round', 'rebuild_generation', 'base_motion_epoch'):
        expected = getattr(current, name)
        if expected is not None:
            require(getattr(message, name) == expected, name, 'obsolete geometry or recovery context',
                    ErrorCode.STALE_GEOMETRY if name == 'base_motion_epoch' else ErrorCode.STALE_RECOVERY)
    require(message.versions == current.versions, 'versions', 'explicit revalidation required', ErrorCode.VERSION_MISMATCH)
    monotonic(current.now, 'now')
    require(current.now - message.emitted_at >= 0, 'emitted_at', 'future message', ErrorCode.FUTURE_RESULT)
    require(message.expires_at - current.now > 0, 'expires_at', 'result expired', ErrorCode.EXPIRED)


def remaining_seconds(run, now):
    monotonic(now, 'now')
    require(now - run.started_at >= 0, 'now', 'clock predates start', ErrorCode.CLOCK_MISMATCH)
    return max(0.0, run.deadline - now)


def require_physical_evidence(evidence, envelope, now, origin):
    require(evidence.kind is EvidenceKind.PHYSICAL, 'evidence.kind',
            'interface/submission/echo is not physical proof', ErrorCode.EVIDENCE_NOT_PHYSICAL)
    require(evidence.origin is origin and envelope.origin is origin, 'evidence.origin',
            'simulation/replay cannot authorize live operation', ErrorCode.SIMULATED_EVIDENCE)
    require(evidence.run_id == envelope.run_id, 'evidence.run_id', 'wrong run', ErrorCode.STALE_RUN)
    for name in ('target_id', 'action_id'):
        expected = getattr(envelope, name)
        if expected is not None:
            require(getattr(evidence, name) == expected, 'evidence.' + name,
                    'observation does not match action/target', ErrorCode.STALE_REQUEST)
    require(now - evidence.received_at >= 0, 'evidence.received_at', 'future receipt', ErrorCode.FUTURE_RESULT)
    require(evidence.valid_until - now > 0, 'evidence.valid_until', 'expired evidence', ErrorCode.EXPIRED)


def require_space_version(envelope, snapshot):
    require(envelope.run_id == snapshot.envelope.run_id, 'space.run_id', 'wrong run', ErrorCode.STALE_RUN)
    require(envelope.origin is snapshot.envelope.origin, 'space.origin', 'wrong provenance', ErrorCode.SIMULATED_EVIDENCE)
    require(envelope.versions.space is not None and envelope.versions.space == snapshot.version,
            'space.version', 'P03/P06 must consume the same committed snapshot', ErrorCode.VERSION_MISMATCH)
    require(envelope.versions.map == snapshot.envelope.versions.map,
            'space.map', 'space coordinates belong to another map', ErrorCode.VERSION_MISMATCH)


def require_measured_parameter(parameter):
    require(parameter.value is not None and parameter.minimum is not None and
            parameter.maximum is not None and parameter.range_source is not None and
            parameter.measurement is Measurement.VERIFIED and bool(parameter.evidence),
            parameter.name, 'verified value, range source and evidence required', ErrorCode.UNCONFIGURED)
    require(all(e.origin is Origin.LIVE for e in parameter.evidence),
            parameter.name, 'simulated parameter evidence cannot enable live operation', ErrorCode.SIMULATED_EVIDENCE)


def require_layout_quality(layout):
    """Minimal schema-level layout quality. P06/P08 establish geometry and evidence."""
    require(layout.quality is Fact.CONFIRMED, 'quality', 'quality not confirmed', ErrorCode.INVALID_LAYOUT)
    require(layout.area is not Area.A, 'area', 'ordinary layouts are B/C/D', ErrorCode.INVALID_LAYOUT)
    require(len(layout.bricks) == 3 and layout.confirmed_count == 3, 'bricks', 'complete layout needs three bricks', ErrorCode.INVALID_LAYOUT)
    bricks = sorted(layout.bricks, key=lambda b: b.slot)
    require([b.slot for b in bricks] == [1, 2, 3], 'slot', 'slots must be 1/2/3', ErrorCode.INVALID_LAYOUT)
    require(bricks[0].near_center is Fact.CONFIRMED, 'near_center', 'first brick must be near center', ErrorCode.INVALID_LAYOUT)
    for brick in bricks:
        require(brick.released is Fact.CONFIRMED and brick.in_region is Fact.CONFIRMED and
                brick.horizontal is Fact.CONFIRMED and bool(brick.evidence),
                'bricks', 'release, region, level and evidence required', ErrorCode.INVALID_LAYOUT)
        if brick.slot > 1:
            require(brick.placed_recovery_round == layout.layout_recovery_round,
                    'placed_recovery_round', 'second and third need actual placement in this round', ErrorCode.STALE_RECOVERY)
    if layout.kind is LayoutKind.V1:
        require(all(b.layer == 1 and not b.support_ids for b in bricks), 'layer', 'V1 is three ground-level bricks', ErrorCode.INVALID_LAYOUT)
    elif layout.kind is LayoutKind.V2:
        require([b.layer for b in bricks] == [1, 1, 2] and
                not bricks[0].support_ids and not bricks[1].support_ids and
                set(bricks[2].support_ids) == {bricks[0].target_id, bricks[1].target_id} and
                layout.bases_adjacent is Fact.CONFIRMED,
                'support_ids', 'V2 needs adjacent bases and shared top support', ErrorCode.INVALID_LAYOUT)
    else:
        require([b.layer for b in bricks] == [1, 2, 3] and not bricks[0].support_ids,
                'layer', 'V3 needs successive layers', ErrorCode.INVALID_LAYOUT)
        for lower, upper in zip(bricks, bricks[1:]):
            require(upper.support_ids == (lower.target_id,) and upper.support_ratio_lower is not None and
                    upper.support_ratio_lower > 0.8, 'support_ratio_lower',
                    'actual supported bottom area lower bound must exceed 0.8', ErrorCode.INVALID_LAYOUT)


def validate_stillness(cert, envelope, now, required_targets, current_scene_epoch):
    require_space_version(envelope, _SpaceVersionView(cert.envelope))
    require(cert.envelope.instance_id == envelope.instance_id and
            cert.envelope.control_epoch == envelope.control_epoch,
            'control_epoch', 'certificate from old controller', ErrorCode.STALE_CONTROL)
    require(cert.envelope.layout_recovery_round == envelope.layout_recovery_round and
            cert.envelope.rebuild_generation == envelope.rebuild_generation,
            'layout_recovery_round', 'certificate from old recovery', ErrorCode.STALE_RECOVERY)
    require(cert.scene_epoch == current_scene_epoch and cert.envelope.base_motion_epoch == envelope.base_motion_epoch,
            'scene_epoch', 'scene changed since observation', ErrorCode.STALE_GEOMETRY)
    require(cert.envelope.expires_at - now > 0 and now - cert.envelope.emitted_at >= 0,
            'expires_at', 'stale/future certificate', ErrorCode.EXPIRED)
    require(cert.continuous is Fact.CONFIRMED and cert.all_stopped is Fact.CONFIRMED,
            'continuous', 'uninterrupted valid frames after actual stop required', ErrorCode.EVIDENCE_MISSING)
    require(cert.capture_end - cert.capture_start >= 0.5, 'capture_end', 'at least 500 ms in capture clock')
    require(bool(required_targets) and set(required_targets) <= set(cert.covered_target_ids) and
            len(cert.coverage_polygon_m) >= 3, 'covered_target_ids', 'all affected targets and scattered range required')
    require(len(cert.observation_ids) >= 2 and len(set(cert.observation_ids)) == len(cert.observation_ids),
            'observation_ids', 'distinct new frames required')
    require(0 <= cert.max_adjacent_motion_m <= cert.tolerance_m and
            0 <= cert.max_total_motion_m <= cert.tolerance_m, 'tolerance_m', 'movement exceeds measured tolerance')
    require(bool(cert.evidence), 'evidence', 'certificate needs traceable observations')
    for evidence in cert.evidence:
        require_physical_evidence(evidence, cert.envelope, now, envelope.origin)
    captures = [e.captured_at for e in cert.evidence]
    require(len(captures) == len(cert.observation_ids), 'evidence', 'one capture reference per observation')
    for a, b in zip(captures, captures[1:]):
        require(b - a > 0, 'captured_at', 'repeated/out-of-order capture is not a new frame')
    require(captures[0] == cert.capture_start and captures[-1] == cert.capture_end,
            'capture_end', 'claimed window must match actual capture references')


class _SpaceVersionView:
    """Read-only adapter for version comparison, never a second occupancy map."""
    def __init__(self, envelope):
        self.envelope = envelope
        self.version = envelope.versions.space


def require_simulated(record):
    """Reject any live/replay provenance nested inside a public simulated fixture."""
    require_provenance(record, Origin.SIMULATED)


def require_provenance(record, origin):
    if isinstance(record, (Envelope, EvidenceRef)):
        require(record.origin is origin, 'origin', 'nested provenance mismatch', ErrorCode.SIMULATED_EVIDENCE)
    if isinstance(record, Record) and is_dataclass(record):
        for f in fields(record):
            value = getattr(record, f.name)
            if f.name == 'origin' and isinstance(value, Origin):
                require(value is origin, 'origin', 'nested provenance mismatch', ErrorCode.SIMULATED_EVIDENCE)
            if isinstance(value, tuple):
                for item in value:
                    require_provenance(item, origin)
            else:
                require_provenance(value, origin)


def validate_transaction(tx, previous, now):
    """Validate a proposed complete update before P10B's durable compare-and-swap.

    P08 owns space, P07 holding, P09 inventory/delivery coordination. Only P10B
    commits/publishes. Replay deduplication is checked by storage before this guard.
    """
    run = previous.run
    require_provenance(tx, run.envelope.origin)
    remaining_seconds(run, now)  # Establish the trusted run clock, including boot identity.
    require(tx.envelope.producer is Owner.P09, 'envelope.producer', 'P09 coordinates state transaction')
    current = ConsumerContext(run.envelope.run_id, run.envelope.instance_id, run.envelope.control_epoch,
                              now, tx.envelope.versions, None, None, None,
                              None, None, None, run.envelope.origin)
    validate_envelope(tx.envelope, current)
    require(tx.expected_revision == previous.revision and tx.expected_space_version == previous.space.version,
            'expected_revision', 'compare-and-swap conflict; re-read snapshot', ErrorCode.CONFLICT)
    require(tx.envelope.versions == previous.space.envelope.versions,
            'versions', 'transaction base versions changed', ErrorCode.VERSION_MISMATCH)
    require(tx.space.version == previous.space.version + 1 and tx.space.envelope.producer is Owner.P08,
            'space.version', 'P08 proposes exactly the next space version')
    require(tx.space.envelope.run_id == run.envelope.run_id and tx.space.envelope.origin is run.envelope.origin,
            'space.envelope', 'space must belong to same run and provenance')
    require(tx.space.envelope.instance_id == run.envelope.instance_id and
            tx.space.envelope.control_epoch == run.envelope.control_epoch,
            'space.control_epoch', 'space proposal belongs to old controller', ErrorCode.STALE_CONTROL)
    validate_envelope(tx.space.envelope, replace(current, versions=tx.space.envelope.versions))
    for name in ('config', 'map', 'calibration', 'model', 'layout'):
        require(getattr(tx.space.envelope.versions, name) == getattr(tx.envelope.versions, name),
                'space.versions.' + name, 'use explicit migration for version changes', ErrorCode.VERSION_MISMATCH)
    if previous.mapping is not None:
        require(tx.mapping == previous.mapping, 'mapping', 'mapping frozen for run', ErrorCode.INVALID_MAPPING)
    else:
        require(not tx.release_events, 'release_events', 'freeze mapping in a committed snapshot before dispatch')
    if tx.mapping is not None:
        require(tx.mapping.envelope.run_id == run.envelope.run_id and
                tx.mapping.envelope.origin is run.envelope.origin and
                tx.mapping.mapping_version == tx.space.envelope.versions.mapping,
                'mapping', 'wrong run/provenance/mapping version', ErrorCode.INVALID_MAPPING)
        if previous.mapping is None:
            require(tx.mapping.envelope.producer is Owner.P05, 'mapping.envelope.producer', 'P05 freezes mapping')
            validate_envelope(tx.mapping.envelope, replace(current, versions=tx.mapping.envelope.versions))
    else:
        require(tx.space.envelope.versions.mapping is None, 'mapping', 'no mapping version without mapping')
    if previous.mapping is not None:
        require(tx.space.envelope.versions.mapping == tx.envelope.versions.mapping,
                'mapping', 'cannot silently change frozen mapping version')
    if tx.run_update is not None:
        proposed_run = tx.run_update
        require(proposed_run.envelope.producer is Owner.P10A, 'run_update', 'P10A owns run state')
        require((proposed_run.envelope.run_id, proposed_run.envelope.instance_id, proposed_run.envelope.control_epoch) ==
                (run.envelope.run_id, run.envelope.instance_id, run.envelope.control_epoch),
                'run_update', 'new controller registration uses register_recovery')
        require(proposed_run.started_at == run.started_at and proposed_run.deadline == run.deadline and
                proposed_run.start_signal_id == run.start_signal_id,
                'run_update.deadline', 'original start/deadline/start signal are immutable', ErrorCode.DEADLINE)
        require(not run.ever_restarted or proposed_run.ever_restarted,
                'run_update.ever_restarted', 'restart history cannot be erased')
        require(proposed_run.envelope.origin is run.envelope.origin, 'run_update.origin', 'provenance cannot change')
        validate_envelope(proposed_run.envelope, replace(current, versions=proposed_run.envelope.versions))
    old_layouts = {layout.area: layout for layout in previous.layouts}
    require(len({layout.area for layout in tx.layouts}) == len(tx.layouts), 'layouts', 'one layout per region')
    require(old_layouts.keys() <= {layout.area for layout in tx.layouts}, 'layouts', 'retain existing regional layout context')
    for layout in tx.layouts:
        if old_layouts.get(layout.area) != layout:
            require(layout.envelope.producer is Owner.P08, 'layouts.envelope.producer', 'P08 owns layout facts')
            validate_envelope(layout.envelope, replace(current, versions=layout.envelope.versions))
            require(layout.envelope.versions.space in (previous.space.version, tx.space.version),
                    'layouts.versions.space', 'layout must bind transaction space')
    old_budgets = {b.name: b for b in previous.budgets}
    budgets = {b.name: b for b in tx.budgets}
    require(len(budgets) == len(tx.budgets) and old_budgets.keys() <= budgets.keys(),
            'budgets', 'duplicate or discarded cumulative budget')
    for name, old in old_budgets.items():
        new = budgets[name]
        require(new.attempts_used >= old.attempts_used and new.seconds_used >= old.seconds_used and
                (new.attempts_limit, new.seconds_limit) == (old.attempts_limit, old.seconds_limit),
                'budgets.' + name, 'recovery cannot reset usage or change configured limits')
    ids = [i.inventory_id for i in tx.inventory]
    require(len(ids) == len(set(ids)), 'inventory_id', 'duplicate inventory identity', ErrorCode.ID_CONFLICT)
    aliases = [t for i in tx.inventory for t in i.target_ids]
    require(len(aliases) == len(set(aliases)), 'target_ids', 'target has multiple inventory identities', ErrorCode.ID_CONFLICT)
    old_inventory = {i.inventory_id: i for i in previous.inventory}
    inventory = {i.inventory_id: i for i in tx.inventory}
    for key, item in old_inventory.items():
        require(key in inventory and set(item.target_ids) <= set(inventory[key].target_ids),
                'inventory', 'rebuild cannot discard known inventory/aliases', ErrorCode.UNKNOWN_IDENTITY)
        require(item.category is None or inventory[key].category is item.category,
                'inventory.category', 'rebuild cannot reclassify established inventory', ErrorCode.ID_CONFLICT)
    old_occupancy = {o.target_id: o for o in previous.space.occupancies}
    new_occupancy = {o.target_id: o for o in tx.space.occupancies}
    for target in new_occupancy.keys() - old_occupancy.keys():
        new = new_occupancy[target]
        if new.state is OccupancyState.ACTUAL:
            require(bool(new.evidence), 'occupancies.evidence', 'new actual occupancy requires physical observation')
            for proof in new.evidence:
                require_physical_evidence(proof, replace(tx.envelope, target_id=target, action_id=None), now, run.envelope.origin)
    for target in old_occupancy.keys() - new_occupancy.keys():
        clearances = [e for e in tx.clearing_evidence if e.target_id == target and e.confirmed is Fact.CONFIRMED]
        proofs = [e for clearance in clearances for e in clearance.evidence]
        require(bool(proofs), 'clearing_evidence', 'disappearance cannot clear occupancy', ErrorCode.OCCUPANCY_UNRESOLVED)
        for proof in proofs:
            require_physical_evidence(proof, replace(tx.envelope, target_id=target), now, run.envelope.origin)
            require(proof.received_at - old_occupancy[target].observed_at > 0,
                    'clearing_evidence', 'clearance needs newer observation')
    for target in old_occupancy.keys() & new_occupancy.keys():
        old, new = old_occupancy[target], new_occupancy[target]
        changed = (old.footprint_m, old.z_min_m, old.z_max_m, old.error_m, old.state, old.frame_id) != (
                   new.footprint_m, new.z_min_m, new.z_max_m, new.error_m, new.state, new.frame_id)
        if changed:
            require(new.state is not OccupancyState.RESERVED or old.state is OccupancyState.RESERVED,
                    'occupancies.state', 'actual/unknown cannot become a mere reservation', ErrorCode.OCCUPANCY_UNRESOLVED)
            require(bool(new.evidence), 'occupancies.evidence', 'changed footprint needs new physical observation')
            for proof in new.evidence:
                require_physical_evidence(proof, replace(tx.envelope, target_id=target), now, run.envelope.origin)
                require(proof.received_at - old.observed_at > 0, 'occupancies.evidence', 'old observation cannot shrink occupancy')
    receipts = {d.key: d for d in previous.deliveries}
    seen_events = set()
    for event in tx.release_events:
        require(event.envelope.event_id not in seen_events, 'event_id', 'duplicate event within transaction', ErrorCode.ID_CONFLICT)
        seen_events.add(event.envelope.event_id)
        require(event.envelope.producer is Owner.P08, 'release.producer', 'only P08 produces release evidence')
        for name in ('request_id', 'action_id', 'target_id', 'layout_recovery_round', 'rebuild_generation', 'base_motion_epoch'):
            require(getattr(event.envelope, name) == getattr(tx.envelope, name),
                    'release.' + name, 'release belongs to obsolete work/recovery', ErrorCode.STALE_REQUEST)
        require(event.envelope.versions == tx.envelope.versions, 'release.versions', 'release uses another snapshot', ErrorCode.VERSION_MISMATCH)
        require(tx.holding.state is Holding.EMPTY, 'holding', 'confirmed release requires P07 empty proposal')
        require(event.delivery_key.run_id == run.envelope.run_id and event.envelope.run_id == run.envelope.run_id,
                'delivery_key.run_id', 'wrong run', ErrorCode.STALE_RUN)
        require(event.envelope.instance_id == run.envelope.instance_id and
                event.envelope.control_epoch == run.envelope.control_epoch,
                'release.control_epoch', 'old control result', ErrorCode.STALE_CONTROL)
        require(event.envelope.expires_at - now > 0 and
                tx.envelope.emitted_at - event.envelope.emitted_at >= 0,
                'release.expires_at', 'release result is stale/future', ErrorCode.EXPIRED)
        require(event.separated is Fact.CONFIRMED and event.correct_region is Fact.CONFIRMED,
                'release', 'actual separation and correct region must be confirmed', ErrorCode.EVIDENCE_MISSING)
        require(tx.mapping is not None and event.mapping_version == tx.mapping.mapping_version and
                tx.mapping.destination(event.category) is event.area,
                'release.area', 'wrong mapping/region', ErrorCode.INVALID_MAPPING)
        require(event.released_at - run.started_at >= 0 and run.deadline - event.released_at > 0 and
                tx.envelope.emitted_at - event.released_at >= 0,
                'released_at', 'release must actually precede original deadline and receipt', ErrorCode.DEADLINE)
        item = inventory.get(event.delivery_key.inventory_id)
        require(item is not None and item.identity_confirmed is Fact.CONFIRMED and
                event.envelope.target_id in item.target_ids and item.category is event.category,
                'delivery_key.inventory_id', 'unresolved/reassigned physical identity', ErrorCode.UNKNOWN_IDENTITY)
        require(item.location is InventoryLocation.REGION and item.area is event.area,
                'inventory.location', 'release and inventory must agree')
        occupancy = new_occupancy.get(event.envelope.target_id)
        require(occupancy is not None and occupancy.state is OccupancyState.ACTUAL and occupancy.area is event.area,
                'space.occupancies', 'arrival requires matching physical occupancy')
        require(bool(event.evidence), 'release.evidence', 'physical release evidence required')
        for evidence in event.evidence:
            require_physical_evidence(evidence, event.envelope, now, run.envelope.origin)
        if event.delivery_key in receipts:
            old = receipts[event.delivery_key]
            require(old.category is event.category and old.area is event.area,
                    'delivery_key', 'same inventory cannot earn another category/region', ErrorCode.ID_CONFLICT)
        else:
            receipts[event.delivery_key] = DeliveryReceipt(event.delivery_key, event.envelope.event_id,
                event.category, event.area, event.released_at, tuple(e.evidence_id for e in event.evidence))
    for category in Category:
        limit = 1 if category is Category.CAKE else 3
        require(sum(d.category is category for d in receipts.values()) <= limit,
                'deliveries', 'excess arrivals require review, never clamp counts', ErrorCode.LIMIT_EXCEEDED)
    if tx.holding != previous.holding:
        proposal = tx.holding_update
        require(proposal is not None, 'holding_update', 'changed holding requires P07 proposal')
        require(proposal.envelope.producer is Owner.P07 and proposal.previous is previous.holding.state and
                proposal.current == tx.holding, 'holding_update', 'P07 must propose the exact linked transition')
        validate_envelope(proposal.envelope, replace(current, request_id=tx.envelope.request_id,
                          action_id=tx.envelope.action_id, target_id=tx.envelope.target_id,
                          layout_recovery_round=tx.envelope.layout_recovery_round,
                          rebuild_generation=tx.envelope.rebuild_generation,
                          base_motion_epoch=tx.envelope.base_motion_epoch))
        if tx.holding.state in (Holding.EMPTY, Holding.HELD, Holding.DROPPED):
            require(bool(tx.holding.evidence), 'holding.evidence', 'physical holding conclusion needs new evidence')
            for proof in tx.holding.evidence:
                require_physical_evidence(proof, tx.envelope, now, run.envelope.origin)
        if previous.holding.state is not Holding.EMPTY and tx.holding.state is Holding.EMPTY:
            if proposal.cause is HoldingCause.DROP_RECONCILED:
                require(not tx.release_events and previous.holding.target_id in new_occupancy,
                        'holding', 'drop reconciliation retains actual/unknown occupancy without arrival')
            else:
                require(proposal.cause is HoldingCause.RELEASE and
                        any(e.envelope.target_id == previous.holding.target_id for e in tx.release_events),
                        'holding', 'empty transition needs correlated release', ErrorCode.EVIDENCE_MISSING)
    require(not any(i.location is InventoryLocation.HELD for i in tx.inventory) or tx.holding.state is not Holding.EMPTY,
            'inventory.location', 'empty holding cannot leave inventory in hand')
    held_items = [i for i in tx.inventory if i.location is InventoryLocation.HELD]
    require(len(held_items) <= 1, 'inventory.location', 'at most one held inventory item')
    if tx.holding.state is Holding.HELD:
        require(len(held_items) == 1 and tx.holding.target_id in held_items[0].target_ids,
                'inventory.location', 'HELD state and canonical inventory must agree')
    return tuple(receipts.values())
