"""Current-context and physical-evidence gates for simulation assembly."""
from dataclasses import replace
from treasure_auto.contracts import core as c,models as m,rules as r


def context(e,now,*,identities=False):
    return m.ConsumerContext(e.run_id,e.instance_id,e.control_epoch,now,e.versions,
        e.request_id if identities else None,e.action_id if identities else None,e.target_id if identities else None,
        e.layout_recovery_round,e.rebuild_generation,e.base_motion_epoch,e.origin)


def physical(proofs,e,now,max_age):
    c.require(bool(proofs),'evidence','physical observation required',c.ErrorCode.EVIDENCE_MISSING)
    for ev in proofs:
        r.require_physical_evidence(ev,e,now,e.origin)
        c.require(ev.captured_at.basis is c.ClockBasis.MONOTONIC,'capture','trusted capture-to-run clock mapping missing',c.ErrorCode.CLOCK_MISMATCH)
        c.require(0<=now-ev.captured_at<=max_age and ev.received_at-ev.captured_at>=0,'capture','stale/future capture',c.ErrorCode.EXPIRED)


def stopped(stop,e,now,max_age):
    c.require(stop is not None and stop.envelope.producer is c.Owner.P01 and
              stop.physically_stopped is c.Fact.CONFIRMED and stop.old_queue_empty is c.Fact.CONFIRMED,
              'stop','actual stopped base and empty old queue required',c.ErrorCode.EVIDENCE_MISSING)
    r.validate_envelope(stop.envelope,context(e,now))
    physical(stop.evidence,stop.envelope,now,max_age)


def ready(readiness,snapshot,e,now,max_age,state):
    c.require(readiness is not None and readiness.envelope.producer is c.Owner.P07 and
              readiness.holding==snapshot.holding and readiness.holding.state is state and
              readiness.posture_confirmed is c.Fact.CONFIRMED,'readiness','P07 current holding and posture required',c.ErrorCode.EVIDENCE_MISSING)
    r.validate_envelope(readiness.envelope,context(e,now))
    physical(readiness.evidence,readiness.envelope,now,max_age)
    physical(snapshot.holding.evidence,replace(e,target_id=snapshot.holding.target_id,action_id=None),now,max_age)


def layout_prefix(layout,snapshot,e,now,max_age,target=None):
    c.require(layout is not None and layout.envelope.producer is c.Owner.P08,'layout','P08 layout required')
    r.validate_envelope(layout.envelope,context(e,now))
    c.require(layout.kind.value==snapshot.space.envelope.versions.layout,'layout','frozen layout kind changed')
    c.require(layout.quality is not c.Fact.REJECTED and layout.confirmed_count==len(layout.bricks)>0,'layout','current prefix not confirmed')
    bricks=sorted(layout.bricks,key=lambda b:b.slot)
    c.require([b.slot for b in bricks]==list(range(1,len(bricks)+1)),'layout','nonsequential prefix')
    c.require(target is None or target in {b.target_id for b in bricks},'layout','active brick absent from current prefix')
    for b in bricks:
        c.require(b.released is c.Fact.CONFIRMED and b.in_region is c.Fact.CONFIRMED and b.horizontal is c.Fact.CONFIRMED,
                  'layout','release/region/horizontal unknown')
        c.require(b.slot==1 or b.placed_recovery_round==layout.layout_recovery_round,'layout','old second/third placement')
        physical(b.evidence,replace(layout.envelope,target_id=b.target_id,action_id=None),now,max_age)
    c.require(bricks[0].near_center is c.Fact.CONFIRMED,'layout','first brick center unconfirmed')
    if layout.area is m.Area.A:
        c.require(layout.confirmed_count==1 and layout.quality is c.Fact.CONFIRMED,'cake','stable cake required')
    elif layout.confirmed_count==3:r.require_layout_quality(layout)


def all_layouts(snapshot,e,now,max_age):
    try:
        c.require({l.area for l in snapshot.layouts}==set(m.Area),'layouts','all areas required')
        for layout in snapshot.layouts:
            layout_prefix(layout,snapshot,e,now,max_age)
            c.require(layout.confirmed_count==(1 if layout.area is m.Area.A else 3),'layouts','incomplete')
        c.require(snapshot.holding.state is c.Holding.EMPTY and
                  not any(o.state is not m.OccupancyState.ACTUAL for o in snapshot.space.occupancies) and
                  not any(i.location in (m.InventoryLocation.UNKNOWN,m.InventoryLocation.HELD,m.InventoryLocation.TEMPORARY) for i in snapshot.inventory),
                  'unresolved','holding/space/inventory unresolved')
        physical(snapshot.holding.evidence,replace(e,target_id=None,action_id=None),now,max_age)
        return c.Fact.CONFIRMED
    except c.ContractError:return c.Fact.UNKNOWN
