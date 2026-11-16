"""Finite recovery decisions; inventory and cumulative budgets remain P09 facts."""
from treasure_auto.contracts import core as c, models as m, rules as r
from . import guards
from .inputs import RecoveryDecision,TemporaryPlacement
from .targets import whole_in_region


def recovery_step(target_id,inventory,holding,space,temporary,ctx,config,budget,deadline):
    def result(op,target=None,field='recovery'):
        return RecoveryDecision(op,target,c.Reason(c.ErrorCode.OCCUPANCY_UNRESOLVED,field,op))
    if deadline-ctx.now<=0:return result('deadline')
    if (budget.attempts_limit is None or budget.seconds_limit is None or
        budget.attempts_used>=budget.attempts_limit or budget.seconds_used>=budget.seconds_limit):return result('budget_exhausted')
    if holding.state is not c.Holding.EMPTY:return result('resolve_holding',holding.target_id,'holding')
    aliases=[t for i in inventory for t in i.target_ids]
    if len(set(aliases))!=len(aliases) or any(i.identity_confirmed is not c.Fact.CONFIRMED or i.location is m.InventoryLocation.UNKNOWN for i in inventory):
        return result('reconcile_inventory')
    item=next((i for i in inventory if target_id in i.target_ids),None)
    if item is None:return result('reconcile_inventory')
    if item.location is m.InventoryLocation.HELD:return result('resolve_holding',target_id)
    if item.location is m.InventoryLocation.CENTRAL:return RecoveryDecision('fetch_central',target_id,None)
    objects={o.target_id:o for o in space.occupancies}
    if target_id not in objects:return result('reconcile_inventory')
    try:
        r.require_space_version(space.envelope,space)
        for o in objects.values():
            if o.state is not m.OccupancyState.RESERVED:guards.actual(o,space.envelope,ctx.now,config)
        # Follow loads before choosing a removable top. Cycles are unresolved,
        # never a reason to choose an arbitrary bottom brick.
        def top(target,visited):
            c.require(target not in visited,'support','cyclic support graph')
            children=[o for o in objects.values() if target in o.support_ids]
            if not children:return objects[target]
            leaves=[top(o.target_id,visited|{target}) for o in children]
            return max(leaves,key=lambda o:(o.z_max_m,o.target_id))
        selected=top(target_id,set())
        if selected.target_id!=target_id:
            if temporary is None:return result('await_temporary')
            c.require(type(temporary) is TemporaryPlacement,'temporary','explicit P06 target/offset/reservation binding required',c.ErrorCode.UNAVAILABLE)
            reservation,plan,binding=temporary.reservation,temporary.plan,temporary.binding
            c.require(objects.get(reservation.target_id)==reservation and reservation.state is m.OccupancyState.RESERVED,
                      'temporary','temporary place must be in committed snapshot')
            r.validate_envelope(plan.envelope,guards.generic(ctx))
            c.require(plan.envelope.target_id==selected.target_id and binding.path_id==plan.path_id and binding.target==reservation and
                      binding.base_pose.frame_id==reservation.frame_id and whole_in_region(reservation,temporary.region),
                      'temporary.binding','route must bind selected top, measured offset and entire reserved destination')
            c.require(binding.held_offset is not None,'temporary.offset','held offset required for route certification')
            c.require(plan.envelope.producer is c.Owner.P06 and plan.reachable is c.Fact.CONFIRMED and plan.collision_free is c.Fact.CONFIRMED and
                      any(s.phase is m.Phase.TEMPORARY for s in plan.segments),'temporary','certified temporary route required')
            guards.physical(plan.evidence,plan.envelope,ctx.now,config)
            guards.physical(binding.localization_evidence,plan.envelope,ctx.now,config)
            return RecoveryDecision('dismantle',selected.target_id,None)
        return RecoveryDecision('reposition',target_id,None)
    except c.ContractError as error:
        return RecoveryDecision('unresolved',None,c.Reason(error.code,error.field,error.detail))
