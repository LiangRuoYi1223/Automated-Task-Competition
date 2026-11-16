"""Synthetic field-policy scenarios through the actual MissionEngine entry.

All map coordinates, timestamps, observations and physical snapshots here are
explicit SIMULATED fixtures. No devices, transport or live camera is created.
"""
from dataclasses import replace
from treasure_auto.contracts import core as c, models as m
from treasure_auto.mission.simulator import SimulatedWorld
from treasure_auto.mission.engine import MissionEngine
from treasure_auto.mission.checkpoint import SimulationJournal
from treasure_auto.mission.field_strategy import FieldInput, FieldStrategyController
from treasure_auto.navigation.field_regions import FieldRegions
from treasure_auto.navigation.controlled_push import PushLimits, PushPlan, PushObservation
from treasure_auto.manipulation.strategy import HandlingObservation


FIELD_SCENARIOS = ('field_push', 'field_stall', 'field_budget', 'field_regions', 'field_recenter')


def _box(x,y,w=.1,h=.1):
    return tuple(m.Point2(a,b) for a,b in ((x,y),(x+w,y),(x+w,y+h),(x,y+h)))


def _controller():
    positions=(('A',1,1),('B',8,1),('C',1,8),('D',8,8))
    regions=FieldRegions('map',_box(0,0,10,10),_box(3,3,4,4),
        {a:_box(x,y,1,1) for a,x,y in positions},{a:_box(x,y,.45,.45) for a,x,y in positions},
        (_box(6,6,.3,.3),),(_box(7,5,.1,1),))
    # Test vectors only. The shipped unmeasured configuration has None instead.
    limits=PushLimits(.2,2.,.1,.5,.5,.005,.001,.001,.2,.5)
    return FieldStrategyController(regions,limits)


def _plan():
    return PushPlan('pile','push-1','map',('target-0','target-1'),_box(4,4,.3,.3),
        _box(3.5,4,.5,.3),_box(4,4,.6,.3),_box(3.3,4,.7,.3),.01,(1.,0.),.2,2.,.1)


def _observation(t,**changes):
    values=dict(sample_id='field-'+str(t),captured_at=t,frame_id='map',group_id='pile',
        affected_ids=('target-0','target-1'),fixed_position_xy=(3.5,4.),wheel_distance_m=0.,
        linear_m_s=0.,angular_rad_s=0.,body_footprint=_box(3.5,4,.2,.3),brick_footprint=_box(4,4,.3,.3),
        localization_valid=True,observation_valid=True,space_confirmed=True,affected_visible=True,
        affected_stable=True,graspable_count=0,old_action_stopped=True,queue_empty=True,stable_since=t-1.)
    values.update(changes)
    return PushObservation(**values)


def run_field_replay(layout,scenario,config=None):
    world=SimulatedWorld(layout)
    if config is not None:
        c.require(config.version==world.engine.config.version,'config','simulation configuration differs')
        world.engine=MissionEngine(config,world.clock,SimulationJournal())
    controller=_controller();world.engine.field_strategy=controller
    actions=[];stops=[];world.engine.stop_sink=stops.append

    def advance(frame,field_input=None):
        out=world.engine.advance(replace(frame,field_input=field_input))
        if out.field_decision is not None:
            decision=out.field_decision
            actions.append(dict(action=decision.action,reason=decision.reason,
                                at_s=world.clock.seconds,target_id=decision.target_id))
        return out

    out=advance(world.frame())
    # Finish actual scheduler navigation/read-board stages using synthetic domain results.
    for _ in range(2):
        world.clock.advance(.1)
        out=advance(world.complete(out.directive))
    navigation=out.directive
    world.clock.advance(.1)
    request=FieldInput('select','target-0','map',_box(4,4,.3,.3),.01,
        abnormal=True,reliable_grasp=True,push_plan=_plan(),push_observation=_observation(world.clock.seconds))
    if scenario in ('field_regions','field_recenter'):
        request=replace(request,abnormal=False,normal_known_layout=True)
    out=advance(world.complete(navigation),request)
    started=world.clock.seconds

    if scenario=='field_regions':
        for polygon,abnormal,normal in ((_box(2.5,2.5),False,False),(_box(1.7,1.7),True,False)):
            world.clock.advance(.1)
            out=advance(world.frame(),replace(request,footprint=polygon,abnormal=abnormal,normal_known_layout=normal))
    elif scenario=='field_budget':
        world.clock.advance(30.)
        out=advance(world.frame(),FieldInput('tick','target-0','map'))
    elif scenario in ('field_push','field_stall'):
        for n in range(1,12):
            elapsed=n*.2
            world.clock.seconds=started+elapsed
            dx=0. if scenario=='field_stall' else min(.2,elapsed*.1)
            observation=_observation(world.clock.seconds,fixed_position_xy=(3.5+dx,4.),
                wheel_distance_m=elapsed*.1,linear_m_s=.1,old_action_stopped=False,queue_empty=False,
                affected_stable=False,stable_since=None,body_footprint=_box(3.5+dx,4.,.2,.3))
            out=advance(world.frame(),FieldInput('push_observation','target-0','map',push_observation=observation))
            if out.field_decision is None or out.field_decision.action=='stop':break
        if scenario=='field_push' and out.field_decision is not None:
            stopped=world.clock.seconds
            for dt in (.1,.32,.9):
                world.clock.seconds=stopped+dt
                observation=_observation(world.clock.seconds,fixed_position_xy=(3.7,4.),
                    body_footprint=_box(3.7,4.,.2,.3),stable_since=stopped+.1,graspable_count=1)
                out=advance(world.frame(),FieldInput('push_observation','target-0','map',push_observation=observation))
    elif scenario=='field_recenter':
        offset=c.ArmPoint('tool',12.,2.,0.,'mm',3.)
        target='target-0'

        def handling(phase,number,**changes):
            world.clock.advance(.1)
            observation=HandlingObservation(target,'field-handling-'+str(world.clock.seconds),world.clock.seconds,
                phase,world.snapshot.holding.state,visible=c.Fact.CONFIRMED,neighbors_stable=c.Fact.CONFIRMED,
                action_id='recenter-'+str(number),**changes)
            return advance(world.frame(),FieldInput('handling',target,'map',handling_observation=observation))

        for number in range(1,4):
            # This is the fake world's committed physical release, not a policy-produced receipt.
            item=next(i for i in world.snapshot.inventory if target in i.target_ids)
            delivery=m.DeliveryReceipt(m.DeliveryKey(world.run.envelope.run_id,item.inventory_id),
                'temporary-release',m.Category.CAKE,m.Area.A,world.clock.now(),('sim-release-proof',))
            occupancy=m.Occupancy(target,m.OccupancyState.ACTUAL,'map',_box(1.7,1.7),0.,.1,.001,(),
                world.clock.now(),m.Visibility.VISIBLE,m.Area.A,(world.proof(target),))
            world.update(holding=c.HoldingSnapshot(c.Holding.EMPTY,None,None,(world.proof(),)),
                inventory=tuple(replace(i,location=m.InventoryLocation.REGION,area=m.Area.A) if target in i.target_ids else i
                                for i in world.snapshot.inventory),
                deliveries=world.snapshot.deliveries or (delivery,),
                space=replace(world.snapshot.space,occupancies=(occupancy,)))
            handling('temporary_release',number)
            handling('recenter_start',number)
            world.update(holding=c.HoldingSnapshot(c.Holding.HELD,target,offset,(world.proof(target),)))
            out=handling('recenter_result',number,independent_lift=c.Fact.CONFIRMED,following=c.Fact.CONFIRMED,
                         offset_verified=c.Fact.REJECTED,measured_offset=offset)

    state=controller.checkpoint(world.clock.seconds)
    flow_issues=list(out.issues)
    world.clock.seconds=world.run.deadline.seconds
    deadline=advance(world.frame())
    return dict(origin='simulated',scenario=scenario,layout=layout,hardware_outputs=0,field_actions=actions,
        field_state=state,issues=flow_issues,trace=world.engine.trace,stop_requests=len(stops),
        arrivals=len(world.snapshot.deliveries),delivery_keys=[d.key.inventory_id for d in world.snapshot.deliveries],
        elapsed_s=600.,original_deadline_s=world.run.deadline.seconds,
        deadline_probe=dict(issued=deadline.directive is not None or deadline.field_decision is not None,
                            phase=deadline.decision.phase.value))


def validate_field_replay(result):
    failures=[]
    def check(ok,message):
        if not ok:failures.append(message)
    actions=[item['action'] for item in result['field_actions']]
    check(result['origin']=='simulated' and result['hardware_outputs']==0,'offline field boundary')
    check(not result['deadline_probe']['issued'] and result['deadline_probe']['phase']=='stopped','original deadline stops field actions')
    check(not result['issues'],'field integration rejected a required observation')
    scenario=result['scenario']
    if scenario=='field_push':check(actions.count('push_step')==1 and actions[-1]=='grasp','one bounded push then fresh grasp readiness')
    elif scenario=='field_stall':check(actions.count('push_step')==1 and actions[-1]=='stop','fixed-environment stall stops despite wheel progress')
    elif scenario=='field_budget':
        check(actions[-1]=='stop','thirty second recovery cutoff')
        check(result['field_state']['push']['groups']['pile']['elapsed_s']>=30.,'observation time charged')
    elif scenario=='field_regions':check(actions==['grasp','grasp','arm_recovery'],'opening/outside/whole-A priorities')
    elif scenario=='field_recenter':
        check(actions.count('recenter')==3 and actions[-1]=='approximate_place','third recenter continues approximate placement')
        state=result['field_state']['handling']['targets']['target-0']
        check(state['bias'] and state['recenter_attempts']==3,'bias and budget retained')
        check(result['arrivals']==1 and len(set(result['delivery_keys']))==1,'temporary regrasp does not double delivery')
    else:check(False,'unknown field scenario')
    return tuple(failures)
