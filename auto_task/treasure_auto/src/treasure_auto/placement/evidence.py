"""Release separation and continuous static-scene evidence from metric frames."""
import math
from dataclasses import replace
from treasure_auto.contracts import core as c, models as m, rules as r
from treasure_auto.geometry import planar
from . import guards
from .targets import whole_in_region


def distance(a,b):return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))


def motion(a,b):
    c.require(a.target_id==b.target_id and a.frame_id==b.frame_id and len(a.footprint_m)==len(b.footprint_m),
              'geometry','incompatible object geometry')
    # Corresponding ordered corners are required; orientation changes are also
    # independently checked. Reordering conservatively restarts the window.
    return max([math.hypot(p.x-q.x,p.y-q.y) for p,q in zip(a.footprint_m,b.footprint_m)]+
               [abs(a.z_min_m-b.z_min_m),abs(a.z_max_m-b.z_max_m)])+a.error_m+b.error_m


def angle(a,b):return abs((a-b+math.pi)%(2*math.pi)-math.pi)


class _Frames:
    def __init__(self,config):
        self.config=config;self.frames=[];self.seen_ids=set();self.seen_uris=set();self.seen_captures=set();self.last_reason=None
        self.barrier=None;self.max_adjacent=0.;self.max_total=0.

    def _new(self,sample,ctx):
        guards.frame(sample,ctx,self.config)
        ev=sample.evidence;cap=ev.captured_at
        c.require(self.barrier is None or cap-self.barrier>0,'captured_at','capture predates invalidation barrier')
        c.require(sample.observation_id not in self.seen_ids and ev.evidence_id not in self.seen_ids and
                  ev.uri not in self.seen_uris and cap not in self.seen_captures,'frame','duplicate exposure',c.ErrorCode.DUPLICATE)
        self.seen_ids.update((sample.observation_id,ev.evidence_id));self.seen_uris.add(ev.uri);self.seen_captures.add(cap)
        if self.frames:
            prev=self.frames[-1]
            c.require(sample.envelope.sequence>prev.envelope.sequence,'sequence','out-of-order frame')
            c.require(0<cap-prev.evidence.captured_at<=self.config.frame_gap_s,'capture_gap','missing/out-of-order frames')
        return cap

    def _reset(self,error,now):
        self.frames=[];self.last_reason=c.Reason(error.code,error.field,error.detail)
        self.barrier=now;self.max_adjacent=0.;self.max_total=0.


class ReleaseWindow(_Frames):
    def __init__(self,request,config,started_at):
        super().__init__(config);self.request=request;self.started_at=started_at;self.contact=None;self.separated=[];self.event=None

    def push(self,sample,ctx,run):
        if self.event is not None:return self.event  # immutable historical event, never refreshed
        try:
            cap=self._new(sample,ctx)
            r.validate_envelope(self.request.envelope,ctx)
            c.require(run.envelope.run_id==ctx.run_id and cap-self.started_at>=0 and run.deadline-cap>0 and run.deadline-ctx.now>0,
                      'deadline','release must be observed before original cutoff',c.ErrorCode.DEADLINE)
            guards.physical((sample.evidence,),self.request.envelope,ctx.now,self.config)
            obj=next((o for o in sample.objects if o.target_id==self.request.envelope.target_id),None)
            c.require(obj is not None and sample.cup_xyz_m is not None,'release','same brick and cup must both be visible',c.ErrorCode.EVIDENCE_MISSING)
            c.require(obj.area is self.request.region.area and whole_in_region(obj,self.request.region),'region','brick not fully in correct region')
            c.require(sample.frame_id==self.request.region.frame_id,'frame','cup and brick must share map coordinates')
            if self.contact is None:
                cup=sample.cup_xyz_m
                c.require(abs(cup[2]-obj.z_max_m)+sample.cup_error_m+obj.error_m<=self.config.height_tolerance_m and
                          planar.clearance(guards.polygon(obj.footprint_m),cup[:2])>=0,
                          'contact','pre-withdraw cup/brick contact frame required',c.ErrorCode.EVIDENCE_MISSING)
                self.contact=sample;self.frames.append(sample);return None
            initial=next(o for o in self.contact.objects if o.target_id==obj.target_id)
            c.require(sample.scene_epoch==self.contact.scene_epoch,'scene_epoch','scene changed during release')
            c.require(motion(initial,obj)<=self.config.brick_motion_m,'brick_motion','brick moved with cup or slipped')
            original_orientation=dict((i,(y,t)) for i,y,t in self.contact.orientations)[obj.target_id]
            current_orientation=dict((i,(y,t)) for i,y,t in sample.orientations)[obj.target_id]
            c.require(all(angle(a,b)<=self.config.still_angle_rad for a,b in zip(original_orientation,current_orientation)),
                      'brick_orientation','brick rotated/tilted during withdrawal')
            cup=sample.cup_xyz_m
            separated=(cup[2]-obj.z_max_m-sample.cup_error_m-obj.error_m>=self.config.cup_retreat_m and
                       distance(cup,self.contact.cup_xyz_m)-sample.cup_error_m-self.contact.cup_error_m>=self.config.cup_retreat_m)
            self.frames.append(sample)
            if not separated:self.separated=[];return None
            self.separated.append(sample)
            if cap-self.separated[0].evidence.captured_at<self.config.release_window_s:return None
            evidence=tuple(f.evidence for f in [self.contact]+self.separated)
            for ev in evidence:guards.physical((ev,),self.request.envelope,ctx.now,self.config)
            e=guards.next_envelope(self.request.envelope,ctx.now,'release')
            e=replace(e,expires_at=min([e.expires_at]+[x.valid_until for x in evidence],key=lambda x:x.seconds))
            self.event=m.ReleaseEvent(e,m.DeliveryKey(ctx.run_id,self.request.order.inventory_id),self.request.order.category,
                self.request.region.area,self.request.mapping.mapping_version,self.separated[0].evidence.captured_at,
                c.Fact.CONFIRMED,c.Fact.CONFIRMED,evidence)
            return self.event
        except c.ContractError as error:
            self._reset(error,ctx.now);self.contact=None;self.separated=[];return None


class StillnessWindow(_Frames):
    def __init__(self,config,required_targets,coverage_polygon,frame_id,scene_epoch,stop,ctx):
        super().__init__(config)
        c.require(bool(required_targets) and len(set(required_targets))==len(required_targets),'targets','explicit unique affected inventory required')
        guards.polygon(coverage_polygon);guards.stopped(stop,ctx,config)
        self.required=tuple(required_targets);self.coverage=coverage_polygon;self.frame_id=frame_id
        self.scene_epoch=scene_epoch;self.stop=stop;self.initial_context=ctx;self.certificate=None

    def push(self,sample,ctx):
        self.certificate=None
        try:
            c.require((ctx.run_id,ctx.instance_id,ctx.control_epoch,ctx.versions,ctx.base_motion_epoch,ctx.layout_recovery_round,ctx.rebuild_generation)==
                      (self.initial_context.run_id,self.initial_context.instance_id,self.initial_context.control_epoch,self.initial_context.versions,
                       self.initial_context.base_motion_epoch,self.initial_context.layout_recovery_round,self.initial_context.rebuild_generation),
                      'context','scene version changed',c.ErrorCode.STALE_GEOMETRY)
            guards.stopped(self.stop,ctx,self.config)
            cap=self._new(sample,ctx)
            c.require(cap-self.stop.envelope.emitted_at>0 and all(cap-e.captured_at>0 for e in self.stop.evidence),
                      'captured_at','window begins only after actual old-queue stop')
            c.require(sample.scene_epoch==self.scene_epoch and sample.frame_id==self.frame_id,'scene','arm/scene/frame changed')
            poly=guards.polygon(sample.coverage_polygon_m)
            c.require(all(planar.clearance(poly,(p.x,p.y))>=0 for p in self.coverage),'coverage','missing scattered range',c.ErrorCode.EVIDENCE_MISSING)
            objects={o.target_id:o for o in sample.objects}
            c.require(set(self.required)==objects.keys(),'coverage','missing or unaccounted related brick',c.ErrorCode.UNKNOWN_IDENTITY)
            for target in self.required:
                c.require(all(planar.clearance(poly,(p.x,p.y))>=objects[target].error_m for p in objects[target].footprint_m),
                          'coverage','object uncertainty outside visible range')
            adjacent=total=0.
            for old in ([self.frames[0],self.frames[-1]] if self.frames else []):
                old_items={o.target_id:o for o in old.objects};old_angles={i:(a,t) for i,a,t in old.orientations}
                new_angles={i:(a,t) for i,a,t in sample.orientations}
                maximum=0.
                for target in self.required:
                    a,b=old_items[target],objects[target]
                    c.require(a.support_ids==b.support_ids,'support','support changed')
                    maximum=max(maximum,motion(a,b))
                    c.require(all(angle(a,b)<=self.config.still_angle_rad for a,b in zip(old_angles[target],new_angles[target])),
                              'orientation','rotation/tilt exceeded tolerance')
                c.require(maximum<=self.config.still_translation_m,'motion','adjacent or cumulative drift exceeded tolerance')
                if old is self.frames[0]:total=max(total,maximum)
                if old is self.frames[-1]:adjacent=max(adjacent,maximum)
            self.frames.append(sample)
            self.max_adjacent=max(self.max_adjacent,adjacent);self.max_total=max(self.max_total,total)
            if cap-self.frames[0].evidence.captured_at<.5:return None
            refs=tuple(f.evidence for f in self.frames)
            for ev in refs:guards.physical((ev,),replace(sample.envelope,target_id=None,action_id=None),ctx.now,self.config)
            expires=min([replace(ctx.now,seconds=ctx.now.seconds+self.config.certificate_ttl_s),self.stop.envelope.expires_at,
                         sample.envelope.expires_at]+[e.valid_until for e in refs],key=lambda t:t.seconds)
            e=replace(guards.next_envelope(sample.envelope,ctx.now,'stillness'),expires_at=expires,action_id=None,target_id=None)
            self.certificate=m.StillnessCertificate(e,self.required,self.coverage,self.frame_id,self.frames[0].evidence.captured_at,cap,
                tuple(f.observation_id for f in self.frames),c.Fact.CONFIRMED,c.Fact.CONFIRMED,self.max_adjacent,self.max_total,
                self.config.still_translation_m,self.scene_epoch,refs)
            r.validate_stillness(self.certificate,e,ctx.now,self.required,self.scene_epoch)
            return self.certificate
        except c.ContractError as error:
            self._reset(error,ctx.now);return None
