"""One run-scoped A-exit proposal; P09/P10B own durable acceptance and score."""
import hashlib
import math
from dataclasses import replace
from treasure_auto.contracts import core as c, models as m
from treasure_auto.mapping.geometry import transform_polygon,collides,validate_polygon
from .observations import validate_localization,validate_readiness,reason


class ExitTracker:
    def __init__(self,run,metadata,profile,max_age_s,committed=None):
        self.run,self.metadata,self.profile,self.max_age_s=run,metadata,profile,max_age_s
        self.key='p03:exit:A:'+hashlib.sha256(run.envelope.run_id.encode('utf-8')).hexdigest()
        self.pending=None;self.reason=None;self.committed=False
        if committed is not None:
            c.require(committed.envelope.run_id==run.envelope.run_id and committed.envelope.origin is run.envelope.origin and
                      committed.envelope.producer is c.Owner.P03 and committed.envelope.event_id==self.key and
                      committed.area is m.Area.A and committed.entire_envelope_outside is c.Fact.CONFIRMED and
                      bool(committed.evidence),'committed_exit','wrong/invalid historical exit',c.ErrorCode.ID_CONFLICT)
            self.committed=True

    def observe(self,localization,readiness,now):
        if self.committed or self.pending is not None:return None
        try:
            e=replace(self.run.envelope,versions=replace(self.run.envelope.versions,space=localization.envelope.versions.space),
                      base_motion_epoch=localization.base_motion_epoch)
            c.require(e.versions.map==self.metadata.map_version and self.metadata.measurement is m.Measurement.SIMULATED,
                      'map','matching configured simulated map required',c.ErrorCode.UNCONFIGURED)
            c.require(now-self.run.started_at>=0 and self.run.deadline-now>0,'deadline','outside run',c.ErrorCode.DEADLINE)
            validate_localization(localization,e,now,self.max_age_s)
            validate_readiness(readiness,e,now,self.max_age_s)
            regions=[r for r in self.metadata.regions if r.area is m.Area.A]
            c.require(len(regions)==1,'area.A','one A boundary required',c.ErrorCode.UNCONFIGURED)
            region=regions[0];validate_polygon(region.polygon_m)
            pose=localization.pose
            c.require(region.frame_id==pose.frame_id,'frame','boundary/pose frame mismatch')
            for shape in self.profile.shapes(readiness.holding.state):
                radius=max(math.hypot(p.x,p.y) for p in shape)
                margin=self.profile.error_m+pose.error_m+region.error_m
                margin+=2*radius*math.sin(min(math.pi,pose.error_rad)/2)
                c.require(not collides(transform_polygon(shape,pose),region.polygon_m,margin),
                          'entire_envelope','tail/arm/load/error still touches A',c.ErrorCode.CONFLICT)
            evidence=localization.evidence+readiness.evidence+readiness.holding.evidence
            out=replace(e,producer=c.Owner.P03,event_id=self.key,request_id=None,action_id=None,target_id=None,
                        emitted_at=now,expires_at=min((p.valid_until for p in evidence),key=lambda t:t.seconds))
            self.pending=m.ExitEvidence(out,m.Area.A,c.Fact.CONFIRMED,evidence)
            self.reason=None
            return self.pending
        except c.ContractError as error:self.reason=reason(error);return None
