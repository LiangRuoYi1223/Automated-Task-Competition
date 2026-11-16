"""Private read-progress checkpoint. Persist via P10B after P00 freezes its port.

This is not a new public mapping contract. It never contains a confirmation
window or a motion request. C0 records embedded here keep their original types.
"""
import json
from dataclasses import dataclass, fields
from typing import Optional, Tuple
from treasure_auto.contracts import core as c, models as m
from .validation import data


@dataclass(frozen=True)
class ReadProgress:
    run_id: str
    request_id: str
    fingerprint: str
    started_at: c.ClockStamp
    run_deadline: c.ClockStamp
    read_deadline: c.ClockStamp
    views_used: int
    view_limit: int
    capture_watermark: Optional[c.ClockStamp]
    seen: Tuple[Tuple[str,str,str], ...]
    requests: Tuple[Tuple[str,str], ...]
    pending: Optional[m.MappingSnapshot]
    receipt: Optional[m.CommitReceipt]
    paused: bool
    terminal_status: Optional[c.Outcome]
    terminal_reason: Optional[c.Reason]
    last_now: Optional[c.ClockStamp]

    def __post_init__(self):
        for name in ('run_id','request_id','fingerprint'):
            c.require(type(getattr(self,name)) is str and bool(getattr(self,name).strip()),name,'checkpoint identity required')
        c.require(type(self.views_used) is int and type(self.view_limit) is int and
                  0<=self.views_used<=self.view_limit,'views_used','invalid checkpoint budget')
        c.require(type(self.paused) is bool,'paused','explicit pause latch required')
        c.require((self.terminal_status is None)==(self.terminal_reason is None),
                  'terminal','status/reason must be paired')
        c.require(self.terminal_status in (None,c.Outcome.FAILED,c.Outcome.TIMED_OUT,c.Outcome.CANCELLED),
                  'terminal_status','only terminal failure/cancel/timeout can be restored')
        for name,size in (('seen',3),('requests',2)):
            value=getattr(self,name)
            c.require(type(value) is tuple and all(type(v) is tuple and len(v)==size and
                      all(type(x) is str and bool(x) for x in v) for v in value),name,'immutable identity tuples required')
            c.require(len({v[:-1] for v in value})==len(value),name,'duplicate checkpoint identity')
        c.require(all(kind in ('observation','acquisition','evidence') and len(digest)==64 and
                      all(ch in '0123456789abcdef' for ch in digest) for kind,_,digest in self.seen),
                  'seen','invalid deduplication fingerprints')
        c.require(self.run_deadline-self.started_at==600 and self.run_deadline-self.read_deadline>=0,
                  'read_deadline','checkpoint cannot change original deadline')

    def to_json(self):
        return json.dumps({'format':'p05-read-progress-1','contract':c.CONTRACT_VERSION,'progress':data(self)},
                          sort_keys=True,ensure_ascii=False,allow_nan=False)

    @classmethod
    def from_json(cls,payload):
        try:
            raw=json.loads(payload)
            c.require(set(raw)=={'format','contract','progress'} and raw['format']=='p05-read-progress-1' and
                      raw['contract']==c.CONTRACT_VERSION,'checkpoint','unsupported private progress format',c.ErrorCode.VERSION_MISMATCH)
            values=raw['progress']
            c.require(type(values) is dict and set(values)=={f.name for f in fields(cls)},'checkpoint','explicit known fields required')
            for name in ('started_at','run_deadline','read_deadline','capture_watermark','last_now'):
                if values[name] is not None: values[name]=c.from_data(c.ClockStamp,values[name])
            for name,typ in (('pending',m.MappingSnapshot),('receipt',m.CommitReceipt),('terminal_reason',c.Reason)):
                if values[name] is not None: values[name]=c.from_data(typ,values[name])
            if values['terminal_status'] is not None: values['terminal_status']=c.Outcome(values['terminal_status'])
            for name in ('seen','requests'): values[name]=tuple(tuple(v) for v in values[name])
            return cls(**values)
        except (TypeError,KeyError,ValueError) as error:
            if isinstance(error,c.ContractError): raise
            raise c.ContractError(c.ErrorCode.INVALID_FIELD,'checkpoint',str(error)) from error
