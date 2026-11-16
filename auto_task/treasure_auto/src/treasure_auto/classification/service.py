"""Bounded, polling C0 ClassificationPort implementation with fail-closed seams.

resolve_frame and submit_mapping are PRIVATE offline injection seams, not new C0
ports. No production P03/P04/P09 adapter is claimed. The service never moves the
robot or creates P08/P09 state transactions. P10A's independent stop cannot depend
on this polling loop or on a caller-supplied storage callback completing.
"""
import json
import uuid
import hashlib
import threading
from dataclasses import replace
from treasure_auto.contracts import core as c, models as m, rules
from .config import fingerprint
from .geometry import analyze_frame
from .observations import frame_evidence
from .validation import data, validate_frame, validate_saved
from .recovery import ReadProgress


def reason(code, field, detail):
    return c.Reason(code,field,detail)


class ClassificationService:
    def __init__(self, config, template, run, context, clock, *, resolve_frame=None,
                 store=None, submit_mapping=None):
        c.require(type(run) is m.RunContext and type(context) is m.ConsumerContext,
                  'context','C0 run and consumer context required')
        c.require((run.envelope.run_id,run.envelope.instance_id,run.envelope.control_epoch,run.envelope.origin)==
                  (context.run_id,context.instance_id,context.control_epoch,context.origin),
                  'context','run/consumer identity mismatch')
        c.require(context.request_id is not None and context.base_motion_epoch is not None,
                  'context','trusted read identity and observation motion epoch required')
        self.config,self.template,self.run,self.context,self.clock=config,template,run,context,clock
        self.resolve_frame,self.store,self.submit_mapping=resolve_frame,store,submit_mapping
        self.prefix='p05-v1-'+fingerprint(config,template)+'-'
        self.frozen=None
        self.views_used=0
        self.audit=[]
        self._window=[]
        self._seen={}
        self._capture=None
        self._requests={}
        self._read_deadline=None
        self._view_limit=None
        self._pending=None
        self._receipt=None
        self._paused=False
        self._restored=False
        self._terminal=None
        self._last_now=None
        self._output_sequence=0
        self._reading=threading.Lock()
        self._cancelled=threading.Event()

    @property
    def dispatch_allowed(self):
        """Local mapping gate only; never a control/drive permission."""
        try:
            return (self.frozen is not None and not self._paused and not self._restored and
                    not self._reading.locked() and not self._cancelled.is_set() and self._terminal is None and
                    self.run.deadline-self.clock.now()>0 and
                    (self._last_now is None or self.clock.now()-self._last_now>=0))
        except c.ContractError:
            return False

    def _now(self):
        now=self.clock.now()
        rules.remaining_seconds(self.run,now)
        c.require(self._last_now is None or now-self._last_now>=0,'now','monotonic clock regressed',c.ErrorCode.CLOCK_MISMATCH)
        self._last_now=now
        return now

    def _identity(self, request, now):
        current=replace(self.context,now=now)
        # A frozen mapping changes the published mapping/space version. The
        # caller must install that trusted context in a new service at handoff;
        # this read session keeps its original image version fence.
        rules.validate_envelope(request.envelope,current)
        c.require(type(request.max_additional_views) is int and request.max_additional_views>=0,
                  'max_additional_views','nonnegative budget required')
        c.monotonic(request.read_deadline,'read_deadline')
        request.read_deadline-now  # establish identical clock/boot domain
        c.require(self.run.state in (m.RunState.OFFLINE,m.RunState.RUNNING,m.RunState.RECOVERING),
                  'run.state','read session is not active',c.ErrorCode.NOT_AUTHORIZED)
        self.config.require_ready(current.origin); self.template.require_ready(current.origin)
        c.require(current.versions.config==self.config.version,'config.version','wrong current configuration',c.ErrorCode.VERSION_MISMATCH)
        # C0 lacks frozen raw observation/stop/notification semantics. Even a
        # hand-labeled LIVE frame cannot bypass that integration boundary.
        c.require(current.origin is not c.Origin.LIVE,'adapter.live',
                  'P00/P03/P04/P09 production adapters not frozen; offline only',c.ErrorCode.UNAVAILABLE)

    def _result(self,request,status,why=None,view=None):
        now=self.clock.now()
        self._output_sequence+=1
        expiry=replace(now,seconds=now.seconds+1)
        elapsed=0.0
        try:
            if self.run.deadline-now>0:
                expiry=replace(now,seconds=min(now.seconds+1,self.run.deadline.seconds))
            elapsed=max(0,now-self.run.started_at)
        except c.ContractError:
            pass  # report the clock failure without attempting a cross-boot subtraction again
        header=replace(request.envelope,producer=c.Owner.P05,event_id='p05-result-'+uuid.uuid4().hex,
                       sequence=self._output_sequence,emitted_at=now,expires_at=expiry)
        evidence=self.frozen.evidence if self.frozen is not None else ()
        action=c.ActionResult(status,'classification',why,c.Fact.CONFIRMED,c.Fact.UNKNOWN,
                              c.Fact.UNKNOWN,c.Fact.UNKNOWN,evidence,elapsed)
        result=m.ClassificationResult(header,action,self.frozen,view)
        self.audit.append({'type':'result','value':data(result),'dispatch_allowed':self.dispatch_allowed,
                           'read_in_progress':self._reading.locked()})
        return result

    def _current_result(self,request):
        if self._paused or self._restored:
            return self._result(request,c.Outcome.FAILED,reason(c.ErrorCode.CONFLICT,'mapping',
                'pause new dispatch; runtime review required; historical destination retained'))
        return self._result(request,c.Outcome.SUCCESS)

    def _timed_out(self,request,now):
        if self._cancelled.is_set():
            self._terminal=(c.Outcome.CANCELLED,reason(c.ErrorCode.CANCELLED,'read','cancelled during read/commit'))
            return True
        expired=self.run.deadline-now<=0 or (self.frozen is None and
                 self._read_deadline is not None and self._read_deadline-now<=0)
        if expired:
            self._window=[]
            self._terminal=(c.Outcome.TIMED_OUT,reason(c.ErrorCode.DEADLINE,'read_deadline','original read/run budget exhausted'))
        return expired

    def read_mapping(self,request):
        """One bounded input batch; no wait, navigation, model or automatic retry."""
        c.require(type(request) is m.ClassificationRequest,'request','C0 ClassificationRequest required')
        if not self._reading.acquire(blocking=False):
            return self._result(request,c.Outcome.UNKNOWN,reason(c.ErrorCode.CONFLICT,'read','another read is in progress'))
        try:
            return self._read_locked(request)
        finally:
            self._reading.release()

    def _read_locked(self,request):
        try:
            c.require(type(request) is m.ClassificationRequest,'request','C0 ClassificationRequest required')
            now=self._now()
            self._identity(request,now)
            payload=c.dumps(request)
            key=request.envelope.event_id
            cached=self._requests.get(key)
            if cached is not None:
                c.require(cached[0]==payload,'event_id','same event has different content',c.ErrorCode.ID_CONFLICT)
            if self._read_deadline is None:
                self._read_deadline=replace(now,seconds=min(request.read_deadline.seconds,self.run.deadline.seconds))
                self._view_limit=min(request.max_additional_views,self.config.max_additional_views)
            else:
                self._read_deadline=replace(now,seconds=min(self._read_deadline.seconds,request.read_deadline.seconds))
                self._view_limit=max(self.views_used,min(self._view_limit,request.max_additional_views))
            self._timed_out(request,now)
            if self._terminal: return self._result(request,*self._terminal)
            if cached is not None:
                if self.frozen: return self._current_result(request)
                return cached[1] or self._result(request,c.Outcome.UNKNOWN,reason(c.ErrorCode.DUPLICATE,'event_id',
                                                'restored event ignored; never replay an old observation request'))
            if self._pending is not None:
                # A mapping may already have committed even when acknowledgement
                # is unknown. Observe disputes before readback can publish it.
                if request.observations: self._read_batch(request,inspect_only=True)
                result=self._result(request,*self._terminal) if self._terminal else self._reconcile(request)
            else:
                result=self._read_batch(request)
            self._requests[key]=(payload,result)
            return result
        except c.ContractError as error:
            # Rejected foreign/old messages cannot mutate the run, window or mapping.
            return self._result(request,c.Outcome.REJECTED,reason(error.code,error.field,error.detail))

    def _read_batch(self,request,inspect_only=False):
        last_reason=None
        invalid=False
        any_new=False
        for observation in request.observations:
            try:
                c.require(callable(self.resolve_frame),'adapter','P04 full board/pattern adapter missing',c.ErrorCode.UNAVAILABLE)
                frame=self.resolve_frame(observation)
                now=self._now()
                refs=validate_frame(frame,observation,replace(self.context,now=now),self.config,self.template,now)
                captured=frame.image.image.captured_at
                signature=hashlib.sha256(json.dumps(data((frame.image.image.sha256,captured,frame.boards,frame.patterns)),
                    sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()
                keys=(('observation',frame.image.observation_id),('acquisition',frame.acquisition_id),
                      ('evidence',frame.image.image.evidence_id))
                duplicate=False
                for key in keys:
                    if key in self._seen:
                        c.require(self._seen[key]==signature,'frame.identity','id reused with conflicting content',c.ErrorCode.ID_CONFLICT)
                        duplicate=True
                if self._capture is not None and captured-self._capture<=0: duplicate=True
                if duplicate:
                    self.audit.append({'type':'duplicate','observation_id':frame.image.observation_id})
                    continue
                for key in keys: self._seen[key]=signature
                self._capture=captured
                any_new=True
                analysis=analyze_frame(frame,self.template,self.config)
                # C0's abbreviated claims may be cross-checks, never authority.
                if observation.direction is not c.Fact.CONFIRMED:
                    analysis=replace(analysis,valid=False,conflict=observation.direction is c.Fact.REJECTED,
                        reason=reason(c.ErrorCode.CONFLICT,'board.direction','C0 direction is unconfirmed/conflicting'))
                if observation.slots and tuple(sorted(observation.slots,key=lambda a:a.area.value))!=analysis.assignments:
                    analysis=replace(analysis,valid=False,conflict=True,
                        reason=reason(c.ErrorCode.CONFLICT,'slots','C0 slots disagree with actual geometry'))
                self.audit.append({'type':'frame','frame':data(frame),'analysis':data(analysis)})
                immutable=self.frozen or self._pending
                if immutable is not None:
                    old={a.area:a.category for a in immutable.assignments}
                    structural_conflict=(analysis.reason is not None and analysis.reason.code is c.ErrorCode.INVALID_MAPPING and
                                         analysis.reason.field!='slots')
                    if analysis.conflict or structural_conflict or any(old[a.area] is not a.category for a in analysis.assignments):
                        self._paused=True
                    continue
                if not analysis.valid:
                    self._window=[]; invalid=True; last_reason=analysis.reason
                    continue
                if self._window:
                    previous,old_analysis,_=self._window[-1]
                    if (frame.segment_id!=previous.segment_id or frame.capture_not_before!=previous.capture_not_before or
                        frame.image.capture_pose!=previous.image.capture_pose or analysis.assignments!=old_analysis.assignments or
                        captured-previous.image.image.captured_at>self.config.max_confirmation_gap_s):
                        self._window=[]
                        last_reason=reason(c.ErrorCode.CONFLICT,'confirmation','segment/pose/permutation/gap changed; new window')
                self._window.append((frame,analysis,refs))
            except c.ContractError as error:
                self._window=[]; invalid=True
                if (self.frozen or self._pending) is not None and error.code is c.ErrorCode.ID_CONFLICT: self._paused=True
                last_reason=reason(error.code,error.field,error.detail)
                self.audit.append({'type':'rejected_frame','observation':data(observation),'reason':data(last_reason)})
            except Exception as error:
                self._window=[]; invalid=True
                last_reason=reason(c.ErrorCode.UNAVAILABLE,'adapter',type(error).__name__+': '+str(error))
                self.audit.append({'type':'adapter_failure','observation':data(observation),'reason':data(last_reason)})
        now=self._now()
        if self._timed_out(request,now): return self._result(request,*self._terminal)
        if inspect_only: return None
        if self.frozen is not None: return self._current_result(request)
        if self._window:
            first,last=self._window[0][0],self._window[-1][0]
            span=last.image.image.captured_at-first.image.image.captured_at
            if len(self._window)>=self.config.min_confirmations and span>=self.config.min_confirmation_span_s:
                # Revalidate at publication time, including every supporting frame.
                for f,_,refs in self._window:
                    c.require(now-f.image.image.received_at<=self.config.max_frame_age_s and
                              last.sensor_now-f.image.image.captured_at<=self.config.max_frame_age_s,
                              'confirmation','earliest confirmation expired',c.ErrorCode.EXPIRED)
                    for ref in refs: rules.require_physical_evidence(ref,request.envelope,now,self.context.origin)
                return self._prepare(request,now)
        if invalid or not request.observations:
            return self._additional_view(request,last_reason or reason(c.ErrorCode.UNAVAILABLE,'board','no board observation'))
        return self._result(request,c.Outcome.UNKNOWN,last_reason or reason(c.ErrorCode.EVIDENCE_MISSING,'confirmation',
            'waiting for distinct consecutive new captures' if any_new else 'duplicate/old frames do not advance confirmation'))

    def _additional_view(self,request,why):
        if self.views_used>=self._view_limit:
            self._terminal=(c.Outcome.FAILED,reason(c.ErrorCode.LIMIT_EXCEEDED,'additional_views',
                                                  'observation budget exhausted: '+why.detail))
            return self._result(request,*self._terminal)
        self.views_used+=1
        now=self._now()
        header=replace(request.envelope,producer=c.Owner.P05,
                       event_id='p05-view-'+uuid.uuid4().hex,sequence=self.views_used,
                       emitted_at=now,expires_at=self._read_deadline)
        view=m.WaypointQuery(header,self.config.observation_name,'classification_reobserve')
        return self._result(request,c.Outcome.UNKNOWN,why,view)

    def _prepare(self,request,now):
        unique={}
        for _,_,refs in self._window:
            for ref in refs:
                c.require(ref.evidence_id not in unique or unique[ref.evidence_id]==ref,
                          'evidence_id','conflicting freeze evidence',c.ErrorCode.ID_CONFLICT)
                unique[ref.evidence_id]=ref
        mapping_id=self.prefix+uuid.uuid4().hex
        header=replace(request.envelope,producer=c.Owner.P05,event_id=mapping_id,
                       versions=replace(request.envelope.versions,mapping=mapping_id),
                       emitted_at=now,expires_at=self._read_deadline)
        proposal=m.MappingSnapshot(header,mapping_id,self._window[-1][1].assignments,c.Fact.CONFIRMED,now,tuple(unique.values()))
        if not callable(self.submit_mapping) or self.store is None:
            return self._result(request,c.Outcome.UNKNOWN,reason(c.ErrorCode.UNAVAILABLE,'commit',
                                'P09 coordinator/P10B store missing; no frozen mapping published'))
        self._pending=proposal
        audit=json.dumps({'format':'p05-offline-audit-1','contract':c.CONTRACT_VERSION,
                          'configuration_fingerprint':self.prefix,'mapping':c.to_data(proposal),
                          'events':self.audit},sort_keys=True,ensure_ascii=False,allow_nan=False).encode('utf-8')
        try:
            receipt=self.submit_mapping(proposal,audit)
            c.require(type(receipt) is m.CommitReceipt and receipt.transaction_id==mapping_id,
                      'commit.receipt','receipt is not bound to this complete proposal',c.ErrorCode.ID_CONFLICT)
            self._receipt=receipt
        except Exception as error:
            return self._result(request,c.Outcome.UNKNOWN,reason(c.ErrorCode.STORAGE_FAILURE,'commit',
                                'ambiguous submit; query before any retry: '+str(error)))
        if self._timed_out(request,self._now()): return self._result(request,*self._terminal)
        if receipt.status is m.CommitStatus.REJECTED:
            self._terminal=(c.Outcome.FAILED,receipt.reason)
            return self._result(request,*self._terminal)
        if receipt.status is m.CommitStatus.UNKNOWN:
            return self._result(request,c.Outcome.UNKNOWN,receipt.reason)
        return self._reconcile(request)

    def _reconcile(self,request):
        try:
            read=self.store.read(self.context.run_id)
            now=self._now()
            if self._timed_out(request,now): return self._result(request,*self._terminal)
            c.require(type(read) is m.ReadResult and read.status is c.Outcome.SUCCESS,
                      'commit.read','committed state cannot be read',c.ErrorCode.STORAGE_FAILURE)
            mapping=validate_saved(read.snapshot,self.run,self.context,self.prefix,now)
            c.require(mapping==self._pending,'commit.read.mapping','stored mapping differs from complete proposal',c.ErrorCode.CONFLICT)
            if self._receipt is not None and self._receipt.status is m.CommitStatus.COMMITTED:
                c.require(read.snapshot.revision>=self._receipt.revision and read.snapshot.space.version>=self._receipt.space_version,
                          'commit.read.revision','readback predates acknowledgement',c.ErrorCode.STORAGE_FAILURE)
            self.frozen=mapping
            self._pending=None; self._window=[]
            return self._current_result(request)
        except Exception as error:
            return self._result(request,c.Outcome.UNKNOWN,reason(c.ErrorCode.STORAGE_FAILURE,'commit.read',
                                'no confirmed publication; preserve pending proposal: '+str(error)))

    def cancel(self,request):
        try:
            self._identity(request,self._now())
            self._cancelled.set()
            self._terminal=(c.Outcome.CANCELLED,reason(c.ErrorCode.CANCELLED,'read','cancel observation requests; no more retries'))
            self._window=[]
            return self._result(request,*self._terminal)
        except c.ContractError as error:
            return self._result(request,c.Outcome.REJECTED,reason(error.code,error.field,error.detail))

    def export_progress(self):
        """Caller must persist this via P10B; no filesystem writes happen here."""
        c.require(self._reading.acquire(blocking=False),'progress','cannot checkpoint an in-flight call',c.ErrorCode.CONFLICT)
        try:
            return self._export_locked()
        finally:
            self._reading.release()

    def _export_locked(self):
        c.require(self._read_deadline is not None,'progress','no read session to checkpoint')
        return ReadProgress(self.context.run_id,self.context.request_id,self.prefix,self.run.started_at,self.run.deadline,
            self._read_deadline,self.views_used,self._view_limit,self._capture,
            tuple((kind,key,value) for (kind,key),value in sorted(self._seen.items())),
            tuple((key,value[0]) for key,value in sorted(self._requests.items())),self._pending,self._receipt,self._paused,
            self._terminal[0] if self._terminal else None,self._terminal[1] if self._terminal else None,self._last_now)

    def restore(self,progress=None):
        """Restore context only. Runtime review is still required for any new dispatch."""
        c.require(self._reading.acquire(blocking=False),'restore','another read/restore is in progress',c.ErrorCode.CONFLICT)
        try:
            return self._restore_locked(progress)
        finally:
            self._reading.release()

    def _restore_locked(self,progress):
        c.require(self._read_deadline is None and self.frozen is None,'restore','restore only into a fresh service')
        now=self._now()
        self.config.require_ready(self.context.origin); self.template.require_ready(self.context.origin)
        c.require(self.store is not None,'restore','P10B StateStorePort required',c.ErrorCode.UNAVAILABLE)
        read=self.store.read(self.context.run_id)
        c.require(type(read) is m.ReadResult and read.status is c.Outcome.SUCCESS and read.snapshot is not None,
                  'restore','committed same-run state is unavailable',c.ErrorCode.STORAGE_FAILURE)
        mapping=None
        if read.snapshot.mapping is not None:
            mapping=validate_saved(read.snapshot,self.run,self.context,self.prefix,now)
        else:
            c.require(read.snapshot.run.envelope.run_id==self.context.run_id and
                      read.snapshot.run.envelope.instance_id==self.context.instance_id and
                      read.snapshot.run.envelope.control_epoch==self.context.control_epoch and
                      read.snapshot.run.started_at==self.run.started_at and read.snapshot.run.deadline==self.run.deadline,
                      'restore.run','current run/recovery registration mismatch',c.ErrorCode.STALE_RUN)
            c.require(progress is not None,'restore.progress','missing unfinished read budget; do not reset',c.ErrorCode.UNAVAILABLE)
        if progress is not None:
            c.require(type(progress) is ReadProgress,'restore.progress','private P05 progress required')
            c.require((progress.run_id,progress.request_id,progress.fingerprint,progress.started_at,progress.run_deadline)==
                      (self.context.run_id,self.context.request_id,self.prefix,self.run.started_at,self.run.deadline),
                      'restore.progress','run/read/template/original deadline mismatch',c.ErrorCode.VERSION_MISMATCH)
            c.require(progress.last_now is None or now-progress.last_now>=0,'restore.clock','clock continuity lost',c.ErrorCode.CLOCK_MISMATCH)
            c.require(progress.view_limit<=self.config.max_additional_views,'restore.budget','budget exceeds configured limit')
            if progress.pending is not None:
                c.require(progress.pending.envelope.run_id==self.context.run_id and progress.pending.mapping_version.startswith(self.prefix),
                          'restore.pending','unrelated pending proposal')
            self._read_deadline=progress.read_deadline
            self._view_limit=progress.view_limit; self.views_used=progress.views_used
            self._capture=progress.capture_watermark
            self._seen={(kind,key):digest for kind,key,digest in progress.seen}
            self._requests={key:(payload,None) for key,payload in progress.requests}
            self._pending=progress.pending if mapping is None else None
            self._receipt=progress.receipt
            self._paused=progress.paused
            self._terminal=(progress.terminal_status,progress.terminal_reason) if progress.terminal_status else None
        self.frozen=mapping
        self._restored=mapping is not None
        self._window=[]  # never reconstruct a confirmation window from disk
        self.audit.append({'type':'restore','mapping':data(mapping),'window_restored':False,'views_used':self.views_used})
        return mapping
