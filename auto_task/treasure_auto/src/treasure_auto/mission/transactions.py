"""Coordinate original C0 proposals; only P10B commits and publishes facts."""
from dataclasses import dataclass,replace
from typing import Optional,Tuple
from treasure_auto.contracts import core as c, models as m, rules as r
from .accounting import arrival_counts


@dataclass(frozen=True)
class CoordinationResult:
    receipt: m.CommitReceipt
    snapshot: Optional[m.StateSnapshot]
    observed_counts: Tuple[Tuple[m.Category,int],...]=()


class ReleaseCoordinator:
    def __init__(self,store,clock, *, run_update_provider=None):
        self.store,self.clock=store,clock
        self.run_update_provider=run_update_provider
        self.blocked=False
        self._pending=None
        self._events={}

    def _failure(self,ident,error,status=m.CommitStatus.REJECTED,counts=()):
        self.blocked=True
        reason=c.Reason(error.code,error.field,error.detail)
        return CoordinationResult(m.CommitReceipt(ident,status,None,None,reason),None,counts)

    def _read(self,run_id):
        read=self.store.read(run_id)
        c.require(read.status is c.Outcome.SUCCESS and read.snapshot is not None and read.reason is None,'snapshot','committed state unavailable',c.ErrorCode.STORAGE_FAILURE)
        return read.snapshot

    def commit_release(self,order,placed,holding_update,*,budgets=None,clearances=()):
        ident='p09-release-'+(placed.release.envelope.event_id if placed.release else placed.envelope.event_id)
        counts=()
        try:
            c.require(placed.release is not None,'release','physical release event required',c.ErrorCode.EVIDENCE_MISSING)
            event=placed.release
            digest=(c.dumps(order),c.dumps(placed),c.dumps(holding_update),budgets,clearances)
            prior=self._events.get(event.envelope.event_id)
            if prior:
                c.require(prior[0]==digest,'event_id','same event changed payload',c.ErrorCode.ID_CONFLICT)
                return prior[1]
            if self._pending is not None:
                c.require(self._pending[0]==digest,'pending','unknown commit must be reconciled first',c.ErrorCode.STORAGE_FAILURE)
                return self.reconcile()
            c.require(not self.blocked,'review','previous rejection requires explicit external reconciliation',c.ErrorCode.CONFLICT)
            previous=self._read(event.envelope.run_id)
            r.require_simulated(previous);r.require_simulated(placed);r.require_simulated(order)
            c.require(not arrival_counts(previous).review,'inventory','existing ledger requires review',c.ErrorCode.CONFLICT)
            c.require(event.envelope.request_id==order.envelope.request_id and
                      event.envelope.target_id==order.envelope.target_id and
                      event.delivery_key.inventory_id==order.inventory_id and
                      event.category is order.category and event.area is order.destination and
                      event.mapping_version==order.mapping_version,'order','release must bind the original work order',c.ErrorCode.STALE_REQUEST)
            item=next((i for i in previous.inventory if i.inventory_id==order.inventory_id),None)
            c.require(item is not None and event.envelope.target_id in item.target_ids,'inventory','unknown physical identity',c.ErrorCode.UNKNOWN_IDENTITY)
            inventory=tuple(replace(i,location=m.InventoryLocation.REGION,area=event.area) if i.inventory_id==item.inventory_id else i for i in previous.inventory)
            layouts=tuple(l for l in previous.layouts if l.area is not placed.layout.area)+(placed.layout,)
            now=self.clock.now()
            e=replace(event.envelope,producer=c.Owner.P09,event_id=ident,emitted_at=now)
            c.require(callable(self.run_update_provider),'run_update',
                      'P10A run version proposal adapter required',c.ErrorCode.UNCONFIGURED)
            run_update=self.run_update_provider(previous.run,placed.proposed_space)
            c.require(type(run_update) is m.RunContext and run_update.envelope.producer is c.Owner.P10A and
                      run_update.envelope.versions==placed.proposed_space.envelope.versions,
                      'run_update','P10A proposal must bind the published spatial version',c.ErrorCode.VERSION_MISMATCH)
            now=self.clock.now()
            e=replace(e,emitted_at=now)
            tx=m.StateTransaction(e,ident,previous.revision,previous.space.version,placed.proposed_space,
                placed.proposed_holding,inventory,layouts,previous.mapping,(event,),tuple(clearances),holding_update,
                previous.budgets if budgets is None else tuple(budgets),run_update)
            counts=dict(arrival_counts(previous).counts)
            if event.delivery_key not in {d.key for d in previous.deliveries}:counts[event.category]+=1
            counts=tuple(counts.items())
            expected=r.validate_transaction(tx,previous,now)
            self._pending=(digest,tx,expected,counts)
            try:receipt=self.store.commit(tx)
            except Exception as error:
                return self._failure(ident,c.ContractError(c.ErrorCode.STORAGE_FAILURE,'commit',str(error)),m.CommitStatus.UNKNOWN,counts)
            if receipt.status is m.CommitStatus.REJECTED:
                self._pending=None;self.blocked=True
                return CoordinationResult(receipt,None,counts)
            return self.reconcile()
        except c.ContractError as error:return self._failure(ident,error,counts=counts)
        except Exception as error:return self._failure(ident,c.ContractError(c.ErrorCode.STORAGE_FAILURE,'coordinator',str(error)),m.CommitStatus.UNKNOWN,counts)

    def reconcile(self):
        c.require(self._pending is not None,'pending','no transaction to reconcile')
        digest,tx,deliveries,counts=self._pending
        try:
            snap=self._read(tx.envelope.run_id)
            c.require(snap.run==tx.run_update and snap.revision==tx.expected_revision+1 and snap.space==tx.space and
                      snap.holding==tx.holding and snap.inventory==tx.inventory and snap.layouts==tx.layouts and
                      snap.mapping==tx.mapping and snap.budgets==tx.budgets and snap.deliveries==deliveries,
                      'readback','exact atomic state not confirmed; do not retry physical action',c.ErrorCode.STORAGE_FAILURE)
            receipt=m.CommitReceipt(tx.transaction_id,m.CommitStatus.COMMITTED,snap.revision,snap.space.version,None)
            out=CoordinationResult(receipt,snap,counts)
            self._events[tx.release_events[0].envelope.event_id]=(digest,out)
            self._pending=None;self.blocked=False
            return out
        except Exception as error:
            if not isinstance(error,c.ContractError):error=c.ContractError(c.ErrorCode.STORAGE_FAILURE,'readback',str(error))
            return self._failure(tx.transaction_id,error,m.CommitStatus.UNKNOWN,counts)
