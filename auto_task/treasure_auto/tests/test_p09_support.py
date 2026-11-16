"""P09 simulation fixtures. No physical parameters or devices."""
import sys
from pathlib import Path
from dataclasses import replace
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from treasure_auto.contracts import core as c, models as m, rules as r
from contract_fixtures import samples as s
from contract_fixtures.fakes import FakeClock, FakeStateStore


def config_data():
    return dict(version='sim-config-v1', origin='simulated', measurement='simulated',
                navigation_s=2., observation_s=1., grasp_s=2., placement_s=2.,
                recovery_s=2., stop_margin_s=1., recovery_attempts=3,
                recovery_total_s=30., observation_attempts=3, evidence_age_s=2.)


def proof(t=110., target=None, action=None, ident='p09-proof'):
    return s.evidence(evidence_id=ident, target_id=target, action_id=action,
                      captured_at=s.stamp(t), received_at=s.stamp(t),
                      valid_until=s.stamp(t+5), uri='sim://p09/'+ident)


def envelope(owner=c.Owner.P09, t=110., **kw):
    return s.envelope(owner, emitted_at=s.stamp(t), expires_at=s.stamp(t+5), **kw)


def empty_snapshot(mapping=True):
    snap=s.state_snapshot()
    v=s.versions() if mapping else replace(s.versions(),mapping=None)
    return replace(snap, space=m.SpaceSnapshot(envelope(c.Owner.P08,versions=v),0,()),
        holding=c.HoldingSnapshot(c.Holding.EMPTY,None,None,(proof(),)),
        inventory=(), deliveries=(), layouts=(), budgets=(),
        mapping=snap.mapping if mapping else None)


def release_case():
    tx=s.release_transaction(); before=s.state_snapshot()
    order=m.WorkOrder(replace(tx.envelope,producer=c.Owner.P09),m.Category.CAKE,m.Area.A,
                      'sim-mapping-v1','inventory-1',m.Budget('recovery-A',3,30,0,0))
    lay=m.LayoutSnapshot(replace(tx.envelope,producer=c.Owner.P08),m.Area.A,m.LayoutKind.V1,0,0,0,(),c.Fact.UNKNOWN,c.Fact.UNKNOWN)
    result=c.ActionResult(c.Outcome.UNKNOWN,'withdraw',c.Reason(c.ErrorCode.UNAVAILABLE,'layout','pending'),
        c.Fact.CONFIRMED,c.Fact.CONFIRMED,c.Fact.UNKNOWN,c.Fact.UNKNOWN,(),0.)
    placed=m.PlacementResult(tx.release_events[0].envelope,result,tx.release_events[0],lay,tx.space,tx.holding,None)
    return before,order,placed,tx.holding_update


def receipt(category, n, run='sim-run-1'):
    ident=category.value+'-'+str(n)
    return m.DeliveryReceipt(m.DeliveryKey(run,ident),'release-'+ident,category,
        m.Area.A if category is m.Category.CAKE else s.mapping_samples()[0].destination(category),
        s.stamp(109),('proof-'+ident,))
