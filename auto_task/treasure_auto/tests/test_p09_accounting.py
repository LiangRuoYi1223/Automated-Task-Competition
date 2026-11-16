from publication_support import simulated_run_proposal
import unittest
from dataclasses import replace
from test_p09_support import *
from treasure_auto.mission.config import MissionConfig
from treasure_auto.mission.accounting import arrival_counts
from treasure_auto.mission.transactions import ReleaseCoordinator


class BudgetTests(unittest.TestCase):
    def test_T08_budget_covers_all_stages_and_stop(self):
        cfg=MissionConfig.from_dict(config_data())
        self.assertEqual(cfg.next_order_seconds,12.)  # outbound+return navigation
        self.assertFalse(cfg.can_start(11.999))
        self.assertTrue(cfg.can_start(12.))

    def test_unmeasured_nonfinite_boolean_negative_missing_and_live_rejected(self):
        for name,value in [('navigation_s',None),('grasp_s',float('nan')),('stop_margin_s',0),
                           ('placement_s',True),('recovery_attempts',-1),('origin','live'),('measurement','verified')]:
            with self.subTest(name=name,value=value),self.assertRaises(c.ContractError):
                MissionConfig.from_dict(dict(config_data(),**{name:value}))
        data=config_data();del data['recovery_total_s']
        with self.assertRaises(c.ContractError):MissionConfig.from_dict(data)


class AccountingTests(unittest.TestCase):
    def test_T04_overflow_visible_never_clamped(self):
        ds=tuple(receipt(k,i) for k in m.Category for i in range(1, (2 if k is m.Category.CAKE else 4)))
        snap=replace(empty_snapshot(),deliveries=ds)
        self.assertEqual(arrival_counts(snap).complete,c.Fact.CONFIRMED)
        report=arrival_counts(replace(snap,deliveries=ds+(receipt(m.Category.CAKE,2),)))
        self.assertEqual(report.total,11)
        self.assertEqual(dict(report.counts)[m.Category.CAKE],2)
        self.assertEqual(report.complete,c.Fact.UNKNOWN)
        self.assertTrue(report.review)

    def test_duplicate_physical_alias_is_review_even_with_distinct_keys(self):
        items=tuple(m.InventoryItem('i'+str(i),('same',),c.Fact.CONFIRMED,m.Category.DOLL,m.InventoryLocation.REGION,m.Area.B) for i in range(2))
        self.assertTrue(arrival_counts(replace(empty_snapshot(),inventory=items)).review)

    def test_wrong_run_or_wrong_area_receipt_is_review(self):
        for d in (receipt(m.Category.DOLL,1,run='other'),replace(receipt(m.Category.DOLL,1),area=m.Area.C)):
            self.assertTrue(arrival_counts(replace(empty_snapshot(),deliveries=(d,))).review)


class TransactionTests(unittest.TestCase):
    def setup_case(self):
        before,order,placed,update=release_case()
        clock=FakeClock(s.stamp(111));store=FakeStateStore(before,clock)
        return before,order,placed,update,store,ReleaseCoordinator(store,clock,run_update_provider=simulated_run_proposal)

    def test_T03_release_with_layout_failure_counts_once_and_atomic_empty(self):
        before,order,p,u,store,coord=self.setup_case()
        out=coord.commit_release(order,p,u)
        self.assertEqual(out.receipt.status,m.CommitStatus.COMMITTED,out.receipt.reason)
        self.assertEqual(len(out.snapshot.deliveries),1)
        self.assertEqual(out.snapshot.holding.state,c.Holding.EMPTY)
        self.assertEqual(out.snapshot.layouts[0].confirmed_count,0)
        self.assertEqual(coord.commit_release(order,p,u),out)
        self.assertEqual(len(store.read('sim-run-1').snapshot.deliveries),1)

    def test_missing_unknown_wrong_region_echo_and_old_release_do_not_count(self):
        for mode in ('missing','unknown','wrong','echo','old'):
            before,order,p,u,store,coord=self.setup_case()
            e=p.release
            if mode=='missing':e=None
            if mode=='unknown':e=replace(e,separated=c.Fact.UNKNOWN)
            if mode=='wrong':e=replace(e,area=m.Area.B)
            if mode=='echo':e=replace(e,evidence=(replace(e.evidence[0],kind=c.EvidenceKind.ECHO),))
            if mode=='old':e=replace(e,envelope=replace(e.envelope,control_epoch=0))
            out=coord.commit_release(order,replace(p,release=e),u)
            self.assertEqual(out.receipt.status,m.CommitStatus.REJECTED,mode)
            self.assertEqual(store.read('sim-run-1').snapshot,before)

    def test_commit_failure_does_not_publish_partial_state(self):
        before,order,p,u,store,coord=self.setup_case()
        def reject(tx):return m.CommitReceipt(tx.transaction_id,m.CommitStatus.REJECTED,None,None,c.Reason(c.ErrorCode.STORAGE_FAILURE,'disk','failed'))
        store.commit=reject
        out=coord.commit_release(order,p,u)
        self.assertIsNone(out.snapshot)
        self.assertTrue(coord.blocked)
        self.assertEqual(store.read('sim-run-1').snapshot,before)

    def test_unknown_commit_needs_exact_readback(self):
        before,order,p,u,store,coord=self.setup_case();commit=store.commit
        def unknown(tx):
            commit(tx)
            return m.CommitReceipt(tx.transaction_id,m.CommitStatus.UNKNOWN,None,None,c.Reason(c.ErrorCode.STORAGE_FAILURE,'disk','timeout'))
        store.commit=unknown
        out=coord.commit_release(order,p,u)
        self.assertEqual(out.receipt.status,m.CommitStatus.COMMITTED)
        self.assertEqual(len(out.snapshot.deliveries),1)
        self.assertFalse(coord.blocked)

    def test_unknown_without_readback_locks_pending_transaction(self):
        before,order,p,u,store,coord=self.setup_case()
        def unknown(tx):return m.CommitReceipt(tx.transaction_id,m.CommitStatus.UNKNOWN,None,None,c.Reason(c.ErrorCode.STORAGE_FAILURE,'disk','timeout'))
        store.commit=unknown
        self.assertEqual(coord.commit_release(order,p,u).receipt.status,m.CommitStatus.UNKNOWN)
        self.assertTrue(coord.blocked)
        other=replace(p,release=replace(p.release,envelope=replace(p.release.envelope,event_id='another')))
        self.assertEqual(coord.commit_release(order,other,u).receipt.status,m.CommitStatus.REJECTED)

    def test_same_event_changed_payload_rejected(self):
        _,order,p,u,_,coord=self.setup_case();coord.commit_release(order,p,u)
        out=coord.commit_release(order,replace(p,release=replace(p.release,category=m.Category.DOLL)),u)
        self.assertEqual(out.receipt.status,m.CommitStatus.REJECTED)

    def test_T04_excess_proposal_reported_but_C0_not_bypassed(self):
        before,order,p,u,store,coord=self.setup_case()
        store=FakeStateStore(replace(before,deliveries=(receipt(m.Category.CAKE,99),)),FakeClock(s.stamp(111)))
        coord=ReleaseCoordinator(store,store.clock,run_update_provider=simulated_run_proposal)
        out=coord.commit_release(order,p,u)
        self.assertEqual(out.receipt.status,m.CommitStatus.REJECTED)
        self.assertEqual(dict(out.observed_counts)[m.Category.CAKE],2)
        self.assertTrue(coord.blocked)


if __name__=='__main__':unittest.main()
