import unittest
from dataclasses import replace
from test_p10a_support import *


def mapping_for(run, clock):
    sample=s.mapping_samples()[0]
    ev=s.evidence(run_id=run.envelope.run_id,target_id=None,action_id=None,
                  received_at=clock.now(),valid_until=replace(clock.now(),seconds=clock.now().seconds+10))
    env=replace(run.envelope,producer=c.Owner.P05,emitted_at=clock.now(),
                expires_at=replace(clock.now(),seconds=clock.now().seconds+10))
    return replace(sample,envelope=env,mapping_version=env.versions.mapping,frozen_at=clock.now(),evidence=(ev,))


class UpdateTests(unittest.TestCase):
    def test_scene_notification_cannot_adopt_uncommitted_versions_or_change_config(self):
        rt,clock=started(); req=request(rt,clock)
        space=replace(req.space,version=1,envelope=replace(req.space.envelope,versions=replace(req.space.envelope.versions,space=1)))
        rt.scene_changed(space,1,('brick-1',))
        self.assertEqual(rt.run.envelope.versions.space,0)
        bad=replace(space,envelope=replace(space.envelope,versions=replace(space.envelope.versions,config='injected')))
        with self.assertRaises(c.ContractError):rt.scene_changed(bad,2,('brick-1',))

    def test_committed_mapping_snapshot_enables_central_work_and_invalidates_old_permit(self):
        rt,clock=started(mapping=False); req=request(rt,clock)
        permit=rt.request_permission(req).permit
        versions=replace(rt.run.envelope.versions,space=1,mapping='sim-new-map')
        run=replace(rt.run,envelope=replace(rt.run.envelope,versions=versions))
        snapshot=initial_snapshot(run)
        snapshot=replace(snapshot,revision=1,space=replace(snapshot.space,version=1),mapping=mapping_for(run,clock))
        receipt=m.CommitReceipt('new-mapping',m.CommitStatus.COMMITTED,1,1,None)
        rt.accept_snapshot(snapshot,receipt)
        self.assertEqual(rt.run.envelope.versions.mapping,'sim-new-map')
        self.assertFalse(rt.authorize_output(permit,req.envelope,req.source,real_output=False))
        current=request(rt,clock,purpose='central_pick',seq=20)
        self.assertIsNotNone(rt.resume(current).permit)

    def test_holding_identity_change_revokes_existing_permit(self):
        rt,clock=started(); req=request(rt,clock); permit=rt.request_permission(req).permit
        ev=replace(req.readiness.evidence[0],target_id='brick-1')
        holding=c.HoldingSnapshot(c.Holding.HELD,'brick-1',None,(ev,))
        rt.observe_phase(m.Phase.TRIAL_LIFT,replace(req.readiness,holding=holding))
        self.assertFalse(rt.authorize_output(permit,req.envelope,req.source,real_output=False))

    def test_base_motion_invalidates_old_targets_but_does_not_clear_space(self):
        rt,clock=started(); req=request(rt,clock); permit=rt.request_permission(req).permit
        loc=replace(req.localization,base_motion_epoch=1,envelope=replace(req.localization.envelope,base_motion_epoch=1))
        rt.observe_localization(loc)
        self.assertFalse(rt.authorize_output(permit,req.envelope,req.source,real_output=False))
        self.assertIsNone(rt.resume(req).permit)
        self.assertEqual(rt.run.envelope.versions.space,0)

    def test_T07_debug_offline_review_requires_return_to_A_and_registered_new_instance(self):
        rt,clock=started(); deadline=rt.run.deadline
        rt.signal(signal(rt,clock,m.SignalKind.MANUAL_DEBUG,'signal-2'))
        clock.advance(.5)
        proposal=rt.manual_debug_proposal()
        self.assertNotEqual(proposal.envelope.instance_id,'sim-instance-1')
        self.assertEqual(proposal.envelope.control_epoch,2)
        self.assertFalse(proposal.ever_restarted)
        snap=initial_snapshot(proposal)
        result=m.RecoveryResult(c.Outcome.SUCCESS,snap,599.5,m.CommitReceipt('debug-1',m.CommitStatus.COMMITTED,0,0,None),None)
        original_req=request(rt,clock)
        loc=replace(original_req.localization,envelope=replace(proposal.envelope,producer=c.Owner.P02),
                    evidence=original_req.localization.evidence)
        self.assertIsNotNone(rt.accept_manual_registration(result,loc,c.Fact.UNKNOWN).reason)
        rt.accept_manual_registration(result,loc,c.Fact.CONFIRMED)
        self.assertEqual(rt.manual_debug_count,1)
        self.assertEqual(rt.run.deadline,deadline)
        self.assertIsNone(rt.run.envelope.versions.mapping)
        req=request(rt,clock,seq=20)
        self.assertIsNotNone(rt.resume(req).permit)
        central=request(rt,clock,purpose='central_pick',seq=30)
        self.assertIsNone(rt.request_permission(central).permit)


if __name__=='__main__':unittest.main()
