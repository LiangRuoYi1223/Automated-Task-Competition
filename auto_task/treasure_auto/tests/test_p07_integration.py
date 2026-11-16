"""Narrow offline wiring of delivered P01/P10B code, never hardware."""
import json
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
from test_p07_support import Harness, c, m, r, s, api


class IntegrationTests(unittest.TestCase):
    def store(self,h):
        from treasure_auto.persistence import SQLitePersistence, PersistenceConfig
        temp=tempfile.TemporaryDirectory(prefix='p07-p10b-')
        self.addCleanup(temp.cleanup)
        store=SQLitePersistence(PersistenceConfig(str(Path(temp.name)/'runtime'),16,2.,1.,4096),
                                h.clock,remaining=r.remaining_seconds)
        self.addCleanup(store.close)
        snap=replace(s.state_snapshot(),holding=h.service.holding)
        signal=m.RuntimeSignal(replace(snap.run.envelope,event_id=snap.run.start_signal_id),m.SignalKind.START,c.Fact.CONFIRMED)
        receipt=store.initialize_run(m.InitializeRunRequest(snap.run,snap,signal,None))
        self.assertEqual(receipt.status,m.CommitStatus.COMMITTED)
        return store

    def test_real_P10B_intent_is_recorded_before_fake_device_and_deduplicated(self):
        h=Harness(); store=self.store(h); h.service.persistence=store
        req=h.request(); h.service.grasp(req); h.service.grasp(req)
        self.assertEqual(len(h.commands),1)
        history=store.history('sim-run-1')
        intents=[c.loads(row['payload']) for row in history if row['kind']=='intent']
        self.assertEqual(len(intents),1)
        self.assertEqual(intents[0].arm_plan.segments[0].target.z,20)
        self.assertEqual(h.service.result.result.submitted,c.Fact.CONFIRMED)

    def test_real_P01_gateway_receives_exact_intent_and_converts_mm_per_min(self):
        from treasure_auto.devices.gateway import DeviceGateway
        from treasure_auto.devices.config import ControlConfig
        from treasure_auto.devices.events import EventOutbox
        from treasure_auto.devices.drivers import SerialLease
        from test_p01_gateway import BoundaryTransport
        h=Harness(); transport=BoundaryTransport()
        lease=SerialLease('sim://p07-integration').acquire(); self.addCleanup(lease.release)
        cfg=ControlConfig('sim-config-v1',.5,1.,'sim-arm',((-200.,200.),(-200.,200.),(0.,200.)),
                          tuple((p,(.5,10.,5.)) for p in m.Phase))
        def health():
            return m.HealthSnapshot(h.env(c.Owner.P01),tuple(m.Capability(k,c.Fact.CONFIRMED,None,()) for k in ('base','arm','pump')))
        gateway=DeviceGateway(h.clock,s.run_context(),cfg,transport,health,EventOutbox(100),lease)
        def prepare(command,intent,receipt):
            gateway.register_intent(command,intent,receipt)
            gateway.install_permit(command.permit)
        h.service.device=gateway; h.service.prepare=prepare
        req=h.request(); result=h.service.grasp(req); h.service.grasp(req)
        self.assertEqual(result.result.submitted,c.Fact.CONFIRMED)
        self.assertEqual(len(transport.arm),1)
        self.assertEqual((transport.arm[0].z_mm,transport.arm[0].feed_mm_min,transport.arm[0].timeout_s),(20,60,2))

    def test_real_P10B_release_empty_arrival_commit_then_unlock(self):
        h=Harness(); h.reach('transport'); h.confirm_stage(); h.service.tick()
        store=self.store(h); h.service.persistence=store
        tx=s.release_transaction()
        ev=h.ev('sim-action-1','brick-1')
        release=replace(tx.release_events[0],envelope=h.env(c.Owner.P08,action_id='sim-action-1'),
                        released_at=h.clock.now(),evidence=(ev,))
        update=h.service.propose_release(release,h.service.holding)
        tx=replace(tx,envelope=replace(tx.envelope,emitted_at=h.clock.now()),
                   holding=update.current,holding_update=update,release_events=(release,))
        receipt=store.commit(tx)
        self.assertEqual(receipt.status,m.CommitStatus.COMMITTED,receipt.reason)
        result=h.service.accept_holding_commit(receipt)
        self.assertEqual(result.state,c.Holding.EMPTY)
        self.assertIsNone(h.service.readiness)
        self.assertEqual(len(store.read('sim-run-1').snapshot.deliveries),1)
        h.context=replace(h.context,versions=s.versions(1))
        plan=h.stage_plan(m.Phase.WITHDRAW,h.env(action_id='empty-posture',target_id=None),None,h.service.holding)
        first=h.service.confirm_empty_readiness(h.observation(empty=True,action_id='empty-posture',target_id=None,z=30,posture_id='sim-empty'),plan,'sim-empty')
        self.assertIsNone(first,'old held frames cannot count as new empty-arm frames')
        ready=h.service.confirm_empty_readiness(h.observation(empty=True,action_id='empty-posture',target_id=None,z=30,posture_id='sim-empty'),plan,'sim-empty')
        self.assertIsNotNone(ready)
        for ev in ready.evidence:
            r.require_physical_evidence(ev,ready.envelope,h.clock.now(),c.Origin.SIMULATED)

    def test_config_rejects_unmeasured_and_accepts_only_explicit_simulation_values(self):
        p=api(); root=Path(__file__).resolve().parents[1]/'config/manipulation'
        with self.assertRaises(c.ContractError): p.ManipulationConfig.load(root/'unmeasured.json')
        config=p.ManipulationConfig.load(root/'offline.example.json')
        self.assertEqual(config.transport_pose_id,'sim-transport-42')
        for changes in (dict(evidence_age_s=None),dict(confirmation_frames=True),dict(min_lift_mm=float('nan')),
                        dict(max_lift_mm=1.),dict(pose_tolerance_mm=-1.),dict(transport_pose_id='')):
            with self.subTest(changes=changes),self.assertRaises(c.ContractError): replace(config,**changes)

    def test_live_run_never_activates_even_with_filled_configuration(self):
        h=Harness()
        with self.assertRaises(c.ContractError):
            h.p.ManipulationService(h.config,replace(s.run_context(),envelope=replace(s.run_context().envelope,origin=c.Origin.LIVE)),
                h.clock,context_provider=lambda:h.context,capture_now=h.clock.now,stop_provider=h.stop_report,
                stage_planner=h.stage_plan,authorize=h.authorize,prepare=h.prepare,device=h,persistence=h,checkpoints=h.checkpoints)
