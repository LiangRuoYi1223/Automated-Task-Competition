"""Narrow offline integration with concurrently delivered P01/P10B source.

No ROS/serial; SQLite uses a dedicated TemporaryDirectory only.
"""
import tempfile
import unittest
from dataclasses import replace
from test_p10a_support import *


class Transport:
    origin=c.Origin.SIMULATED
    def __init__(self):self.base=[];self.pump=[]
    def send_base(self,linear,angular):self.base.append((linear,angular))
    def send_arm(self,dispatch):raise AssertionError('this test never moves arm')
    def send_pump(self,enabled):self.pump.append(enabled)
    def cancel_arm(self):pass


class IntegrationTests(unittest.TestCase):
    def test_real_sqlite_initialization_and_original_deadline_readback(self):
        from treasure_auto.persistence import SQLitePersistence,PersistenceConfig
        rt,clock=make_runtime(mapping=False);rt.arm();rt.signal(signal(rt,clock))
        with tempfile.TemporaryDirectory(prefix='p10a-integration-') as folder:
            store=SQLitePersistence(PersistenceConfig(str(Path(folder)/'run'),8,2.,1.,4096),clock,remaining=rt.remaining)
            try:
                decision=rt.initialize(store,initial_snapshot(rt.run))
                self.assertIsNone(decision.reason)
                persisted=store.read(rt.run.envelope.run_id)
                self.assertEqual(persisted.snapshot.run.deadline.seconds,700)
                self.assertEqual(persisted.snapshot.run.state,m.RunState.OFFLINE)
                self.assertEqual(store.storage_settings()['journal_mode'],'wal')
            finally:self.assertTrue(store.close())

    def test_real_sqlite_recovery_registration_retains_original_deadline(self):
        from treasure_auto.persistence import SQLitePersistence,PersistenceConfig
        original,clock=make_runtime(mapping=False);original.arm();original.signal(signal(original,clock))
        with tempfile.TemporaryDirectory(prefix='p10a-recovery-') as folder:
            store=SQLitePersistence(PersistenceConfig(str(Path(folder)/'run'),8,2.,1.,4096),clock,remaining=original.remaining)
            try:
                original.initialize(store,initial_snapshot(original.run))
                saved=store.read(original.run.envelope.run_id).snapshot.run
                clock.advance(190)
                rt,_=make_runtime(mapping=False,instance_id='new-instance',control_epoch=2);rt.clock=clock
                rt.begin_recovery(saved)
                old_stop=request(original,clock).old_stop
                result=store.register_recovery(m.RecoveryRequest(rt.run.envelope,rt.run.started_at,rt.run.deadline,old_stop))
                self.assertEqual(result.status,c.Outcome.SUCCESS,result.reason)
                rt.recover(result)
                self.assertEqual(rt.remaining_s,410)
                self.assertTrue(rt.run.ever_restarted)
                self.assertIsNone(rt.request_permission(request(rt,clock)).permit)
                self.assertIsNotNone(rt.resume(request(rt,clock,seq=20)).permit)
            finally:self.assertTrue(store.close())

    def test_p01_gateway_consumes_p10a_permit_and_stops_at_original_deadline(self):
        from treasure_auto.devices.gateway import DeviceGateway
        from treasure_auto.devices.config import ControlConfig
        from treasure_auto.devices.events import EventOutbox
        rt,clock=started();req=request(rt,clock)
        # P03 owns base-command/intent envelopes; P09 remains mission heartbeats.
        env=replace(req.envelope,producer=c.Owner.P03,action_id='sim-p03-action')
        intent=m.ActionIntent(env,m.IntentOperation.BASE_VELOCITY,None,None,None,(.1,0.),None)
        receipt=s.intent_receipt(intent)
        rt.remember_intent(intent,receipt,source=req.source)
        req=replace(req,envelope=env,intent_receipt=receipt)
        permit=rt.request_permission(req).permit
        self.assertIsNotNone(permit)
        transport=Transport()
        config=ControlConfig('sim-config-v1',.5,1.,'sim-arm',((-200.,200.),(-200.,200.),(0.,200.)),
                              tuple((phase,(.5,10.,5.)) for phase in m.Phase))
        gateway=DeviceGateway(clock,rt.run,config,transport,lambda:req.health,EventOutbox(100),None)
        command=m.BaseCommand(env,permit,receipt.commit.transaction_id,.1,0.,req.source)
        gateway.register_intent(command,intent,receipt);gateway.install_permit(permit)
        self.assertEqual(gateway.submit_base(command).result.submitted,c.Fact.CONFIRMED)
        self.assertEqual(transport.base[-1],(.1,0.))
        clock.advance(600)
        stop=rt.supervise(clock.now(),None).stop
        report=gateway.stop(stop)
        self.assertEqual(transport.base[-1],(0.,0.))
        self.assertEqual(report.physically_stopped,c.Fact.UNKNOWN)
        self.assertNotEqual(gateway.submit_base(command).result.status,c.Outcome.SUCCESS)
        self.assertEqual(transport.pump,[])


if __name__=='__main__':unittest.main()
