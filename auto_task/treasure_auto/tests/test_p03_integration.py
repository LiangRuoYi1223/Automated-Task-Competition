"""Real P01 gateway, synthetic transport only. No physical action authorized."""
from publication_support import simulated_space_readback
import unittest
from test_p03_support import *
from treasure_auto.navigation.service import NavigationService
from treasure_auto.devices.gateway import DeviceGateway
from treasure_auto.devices.config import ControlConfig
from treasure_auto.devices.events import EventOutbox

class Transport:
    origin=c.Origin.SIMULATED
    def __init__(self):self.base=[]
    def send_base(self,v,w):self.base.append((v,w))
    def send_arm(self,*args):raise AssertionError('arm not in P03 scope')
    def send_pump(self,*args):raise AssertionError('pump not in P03 scope')
    def cancel_arm(self):raise AssertionError('P03 only revokes base channel')

class IntegrationTests(unittest.TestCase):
    def test_T02_real_P01_gateway_rejects_old_velocity_after_handoff(self):
        clock=Clock();transport=Transport()
        config=ControlConfig('sim-config-v1',.5,1.,'arm',((-200.,200.),(-200.,200.),(0.,200.)),
                             tuple((phase,(.5,10.,5.)) for phase in m.Phase))
        health=lambda:m.HealthSnapshot(env(clock.t,producer=c.Owner.P01),(m.Capability('base',c.Fact.CONFIRMED,None,()),))
        gateway=DeviceGateway(clock,run(),config,transport,health,EventOutbox(100),None)
        nav=NavigationService(run(),clock,policy(),profile(),bundle(),gateway,lambda r:(r.localization.pose,r.goal.pose),space_readback=simulated_space_readback)
        req=request();gateway.install_permit(req.permit);start(nav,req);clock.advance(.1)
        intent=nav.prepare_velocity(.05,0,m.MotionSource.NAVIGATION);rec=receipt(intent)
        self.assertIsNotNone(nav.dispatch_velocity(intent,rec,req.permit))
        self.assertGreater(transport.base[-1][0],0.)
        old=m.BaseCommand(intent.envelope,req.permit,rec.commit.transaction_id,*intent.base_velocity_m_s_rad_s,m.MotionSource.NAVIGATION)
        nav.coarse_event(arrived(nav,clock))
        self.assertEqual(transport.base[-1],(0.,0.))
        count=len(transport.base)
        self.assertEqual(gateway.submit_base(old).result.accepted,c.Fact.REJECTED)
        self.assertEqual(len(transport.base),count)
        self.assertEqual(nav.current_result().work_ready,c.Fact.UNKNOWN)

    def test_live_service_cannot_be_enabled_by_changing_provenance_label(self):
        live=replace(run(),envelope=replace(run().envelope,origin=c.Origin.LIVE),state=m.RunState.RUNNING)
        with self.assertRaises(c.ContractError):
            NavigationService(live,Clock(),policy(),profile(),bundle(),GatewayBoundary(),lambda r:())
