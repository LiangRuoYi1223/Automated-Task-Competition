import unittest
from dataclasses import replace
from test_p10a_support import *


class LifecycleTests(unittest.TestCase):
    def test_T01_only_one_start_and_deadline_without_mapping(self):
        rt, clock = make_runtime(mapping=False)
        self.assertEqual(rt.state, m.RunState.PREPARATION)
        rt.arm()
        sig = signal(rt, clock)
        first = rt.signal(sig)
        clock.advance(30)
        duplicate = rt.signal(sig)
        self.assertEqual(first.run.deadline.seconds, 700)
        self.assertEqual(first.run.envelope.run_id, duplicate.run.envelope.run_id)
        self.assertEqual(duplicate.run.deadline, first.run.deadline)
        self.assertIsNone(first.run.envelope.versions.mapping)

    def test_T01_unauthorized_or_old_start_cannot_create_run(self):
        for field in ('auth', 'run_id', 'instance_id', 'expires_at'):
            rt, clock = make_runtime()
            rt.arm()
            sig = signal(rt, clock)
            if field == 'auth':
                sig = replace(sig, referee_authorized=c.Fact.UNKNOWN)
            else:
                value = s.stamp(99) if field == 'expires_at' else 'old'
                changes = {field: value}
                if field == 'expires_at': changes['emitted_at'] = s.stamp(98)
                sig = replace(sig, envelope=replace(sig.envelope, **changes))
            with self.subTest(field=field), self.assertRaises(c.ContractError): rt.signal(sig)
            self.assertIsNone(rt.run)

    def test_T01_ended_run_never_reopens_and_old_finish_does_not_stop_current(self):
        rt, clock = started()
        old = signal(rt, clock, m.SignalKind.FINISH, 'signal-2')
        old = replace(old, envelope=replace(old.envelope, run_id='old-run'))
        self.assertIsNotNone(rt.signal(old).reason)
        self.assertEqual(rt.state, m.RunState.OFFLINE)
        clock.advance(600)
        rt.supervise(clock.now(), request(rt, clock).health)
        run = rt.signal(signal(rt, clock, event='signal-3')).run
        self.assertEqual(run.deadline.seconds, 700)
        self.assertEqual(rt.state, m.RunState.STOPPING)
        with self.assertRaises(c.ContractError): rt.arm()

    def test_T01_duplicate_id_with_changed_payload_rejected(self):
        rt, clock = started()
        sig = signal(rt, clock, m.SignalKind.MANUAL_DEBUG, 'signal-2')
        rt.signal(sig)
        bad = replace(sig, kind=m.SignalKind.FINISH)
        self.assertEqual(rt.signal(bad).reason.code, c.ErrorCode.ID_CONFLICT)

    def test_storage_unknown_and_failed_initialization_never_release_permission(self):
        for status in (m.CommitStatus.REJECTED, m.CommitStatus.UNKNOWN):
            rt, clock = make_runtime()
            rt.arm(); rt.signal(signal(rt, clock))
            decision = rt.initialize(Store(status), initial_snapshot(rt.run))
            self.assertEqual(decision.reason.code, c.ErrorCode.STORAGE_FAILURE)
            self.assertIsNone(rt.request_permission(request(rt, clock)).permit)

    def test_offline_permit_cannot_authorize_real_output(self):
        rt, clock = started()
        req = request(rt, clock)
        permit = rt.request_permission(req).permit
        self.assertIsNotNone(permit)
        self.assertFalse(rt.authorize_output(permit, req.envelope, req.source, real_output=True))
        self.assertTrue(rt.authorize_output(permit, req.envelope, req.source, real_output=False))

    def test_config_missing_nan_bool_and_unverified_formal_rejected(self):
        from treasure_auto.runtime import RuntimeConfig
        for key, value in [('lease_s', None), ('lease_s', float('nan')), ('supervision_period_s', True),
                           ('heartbeat_timeout_s', -1), ('mode', 'formal'), ('required_capabilities', []), ('unknown', 1)]:
            data = config_data(); data[key] = value
            with self.subTest(key=key,value=value), self.assertRaises(c.ContractError): RuntimeConfig.from_dict(data)

    def test_formal_state_rehearsal_is_simulated_and_cannot_output_to_hardware(self):
        from treasure_auto.runtime import RuntimeConfig,RuntimeController
        data=config_data();data['mode']='formal_simulation'
        clock=FakeClock(s.stamp(100))
        rt=RuntimeController(RuntimeConfig.from_dict(data),clock,s.versions(),'sim-instance',1)
        rt.arm();self.assertEqual(rt.state,m.RunState.ARMED)
        rt.signal(signal(rt,clock));self.assertEqual(rt.state,m.RunState.RECOVERING)
        rt.initialize(Store(),initial_snapshot(rt.run));self.assertEqual(rt.state,m.RunState.RUNNING)
        req=request(rt,clock);permit=rt.request_permission(req).permit
        self.assertIsNotNone(permit)
        self.assertEqual(permit.envelope.origin,c.Origin.SIMULATED)
        self.assertFalse(rt.authorize_output(permit,req.envelope,req.source,real_output=True))


if __name__ == '__main__': unittest.main()
