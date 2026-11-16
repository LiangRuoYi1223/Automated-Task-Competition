"""P10B test-only simulated inputs. Never imported by production code."""
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from treasure_auto.contracts import core as c, models as m, rules as r
from contract_fixtures import samples as s
from contract_fixtures.fakes import FakeClock


def persistence():
    import treasure_auto.persistence as p
    return p


def initial_request():
    snap = s.state_snapshot()
    signal = m.RuntimeSignal(replace(snap.run.envelope, event_id=snap.run.start_signal_id),
                             m.SignalKind.START, c.Fact.CONFIRMED)
    return m.InitializeRunRequest(snap.run, snap, signal, None)


def stop_report(now=110, confirmed=True):
    fact = c.Fact.CONFIRMED if confirmed else c.Fact.UNKNOWN
    e = s.envelope(c.Owner.P01, event_id='stop', target_id=None, action_id=None,
                   emitted_at=s.stamp(now), expires_at=s.stamp(now + 30))
    return m.StopReport(e, c.Fact.CONFIRMED, fact, fact, fact, fact,
                        (s.evidence(target_id=None, action_id=None, received_at=s.stamp(now),
                                    valid_until=s.stamp(now + 30)),) if confirmed else ())


class StoreCase(unittest.TestCase):
    # No test methods here: importing this base does not duplicate collected tests.
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='p10b-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'runtime'
        self.clock = FakeClock(s.stamp(110))
        self.stores = []
        self.addCleanup(self.close_all)
        self.p = persistence()
        self.store = self.open_store()
        self.initial = initial_request()
        self.assertEqual(self.store.initialize_run(self.initial).status, m.CommitStatus.COMMITTED)

    def close_all(self):
        for store in self.stores:
            store.close()

    def open_store(self, **changes):
        options = dict(runtime_root=str(self.root), queue_capacity=8, operation_timeout_s=2.0,
                       timer_refresh_period_s=1.0, max_keyframe_bytes=4096)
        options.update(changes)
        store = self.p.SQLitePersistence(self.p.PersistenceConfig(**options), self.clock,
                                         remaining=r.remaining_seconds)
        self.stores.append(store)
        return store

    def reopen(self):
        self.store.close()
        self.store = self.open_store()
        return self.store

    def recovery(self, instance='sim-instance-2', epoch=2, now=None):
        now = self.clock.now().seconds if now is None else now
        snapshot = self.store.read('sim-run-1').snapshot
        e = s.envelope(c.Owner.P10A, event_id='recover-' + instance, instance_id=instance,
                       control_epoch=epoch, emitted_at=s.stamp(now), expires_at=s.stamp(now + 30),
                       versions=snapshot.space.envelope.versions if snapshot else s.versions())
        return m.RecoveryRequest(e, s.stamp(100), s.stamp(700), stop_report(now))

    def seal(self, code=c.ErrorCode.DEADLINE, confirmed=True):
        now = self.clock.now().seconds
        snap = self.store.read('sim-run-1').snapshot
        e = replace(snap.run.envelope, event_id='seal', emitted_at=self.clock.now(),
                    expires_at=s.stamp(now + 30), versions=snap.space.envelope.versions)
        run = replace(snap.run, envelope=e, state=m.RunState.ENDED)
        return self.store.seal(m.SealRequest(e, run, stop_report(now, confirmed),
                                            c.Reason(code, 'run', 'simulated end')))

    def cleanup_request(self, marker, automatic=True):
        return m.CleanupRequest('sim-run-1', str(self.root), marker, automatic)
