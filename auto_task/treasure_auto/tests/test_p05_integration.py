"""Actual P10B SQLite in an isolated scratch directory; P08/P09 are test fakes."""
import hashlib
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from test_p05_support import P05Case, Harness, frame, c, m
from treasure_auto.contracts import rules


class IntegrationTests(P05Case):
    def test_R04_real_P10B_atomic_freeze_keyframes_and_reopen(self):
        from treasure_auto.persistence import SQLitePersistence, PersistenceConfig
        h=Harness()
        initial=h.store.read(h.run.envelope.run_id).snapshot
        scratch=Path(__file__).resolve().parents[1]/'src/treasure_auto/classification/verification'
        with tempfile.TemporaryDirectory(prefix='p05-sqlite-',dir=str(scratch)) as tmp:
            root=Path(tmp).resolve()/'runtime'
            settings=PersistenceConfig(str(root),8,2.0,1.0,1024*1024)
            store=SQLitePersistence(settings,h.clock,remaining=rules.remaining_seconds)
            try:
                signal=m.RuntimeSignal(replace(h.run.envelope,event_id=h.run.start_signal_id),m.SignalKind.START,c.Fact.CONFIRMED)
                initialized=store.initialize_run(m.InitializeRunRequest(h.run,initial,signal,None))
                self.assertEqual(initialized.status,m.CommitStatus.COMMITTED,initialized.reason)
                h.store=store; h.service=h.new_service()
                frames=[]
                for i in range(2):
                    f=frame(i)
                    saved=store.save_keyframe(f.image.image,f.image.image.evidence_id.encode())
                    board=replace(f.boards[0],directions=(replace(f.boards[0].directions[0],evidence=(saved,)),))
                    frames.append(replace(f,image=replace(f.image,image=saved),boards=(board,),stationary_evidence=(saved,),
                                          patterns=tuple(replace(p,evidence=(saved,)) for p in f.patterns)))
                def coordinate(mapping,audit):
                    # Explicitly fake P09 coordination. P05 itself never claims this ownership.
                    ref=replace(frames[-1].image.image,evidence_id='audit-'+mapping.mapping_version,
                                kind=c.EvidenceKind.INTERFACE,captured_at=None,
                                sha256=hashlib.sha256(audit).hexdigest(),uri='sim://p05/audit')
                    store.save_keyframe(ref,audit)
                    return h.commit(mapping,audit)
                h.service.submit_mapping=coordinate
                result=h.send(tuple(frames))
                self.assertEqual(result.result.status,c.Outcome.SUCCESS,result.result.reason)
                audit=json.loads(h.calls[0][1])
                self.assertEqual(audit['mapping']['mapping_version'],result.mapping.mapping_version)
                self.assertEqual(len([e for e in audit['events'] if e['type']=='frame']),2)
                self.assertEqual(len(list(root.rglob('*.bin'))),3)
                original=result.mapping
            finally:
                store.close()
            reopened=SQLitePersistence(settings,h.clock,remaining=rules.remaining_seconds)
            try:
                actual=reopened.read(h.run.envelope.run_id)
                self.assertEqual(actual.status,c.Outcome.SUCCESS,actual.reason)
                self.assertEqual(actual.snapshot.mapping,original)
                self.assertEqual(actual.snapshot.run.deadline,h.run.deadline)
            finally:
                reopened.close()

