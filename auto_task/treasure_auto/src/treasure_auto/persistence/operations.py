"""C0 state publication, separate action journal, and verified keyframes."""
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname
from treasure_auto.contracts import core as c, models as m, rules as r
from . import filesystem as fs


class Operations:
    def _writable(self, run_id):
        c.require(not self._faulted.is_set(), 'storage', 'storage fault is latched; preserve and recover', c.ErrorCode.STORAGE_FAILURE)
        c.require(self._authorized and self._catalog['active'] == run_id and
                  self._catalog['runs'].get(run_id, {}).get('phase') == 'active',
                  'run_id', 'not active or recovery registration required', c.ErrorCode.NOT_AUTHORIZED)
        c.require(not self._marker_path(run_id).exists(), 'run_id', 'ended marker prevents new writes', c.ErrorCode.NOT_AUTHORIZED)
        return self._snapshot(run_id)

    def _current_envelope(self, envelope, snapshot):
        run = snapshot.run
        context = m.ConsumerContext(run.envelope.run_id, run.envelope.instance_id, run.envelope.control_epoch,
                                    self.clock.now(), snapshot.space.envelope.versions, None, None, None,
                                    None, None, None, run.envelope.origin)
        r.validate_envelope(envelope, context)

    def _references(self, value):
        if isinstance(value, c.EvidenceRef):
            yield value
        elif is_dataclass(value):
            for field in fields(value):
                yield from self._references(getattr(value, field.name))
        elif isinstance(value, (tuple, list)):
            for item in value:
                yield from self._references(item)

    def _check_frames(self, value):
        for reference in self._references(value):
            parsed = urlparse(reference.uri)
            if parsed.scheme == 'sim' and reference.origin is c.Origin.SIMULATED:
                continue  # Explicit C0 simulated evidence, never LIVE proof.
            c.require(parsed.scheme == 'file' and not parsed.netloc and reference.sha256 is not None,
                      'evidence.uri', 'verified local keyframe required', c.ErrorCode.EVIDENCE_MISSING)
            path = fs.checked(Path(url2pathname(parsed.path)), self.root)
            c.require(path.parent == self._run_path(reference.run_id) / 'keyframes' and path.is_file(),
                      'evidence.uri', 'keyframe missing/outside this run', c.ErrorCode.EVIDENCE_MISSING)
            c.require(fs.digest(path.read_bytes()) == reference.sha256,
                      'evidence.sha256', 'keyframe content changed', c.ErrorCode.EVIDENCE_MISSING)
            row = self._db.execute('SELECT result FROM receipts WHERE kind=? AND key=?',
                                   ('keyframe', reference.evidence_id)).fetchone()
            c.require(row is not None and c.loads(row[0]) == reference,
                      'evidence', 'file reference was not committed by this store', c.ErrorCode.EVIDENCE_MISSING)

    def _event_key(self, envelope):
        return '{}:{}'.format(envelope.producer.value, envelope.event_id)

    def _commit(self, tx):
        previous = self._writable(tx.envelope.run_id)
        known = self._lookup('transaction', tx.transaction_id, tx)
        if known is not None:
            return known
        deliveries = r.validate_transaction(tx, previous, self.clock.now())
        self._check_frames(tx)
        proposed = replace(previous, revision=previous.revision + 1, space=tx.space, holding=tx.holding,
                           inventory=tx.inventory, deliveries=deliveries, layouts=tx.layouts,
                           mapping=tx.mapping, budgets=tx.budgets, run=tx.run_update or previous.run)
        # Run terminal lifecycle belongs to seal(); a business transaction must
        # never evade the external marker or change original run/clock identity.
        c.require(proposed.run.state not in (m.RunState.ENDED, m.RunState.STOPPING),
                  'run_update', 'use seal for terminal lifecycle')
        receipt = m.CommitReceipt(tx.transaction_id, m.CommitStatus.COMMITTED, proposed.revision, proposed.space.version, None)
        def work():
            for event in (tx,) + tx.release_events:
                key = self._event_key(event.envelope)
                if self._lookup('event', key, event) is None:
                    self._remember('event', key, event, receipt)
            self._write_snapshot(proposed)
            self._log('transaction', tx, receipt)
            self._remember('transaction', tx.transaction_id, tx, receipt)
            return receipt
        result = self._atomic(work)
        if proposed.run.state in (m.RunState.RUNNING, m.RunState.OFFLINE):
            self._blocked.clear()
        return result

    def _commit_intent(self, intent):
        snapshot = self._writable(intent.envelope.run_id)
        self._current_envelope(intent.envelope, snapshot)
        c.require(snapshot.run.state in (m.RunState.RUNNING, m.RunState.OFFLINE),
                  'state', 'recovery needs fresh P10A authorization', c.ErrorCode.NOT_AUTHORIZED)
        c.require(self._remaining(snapshot.run, self.clock.now()) > 0,
                  'deadline', 'no new action at/after deadline', c.ErrorCode.DEADLINE)
        known = self._lookup('intent', intent.envelope.action_id, intent)
        if known is not None:
            return known
        self._check_frames(intent)
        if intent.arm_plan is not None:
            self._current_envelope(intent.arm_plan.envelope, snapshot)
            c.require(intent.arm_plan.envelope.action_id == intent.envelope.action_id,
                      'arm_plan', 'plan/action mismatch', c.ErrorCode.ID_CONFLICT)
        commit = m.CommitReceipt('intent:' + intent.envelope.action_id, m.CommitStatus.COMMITTED,
                                  snapshot.revision, snapshot.space.version, None)
        result = m.IntentReceipt(intent.envelope.run_id, intent.envelope.action_id, r.intent_digest(intent), commit)
        def work():
            key = self._event_key(intent.envelope)
            self._lookup('event', key, intent)
            self._remember('event', key, intent, commit)
            self._remember('intent', intent.envelope.action_id, intent, result)
            self._log('intent', intent, result)
            return result
        return self._atomic(work)

    def _append_event(self, event):
        c.require(isinstance(event, m.ActionEvent), 'event', 'C0 ActionEvent required')
        return self._append_record(event)

    def _append_record(self, value):
        """Additional local API: journal a C0 record without publishing its facts.

        Late/old controller results remain traceable, not actionable. P00 must
        freeze this broader port before cross-package use of non-ActionEvents.
        """
        c.require(isinstance(value, c.Record) and isinstance(getattr(value, 'envelope', None), c.Envelope),
                  'record', 'C0 record with envelope required')
        snapshot = self._writable(value.envelope.run_id)
        r.require_provenance(value, snapshot.run.envelope.origin)
        key = self._event_key(value.envelope)
        known = self._lookup('event', key, value)
        if known is not None:
            return known
        self._check_frames(value)
        receipt = m.CommitReceipt('event:' + key, m.CommitStatus.COMMITTED, snapshot.revision, snapshot.space.version, None)
        def work():
            self._log('event', value, receipt)
            self._remember('event', key, value, receipt)
            return receipt
        return self._atomic(work)

    def _save_keyframe(self, pair):
        reference, content = pair
        snapshot = self._writable(reference.run_id)
        c.require(reference.origin is snapshot.run.envelope.origin, 'origin', 'wrong evidence origin')
        c.require(type(content) is bytes and 0 < len(content) <= self.config.max_keyframe_bytes,
                  'content', 'empty or oversized keyframe')
        checksum = fs.digest(content)
        c.require(reference.sha256 is None or reference.sha256 == checksum,
                  'sha256', 'content does not match claimed hash', c.ErrorCode.EVIDENCE_MISSING)
        source = replace(reference, sha256=checksum)
        known = self._lookup('keyframe', reference.evidence_id, source)
        if known is not None:
            self._check_frames(known)
            return known
        path = self._run_path(reference.run_id) / 'keyframes' / (fs.digest(reference.evidence_id.encode('utf-8')) + '.bin')
        fs.checked(path, self.root)
        fs.atomic_write(path, content)
        result = replace(source, uri=path.as_uri())
        def work():
            self._remember('keyframe', reference.evidence_id, source, result)
            self._log('keyframe', result)
            return result
        return self._atomic(work)
