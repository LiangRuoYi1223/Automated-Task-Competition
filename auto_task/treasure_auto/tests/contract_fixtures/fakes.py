"""Deterministic in-memory test doubles, not P10A/P10B production implementations."""
from dataclasses import replace
from treasure_auto.contracts import core as c, models as m, rules as r


class FakeClock:
    origin = c.Origin.SIMULATED

    def __init__(self, initial):
        c.monotonic(initial, 'initial')
        self._now = initial

    def now(self):
        return self._now

    def advance(self, seconds):
        c.require(type(seconds) in (int, float) and seconds >= 0, 'seconds', 'cannot rewind clock')
        self._now = replace(self._now, seconds=self._now.seconds + seconds)
        return self._now


class ResultConsumer:
    """Read-only consumption example with event collision and sequence protection."""
    def __init__(self, context):
        self.context, self.current, self.updates = context, None, 0
        self._events, self._sequence = {}, {}

    def consume(self, envelope, result):
        r.validate_envelope(envelope, self.context)
        r.validate_action_result(envelope, result, self.context.now)
        key = (envelope.run_id, envelope.producer, envelope.event_id)
        payload = c.dumps(m.ActionEvent(envelope, result))
        if key in self._events:
            c.require(self._events[key] == payload, 'event_id', 'same event id, different payload', c.ErrorCode.ID_CONFLICT)
            return False
        stream = (envelope.producer, envelope.instance_id, envelope.control_epoch,
                  envelope.request_id, envelope.action_id)
        c.require(envelope.sequence > self._sequence.get(stream, -1), 'sequence', 'out-of-order event', c.ErrorCode.STALE_REQUEST)
        self._events[key] = payload
        self._sequence[stream] = envelope.sequence
        self.current, self.updates = result, self.updates + 1
        return True


class FakeStateStore:
    """Single-thread deterministic atomicity/CAS oracle; no disk, ROS or sleep.

    Does not model durable commits, process crashes, concurrent writers, timer files
    or cleanup. fail_next simulates a known failure before commit, never an unknown
    outcome after disk I/O. P10B must implement and test those separately.
    """
    def __init__(self, initial, clock=None):
        r.require_simulated(initial)
        self._snapshot = initial
        self.clock = clock if clock is not None else FakeClock(initial.run.envelope.emitted_at)
        self._transactions, self._events = {}, {}
        self.fail_next = False

    def read_snapshot(self):
        return self._snapshot

    def read(self, run_id):
        if run_id != self._snapshot.run.envelope.run_id:
            return m.ReadResult(c.Outcome.REJECTED, None, c.Reason(c.ErrorCode.STALE_RUN, 'run_id', 'wrong run'))
        return m.ReadResult(c.Outcome.SUCCESS, self._snapshot, None)

    def commit(self, transaction):
        tx = transaction
        try:
            r.require_simulated(tx)
            payload = c.dumps(tx)
            if tx.transaction_id in self._transactions:
                old_payload, receipt = self._transactions[tx.transaction_id]
                c.require(payload == old_payload, 'transaction_id', 'id reused with different payload', c.ErrorCode.ID_CONFLICT)
                return receipt
            deliveries = r.validate_transaction(tx, self._snapshot, self.clock.now())
            events = dict(self._events)
            for event in tx.release_events:
                key = (event.envelope.run_id, event.envelope.producer, event.envelope.event_id)
                event_payload = c.dumps(event)
                c.require(key not in events or events[key] == event_payload,
                          'event_id', 'event id reused', c.ErrorCode.ID_CONFLICT)
                events[key] = event_payload
            if self.fail_next:
                self.fail_next = False
                raise c.ContractError(c.ErrorCode.STORAGE_FAILURE, 'commit', 'simulated failure before commit')
            proposed = replace(self._snapshot, revision=self._snapshot.revision + 1,
                               space=tx.space, holding=tx.holding, inventory=tx.inventory,
                               deliveries=deliveries, layouts=tx.layouts, mapping=tx.mapping,
                               budgets=tx.budgets, run=tx.run_update or self._snapshot.run)
            receipt = m.CommitReceipt(tx.transaction_id, m.CommitStatus.COMMITTED,
                                      proposed.revision, proposed.space.version, None)
            # The only publication point; everything above operates on local copies.
            self._snapshot = proposed
            self._events = events
            self._transactions[tx.transaction_id] = (payload, receipt)
            return receipt
        except c.ContractError as error:
            return m.CommitReceipt(tx.transaction_id, m.CommitStatus.REJECTED, None, None,
                                    c.Reason(error.code, error.field, error.detail))
