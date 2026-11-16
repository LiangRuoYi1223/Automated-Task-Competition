"""Explicit simulated P10B read and P10A proposal boundaries for domain unit tests."""
from dataclasses import replace
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from contract_fixtures import samples as s
from treasure_auto.contracts import core as c, models as m


def snapshot_for_space(space):
    e = replace(space.envelope, producer=c.Owner.P10A, action_id=None, target_id=None)
    start = replace(e.emitted_at, seconds=max(0., e.emitted_at.seconds - 10.))
    run = m.RunContext(e, m.RunState.OFFLINE, 'publication-start', start,
                       replace(start, seconds=start.seconds + 600.), False)
    mapping = None
    if e.versions.mapping is not None:
        mapping = replace(s.mapping_samples()[0],
            envelope=replace(e, producer=c.Owner.P05), mapping_version=e.versions.mapping)
    return m.StateSnapshot(run, 0, space, c.HoldingSnapshot(c.Holding.UNKNOWN, None, None, ()),
                           (), (), (), mapping, ())


def simulated_space_readback(space):
    return m.ReadResult(c.Outcome.SUCCESS, snapshot_for_space(space), None)


def simulated_run_proposal(previous, space):
    return replace(previous, envelope=replace(previous.envelope,
        versions=space.envelope.versions, emitted_at=space.envelope.emitted_at,
        expires_at=space.envelope.expires_at))
