"""Read-only arrival accounting. P10B owns the durable delivery ledger."""
from dataclasses import dataclass
from typing import Tuple
from treasure_auto.contracts import core as c, models as m


@dataclass(frozen=True)
class ArrivalStatus:
    counts: Tuple[Tuple[m.Category,int],...]
    total: int
    complete: c.Fact
    review: Tuple[str,...]


def arrival_counts(snapshot):
    counts={category:0 for category in m.Category};issues=[]
    aliases=[t for item in snapshot.inventory for t in item.target_ids]
    if len(aliases)!=len(set(aliases)):issues.append('physical target has conflicting inventory identities')
    ids=[item.inventory_id for item in snapshot.inventory]
    if len(ids)!=len(set(ids)):issues.append('duplicate inventory identity')
    for d in snapshot.deliveries:
        counts[d.category]+=1
        if d.key.run_id!=snapshot.run.envelope.run_id:issues.append('receipt belongs to another run')
        if snapshot.mapping is None or snapshot.mapping.destination(d.category) is not d.area:issues.append('receipt has no correct frozen mapping')
        if not d.evidence_ids:issues.append('receipt has no release evidence')
    if any(n>(1 if k is m.Category.CAKE else 3) for k,n in counts.items()):issues.append('arrival cap exceeded; review required')
    total=sum(counts.values())
    complete=c.Fact.UNKNOWN if issues else (c.Fact.CONFIRMED if total==10 else c.Fact.REJECTED)
    return ArrivalStatus(tuple(counts.items()),total,complete,tuple(issues))
