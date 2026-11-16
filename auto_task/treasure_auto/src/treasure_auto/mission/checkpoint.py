"""Simulated checkpoint boundary, explicitly not a production state writer."""
import json
from treasure_auto.contracts import core as c


class SimulationJournal:
    origin=c.Origin.SIMULATED
    def __init__(self,payload=None):
        self.payload=payload
        self.fail=False

    def load(self):
        return None if self.payload is None else json.loads(json.dumps(self.payload,allow_nan=False))

    def save(self,payload):
        if self.fail:return c.Fact.UNKNOWN
        self.payload=json.loads(json.dumps(payload,allow_nan=False))
        return c.Fact.CONFIRMED
