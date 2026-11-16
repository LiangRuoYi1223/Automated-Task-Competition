"""Explicit domain -> P10A -> P01 stop assembly; no physical success inference."""
from treasure_auto.contracts import core as c, models as m


class StopRouter:
    def __init__(self, runtime, gateway):
        self.runtime, self.gateway = runtime, gateway

    def __call__(self, request):
        decision = self.runtime.request_domain_stop(request)
        c.require(decision.stop is not None, 'stop', 'no current P10A stop decision', c.ErrorCode.NOT_AUTHORIZED)
        # Authority is already revoked if output fails or returns UNKNOWN.
        report = self.gateway.stop(decision.stop)
        c.require(type(report) is m.StopReport and report.envelope.producer is c.Owner.P01 and
                  (report.envelope.request_id, report.envelope.action_id) ==
                  (decision.stop.envelope.request_id, decision.stop.envelope.action_id),
                  'stop.report', 'P01 report must bind this stop', c.ErrorCode.STALE_REQUEST)
        self.runtime.record_stop_report(report)
        return report
