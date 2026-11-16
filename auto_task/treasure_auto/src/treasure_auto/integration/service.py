"""C0 IntegrationPort for installed, offline replay and exact dependency audit."""
import hashlib
import importlib
import json
from pathlib import Path

from treasure_auto.contracts import core as c, models as m
from .replay import SCENARIOS, run_replay, validate_replay


MODULES = {
    c.Owner.P00: 'contracts', c.Owner.P01: 'devices', c.Owner.P02: 'mapping',
    c.Owner.P03: 'navigation', c.Owner.P04: 'perception', c.Owner.P05: 'classification',
    c.Owner.P06: 'geometry', c.Owner.P07: 'manipulation', c.Owner.P08: 'placement',
    c.Owner.P09: 'mission', c.Owner.P10A: 'runtime', c.Owner.P10B: 'persistence',
}


def _reason(detail, code=c.ErrorCode.UNAVAILABLE, field='dependencies'):
    return c.Reason(code, field, detail)


def audit_dependencies(package_root=None):
    """Availability is code/hash/import compatibility, never physical validation.

Handoff digests identify the reviewed delivery; installed deployments do not
pretend that original handoff markdown is present or that its tests were rerun.
"""
    package = Path(package_root) if package_root is not None else Path(__file__).resolve().parents[1]
    try:
        manifest = json.loads((package / 'integration/dependencies.json').read_text(encoding='utf-8'))
        c.require(manifest['schema_version'] == 1 and manifest['contract_version'] == c.CONTRACT_VERSION,
                  'manifest', 'frozen contract version differs', c.ErrorCode.VERSION_MISMATCH)
        entries = {c.Owner(x['owner']): x for x in manifest['dependencies']}
        c.require(set(entries) == set(MODULES) and len(entries) == len(manifest['dependencies']),
                  'manifest', 'exactly one frozen entry per prerequisite required')
    except (OSError, ValueError, TypeError, KeyError) as error:
        return tuple(m.DependencyStatus(owner, None, None, c.Fact.UNKNOWN, _reason(str(error))) for owner in MODULES)
    statuses = []
    for owner, module in MODULES.items():
        entry = entries[owner]
        version = entry.get('contract_version')
        delivery = entry.get('delivery_sha256')
        reason = None
        try:
            c.require(entry['module'] == module and version == c.CONTRACT_VERSION,
                      'manifest', 'module/contract identity mismatch', c.ErrorCode.VERSION_MISMATCH)
            c.require(type(delivery) is str and len(delivery) == 64 and all(x in '0123456789abcdef' for x in delivery),
                      'manifest', 'missing delivery fingerprint')
            expected = entry['files']
            c.require(type(expected) is dict and all(type(name) is str and type(digest) is str
                      for name, digest in expected.items()), 'manifest.files', 'file hashes must be a mapping')
            actual_names = {module+'/'+p.name for p in (package/module).glob('*.py')}
            c.require(actual_names == {name for name in expected if name.endswith('.py')},
                      'files', 'missing or unreviewed dependency source')
            c.require(module+'/__init__.py' in expected, 'files', 'subpackage initializer missing')
            for name, digest in expected.items():
                path = package / name
                c.require(path.resolve().parent == (package/module).resolve(), 'files', 'source outside declared module')
                c.require(hashlib.sha256(path.read_bytes()).hexdigest() == digest,
                          name, 'file differs from reviewed delivery', c.ErrorCode.VERSION_MISMATCH)
            if package_root is None:
                imported = importlib.import_module('treasure_auto.'+module)
                c.require(Path(imported.__file__).resolve().parent == (package/module).resolve(),
                          'import', 'loaded dependency outside installed package')
        except (OSError, ValueError, TypeError, KeyError, ImportError) as error:
            reason = _reason(str(error), c.ErrorCode.VERSION_MISMATCH)
        statuses.append(m.DependencyStatus(owner, version if isinstance(version, str) and version else None,
                         'sha256:'+delivery if isinstance(delivery, str) and delivery else None,
                         c.Fact.CONFIRMED if reason is None else c.Fact.UNKNOWN, reason))
    return tuple(statuses)


class IntegrationService:
    """Implements the unchanged C0 signature; confirms OFFLINE tier only."""
    def __init__(self, layout='V1', config=None):
        c.require(layout in ('V1', 'V2', 'V3'), 'layout', 'expected V1/V2/V3')
        self.layout, self.config = layout, config
        self.results = {}

    def verify(self, request):
        c.require(type(request) is m.IntegrationRequest, 'request', 'C0 IntegrationRequest required')
        c.require(bool(request.scenario_ids) and len(request.scenario_ids) == len(set(request.scenario_ids)),
                  'scenario_ids', 'nonempty distinct scenarios required')
        self.results = {}
        actual = audit_dependencies()
        gate = None
        gate_status = c.Fact.UNKNOWN
        if request.contract_version != c.CONTRACT_VERSION:
            gate = _reason('exact C0 1.0.0 required', c.ErrorCode.VERSION_MISMATCH, 'contract_version')
            gate_status = c.Fact.REJECTED
        elif request.origin is not c.Origin.SIMULATED:
            gate = _reason('only explicit SIMULATED replay is assembled', c.ErrorCode.NOT_AUTHORIZED, 'origin')
            gate_status = c.Fact.REJECTED
        elif len(request.dependencies) != len(MODULES) or {x.owner for x in request.dependencies} != set(MODULES):
            gate = _reason('missing or duplicate prerequisite dependency')
        elif any(x.available is not c.Fact.CONFIRMED for x in actual):
            gate = _reason('; '.join(x.owner.value+': '+x.reason.detail for x in actual if x.reason))
        elif {x.owner: x for x in request.dependencies} != {x.owner: x for x in actual}:
            gate = _reason('caller dependency claims differ from reviewed available files', c.ErrorCode.VERSION_MISMATCH)
        checks = []
        for scenario in request.scenario_ids:
            status, reason = gate_status, gate
            if gate is None:
                if scenario not in SCENARIOS:
                    status, reason = c.Fact.REJECTED, _reason('unsupported replay scenario', c.ErrorCode.INVALID_FIELD, 'scenario_id')
                else:
                    try:
                        result = run_replay(self.layout, scenario, self.config)
                        self.results[scenario] = result
                        failures = validate_replay(result)
                        status = c.Fact.REJECTED if failures else c.Fact.CONFIRMED
                        reason = _reason('; '.join(failures), field='scenario') if failures else None
                    except Exception as error:
                        status, reason = c.Fact.REJECTED, _reason(type(error).__name__+': '+str(error), field='scenario')
            checks.append(m.VerificationResult(scenario, m.VerificationTier.OFFLINE, status, None, reason))
        return m.IntegrationReport(c.CONTRACT_VERSION, tuple(checks), actual, c.Origin.SIMULATED)
