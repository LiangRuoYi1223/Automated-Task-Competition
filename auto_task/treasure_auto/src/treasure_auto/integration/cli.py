"""Installed/source offline command. There is no hardware enabling flag."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import sysconfig
import tempfile

from treasure_auto.contracts import core as c, models as m
from treasure_auto.mission.config import MissionConfig
from .replay import SCENARIOS
from .resources import get_resource_path
from .service import IntegrationService, audit_dependencies


BLOCKERS = (
    'P00/P03-P09: formal phase requests, callbacks and physical evidence adapters are not frozen (P09-C0-01).',
    'P00/P09/P10B: durable mission checkpoints, exit facts and replay indices are absent (P09-C0-02/05).',
    'P00/P07/P08/P09/P10B: temporary non-arrival release and conservative UNKNOWN occupancy transactions remain blocked (P08-C0-03/04).',
    'P00/P10A/P10B: manual-debug history and recovery purpose need frozen persistence semantics (P10A-C0-05).',
    'P01/P10A/P11: real final-output exclusivity, process fencing, lost-process stop and firmware cancellation are unverified.',
    'P02/P04/P06/P07/P08/P09: field map, calibration, model/images, envelope, posture, evidence tolerances and measured budgets are missing.',
    'ROS1 Noetic/catkin/Python3.8 target execution, physical actions and an actual 600-second match are unverified.',
)


def _base_report(mode):
    return dict(schema_version=1, generated_at=datetime.now(timezone.utc).isoformat(),
                contract_version=c.CONTRACT_VERSION, mode=mode, origin='simulated',
                status='blocked', validation='not_run', results={},
                tiers=dict(offline='not_run', cross_package='not_run', ros='unverified',
                           device_control='unverified', physical_action='unverified', full_match='unverified'),
                blockers=list(BLOCKERS))


def _protect_output(output, config):
    if output is None:
        return
    destination = Path(output).resolve()
    c.require(destination.suffix.lower() == '.json', 'output', 'report destination must be a .json file')
    config = config.resolve()
    same = destination == config or (destination.is_file() and config.is_file()
                                    and os.path.samefile(str(destination), str(config)))
    c.require(not same, 'output', 'report must not overwrite configuration')
    package = Path(__file__).resolve().parents[1]
    protected = [package, Path(sysconfig.get_path('scripts')).resolve()]
    if package.parent.name == 'src':
        protected.append(package.parent.parent / 'scripts')
    for relative in ('config/hardware.json', 'examples/healthy_interfaces.json', 'launch/preflight.launch'):
        protected.append(get_resource_path(relative).parent)
    c.require(not any(destination == root or root in destination.parents for root in protected),
              'output', 'report must not overwrite installed/source code or input resources')


def _publish(report, output):
    payload = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + '\n'
    if output is None:
        print(payload, end='')
        return
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=str(destination.parent),
                                         prefix='.p11-report-', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        os.replace(str(temporary), str(destination))
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    print(json.dumps(dict(status=report['status'], origin='simulated', validation=report['validation'],
                          output=str(destination.resolve())), ensure_ascii=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description='P11 offline replay only; formal robot execution is unavailable.')
    parser.add_argument('--mode', choices=('offline', 'formal'), default='offline')
    parser.add_argument('--layout', choices=('V1', 'V2', 'V3'), default='V1')
    parser.add_argument('--scenario', choices=SCENARIOS+('all',), default='nominal')
    parser.add_argument('--config', help='explicit simulated P09 mission configuration')
    parser.add_argument('--output', help='JSON evidence report; defaults to stdout')
    parser.add_argument('--audit', action='store_true', help='only audit reviewed dependency files and imports')
    args, unknown = parser.parse_known_args(argv)
    if any(not any(x.startswith(key+':=') for key in ('__name', '__log', '__ns')) for x in unknown):
        parser.error('unrecognized arguments: '+' '.join(unknown))
    report = _base_report(args.mode)
    output_safe = False
    try:
        config_path = Path(args.config) if args.config else get_resource_path('config/mission/offline.example.json')
        _protect_output(args.output, config_path)
        output_safe = True
        c.require(args.mode == 'offline', 'mode', 'formal execution blocked; see blockers', c.ErrorCode.NOT_AUTHORIZED)
        dependencies = audit_dependencies()
        report['dependencies'] = c.to_data(dependencies)
        if args.audit:
            report['validation'] = 'dependency_audit'
            report['status'] = 'passed' if all(d.available is c.Fact.CONFIRMED for d in dependencies) else 'blocked'
            report['tiers']['offline'] = 'code_audit_only'
            exit_code = 0 if report['status'] == 'passed' else 2
        else:
            config = MissionConfig.load(config_path)
            scenarios = SCENARIOS if args.scenario == 'all' else (args.scenario,)
            service = IntegrationService(args.layout, config)
            checks = service.verify(m.IntegrationRequest(c.CONTRACT_VERSION, dependencies, scenarios, c.Origin.SIMULATED))
            report.update(validation='replay', contract_report=c.to_data(checks), results=service.results,
                          status='passed' if all(x.status is c.Fact.CONFIRMED for x in checks.checks) else 'failed')
            report['tiers']['offline'] = report['status']
            exit_code = 0 if report['status'] == 'passed' else 1
    except (OSError, ValueError, TypeError) as error:
        report['error'] = str(error)
        exit_code = 2
    try:
        if args.output is not None and not output_safe:
            print(json.dumps(report, ensure_ascii=False), file=sys.stderr)
        else:
            _publish(report, args.output)
    except (OSError, ValueError) as error:
        print('P11 report could not be saved: '+str(error), file=sys.stderr)
        return 2
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
