"""P11-R04: exact C0, delivered module fingerprints, no implicit live approval."""
import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from treasure_auto.contracts import core as c, models as m
from treasure_auto.integration.service import IntegrationService, audit_dependencies


class ServiceTests(unittest.TestCase):
    def request(self, scenarios=('nominal',)):
        return m.IntegrationRequest('1.0.0', audit_dependencies(), scenarios, c.Origin.SIMULATED)

    def test_current_deliveries_are_present_and_only_offline_checks_confirmed(self):
        dependencies = audit_dependencies()
        self.assertEqual(len(dependencies), 12)
        self.assertTrue(all(d.available is c.Fact.CONFIRMED for d in dependencies), dependencies)
        report = IntegrationService().verify(self.request())
        self.assertEqual(c.loads(c.dumps(report)), report)
        self.assertEqual(report.checks[0].status, c.Fact.CONFIRMED)
        self.assertEqual(report.checks[0].tier, m.VerificationTier.OFFLINE)
        self.assertEqual(report.origin, c.Origin.SIMULATED)

    def test_missing_unknown_duplicate_or_wrong_version_dependency_blocks_execution(self):
        original = self.request()
        variants = [(), original.dependencies[:-1], original.dependencies + (original.dependencies[0],),
                    (replace(original.dependencies[0], contract_version='2.0.0'),) + original.dependencies[1:],
                    (replace(original.dependencies[0], available=c.Fact.UNKNOWN),) + original.dependencies[1:]]
        for dependencies in variants:
            report = IntegrationService().verify(replace(original, dependencies=dependencies))
            self.assertEqual(report.checks[0].status, c.Fact.UNKNOWN, dependencies)
            self.assertIsNotNone(report.checks[0].reason)

    def test_incompatible_contract_and_live_origin_are_explicitly_rejected(self):
        for change in ({'contract_version': '1.0.1'}, {'origin': c.Origin.LIVE}, {'origin': c.Origin.REPLAY}):
            report = IntegrationService().verify(replace(self.request(), **change))
            self.assertEqual(report.checks[0].status, c.Fact.REJECTED)

    def test_unknown_scenario_not_silently_substituted_and_empty_set_rejected(self):
        report = IntegrationService().verify(self.request(('missing', 'nominal')))
        self.assertEqual([x.status for x in report.checks], [c.Fact.REJECTED, c.Fact.CONFIRMED])
        with self.assertRaises(c.ContractError):
            IntegrationService().verify(self.request(()))
        with self.assertRaises(c.ContractError):
            IntegrationService().verify(self.request(('nominal', 'nominal')))

    def test_dependency_source_tamper_fails_audit(self):
        import shutil
        import tempfile
        source = Path(__file__).resolve().parents[1] / 'src/treasure_auto'
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / 'treasure_auto'
            shutil.copytree(source, package, ignore=shutil.ignore_patterns('verification', '__pycache__'))
            (package / 'contracts/core.py').write_text('# stale contract\n', encoding='utf-8')
            statuses = audit_dependencies(package)
            self.assertTrue(any(d.available is not c.Fact.CONFIRMED for d in statuses))

    def test_malformed_dependency_file_inventory_reports_unknown(self):
        import json
        import shutil
        import tempfile
        source = Path(__file__).resolve().parents[1] / 'src/treasure_auto'
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / 'treasure_auto'
            shutil.copytree(source, package, ignore=shutil.ignore_patterns('verification', '__pycache__'))
            manifest_path = package / 'integration/dependencies.json'
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            manifest['dependencies'][0]['files'] = list(manifest['dependencies'][0]['files'])
            manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
            statuses = audit_dependencies(package)
            self.assertIs(statuses[0].available, c.Fact.UNKNOWN)
            self.assertIsNotNone(statuses[0].reason)


if __name__ == '__main__':
    unittest.main()
