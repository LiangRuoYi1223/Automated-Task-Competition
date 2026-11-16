"""P00-R02/R05: concrete models, mapping permutations, layout boundaries."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from treasure_auto.contracts import core as c
try:
    from contract_fixtures import samples as s
    from treasure_auto.contracts import models as m, rules as r, ports
except ImportError:
    s = None


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(s, 'P00 models/fixtures are not implemented')

    def test_R05_six_unique_mappings_cake_fixed_A(self):
        samples = s.mapping_samples()
        self.assertEqual(len(samples), 6)
        self.assertEqual(len({tuple(a.category for a in x.assignments) for x in samples}), 6)
        for snapshot in samples:
            self.assertEqual(snapshot.destination(m.Category.CAKE), m.Area.A)
            self.assertEqual(set(a.area for a in snapshot.assignments), {m.Area.B, m.Area.C, m.Area.D})
            self.assertEqual(c.loads(c.dumps(snapshot)), snapshot)

    def test_R05_mapping_missing_duplicate_cake_rejected(self):
        snapshot = s.mapping_samples()[0]
        for assignments in (snapshot.assignments[:2],
                            (snapshot.assignments[0],) * 3,
                            (replace(snapshot.assignments[0], category=m.Category.CAKE),) + snapshot.assignments[1:]):
            with self.subTest(assignments=assignments), self.assertRaises(c.ContractError):
                replace(snapshot, assignments=assignments)

    def test_R05_mapping_unconfirmed_direction_rejected(self):
        with self.assertRaises(c.ContractError):
            replace(s.mapping_samples()[0], direction=c.Fact.UNKNOWN)

    def test_R05_each_layout_valid_sample_and_invalid_sample(self):
        for kind in m.LayoutKind:
            with self.subTest(kind=kind):
                good, bad = s.layout_pair(kind)
                r.require_layout_quality(good)
                with self.assertRaises(c.ContractError):
                    r.require_layout_quality(bad)
                self.assertEqual(c.loads(c.dumps(good)), good)

    def test_R05_v3_exact_80_or_unknown_is_not_pass(self):
        good, _ = s.layout_pair(m.LayoutKind.V3)
        for ratio in (0.8, 0.799, None):
            changed = replace(good.bricks[1], support_ratio_lower=ratio)
            with self.subTest(ratio=ratio), self.assertRaises(c.ContractError):
                r.require_layout_quality(replace(good, bricks=(good.bricks[0], changed, good.bricks[2])))

    def test_R05_rebuild_second_and_third_need_current_round(self):
        good, _ = s.layout_pair(m.LayoutKind.V3)
        with self.assertRaises(c.ContractError):
            r.require_layout_quality(replace(good, layout_recovery_round=1))

    def test_R02_run_keeps_original_600_seconds(self):
        run = s.run_context()
        self.assertEqual(run.deadline - run.started_at, 600)
        with self.assertRaises(c.ContractError):
            replace(run, deadline=s.stamp(701))
        self.assertEqual(r.remaining_seconds(run, s.stamp(290)), 410)

    def test_R02_arm_request_binds_speed_unit_phase_timeout(self):
        command = s.arm_command()
        for kwargs in (dict(speed_mm_s=0), dict(speed_unit='mm/min'),
                       dict(timeout_s=float('nan')), dict(timeout_s=-1)):
            with self.subTest(kwargs=kwargs), self.assertRaises(c.ContractError):
                replace(command, **kwargs)
        self.assertEqual(c.loads(c.dumps(command)), command)

    def test_R02_unmeasured_parameter_does_not_become_default(self):
        param = m.ParameterSpec('arm_speed', c.Owner.P01, 'mm/s', None, None,
                                None, None, m.Measurement.UNMEASURED,
                                'measure robot firmware range', ())
        with self.assertRaises(c.ContractError) as e:
            r.require_measured_parameter(param)
        self.assertEqual(e.exception.code, c.ErrorCode.UNCONFIGURED)

    def test_R02_space_is_immutable_and_retains_unknown(self):
        space = s.space_snapshot()
        self.assertEqual(space.occupancies[0].state, m.OccupancyState.UNKNOWN)
        detached = c.to_data(space)
        detached['occupancies'].clear()
        self.assertEqual(len(space.occupancies), 1)

    def test_R02_invalid_full_image_coordinates_rejected(self):
        surface = s.top_surface()
        with self.assertRaises(c.ContractError):
            replace(surface, corners=(m.PixelPoint(1000, 1),) + surface.corners[1:])
        with self.assertRaises(c.ContractError):
            replace(surface, corners=surface.corners[:3])

    def test_R02_top_surface_unknown_plane_preserved(self):
        surface = s.top_surface()
        unknown = replace(surface, plane_basis=m.PlaneBasis.UNKNOWN, height_mm=None)
        self.assertIsNone(c.loads(c.dumps(unknown)).height_mm)

    def test_R02_ports_cover_all_13_owners_and_have_typed_signatures(self):
        import inspect
        import typing
        self.assertEqual(set(ports.PORT_OWNERS.values()), set(c.Owner))
        for name, owner in ports.PORT_OWNERS.items():
            port = getattr(ports, name)
            methods = [v for k, v in vars(port).items() if not k.startswith('_') and inspect.isfunction(v)]
            self.assertTrue(methods, (name, owner))
            for method in methods:
                hints = typing.get_type_hints(method)
                self.assertIn('return', hints)
                self.assertTrue(all(p == 'self' or p in hints for p in inspect.signature(method).parameters))

    def test_R05_all_sample_records_roundtrip(self):
        samples = s.all_samples()
        self.assertGreaterEqual(len(samples), 13)
        for name, record in samples.items():
            with self.subTest(name=name):
                self.assertEqual(c.loads(c.dumps(record)), record)
                r.require_simulated(record)


if __name__ == '__main__':
    unittest.main()
