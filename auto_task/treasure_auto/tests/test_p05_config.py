import json
from dataclasses import replace
from pathlib import Path
from test_p05_support import P05Case, config, template, c, m


class ConfigTests(P05Case):
    def test_R01_unmeasured_config_never_enables_recognition(self):
        path=Path(__file__).resolve().parents[1]/'config/classification/unmeasured.json'
        value=self.p.ClassificationConfig.from_dict(json.loads(path.read_text(encoding='utf-8')))
        with self.assertRaises(c.ContractError): value.require_ready(c.Origin.SIMULATED)

    def test_R01_simulated_parameters_cannot_enable_live(self):
        config().require_ready(c.Origin.SIMULATED)
        with self.assertRaises(c.ContractError): config().require_ready(c.Origin.LIVE)

    def test_R01_bad_thresholds_and_unknown_fields_rejected(self):
        for field, value in [('min_confirmations',1),('max_additional_views',True),
                             ('max_frame_age_s',float('nan')),('slot_margin',-0.1),
                             ('min_pattern_confidence',1.01),('min_confirmation_span_s',0)]:
            with self.subTest(field=field), self.assertRaises(c.ContractError): replace(config(),**{field:value})
        with self.assertRaises(c.ContractError): self.p.ClassificationConfig.from_dict({'surprise':1})

    def test_R01_template_needs_exact_unique_nonoverlapping_BCD(self):
        t=template()
        for slots in [t.slots[:2],(t.slots[0],)*3,
                      (replace(t.slots[0],right=0.8),)+t.slots[1:]]:
            with self.subTest(slots=slots), self.assertRaises(c.ContractError): replace(t,slots=slots)
        with self.assertRaises(c.ContractError): replace(t.slots[0],area=m.Area.A)

