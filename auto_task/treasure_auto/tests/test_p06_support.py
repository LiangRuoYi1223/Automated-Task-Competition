import unittest
from dataclasses import replace
from test_p06_fixtures import *


class SupportTests(unittest.TestCase):
    def setUp(self):
        self.g=module(self,'support');self.i=module(self,'inputs')
        self.lower=occupancy();self.upper=occupancy('upper',z0=.1,z1=.2)

    def result(self,upper=None,lowers=None,contact=c.Fact.CONFIRMED,horizontal=c.Fact.CONFIRMED,proof=None):
        upper=upper or self.upper;lowers=(self.lower,) if lowers is None else lowers
        proof=proof or self.i.ContactProof(upper.target_id,tuple(x.target_id for x in lowers),0,
                              contact,horizontal,(evidence(target_id='upper'),))
        req=m.SupportGeometryRequest(envelope(c.Owner.P08,target_id='upper'),upper,lowers,
                                    space((upper,)+lowers),(evidence(target_id='upper'),))
        return self.g.compute_support(req,proof,.001)

    def test_T07_exact_80_percent_is_not_strict_quality(self):
        r=self.result(upper=replace(self.upper,footprint_m=square(.02)))
        self.assertAlmostEqual(r.upper_bottom_area_m2,.01)
        self.assertAlmostEqual(r.supported_area_m2,.008)
        self.assertAlmostEqual(r.ratio_lower_bound,.8)
        self.assertFalse(self.g.passes_quality(r))

    def test_T07_error_crossing_boundary_and_complete_overlap(self):
        r=self.result(upper=replace(self.upper,footprint_m=square(.019),error_m=.003))
        self.assertLess(r.ratio_lower_bound,.8)
        self.assertFalse(self.g.passes_quality(r))
        good=self.result()
        self.assertAlmostEqual(good.ratio_lower_bound,1.)
        self.assertTrue(self.g.passes_quality(good))

    def test_R07_union_area_does_not_double_count_two_overlapping_supports(self):
        a=replace(self.lower,footprint_m=(m.Point2(0,0),m.Point2(.06,0),m.Point2(.06,.1),m.Point2(0,.1)))
        b=replace(self.lower,target_id='lower-2',footprint_m=(m.Point2(.04,0),m.Point2(.1,0),m.Point2(.1,.1),m.Point2(.04,.1)))
        r=self.result(lowers=(a,b))
        self.assertAlmostEqual(r.supported_area_m2,.01)
        self.assertAlmostEqual(r.ratio_lower_bound,1.)
        self.assertEqual(set(r.support_ids),{'lower','lower-2'})

    def test_R07_overlap_without_contact_level_or_height_remains_unknown(self):
        for kw in (dict(contact=c.Fact.UNKNOWN),dict(horizontal=c.Fact.UNKNOWN),
                   dict(upper=replace(self.upper,z_min_m=.12)),dict(lowers=(replace(self.lower,state=m.OccupancyState.UNKNOWN),))):
            with self.subTest(kw=kw):
                r=self.result(**kw)
                self.assertIsNone(r.ratio_lower_bound)
                self.assertFalse(self.g.passes_quality(r))
                self.assertIsNotNone(r.reason)

    def test_R07_same_version_wrong_support_geometry_and_duplicate_ids_rejected(self):
        req=m.SupportGeometryRequest(envelope(c.Owner.P08,target_id='upper'),self.upper,(self.lower,),
                  space((self.upper,replace(self.lower,z_max_m=.09))),(evidence(target_id='upper'),))
        proof=self.i.ContactProof('upper',('lower',),0,c.Fact.CONFIRMED,c.Fact.CONFIRMED,(evidence(target_id='upper'),))
        with self.assertRaises(c.ContractError):self.g.compute_support(req,proof,.001)
        with self.assertRaises(c.ContractError):self.result(lowers=(self.lower,self.lower))

    def test_R07_large_uncertainty_cannot_create_area(self):
        r=self.result(upper=replace(self.upper,error_m=.2))
        self.assertEqual(r.ratio_lower_bound,0.)
        self.assertFalse(self.g.passes_quality(r))
