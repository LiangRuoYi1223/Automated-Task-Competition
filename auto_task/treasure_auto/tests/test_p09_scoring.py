import unittest
from dataclasses import replace
from test_p09_support import *
from treasure_auto.mission.scoring import ScoreBrick,ScoreScene,ScoreZone,estimate


def scored(target='b1',category=m.Category.DOLL,area=m.Area.B,layer=1,supports=(),zone=ScoreZone.COLLECTION):
    e=envelope(c.Owner.P08,target_id=target,action_id=None)
    return ScoreBrick(e,target,'inventory-'+target,category,area,zone,layer,supports,
        c.Fact.CONFIRMED,c.Fact.CONFIRMED,s.stamp(109),.1*layer,c.Fact.CONFIRMED,
        (proof(target=target,ident=target),))


def scene(bricks,exit=False,priority=c.Fact.CONFIRMED,deductions=0.):
    ex=m.ExitEvidence(envelope(c.Owner.P03,target_id=None,action_id=None),m.Area.A,c.Fact.CONFIRMED,(proof(),)) if exit else None
    return ScoreScene(s.mapping_samples()[0],tuple(bricks),ex,priority,deductions)


class ScoreTests(unittest.TestCase):
    def points(self,bricks,**kw):return estimate(scene(bricks,**kw),s.run_context(),s.stamp(111))

    def test_T07_V1_V2_V3_area_6_8_13(self):
        cases=[((1,1,1),((),(),()),6),((1,1,2),((),(),('b1','b2')),8),((1,2,3),((),('b1',),('b2',)),13)]
        for layers,supports,want in cases:
            bs=tuple(scored('b'+str(i+1),layer=l,supports=ss) for i,(l,ss) in enumerate(zip(layers,supports)))
            result=self.points(bs)
            self.assertEqual(result.estimate.confirmed_points,want)
            self.assertFalse(result.estimate.unresolved_target_ids)

    def test_T07_full_21_27_42(self):
        for layers,supports,want in [((1,1,1),((),(),()),21),((1,1,2),((),(),(0,1)),27),((1,2,3),((),(0,),(1,)),42)]:
            bs=[scored('cake',m.Category.CAKE,m.Area.A)]
            for category,area in [(m.Category.DOLL,m.Area.B),(m.Category.CLOCK,m.Area.C),(m.Category.BANANA,m.Area.D)]:
                ids=[category.value+str(i) for i in range(3)]
                bs.extend(scored(ids[i],category,area,layers[i],tuple(ids[j] for j in supports[i])) for i in range(3))
            self.assertEqual(self.points(bs,exit=True).estimate.confirmed_points,want)

    def test_transport_and_mixed_support_inherit_transport(self):
        bs=(scored('b1',zone=ScoreZone.TRANSPORT),scored('b2'),scored('top',layer=2,supports=('b1','b2')))
        out=self.points(bs)
        self.assertEqual([t.points for t in out.terms],[1,2,2])
        self.assertEqual(out.terms[-1].zone,ScoreZone.TRANSPORT)

    def test_wrong_support_layer_deducted_once_for_two_wrong_bricks(self):
        bs=(scored('b1',category=m.Category.CLOCK),scored('b2',category=m.Category.BANANA),
            scored('top',layer=2,supports=('b1','b2')))
        out=self.points(bs)
        self.assertEqual([t.points for t in out.terms],[0,0,2])
        self.assertEqual(out.terms[-1].effective_layer,1)

    def test_mixed_correct_wrong_support_reduces_upper_layers(self):
        bs=(scored('b1'),scored('b2',category=m.Category.CLOCK),scored('top',layer=2,supports=('b1','b2')),
            scored('upper',layer=3,supports=('top',)))
        self.assertEqual([t.points for t in self.points(bs).terms],[2,0,2,4])

    def test_nonhorizontal_is_layer_one_not_layout_failure_zero(self):
        bs=(scored('b1'),replace(scored('top',layer=2,supports=('b1',)),horizontal=c.Fact.REJECTED))
        self.assertEqual([t.points for t in self.points(bs).terms],[2,2])

    def test_deadline_height_unreleased_outside_excluded(self):
        for change in [dict(height_top_m=.400001),dict(released=c.Fact.REJECTED),dict(whole_in_scoring=c.Fact.REJECTED)]:
            out=self.points((replace(scored(),**change),))
            self.assertEqual(out.estimate.confirmed_points,0)
        self.assertEqual(self.points((replace(scored(),height_top_m=.4),)).estimate.confirmed_points,2)
        late=replace(scored(),envelope=envelope(c.Owner.P08,t=701,target_id='b1',action_id=None),released_at=s.stamp(700),evidence=(proof(701,target='b1'),))
        self.assertEqual(estimate(scene((late,)),s.run_context(),s.stamp(702)).estimate.confirmed_points,0)

    def test_unknown_fields_stay_unresolved_and_known_part_kept(self):
        for change in [dict(support_ids=None),dict(horizontal=c.Fact.UNKNOWN),dict(released=c.Fact.UNKNOWN),
                       dict(released_at=None),dict(height_top_m=None),dict(zone=None),dict(evidence=())]:
            out=self.points((scored('known'),replace(scored('unknown'),**change)))
            self.assertEqual(out.estimate.confirmed_points,2)
            self.assertEqual(out.estimate.unresolved_target_ids,('unknown',))
        self.assertIsNone(self.points((replace(scored(),zone=None),)).estimate.confirmed_points)

    def test_cycle_missing_support_inconsistent_layer_cross_area_unknown(self):
        cases=[(scored('x',layer=2,supports=('x',)),),
               (scored('x',layer=2,supports=('absent',)),),
               (scored('base'),scored('x',layer=3,supports=('base',))),
               (scored('base'),scored('other',category=m.Category.CLOCK,area=m.Area.C),scored('x',layer=2,supports=('base','other')))]
        for bs in cases:self.assertIn('x',self.points(bs).estimate.unresolved_target_ids)

    def test_cake_priority_does_not_exclude_ordinary(self):
        bs=(scored('cake',m.Category.CAKE,m.Area.A),scored('doll'))
        self.assertEqual([t.points for t in self.points(bs,priority=c.Fact.REJECTED).terms],[0,2])
        self.assertEqual(self.points(bs,priority=c.Fact.UNKNOWN).estimate.unresolved_target_ids,('cake',))

    def test_unknown_penalties_not_promised_as_final_total(self):
        out=self.points((scored(),),deductions=None)
        self.assertEqual(out.estimate.confirmed_points,2)
        self.assertIn('match:deductions',out.estimate.unresolved_target_ids)

    def test_duplicate_inventory_identity_rejected(self):
        with self.assertRaises(c.ContractError):
            self.points((scored('one'),replace(scored('renamed'),inventory_id='inventory-one')))

    def test_expired_echo_or_old_run_evidence_is_unknown(self):
        b=scored()
        for ev in [replace(b.evidence[0],valid_until=s.stamp(110.5)),replace(b.evidence[0],kind=c.EvidenceKind.ECHO),replace(b.evidence[0],run_id='old')]:
            self.assertIsNone(self.points((replace(b,evidence=(ev,)),)).estimate.confirmed_points)


if __name__=='__main__':unittest.main()
