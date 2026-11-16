import unittest
from dataclasses import replace
from test_p08_support import *
from treasure_auto.placement.config import PlacementConfig
from treasure_auto.placement.targets import target_for_slot, validate_request, whole_in_region


class TargetsTests(unittest.TestCase):
    def test_T01_whole_brick_and_error_margin_not_just_center(self):
        self.assertFalse(whole_in_region(occupancy(x=.2),region()))
        self.assertFalse(whole_in_region(occupancy(x=.17,error=.01),region()))
        self.assertTrue(whole_in_region(occupancy(),region()))

    def test_R01_wrong_region_old_mapping_and_motion_rejected(self):
        req=request();target=occupancy(state=m.OccupancyState.RESERVED)
        validate_request(req,context(),space(),target,config())
        for bad in (replace(req,region=region(m.Area.C)),replace(req,mapping=replace(req.mapping,mapping_version='old')),
                    replace(req,plan=replace(req.plan,envelope=replace(req.plan.envelope,base_motion_epoch=1)))):
            with self.subTest(bad=bad),self.assertRaises(c.ContractError):
                validate_request(bad,context(),space(),target,config())

    def test_R01_unknown_holding_echo_only_stop_missing_withdraw_rejected(self):
        req=request()
        cases=(replace(req,holding=replace(req.holding,state=c.Holding.UNKNOWN)),
               replace(req,stop=replace(req.stop,physically_stopped=c.Fact.UNKNOWN)),
               replace(req,plan=replace(req.plan,segments=req.plan.segments[:-1])),
               replace(req,holding=replace(req.holding,tool_offset=None)))
        for bad in cases:
            with self.subTest(bad=bad),self.assertRaises(c.ContractError):
                validate_request(bad,context(),space(),occupancy(),config())

    def test_R02_center_target_and_actual_second_top_for_third(self):
        p=target_for_slot(layout(),space(),region(),occupancy('held',x=2),config(),s.stamp(110))
        self.assertAlmostEqual(p.desired_pose.x,0);self.assertAlmostEqual(p.desired_pose.y,0)
        below=occupancy('brick-2',x=.008,z=.123)
        state=layout(count=2,bricks=(brick(1),brick(2)))
        p=target_for_slot(state,space((below,)),region(),occupancy('held'),config(),s.stamp(110))
        self.assertAlmostEqual(p.occupancy.z_min_m,.223)
        self.assertAlmostEqual(p.desired_pose.x,.008)
        self.assertEqual(p.occupancy.support_ids,('brick-2',))

    def test_R02_V1_flat_V2_dual_base_and_missing_support(self):
        for kind in (m.LayoutKind.V1,m.LayoutKind.V2):
            first=occupancy()
            p=target_for_slot(layout(kind,1,bricks=(brick(kind=kind),)),space((first,)),region(),occupancy('held'),config(),s.stamp(110))
            self.assertAlmostEqual(p.occupancy.z_min_m,0)
            self.assertGreater(p.desired_pose.x,.1)
        low1=occupancy();low2=occupancy('brick-2',x=.105)
        p=target_for_slot(layout(m.LayoutKind.V2,2,bricks=(brick(kind=m.LayoutKind.V2),brick(2,kind=m.LayoutKind.V2))),
                          space((low1,low2)),region(),occupancy('held'),config(),s.stamp(110))
        self.assertEqual(p.occupancy.support_ids,('brick-1','brick-2'))
        with self.assertRaises(c.ContractError):
            target_for_slot(layout(count=2,bricks=(brick(),brick(2))),space(),region(),occupancy('held'),config(),s.stamp(110))

    def test_R02_cake_only_first_slot_and_stale_support(self):
        cake=replace(layout(),area=m.Area.A)
        self.assertEqual(target_for_slot(cake,space(),region(m.Area.A),occupancy('cake'),config(),s.stamp(110)).slot,1)
        with self.assertRaises(c.ContractError):
            target_for_slot(replace(cake,confirmed_count=1),space(),region(m.Area.A),occupancy(),config(),s.stamp(110))
        with self.assertRaises(c.ContractError):
            target_for_slot(layout(count=1,bricks=(brick(),)),space((occupancy(t=108),)),region(),occupancy('new'),config(),s.stamp(110))

    def test_config_unmeasured_nan_unknown_fields_and_live_refused(self):
        for changes in ({'center_tolerance_m':None},{'frame_gap_s':float('nan')},{'extra':1},
                        {'origin':'live'},{'measurement':'verified'},{'frame_gap_s':0}):
            with self.subTest(changes=changes),self.assertRaises(c.ContractError):
                PlacementConfig.from_dict(dict(config_data(),**changes))


if __name__=='__main__':unittest.main()
