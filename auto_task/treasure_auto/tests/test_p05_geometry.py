from dataclasses import replace
from itertools import permutations
from test_p05_support import P05Case, config, template, frame, c, m


class GeometryTests(P05Case):
    def test_T01_all_six_permutations_have_consistent_reverse_lookup(self):
        for order in permutations((m.Category.DOLL,m.Category.CLOCK,m.Category.BANANA)):
            result=self.p.analyze_frame(frame(categories=order),template(),config())
            self.assertTrue(result.valid,result.reason)
            self.assertEqual(tuple(a.category for a in result.assignments),order)
            self.assertEqual({a.category:a.area for a in result.assignments},dict(zip(order,(m.Area.B,m.Area.C,m.Area.D))))

    def test_T03_perspective_and_rotations_preserve_slot_semantics(self):
        for project in [lambda x,y:((120+580*x+35*y)/(1+0.15*x+0.25*y),(110+20*x+290*y)/(1+0.15*x+0.25*y)),
                        lambda x,y:(800-600*x,700-200*y),lambda x,y:(500-200*y,200+600*x)]:
            f=frame(project=project)
            result=self.p.analyze_frame(f,template(),config())
            self.assertTrue(result.valid,result.reason)
            self.assertEqual(result.assignments[0],m.SlotAssignment(m.Area.B,m.Category.DOLL))
            self.assertEqual(len(result.homography),9)
            self.assertEqual(result.raw_corners,f.boards[0].corners)

    def test_T03_mirror_unknown_direction_and_disagreeing_markers_reject(self):
        original=frame()
        board=original.boards[0]
        cases=[frame(project=lambda x,y:(800-600*x,100+200*y)),
               replace(original,boards=(replace(board,directions=()),)),
               replace(original,boards=(replace(board,directions=(replace(board.directions[0],marker_key='ar:unverified'),)),)),
               replace(original,boards=(replace(board,directions=board.directions+(replace(board.directions[0],corner_order=(2,3,0,1)),)),))]
        for value in cases:
            with self.subTest(value=value.boards): self.assertFalse(self.p.analyze_frame(value,template(),config()).valid)

    def test_T02_missing_duplicate_unknown_cake_and_extra_patterns_reject(self):
        f=frame()
        cases=[replace(f,patterns=f.patterns[:2]),frame(categories=(m.Category.DOLL,)*3),
               frame(categories=(m.Category.DOLL,None,m.Category.BANANA)),
               frame(categories=(m.Category.DOLL,m.Category.CAKE,m.Category.BANANA)),
               replace(f,patterns=f.patterns+(f.patterns[0],))]
        for value in cases:
            with self.subTest(patterns=value.patterns): self.assertFalse(self.p.analyze_frame(value,template(),config()).valid)

    def test_R03_cross_slot_and_low_confidence_reject_with_specific_reason(self):
        f=frame()
        cross=replace(f.patterns[0],polygon=(m.PixelPoint(270,150),m.PixelPoint(360,150),m.PixelPoint(360,250),m.PixelPoint(270,250)))
        result=self.p.analyze_frame(replace(f,patterns=(cross,)+f.patterns[1:]),template(),config())
        self.assertFalse(result.valid)
        self.assertIn('slot',result.reason.field)
        self.assertFalse(self.p.analyze_frame(replace(f,patterns=(replace(f.patterns[0],confidence=0.2),)+f.patterns[1:]),template(),config()).valid)

    def test_R01_no_partial_multiple_or_degenerate_board(self):
        f=frame(); b=f.boards[0]
        cases=[(),(b,b),(replace(b,complete=c.Fact.UNKNOWN),),
               (replace(b,corners=tuple(m.PixelPoint(100+i*50,100) for i in range(4))),),
               (replace(b,corners=(b.corners[0],b.corners[2],b.corners[1],b.corners[3])),)]
        for boards in cases:
            with self.subTest(boards=boards): self.assertFalse(self.p.analyze_frame(replace(f,boards=boards),template(),config()).valid)

    def test_R01_full_image_coordinates_and_correction_required(self):
        f=frame()
        for image in [replace(f.image,distortion=m.Distortion.UNKNOWN),replace(f.image,distortion=m.Distortion.RAW),
                      replace(f.image,width=200)]:
            with self.subTest(image=image): self.assertFalse(self.p.analyze_frame(replace(f,image=image),template(),config()).valid)

