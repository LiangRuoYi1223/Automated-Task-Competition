"""Explicit local assembly inputs. None are registered as C0 wire types."""
from dataclasses import dataclass
from enum import Enum
from typing import Optional,Tuple
from treasure_auto.contracts import core as c,models as m


class Stage(str,Enum):
    NAV_BOARD='navigate_board'
    READ_BOARD='read_board'
    NAV_CENTRAL='navigate_central'
    SELECT='select_one'
    OBSERVE='observe_central'
    GRASP='grasp_and_transport_ready'
    CARRY='carry_to_destination'
    PLACE='release_in_destination'
    LAYOUT='assess_layout'
    EMPTY_READY='confirm_empty_ready'
    RETURN='return_central'
    REVALIDATE='revalidate_after_restart'
    STOPPED='stopped'


OWNERS={Stage.NAV_BOARD:c.Owner.P03,Stage.READ_BOARD:c.Owner.P05,Stage.NAV_CENTRAL:c.Owner.P03,
        Stage.OBSERVE:c.Owner.P04,Stage.GRASP:c.Owner.P07,Stage.CARRY:c.Owner.P03,
        Stage.PLACE:c.Owner.P08,Stage.LAYOUT:c.Owner.P08,Stage.EMPTY_READY:c.Owner.P07,Stage.RETURN:c.Owner.P03}


@dataclass(frozen=True)
class Directive:
    stage:Stage
    envelope:c.Envelope
    order:Optional[m.WorkOrder]
    owner:c.Owner


@dataclass(frozen=True)
class MissionFrame:
    inputs:m.MissionInput
    plans:Tuple[m.GeometryPlan,...]=()
    readiness:Optional[m.DriveReadiness]=None
    stop:Optional[m.StopReport]=None
    stillness:Optional[m.StillnessCertificate]=None
    affected_targets:Tuple[str,...]=()
    scene_epoch:int=0
    # Local P04 assembly, never a replacement C0 wire contract.
    perception_result:Optional[m.PerceptionResult]=None
    perception_policy:object=None
    field_input:object=None


@dataclass(frozen=True)
class MissionOutput:
    decision:m.MissionDecision
    directive:Optional[Directive]
    stage:Stage
    issues:Tuple[str,...]
    stop_request:Optional[m.StopRequest]=None
    perception_counts:Optional[Tuple[int,int,int]]=None
    field_decision:object=None
