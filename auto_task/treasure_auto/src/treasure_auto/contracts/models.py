"""C0 domain DTOs. These describe facts/requests, never perform device actions."""
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple
from .core import (Record, Owner, Origin, Outcome, Fact, Holding, Reason, ErrorCode,
                   Envelope, Versions, ClockStamp, EvidenceRef, ArmPoint, MapPose,
                   HoldingSnapshot, ActionResult, require, nonnegative, monotonic)


class Category(str, Enum):
    DOLL = 'doll'
    CLOCK = 'clock'
    BANANA = 'banana'
    CAKE = 'cake'


class Area(str, Enum):
    A = 'A'
    B = 'B'
    C = 'C'
    D = 'D'


class RunState(str, Enum):
    PREPARATION = 'preparation'
    OFFLINE = 'offline'
    ARMED = 'armed'
    RUNNING = 'running'
    RECOVERING = 'recovering'
    STOPPING = 'stopping'
    ENDED = 'ended'


class MotionSource(str, Enum):
    NAVIGATION = 'navigation'
    PRECISION = 'precision'
    RECOVERY = 'recovery'
    DEBUG = 'debug'
    ARM = 'arm'
    PUMP = 'pump'


class Phase(str, Enum):
    APPROACH = 'approach'
    DESCEND = 'descend'
    CONTACT = 'contact'
    TRIAL_LIFT = 'trial_lift'
    TRANSPORT = 'transport'
    PLACE = 'place'
    WITHDRAW = 'withdraw'
    DISMANTLE = 'dismantle'
    TEMPORARY = 'temporary'


class LayoutKind(str, Enum):
    V1 = 'V1'
    V2 = 'V2'
    V3 = 'V3'


class Visibility(str, Enum):
    VISIBLE = 'visible'
    PARTIAL = 'partial'
    OCCLUDED = 'occluded'
    UNKNOWN = 'unknown'


class OccupancyState(str, Enum):
    ACTUAL = 'actual'
    RESERVED = 'reserved'
    UNKNOWN = 'unknown'


class ObservationState(str, Enum):
    DETECTED = 'detected'
    EMPTY = 'empty'
    BAD_IMAGE = 'bad_image'
    BACKEND_FAILURE = 'backend_failure'
    UNKNOWN_CLASS = 'unknown_class'


class Distortion(str, Enum):
    RAW = 'raw'
    CORRECTED = 'corrected'
    UNKNOWN = 'unknown'


class PlaneBasis(str, Enum):
    CALIBRATION = 'calibration'
    SUPPORT = 'support'
    VERIFIED_DEPTH = 'verified_depth'
    KNOWN_DIMENSIONS = 'known_dimensions'
    UNKNOWN = 'unknown'


class Scene(str, Enum):
    BOARD = 'board'
    CENTRAL = 'central'
    PLACEMENT = 'placement'
    TOOL = 'tool'


class Measurement(str, Enum):
    UNMEASURED = 'unmeasured'
    SIMULATED = 'simulated'
    VERIFIED = 'verified'


class CommitStatus(str, Enum):
    COMMITTED = 'committed'
    REJECTED = 'rejected'
    UNKNOWN = 'unknown'


class IntentOperation(str, Enum):
    ARM = 'arm'
    BASE_GOAL = 'base_goal'
    BASE_VELOCITY = 'base_velocity'
    PUMP = 'pump'


class HoldingCause(str, Enum):
    OBSERVATION = 'observation'
    RELEASE = 'release'
    DROP_RECONCILED = 'drop_reconciled'


class ClearanceCause(str, Enum):
    MOVED = 'observed_moved'
    EMPTY = 'observed_empty'


class SignalKind(str, Enum):
    START = 'start'
    EMERGENCY_STOP = 'emergency_stop'
    DEADLINE = 'deadline'
    MANUAL_DEBUG = 'manual_debug'
    FINISH = 'finish'


class MissionPhase(str, Enum):
    WAIT_MAPPING = 'wait_mapping'
    CAKE = 'cake'
    ORDINARY = 'ordinary'
    RECOVERY = 'recovery'
    STOPPED = 'stopped'


class CleanupState(str, Enum):
    RETAINED = 'retained'
    PENDING = 'pending'
    COMPLETE = 'complete'
    UNKNOWN = 'unknown'


class VerificationTier(str, Enum):
    OFFLINE = 'offline'
    CROSS_PACKAGE = 'cross_package'
    ROS = 'ros'
    DEVICE_CONTROL = 'device_control'
    PHYSICAL_ACTION = 'physical_action'
    FULL_MATCH = 'full_match'


class InventoryLocation(str, Enum):
    CENTRAL = 'central'
    HELD = 'held'
    REGION = 'region'
    TEMPORARY = 'temporary'
    UNKNOWN = 'unknown'


@dataclass(frozen=True)
class ParameterSpec(Record):
    name: str
    owner: Owner
    unit: str
    value: Optional[float]
    minimum: Optional[float]
    maximum: Optional[float]
    range_source: Optional[str]
    measurement: Measurement
    acquisition_method: str
    evidence: Tuple[EvidenceRef, ...]

    def _validate(self):
        if self.minimum is not None and self.maximum is not None:
            require(self.minimum <= self.maximum, 'maximum', 'range inverted')
        if self.value is not None:
            require(self.minimum is None or self.value >= self.minimum, 'value', 'below range')
            require(self.maximum is None or self.value <= self.maximum, 'value', 'above range')


@dataclass(frozen=True)
class ConsumerContext(Record):
    run_id: str
    instance_id: str
    control_epoch: int
    now: ClockStamp
    versions: Versions
    request_id: Optional[str]
    action_id: Optional[str]
    target_id: Optional[str]
    layout_recovery_round: Optional[int]
    rebuild_generation: Optional[int]
    base_motion_epoch: Optional[int]
    origin: Origin


@dataclass(frozen=True)
class RunContext(Record):
    envelope: Envelope
    state: RunState
    start_signal_id: str
    started_at: ClockStamp
    deadline: ClockStamp
    ever_restarted: bool

    def _validate(self):
        monotonic(self.started_at, 'started_at')
        monotonic(self.deadline, 'deadline')
        require(self.deadline - self.started_at == 600.0, 'deadline', 'original duration is exactly 600 s')


@dataclass(frozen=True)
class ControlPermit(Record):
    envelope: Envelope
    permit_id: str
    source: MotionSource
    purpose: str
    prerequisite_evidence: Tuple[EvidenceRef, ...]

    def _validate(self):
        require(self.envelope.producer is Owner.P10A, 'envelope.producer', 'only P10A issues permits')


@dataclass(frozen=True)
class Capability(Record):
    name: str
    status: Fact
    reason: Optional[Reason]
    evidence: Tuple[EvidenceRef, ...]


@dataclass(frozen=True)
class HealthSnapshot(Record):
    envelope: Envelope
    capabilities: Tuple[Capability, ...]


@dataclass(frozen=True)
class BaseCommand(Record):
    envelope: Envelope
    permit: ControlPermit
    intent_receipt_id: str
    linear_m_s: float
    angular_rad_s: float
    source: MotionSource


@dataclass(frozen=True)
class ArmCommand(Record):
    envelope: Envelope
    permit: ControlPermit
    intent_receipt_id: str
    phase: Phase
    target: ArmPoint
    speed_mm_s: float
    speed_unit: str
    timeout_s: float
    preconditions: Tuple[EvidenceRef, ...]

    def _validate(self):
        require(self.speed_mm_s > 0, 'speed_mm_s', 'positive stage speed required')
        require(self.timeout_s > 0, 'timeout_s', 'positive stage timeout required')
        require(self.speed_unit == 'mm/s', 'speed_unit', 'adapter alone converts mm/min', ErrorCode.INVALID_UNIT)
        require(self.envelope.action_id is not None, 'envelope.action_id', 'arm action identity required')


@dataclass(frozen=True)
class PumpCommand(Record):
    envelope: Envelope
    permit: ControlPermit
    intent_receipt_id: str
    enabled: bool
    phase: Phase
    holding: HoldingSnapshot


@dataclass(frozen=True)
class ActionEvent(Record):
    envelope: Envelope
    result: ActionResult


@dataclass(frozen=True)
class StopRequest(Record):
    envelope: Envelope
    reason: Reason
    holding: HoldingSnapshot


@dataclass(frozen=True)
class StopReport(Record):
    envelope: Envelope
    requested: Fact
    submitted: Fact
    echoed: Fact
    physically_stopped: Fact
    old_queue_empty: Fact
    evidence: Tuple[EvidenceRef, ...]


@dataclass(frozen=True)
class Point2(Record):
    x: float
    y: float


@dataclass(frozen=True)
class RegionBoundary(Record):
    area: Area
    frame_id: str
    polygon_m: Tuple[Point2, ...]
    placement_center: MapPose
    error_m: float

    def _validate(self):
        require(len(self.polygon_m) >= 3, 'polygon_m', 'at least three corners')
        nonnegative(self.error_m, 'error_m')
        require(self.placement_center.frame_id == self.frame_id, 'frame_id', 'center must share frame')


@dataclass(frozen=True)
class MapMetadata(Record):
    envelope: Envelope
    map_version: str
    image_uri: str
    resolution_m_per_cell: float
    origin: MapPose
    regions: Tuple[RegionBoundary, ...]
    measurement: Measurement

    def _validate(self):
        require(self.resolution_m_per_cell > 0, 'resolution_m_per_cell', 'must be positive')


@dataclass(frozen=True)
class LocalizationSnapshot(Record):
    envelope: Envelope
    pose: Optional[MapPose]
    trusted: Fact
    base_motion_epoch: int
    evidence: Tuple[EvidenceRef, ...]
    reason: Optional[Reason]

    def _validate(self):
        nonnegative(self.base_motion_epoch, 'base_motion_epoch')
        require(self.trusted is not Fact.CONFIRMED or self.pose is not None, 'pose', 'trusted needs pose')


@dataclass(frozen=True)
class WaypointQuery(Record):
    envelope: Envelope
    name: str
    purpose: str


@dataclass(frozen=True)
class WaypointResult(Record):
    envelope: Envelope
    name: str
    purpose: str
    pose: Optional[MapPose]
    valid: Fact
    reason: Optional[Reason]


@dataclass(frozen=True)
class PixelPoint(Record):
    u: float
    v: float


@dataclass(frozen=True)
class PixelTransform(Record):
    crop_u: float
    crop_v: float
    scale_x: float
    scale_y: float
    pad_u: float
    pad_v: float

    def _validate(self):
        require(self.scale_x > 0 and self.scale_y > 0, 'scale_x', 'scales must be positive')


@dataclass(frozen=True)
class ImageObservation(Record):
    envelope: Envelope
    observation_id: str
    image: EvidenceRef
    width: int
    height: int
    encoding: str
    camera_frame: str
    distortion: Distortion
    capture_pose: Optional[MapPose]
    preprocessing: PixelTransform

    def _validate(self):
        require(self.width > 0 and self.height > 0, 'width', 'positive image dimensions required')


@dataclass(frozen=True)
class TopSurface(Record):
    corners: Tuple[PixelPoint, PixelPoint, PixelPoint, PixelPoint]
    contour: Tuple[PixelPoint, ...]
    image_width: int
    image_height: int
    frame_id: str
    distortion: Distortion
    error_px: float
    plane_basis: PlaneBasis
    height_mm: Optional[float]
    visibility: Visibility
    evidence: Tuple[EvidenceRef, ...]

    def _validate(self):
        require(self.image_width > 0 and self.image_height > 0, 'image_width', 'positive image bounds')
        for name in ('corners', 'contour'):
            for i, p in enumerate(getattr(self, name)):
                require(0 <= p.u < self.image_width and 0 <= p.v < self.image_height,
                        '{}[{}]'.format(name, i), 'coordinates must be full-image u/v')
        require(len(set(self.corners)) == 4, 'corners', 'four distinct real corners required')
        nonnegative(self.error_px, 'error_px')


@dataclass(frozen=True)
class TargetObservation(Record):
    observation_id: str
    target_id: Optional[str]
    identity_confirmed: Fact
    category: Optional[Category]
    top: Optional[TopSurface]
    visibility: Visibility
    evidence: Tuple[EvidenceRef, ...]


@dataclass(frozen=True)
class ToolObservation(Record):
    envelope: Envelope
    target_id: Optional[str]
    rim: Tuple[PixelPoint, ...]
    center: Optional[PixelPoint]
    same_object: Fact
    evidence: Tuple[EvidenceRef, ...]


@dataclass(frozen=True)
class PerceptionRequest(Record):
    image: ImageObservation
    scene: Scene
    previous_targets: Tuple[TargetObservation, ...]


@dataclass(frozen=True)
class PerceptionResult(Record):
    envelope: Envelope
    state: ObservationState
    targets: Tuple[TargetObservation, ...]
    tool: Optional[ToolObservation]
    reason: Optional[Reason]


@dataclass(frozen=True)
class SlotAssignment(Record):
    area: Area
    category: Category


@dataclass(frozen=True)
class BoardObservation(Record):
    image: ImageObservation
    template_version: str
    direction: Fact
    slots: Tuple[SlotAssignment, ...]
    evidence: Tuple[EvidenceRef, ...]


@dataclass(frozen=True)
class MappingSnapshot(Record):
    envelope: Envelope
    mapping_version: str
    assignments: Tuple[SlotAssignment, ...]
    direction: Fact
    frozen_at: ClockStamp
    evidence: Tuple[EvidenceRef, ...]

    def _validate(self):
        require(len(self.assignments) == 3 and
                {a.area for a in self.assignments} == {Area.B, Area.C, Area.D} and
                {a.category for a in self.assignments} == {Category.DOLL, Category.CLOCK, Category.BANANA},
                'assignments', 'complete bijection B/C/D required; cake fixed A', ErrorCode.INVALID_MAPPING)
        require(self.direction is Fact.CONFIRMED, 'direction', 'board direction must be confirmed')
        require(bool(self.evidence), 'evidence', 'mapping freeze requires evidence')
        monotonic(self.frozen_at, 'frozen_at')

    def destination(self, category):
        require(type(category) is Category, 'category', 'unknown category')
        return Area.A if category is Category.CAKE else next(a.area for a in self.assignments if a.category is category)


@dataclass(frozen=True)
class ClassificationRequest(Record):
    envelope: Envelope
    read_deadline: ClockStamp
    observations: Tuple[BoardObservation, ...]
    max_additional_views: int


@dataclass(frozen=True)
class ClassificationResult(Record):
    envelope: Envelope
    result: ActionResult
    mapping: Optional[MappingSnapshot]
    additional_view: Optional[WaypointQuery]


@dataclass(frozen=True)
class Occupancy(Record):
    target_id: str
    state: OccupancyState
    frame_id: str
    footprint_m: Tuple[Point2, ...]
    z_min_m: float
    z_max_m: float
    error_m: float
    support_ids: Tuple[str, ...]
    observed_at: ClockStamp
    visibility: Visibility
    area: Optional[Area]
    evidence: Tuple[EvidenceRef, ...]

    def _validate(self):
        require(len(self.footprint_m) >= 3, 'footprint_m', 'footprint needs at least 3 points')
        require(self.z_min_m <= self.z_max_m, 'z_max_m', 'height interval inverted')
        nonnegative(self.error_m, 'error_m')
        require(self.target_id not in self.support_ids, 'support_ids', 'cannot support itself')


@dataclass(frozen=True)
class SpaceSnapshot(Record):
    envelope: Envelope
    version: int
    occupancies: Tuple[Occupancy, ...]

    def _validate(self):
        nonnegative(self.version, 'version')
        require(self.envelope.versions.space == self.version, 'version', 'envelope space version mismatch')
        require(len({o.target_id for o in self.occupancies}) == len(self.occupancies),
                'occupancies', 'duplicate target occupancy')


@dataclass(frozen=True)
class StillnessCertificate(Record):
    envelope: Envelope
    covered_target_ids: Tuple[str, ...]
    coverage_polygon_m: Tuple[Point2, ...]
    frame_id: str
    capture_start: ClockStamp
    capture_end: ClockStamp
    observation_ids: Tuple[str, ...]
    continuous: Fact
    all_stopped: Fact
    max_adjacent_motion_m: float
    max_total_motion_m: float
    tolerance_m: float
    scene_epoch: int
    evidence: Tuple[EvidenceRef, ...]


@dataclass(frozen=True)
class CameraCalibration(Record):
    version: str
    camera_frame: str
    intrinsics: Tuple[float, ...]
    distortion_coefficients: Tuple[float, ...]
    distortion: Distortion
    valid_domain: str
    fit_error_px: float
    validation_error_mm: float
    independent_validation: Tuple[EvidenceRef, ...]

    def _validate(self):
        require(len(self.intrinsics) == 9, 'intrinsics', 'row-major 3x3 matrix required')
        nonnegative(self.fit_error_px, 'fit_error_px')
        nonnegative(self.validation_error_mm, 'validation_error_mm')


@dataclass(frozen=True)
class RigidTransform(Record):
    source_frame: str
    target_frame: str
    matrix: Tuple[float, ...]
    translation_unit: str
    version: str

    def _validate(self):
        require(len(self.matrix) == 16, 'matrix', 'row-major 4x4 matrix required')
        require(self.translation_unit == 'mm', 'translation_unit', 'transform translation requires mm', ErrorCode.INVALID_UNIT)


@dataclass(frozen=True)
class PathSegment(Record):
    phase: Phase
    target: ArmPoint
    speed_mm_s: float
    timeout_s: float

    def _validate(self):
        require(self.speed_mm_s > 0, 'speed_mm_s', 'positive stage speed required')
        require(self.timeout_s > 0, 'timeout_s', 'positive stage timeout required')


@dataclass(frozen=True)
class GeometryRequest(Record):
    envelope: Envelope
    observation: TargetObservation
    capture_pose: MapPose
    current_pose: MapPose
    calibration: CameraCalibration
    tool_transform: RigidTransform
    held_offset: Optional[ArmPoint]
    space: SpaceSnapshot


@dataclass(frozen=True)
class GeometryPlan(Record):
    envelope: Envelope
    path_id: str
    target: ArmPoint
    segments: Tuple[PathSegment, ...]
    reachable: Fact
    collision_free: Fact
    evidence: Tuple[EvidenceRef, ...]


@dataclass(frozen=True)
class GeometryResult(Record):
    envelope: Envelope
    result: ActionResult
    plan: Optional[GeometryPlan]


@dataclass(frozen=True)
class PlacementGeometryRequest(Record):
    envelope: Envelope
    desired_brick_pose: MapPose
    region: RegionBoundary
    current_base_pose: MapPose
    actual_support_ids: Tuple[str, ...]
    tool_transform: RigidTransform
    held_offset: ArmPoint
    space: SpaceSnapshot


@dataclass(frozen=True)
class SupportGeometryRequest(Record):
    envelope: Envelope
    upper: Occupancy
    lowers: Tuple[Occupancy, ...]
    space: SpaceSnapshot
    contact_evidence: Tuple[EvidenceRef, ...]


@dataclass(frozen=True)
class SupportGeometryResult(Record):
    envelope: Envelope
    upper_target_id: str
    support_ids: Tuple[str, ...]
    upper_bottom_area_m2: Optional[float]
    supported_area_m2: Optional[float]
    ratio_lower_bound: Optional[float]
    contact: Fact
    horizontal: Fact
    evidence: Tuple[EvidenceRef, ...]
    reason: Optional[Reason]


@dataclass(frozen=True)
class DriveReadiness(Record):
    envelope: Envelope
    holding: HoldingSnapshot
    posture_confirmed: Fact
    evidence: Tuple[EvidenceRef, ...]


@dataclass(frozen=True)
class NavigationRequest(Record):
    envelope: Envelope
    goal: WaypointResult
    localization: LocalizationSnapshot
    space: SpaceSnapshot
    readiness: DriveReadiness
    permit: ControlPermit


@dataclass(frozen=True)
class ExitEvidence(Record):
    envelope: Envelope
    area: Area
    entire_envelope_outside: Fact
    evidence: Tuple[EvidenceRef, ...]


@dataclass(frozen=True)
class NavigationResult(Record):
    envelope: Envelope
    result: ActionResult
    stop: Optional[StopReport]
    residual_pose: Optional[MapPose]
    work_ready: Fact
    exit_evidence: Optional[ExitEvidence]


@dataclass(frozen=True)
class Budget(Record):
    name: str
    attempts_limit: Optional[int]
    seconds_limit: Optional[float]
    attempts_used: int
    seconds_used: float

    def _validate(self):
        for name in ('attempts_limit', 'seconds_limit', 'attempts_used', 'seconds_used'):
            value = getattr(self, name)
            if value is not None:
                nonnegative(value, name)


@dataclass(frozen=True)
class WorkOrder(Record):
    envelope: Envelope
    category: Category
    destination: Area
    mapping_version: str
    inventory_id: str
    budget: Budget

    def _validate(self):
        require(self.envelope.request_id is not None and self.envelope.target_id is not None,
                'envelope', 'one work order requires one target and request id')
        require((self.category is Category.CAKE) == (self.destination is Area.A),
                'destination', 'A is exclusively for cake')


@dataclass(frozen=True)
class GraspRequest(Record):
    envelope: Envelope
    order: WorkOrder
    plan: GeometryPlan
    stop: StopReport
    holding: HoldingSnapshot
    permit: ControlPermit


@dataclass(frozen=True)
class ManipulationResult(Record):
    envelope: Envelope
    result: ActionResult
    holding: HoldingSnapshot
    readiness: Optional[DriveReadiness]
    recovery_plan: Optional[GeometryPlan]


@dataclass(frozen=True)
class PlacementRequest(Record):
    envelope: Envelope
    order: WorkOrder
    mapping: MappingSnapshot
    plan: GeometryPlan
    stop: StopReport
    holding: HoldingSnapshot
    permit: ControlPermit
    region: RegionBoundary


@dataclass(frozen=True)
class LayoutBrick(Record):
    target_id: str
    slot: int
    layer: int
    support_ids: Tuple[str, ...]
    support_ratio_lower: Optional[float]
    released: Fact
    in_region: Fact
    horizontal: Fact
    near_center: Fact
    placed_recovery_round: int
    evidence: Tuple[EvidenceRef, ...]

    def _validate(self):
        require(1 <= self.slot <= 3 and 1 <= self.layer <= 3, 'slot', 'slot/layer must be 1..3')
        nonnegative(self.placed_recovery_round, 'placed_recovery_round')
        if self.support_ratio_lower is not None:
            require(0 <= self.support_ratio_lower <= 1, 'support_ratio_lower', 'ratio must be in [0,1]')


@dataclass(frozen=True)
class LayoutSnapshot(Record):
    envelope: Envelope
    area: Area
    kind: LayoutKind
    layout_recovery_round: int
    rebuild_generation: int
    confirmed_count: int
    bricks: Tuple[LayoutBrick, ...]
    bases_adjacent: Fact
    quality: Fact

    def _validate(self):
        require(0 <= self.confirmed_count <= 3, 'confirmed_count', 'layout progress must be 0..3')
        nonnegative(self.layout_recovery_round, 'layout_recovery_round')
        nonnegative(self.rebuild_generation, 'rebuild_generation')
        require(len({b.target_id for b in self.bricks}) == len(self.bricks), 'bricks', 'duplicate physical target')


@dataclass(frozen=True)
class DeliveryKey(Record):
    run_id: str
    inventory_id: str


@dataclass(frozen=True)
class InventoryItem(Record):
    inventory_id: str
    target_ids: Tuple[str, ...]
    identity_confirmed: Fact
    category: Optional[Category]
    location: InventoryLocation
    area: Optional[Area]


@dataclass(frozen=True)
class ReleaseEvent(Record):
    envelope: Envelope
    delivery_key: DeliveryKey
    category: Category
    area: Area
    mapping_version: str
    released_at: ClockStamp
    separated: Fact
    correct_region: Fact
    evidence: Tuple[EvidenceRef, ...]


@dataclass(frozen=True)
class DeliveryReceipt(Record):
    key: DeliveryKey
    first_event_id: str
    category: Category
    area: Area
    first_release_at: ClockStamp
    evidence_ids: Tuple[str, ...]


@dataclass(frozen=True)
class PlacementResult(Record):
    envelope: Envelope
    result: ActionResult
    release: Optional[ReleaseEvent]
    layout: LayoutSnapshot
    proposed_space: SpaceSnapshot
    proposed_holding: HoldingSnapshot
    stillness: Optional[StillnessCertificate]


@dataclass(frozen=True)
class StateSnapshot(Record):
    run: RunContext
    revision: int
    space: SpaceSnapshot
    holding: HoldingSnapshot
    inventory: Tuple[InventoryItem, ...]
    deliveries: Tuple[DeliveryReceipt, ...]
    layouts: Tuple[LayoutSnapshot, ...]
    mapping: Optional[MappingSnapshot]
    budgets: Tuple[Budget, ...]

    def _validate(self):
        nonnegative(self.revision, 'revision')
        require(len({d.key for d in self.deliveries}) == len(self.deliveries), 'deliveries', 'duplicate delivery key')


@dataclass(frozen=True)
class HoldingUpdate(Record):
    envelope: Envelope
    previous: Holding
    current: HoldingSnapshot
    cause: HoldingCause


@dataclass(frozen=True)
class OccupancyClearance(Record):
    target_id: str
    cause: ClearanceCause
    confirmed: Fact
    evidence: Tuple[EvidenceRef, ...]


@dataclass(frozen=True)
class StateTransaction(Record):
    envelope: Envelope
    transaction_id: str
    expected_revision: int
    expected_space_version: int
    space: SpaceSnapshot
    holding: HoldingSnapshot
    inventory: Tuple[InventoryItem, ...]
    layouts: Tuple[LayoutSnapshot, ...]
    mapping: Optional[MappingSnapshot]
    release_events: Tuple[ReleaseEvent, ...]
    clearing_evidence: Tuple[OccupancyClearance, ...]
    holding_update: Optional[HoldingUpdate]
    budgets: Tuple[Budget, ...]
    run_update: Optional[RunContext]


@dataclass(frozen=True)
class CommitReceipt(Record):
    transaction_id: str
    status: CommitStatus
    revision: Optional[int]
    space_version: Optional[int]
    reason: Optional[Reason]

    def _validate(self):
        if self.status is CommitStatus.COMMITTED:
            require(self.revision is not None and self.space_version is not None,
                    'revision', 'commit needs state and space revisions')
        else:
            require(self.reason is not None, 'reason', 'non-commit requires cause')


@dataclass(frozen=True)
class ActionIntent(Record):
    envelope: Envelope
    operation: IntentOperation
    order: Optional[WorkOrder]
    arm_plan: Optional[GeometryPlan]
    base_goal: Optional[MapPose]
    base_velocity_m_s_rad_s: Optional[Tuple[float, float]]
    pump_enabled: Optional[bool]

    def _validate(self):
        require(self.envelope.action_id is not None, 'envelope.action_id', 'intent needs stable action identity')
        present = (self.arm_plan is not None, self.base_goal is not None,
                   self.base_velocity_m_s_rad_s is not None, self.pump_enabled is not None)
        expected = (self.operation is IntentOperation.ARM, self.operation is IntentOperation.BASE_GOAL,
                    self.operation is IntentOperation.BASE_VELOCITY, self.operation is IntentOperation.PUMP)
        require(present == expected, 'operation', 'exactly one matching command body required')
        if self.arm_plan is not None:
            require(bool(self.arm_plan.segments), 'arm_plan.segments', 'intent must bind stage speed and timeout')


@dataclass(frozen=True)
class IntentReceipt(Record):
    run_id: str
    action_id: str
    intent_sha256: str
    commit: CommitReceipt


@dataclass(frozen=True)
class ReadResult(Record):
    status: Outcome
    snapshot: Optional[StateSnapshot]
    reason: Optional[Reason]


@dataclass(frozen=True)
class RecoveryRequest(Record):
    envelope: Envelope
    original_started_at: ClockStamp
    original_deadline: ClockStamp
    old_stop: StopReport


@dataclass(frozen=True)
class RecoveryResult(Record):
    status: Outcome
    snapshot: Optional[StateSnapshot]
    remaining_s: Optional[float]
    registration: Optional[CommitReceipt]
    reason: Optional[Reason]


@dataclass(frozen=True)
class TimerRecord(Record):
    format_version: int
    run_id: str
    started_at: ClockStamp
    deadline: ClockStamp
    refreshed_at: ClockStamp
    remaining_s: float
    refresh_period_s: float
    sequence: int
    state: RunState
    ever_restarted: bool

    def _validate(self):
        require(self.format_version == 1, 'format_version', 'timer format v1')
        monotonic(self.started_at, 'started_at')
        monotonic(self.deadline, 'deadline')
        monotonic(self.refreshed_at, 'refreshed_at')
        require(self.deadline - self.started_at == 600, 'deadline', 'original deadline only')
        require(self.refreshed_at - self.started_at >= 0, 'refreshed_at', 'cannot predate original start')
        require(self.remaining_s == max(0, self.deadline - self.refreshed_at),
                'remaining_s', 'must agree with original deadline')
        require(self.refresh_period_s > 0, 'refresh_period_s', 'positive refresh period')
        nonnegative(self.sequence, 'sequence')


@dataclass(frozen=True)
class SealRequest(Record):
    envelope: Envelope
    run: RunContext
    stop: StopReport
    end_reason: Reason


@dataclass(frozen=True)
class EndMarker(Record):
    run_id: str
    original_deadline: ClockStamp
    ended_at: ClockStamp
    ever_restarted: bool
    stop_confirmed: Fact
    writers_closed: Fact
    sealed: Fact
    cleanup_state: CleanupState


@dataclass(frozen=True)
class CleanupRequest(Record):
    run_id: str
    runtime_root: str
    marker: EndMarker
    automatic: bool


@dataclass(frozen=True)
class CleanupResult(Record):
    status: Outcome
    marker: EndMarker
    reason: Optional[Reason]


@dataclass(frozen=True)
class RuntimeSignal(Record):
    envelope: Envelope
    kind: SignalKind
    referee_authorized: Fact


@dataclass(frozen=True)
class PermissionRequest(Record):
    envelope: Envelope
    source: MotionSource
    purpose: str
    health: HealthSnapshot
    localization: LocalizationSnapshot
    readiness: DriveReadiness
    space: SpaceSnapshot
    old_stop: StopReport
    stillness: Optional[StillnessCertificate]
    intent_receipt: IntentReceipt


@dataclass(frozen=True)
class RuntimeDecision(Record):
    run: RunContext
    permit: Optional[ControlPermit]
    stop: Optional[StopRequest]
    reason: Optional[Reason]


@dataclass(frozen=True)
class InitializeRunRequest(Record):
    run: RunContext
    initial_snapshot: StateSnapshot
    start_signal: RuntimeSignal
    previous_end_marker: Optional[EndMarker]


@dataclass(frozen=True)
class ScoreEstimate(Record):
    confirmed_points: Optional[float]
    unresolved_target_ids: Tuple[str, ...]
    evidence: Tuple[EvidenceRef, ...]
    policy_version: str


@dataclass(frozen=True)
class MissionInput(Record):
    envelope: Envelope
    snapshot: StateSnapshot
    candidates: Tuple[TargetObservation, ...]
    action_events: Tuple[ActionEvent, ...]
    exits: Tuple[ExitEvidence, ...]


@dataclass(frozen=True)
class MissionDecision(Record):
    envelope: Envelope
    phase: MissionPhase
    order: Optional[WorkOrder]
    delivery_complete: Fact
    layout_complete: Fact
    score: ScoreEstimate
    recovery_budget: Optional[Budget]
    end_reason: Optional[Reason]


@dataclass(frozen=True)
class DependencyStatus(Record):
    owner: Owner
    contract_version: Optional[str]
    delivery_version: Optional[str]
    available: Fact
    reason: Optional[Reason]


@dataclass(frozen=True)
class IntegrationRequest(Record):
    contract_version: str
    dependencies: Tuple[DependencyStatus, ...]
    scenario_ids: Tuple[str, ...]
    origin: Origin


@dataclass(frozen=True)
class VerificationResult(Record):
    scenario_id: str
    tier: VerificationTier
    status: Fact
    log_uri: Optional[str]
    reason: Optional[Reason]


@dataclass(frozen=True)
class IntegrationReport(Record):
    contract_version: str
    checks: Tuple[VerificationResult, ...]
    dependencies: Tuple[DependencyStatus, ...]
    origin: Origin


@dataclass(frozen=True)
class ContractCheck(Record):
    accepted: bool
    reason: Optional[Reason]
