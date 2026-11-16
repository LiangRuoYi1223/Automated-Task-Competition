"""Immutable wire types and strict validation, using Python 3.8 stdlib only."""
import json
import math
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from functools import lru_cache
from typing import Optional, Tuple, Union, get_args, get_origin, get_type_hints

CONTRACT_VERSION = '1.0.0'
_RECORDS = {}


class ErrorCode(str, Enum):
    INVALID_FIELD = 'invalid_field'
    MISSING_FIELD = 'missing_field'
    UNKNOWN_FIELD = 'unknown_field'
    INVALID_UNIT = 'invalid_unit'
    VERSION_MISMATCH = 'version_mismatch'
    CLOCK_MISMATCH = 'clock_mismatch'
    STALE_RUN = 'stale_run'
    STALE_CONTROL = 'stale_control'
    STALE_REQUEST = 'stale_request'
    STALE_RECOVERY = 'stale_recovery'
    STALE_GEOMETRY = 'stale_geometry'
    EXPIRED = 'expired'
    FUTURE_RESULT = 'future_result'
    DUPLICATE = 'duplicate'
    ID_CONFLICT = 'id_conflict'
    UNKNOWN_IDENTITY = 'unknown_identity'
    EVIDENCE_MISSING = 'evidence_missing'
    EVIDENCE_NOT_PHYSICAL = 'evidence_not_physical'
    SIMULATED_EVIDENCE = 'simulated_evidence'
    TIMEOUT = 'timeout'
    CANCELLED = 'cancelled'
    UNAVAILABLE = 'unavailable'
    UNCONFIGURED = 'unconfigured'
    INVALID_MAPPING = 'invalid_mapping'
    INVALID_LAYOUT = 'invalid_layout'
    OCCUPANCY_UNRESOLVED = 'occupancy_unresolved'
    CONFLICT = 'conflict'
    STORAGE_FAILURE = 'storage_failure'
    NOT_AUTHORIZED = 'not_authorized'
    DEADLINE = 'deadline'
    LIMIT_EXCEEDED = 'limit_exceeded'


class ContractError(ValueError):
    def __init__(self, code, field, detail):
        self.code, self.field, self.detail = code, field, detail
        super().__init__('{}: {}: {}'.format(code.value, field, detail))


def require(condition, field, detail, code=ErrorCode.INVALID_FIELD):
    if not condition:
        raise ContractError(code, field, detail)


class Owner(str, Enum):
    P00 = 'P00'
    P01 = 'P01'
    P02 = 'P02'
    P03 = 'P03'
    P04 = 'P04'
    P05 = 'P05'
    P06 = 'P06'
    P07 = 'P07'
    P08 = 'P08'
    P09 = 'P09'
    P10A = 'P10A'
    P10B = 'P10B'
    P11 = 'P11'


class Origin(str, Enum):
    LIVE = 'live'
    SIMULATED = 'simulated'
    REPLAY = 'replay'


class Outcome(str, Enum):
    SUCCESS = 'success'
    FAILED = 'failed'
    CANCELLED = 'cancelled'
    TIMED_OUT = 'timed_out'
    UNKNOWN = 'unknown'
    REJECTED = 'rejected'


class Fact(str, Enum):
    CONFIRMED = 'confirmed'
    REJECTED = 'rejected'
    UNKNOWN = 'unknown'


class Holding(str, Enum):
    EMPTY = 'empty'
    HELD = 'held'
    UNKNOWN = 'unknown'
    DROPPED = 'dropped_unresolved'


class ClockBasis(str, Enum):
    MONOTONIC = 'monotonic'
    SENSOR = 'sensor'
    ROS = 'ros'
    UTC = 'utc'


class EvidenceKind(str, Enum):
    INTERFACE = 'interface'
    SUBMITTED = 'command_submitted'
    ECHO = 'command_echo'
    PHYSICAL = 'physical_observation'


@lru_cache(maxsize=None)
def _hints(cls):
    return get_type_hints(cls)


def _check(value, typ, path):
    origin, args = get_origin(typ), get_args(typ)
    if origin is Union:
        if value is None and type(None) in args:
            return
        types = [t for t in args if t is not type(None)]
        # Public unions are Optional[T], never ambiguous business payload unions.
        require(len(types) == 1, path, 'unsupported ambiguous union')
        return _check(value, types[0], path)
    if origin is tuple:
        require(type(value) is tuple, path, 'expected immutable tuple')
        variadic = len(args) == 2 and args[1] is Ellipsis
        require(variadic or len(value) == len(args), path, 'wrong tuple length')
        for i, item in enumerate(value):
            _check(item, args[0] if variadic else args[i], '{}[{}]'.format(path, i))
        return
    if typ is float:
        valid = type(value) in (int, float)
        try:
            valid = valid and math.isfinite(value)
        except OverflowError:
            valid = False
        require(valid, path, 'expected finite number, not bool')
    elif typ is str:
        require(type(value) is str and bool(value.strip()), path, 'expected nonempty string')
    elif typ in (int, bool):
        require(type(value) is typ, path, 'expected {}'.format(typ.__name__))
    else:
        require(type(value) is typ, path, 'expected {}'.format(typ.__name__))


class Record:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _RECORDS[cls.__name__] = cls

    def __post_init__(self):
        for field in fields(self):
            _check(getattr(self, field.name), _hints(type(self))[field.name], field.name)
        self._validate()

    def _validate(self):
        pass


def nonnegative(value, field):
    require(value >= 0, field, 'must be nonnegative')


@dataclass(frozen=True)
class Reason(Record):
    code: ErrorCode
    field: str
    detail: str


@dataclass(frozen=True)
class ClockStamp(Record):
    seconds: float
    basis: ClockBasis
    clock_id: str
    boot_id: Optional[str]

    def _validate(self):
        nonnegative(self.seconds, 'seconds')
        if self.basis is ClockBasis.MONOTONIC:
            require(self.boot_id is not None, 'boot_id', 'monotonic clock needs boot identity')

    def __sub__(self, other):
        require(isinstance(other, ClockStamp), 'clock', 'expected ClockStamp')
        require((self.basis, self.clock_id, self.boot_id) ==
                (other.basis, other.clock_id, other.boot_id), 'clock',
                'cannot subtract different clock domains or boots', ErrorCode.CLOCK_MISMATCH)
        return self.seconds - other.seconds


def monotonic(stamp, field):
    require(stamp.basis is ClockBasis.MONOTONIC, field, 'requires monotonic clock',
            ErrorCode.CLOCK_MISMATCH)


@dataclass(frozen=True)
class Versions(Record):
    config: str
    map: Optional[str]
    calibration: Optional[str]
    model: Optional[str]
    mapping: Optional[str]
    layout: Optional[str]
    space: Optional[int]

    def _validate(self):
        if self.space is not None:
            nonnegative(self.space, 'space')


@dataclass(frozen=True)
class Envelope(Record):
    run_id: str
    instance_id: str
    control_epoch: int
    producer: Owner
    event_id: str
    sequence: int
    request_id: Optional[str]
    action_id: Optional[str]
    target_id: Optional[str]
    emitted_at: ClockStamp
    expires_at: ClockStamp
    versions: Versions
    origin: Origin
    layout_recovery_round: Optional[int]
    rebuild_generation: Optional[int]
    base_motion_epoch: Optional[int]

    def _validate(self):
        for name in ('control_epoch', 'sequence', 'layout_recovery_round',
                     'rebuild_generation', 'base_motion_epoch'):
            value = getattr(self, name)
            if value is not None:
                nonnegative(value, name)
        monotonic(self.emitted_at, 'emitted_at')
        monotonic(self.expires_at, 'expires_at')
        require(self.expires_at - self.emitted_at > 0, 'expires_at', 'must follow emitted_at')
        require(self.action_id is None or self.request_id is not None,
                'request_id', 'actions require a work/request identity')


@dataclass(frozen=True)
class ArmPoint(Record):
    frame_id: str
    x: float
    y: float
    z: float
    unit: str
    error_mm: float

    def _validate(self):
        require(self.unit == 'mm', 'unit', 'arm coordinates require mm', ErrorCode.INVALID_UNIT)
        nonnegative(self.error_mm, 'error_mm')


@dataclass(frozen=True)
class MapPose(Record):
    frame_id: str
    x: float
    y: float
    yaw: float
    position_unit: str
    angle_unit: str
    error_m: float
    error_rad: float

    def _validate(self):
        require(self.position_unit == 'm', 'position_unit', 'map coordinates require m', ErrorCode.INVALID_UNIT)
        require(self.angle_unit == 'rad', 'angle_unit', 'angles require rad', ErrorCode.INVALID_UNIT)
        nonnegative(self.error_m, 'error_m')
        nonnegative(self.error_rad, 'error_rad')


@dataclass(frozen=True)
class EvidenceRef(Record):
    evidence_id: str
    run_id: str
    target_id: Optional[str]
    action_id: Optional[str]
    kind: EvidenceKind
    origin: Origin
    captured_at: Optional[ClockStamp]
    received_at: ClockStamp
    valid_until: ClockStamp
    uri: str
    sha256: Optional[str]

    def _validate(self):
        monotonic(self.received_at, 'received_at')
        monotonic(self.valid_until, 'valid_until')
        require(self.valid_until - self.received_at > 0, 'valid_until', 'must follow receipt')
        if self.kind is EvidenceKind.PHYSICAL:
            require(self.captured_at is not None, 'captured_at', 'physical observation needs capture stamp')
        if self.sha256 is not None:
            require(len(self.sha256) == 64 and all(c in '0123456789abcdef' for c in self.sha256),
                    'sha256', 'expected lowercase SHA-256')


@dataclass(frozen=True)
class ActionResult(Record):
    status: Outcome
    phase: str
    reason: Optional[Reason]
    accepted: Fact
    submitted: Fact
    echoed: Fact
    physical: Fact
    evidence: Tuple[EvidenceRef, ...]
    elapsed_s: float

    def _validate(self):
        nonnegative(self.elapsed_s, 'elapsed_s')
        if self.status is not Outcome.SUCCESS:
            require(self.reason is not None, 'reason', 'non-success needs a reason')
        if self.physical is Fact.CONFIRMED:
            require(any(e.kind is EvidenceKind.PHYSICAL for e in self.evidence),
                    'evidence', 'physical completion needs observation', ErrorCode.EVIDENCE_MISSING)


@dataclass(frozen=True)
class HoldingSnapshot(Record):
    state: Holding
    target_id: Optional[str]
    tool_offset: Optional[ArmPoint]
    evidence: Tuple[EvidenceRef, ...]

    def _validate(self):
        if self.state is Holding.HELD:
            require(self.target_id is not None, 'target_id', 'held object needs identity')
        if self.state is Holding.EMPTY:
            require(self.target_id is None and self.tool_offset is None,
                    'target_id', 'empty has no held object or offset')


def to_data(value):
    """Return a detached JSON-compatible value; mutable copies cannot edit the record."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Record) and is_dataclass(value):
        return {f.name: to_data(getattr(value, f.name)) for f in fields(value)}
    if type(value) is tuple:
        return [to_data(item) for item in value]
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise ContractError(ErrorCode.INVALID_FIELD, 'payload', 'unsupported value')


def _decode(typ, value, path):
    origin, args = get_origin(typ), get_args(typ)
    if origin is Union:
        if value is None and type(None) in args:
            return None
        return _decode([t for t in args if t is not type(None)][0], value, path)
    if origin is tuple:
        require(type(value) is list, path, 'expected JSON array')
        variadic = len(args) == 2 and args[1] is Ellipsis
        require(variadic or len(value) == len(args), path, 'wrong array length')
        return tuple(_decode(args[0] if variadic else args[i], v, '{}[{}]'.format(path, i))
                     for i, v in enumerate(value))
    if isinstance(typ, type) and issubclass(typ, Enum):
        require(type(value) is str, path, 'expected enum string')
        try:
            return typ(value)
        except ValueError:
            raise ContractError(ErrorCode.INVALID_FIELD, path, 'unknown enum value') from None
    if isinstance(typ, type) and issubclass(typ, Record):
        require(type(value) is dict, path or 'payload', 'expected JSON object')
        prefix = path + '.' if path else ''
        expected = {f.name for f in fields(typ)}
        for key in sorted(expected - value.keys()):
            raise ContractError(ErrorCode.MISSING_FIELD, prefix + key, 'explicit field required, including null')
        for key in sorted(value.keys() - expected):
            raise ContractError(ErrorCode.UNKNOWN_FIELD, prefix + key, 'field not in frozen contract')
        decoded = {f.name: _decode(_hints(typ)[f.name], value[f.name], prefix + f.name)
                   for f in fields(typ)}
        try:
            return typ(**decoded)
        except ContractError as e:
            raise ContractError(e.code, prefix + e.field, e.detail) from None
    _check(value, typ, path)
    return value


def from_data(typ, value):
    return _decode(typ, value, '')


def dumps(record):
    require(isinstance(record, Record) and is_dataclass(record), 'payload', 'expected contract record')
    return json.dumps(dict(contract_version=CONTRACT_VERSION, type=type(record).__name__,
                           payload=to_data(record)), ensure_ascii=False, allow_nan=False,
                      sort_keys=True, separators=(',', ':'))


def loads(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, key, 'duplicate JSON key', ErrorCode.ID_CONFLICT)
            result[key] = value
        return result

    def constant(value):
        raise ContractError(ErrorCode.INVALID_FIELD, 'payload', 'non-finite JSON value ' + value)

    try:
        data = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, TypeError) as e:
        if isinstance(e, ContractError):
            raise
        raise ContractError(ErrorCode.INVALID_FIELD, 'payload', 'invalid JSON') from None
    require(type(data) is dict and set(data) == {'contract_version', 'type', 'payload'},
            'wire', 'expected version, type and payload only')
    require(data['contract_version'] == CONTRACT_VERSION, 'contract_version',
            'exact version required until explicit migration', ErrorCode.VERSION_MISMATCH)
    # Import definitions only; there are no adapters or device side effects.
    from . import models  # noqa: F401
    require(type(data['type']) is str and data['type'] in _RECORDS,
            'type', 'unknown contract type')
    return from_data(_RECORDS[data['type']], data['payload'])
