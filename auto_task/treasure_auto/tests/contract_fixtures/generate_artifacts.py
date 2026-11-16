"""Regenerate reviewed C0 manifest and SIMULATED JSON cases; never uses devices."""
from dataclasses import fields, is_dataclass
from enum import Enum
import inspect
import json
from pathlib import Path
import sys
from typing import get_args, get_origin, get_type_hints, Union

PACKAGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE / 'src'))
sys.path.insert(0, str(PACKAGE / 'tests'))
from treasure_auto.contracts import core as c, models as m, ports
from contract_fixtures import samples as s


OWNERS = {
    'P01': 'Capability HealthSnapshot BaseCommand ArmCommand PumpCommand StopReport',
    'P02': 'RegionBoundary MapMetadata LocalizationSnapshot WaypointQuery WaypointResult',
    'P03': 'NavigationRequest NavigationResult ExitEvidence',
    'P04': 'PixelPoint PixelTransform ImageObservation TopSurface TargetObservation ToolObservation PerceptionRequest PerceptionResult BoardObservation',
    'P05': 'SlotAssignment MappingSnapshot ClassificationRequest ClassificationResult',
    'P06': 'CameraCalibration RigidTransform PathSegment GeometryRequest GeometryPlan GeometryResult PlacementGeometryRequest SupportGeometryRequest SupportGeometryResult',
    'P07': 'HoldingSnapshot DriveReadiness GraspRequest ManipulationResult HoldingUpdate',
    'P08': 'Occupancy SpaceSnapshot StillnessCertificate PlacementRequest LayoutBrick LayoutSnapshot ReleaseEvent PlacementResult OccupancyClearance',
    'P09': 'Budget WorkOrder DeliveryKey InventoryItem DeliveryReceipt StateTransaction ScoreEstimate MissionInput MissionDecision',
    'P10A': 'RunContext ControlPermit StopRequest RecoveryRequest RuntimeSignal PermissionRequest RuntimeDecision SealRequest InitializeRunRequest',
    'P10B': 'StateSnapshot CommitReceipt ActionIntent IntentReceipt ReadResult RecoveryResult TimerRecord EndMarker CleanupRequest CleanupResult',
    'P11': 'DependencyStatus IntegrationRequest VerificationResult IntegrationReport',
}
OWNER_BY_TYPE = {name: owner for owner, names in OWNERS.items() for name in names.split()}


def type_name(typ):
    args, origin = get_args(typ), get_origin(typ)
    if origin is Union:
        return 'Optional[' + type_name(next(t for t in args if t is not type(None))) + ']'
    if origin is tuple:
        return 'Tuple[' + ', '.join('...' if t is Ellipsis else type_name(t) for t in args) + ']'
    return 'None' if typ is type(None) else typ.__name__


def schema():
    records, enums = {}, {}
    for module in (c, m):
        for name, cls in vars(module).items():
            if not isinstance(cls, type) or cls.__module__ != module.__name__:
                continue
            if is_dataclass(cls):
                hints = get_type_hints(cls)
                records[name] = {'logical_owner': OWNER_BY_TYPE.get(name, 'P00'),
                                 'fields': {f.name: type_name(hints[f.name]) for f in fields(cls)}}
            elif issubclass(cls, Enum):
                enums[name] = [v.value for v in cls]
    interfaces = {}
    for name, owner in ports.PORT_OWNERS.items():
        methods = {}
        for key, value in inspect.getmembers(getattr(ports, name), inspect.isfunction):
            if not key.startswith('_'):
                methods[key] = {k: type_name(v) for k, v in get_type_hints(value).items()}
        interfaces[name] = dict(owner=owner.value, methods=methods)
    return dict(contract_version=c.CONTRACT_VERSION, records=records, enums=enums, ports=interfaces,
                wire_fields=['contract_version', 'type', 'payload'], unknown_fields='reject',
                missing_fields='reject including Optional; send explicit null', version_policy='exact')


NOTES = {
    'envelope': '局、控制实例、请求、时效、版本与模拟来源；消费前校验。',
    'origin': '来源枚举 live/simulated/replay；MapMetadata 中则是地图原点位姿。',
    'frame_id': '来源坐标系的明确名称；缺失拒绝，不推定 map 或 arm。',
    'evidence': '有序证据引用；空元组表示没有证据，不能证明物理成功。',
    'target_id': '本局规范物理目标身份；Optional 的 null 表示未关联/无作用目标，不能新建到货资格。',
    'target_ids': '同一库存的已证实物理身份别名；不能因重编号删除已有别名或新建库存。',
    'inventory_id': 'P09 的本局稳定库存身份；不等于布局槽号或观测号。',
    'action_id': '阶段执行者生成的动作身份；null 仅表示非动作记录。',
    'request_id': 'P09 工作单或调用方请求号；null 表示未关联请求的观测/运行消息。',
    'event_id': '事件产生者生成；同一局和产生者内不可复用为不同内容。',
    'sequence': '非负递增序号；流范围见 README 身份表，不按回调次数计时。',
    'captured_at': '采集端时钟；仅非物理消息允许 null。不能与接收时钟直接相减。',
    'received_at': '本机单调接收时间，含 clock_id/boot_id。',
    'emitted_at': '本机单调消息产生时间；不能用于代替消费者自己的 now。',
    'expires_at': '单调绝对有效截止；now 等于本字段即失效。',
    'valid_until': '证据单调绝对有效截止；不延长原局截止。',
    'released_at': '已证实实际释放的单调时刻/保守最迟上界；无可信期限依据则不生成 ReleaseEvent。',
    'control_epoch': 'P10A 持久登记的控制代次，隔离旧实例和队列。',
    'layout_recovery_round': 'P08 按区域记录确认布局失败次数；代码重启不增加。',
    'rebuild_generation': 'P08 按区域记录全组倒塌后重建代次；与控制代次分开。',
    'base_motion_epoch': 'P02 底盘运动代号；变化使几何/路径无效，真实占用保留。',
    'scene_epoch': 'P08 观察场景代号，调臂/运动/倒塌等使旧静止证书失效。',
    'support_ids': '实际下层物理目标身份；空表示没有已声明支撑，不自动等于地面空闲。',
    'support_ratio_lower': '无量纲；上砖底面受真实下砖顶面支撑面积/上砖底面积的保守下界；null 未知。',
    'ratio_lower_bound': '计入误差的无量纲支撑面积比例下界；null 未计算/无法确认。',
    'height_mm': '当前实际层面的毫米高度；null 未知，不补第一层。',
    'corners': '真实顶面的四个全图像素角点，沿轮廓顺序；不可替成图案框。',
    'contour': '全图 u 列/v 行的真实轮廓像素点。',
    'u': '全图图像列坐标，单位 px。', 'v': '全图图像行坐标，单位 px。',
    'reason': '明确错误字段和原因；null 表示没有拒绝/失败原因，不能单独当成功。',
    'map': '本局地图版本；null 未绑定/不可用。',
    'calibration': '标定版本/记录；null 未绑定/不可用。',
    'model': '模型版本；null 无可用模型，不代表识别成功。',
    'mapping': '完整冻结分类映射或版本；null 尚未冻结。',
    'layout': '选定布局版本；null 尚未选定。',
    'space': '共享快照/版本；Versions.space=null 尚无可消费空间。',
    'pose': '位姿；null 无可靠位姿，不补零。',
    'plan': '完整几何计划；null 未生成可交付计划。',
    'order': '唯一工作单；null 本次不派单。',
    'snapshot': '一致状态；null 读取/恢复未获得可用快照。',
    'readiness': '独立的行驶准备证明；null 尚未确认，不等于空载。',
    'stop': '实际停止分层反馈；null 尚无停止反馈。',
    'permit': 'P10A 短许可；null 未授予许可。',
    'stillness': '适用静止证书；null 不存在可用证书，不自动放行。',
    'run_update': 'P10A 原局状态提议；null 原样保留，不允许后移原截止。',
    'holding_update': 'P07 持物变更提议；null 仅在持物事实未变时允许。',
    'value': '参数实测值；null 未配置。',
    'minimum': '实测合法范围下界；null 待采集。',
    'maximum': '实测合法范围上界；null 待采集。',
    'range_source': '合法范围来源；null 尚无可靠来源。',
    'sha256': '已落地引用的内容摘要；null 尚未形成持久可用引用。',
    'confirmed_points': '已确认部分条件估分；null 无法确定，不能填 0。',
    'boot_id': '系统启动身份；单调时钟必填，其他时基 null 表示无该身份。',
    'clock_id': '时钟实例/域身份；重置或不连续后必须换号。',
    'status': '具名枚举，UNKNOWN 与 FAILED/REJECTED 分开。',
    'state': '该类型的具名状态枚举；超时不能推导 EMPTY。',
    'quality': '布局几何声明状态；不是到货数或正式计分。',
    'physical': '物理完成事实，不能由 accepted/submitted/echoed 推导。',
    'remaining_s': '由原截止与可信单调 now 算出的秒数；Optional 的 null 表示时基不可信。',
}


def field_note(name, typ):
    if name in NOTES:
        return NOTES[name]
    for suffix, unit in (('_mm_s', 'mm/s'), ('_rad_s', 'rad/s'), ('_m_s', 'm/s'),
                         ('_m2', 'm²'), ('_mm', 'mm'), ('_rad', 'rad'), ('_m', 'm'),
                         ('_px', 'px'), ('_s', 's')):
        if name.endswith(suffix):
            return '单位 ' + unit + '；必须为有限值。' + (' null 表示尚未知/未配置。' if typ.startswith('Optional') else '')
    if typ.startswith('Optional'):
        return 'null 表示尚无该项结果/关联；不得据此推断成功或批准动作。'
    if typ.startswith('Tuple'):
        return '有序不可变序列；JSON 使用数组。空序列不提供任何正向证据。'
    if typ == 'str':
        return '非空字符串；版本/标识不按名称猜测或自动迁移，语义见 README 与所属端口。'
    if typ == 'int':
        return '整数标识/计数；不能使用 bool，取值约束由本类型及纯校验执行。'
    return '按具名类型的字段/枚举解释；跨对象消费须调用相应校验，不能只凭构造成功执行动作。'


def dictionary(spec):
    lines = ['# C0 1.0.0 字段字典', '',
             '由已冻结类型生成，配合 README 的时基、身份与事务协议使用。所有字段均必传；Optional 也须显式 null。',
             'logical_owner 是业务事实/协调责任者；全部类型定义仍由 P00 独占。ActionEvent/ActionResult/EvidenceRef 由实际领域产生者填写 Envelope.producer。',
             'ArmPoint 的 x/y/z 与 error_mm 用 mm；MapPose 的 x/y 用 m、yaw 用 rad；Point2 是所属轮廓的 m；所有 *_at/截止是 ClockStamp。', '']
    for name, desc in spec['records'].items():
        lines += ['## ' + name, '', '业务责任：' + desc['logical_owner'] + '。', '',
                  '| 字段 | 类型 | 单位与空值/语义 |', '|---|---|---|']
        for field, typ in desc['fields'].items():
            lines.append('| ' + field + ' | ' + typ + ' | ' + field_note(field, typ) + ' |')
        lines.append('')
    lines += ['## 枚举和值', '']
    for name, values in spec['enums'].items():
        lines += ['- `' + name + '`: ' + ', '.join('`' + v + '`' for v in values)]
    return '\n'.join(lines) + '\n'


def cases():
    output = []
    def add(name, record, expected='accept', validator='structure'):
        output.append(dict(id=name, origin='simulated', expected=expected, validator=validator,
                           wire=json.loads(c.dumps(record))))
    for name, record in s.all_samples().items():
        add(name, record)
    add('arm_intent', s.arm_intent())
    add('intent_receipt', s.intent_receipt(s.arm_intent()))
    for i, mapping in enumerate(s.mapping_samples()):
        add('mapping_permutation_' + str(i + 1), mapping)
    for kind in m.LayoutKind:
        good, bad = s.layout_pair(kind)
        add(kind.value + '_legal', good, validator='layout_quality')
        add(kind.value + '_illegal', bad, 'reject', 'layout_quality')
    add('missing_frame', s.arm_command().target, 'reject')
    output[-1]['wire']['payload']['frame_id'] = ''
    add('invalid_unit', s.arm_command().target, 'reject')
    output[-1]['wire']['payload']['unit'] = 'm'
    add('duplicate_mapping', s.mapping_samples()[0], 'reject')
    output[-1]['wire']['payload']['assignments'][1] = output[-1]['wire']['payload']['assignments'][0]
    add('unknown_contract_version', s.envelope(), 'reject')
    output[-1]['wire']['contract_version'] = '2.0.0'
    return output


def main():
    root = PACKAGE / 'src/treasure_auto/contracts'
    spec = schema()
    (root / 'schema_v1.json').write_text(json.dumps(spec, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (root / 'FIELD_DICTIONARY.md').write_text(dictionary(spec), encoding='utf-8')
    fixture_cases = cases()
    (Path(__file__).parent / 'cases.json').write_text(json.dumps(fixture_cases, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('C0 {}: {} record types, {} enums, {} ports, {} simulated cases'.format(
        c.CONTRACT_VERSION, len(spec['records']), len(spec['enums']), len(spec['ports']), len(fixture_cases)))


if __name__ == '__main__':
    main()
