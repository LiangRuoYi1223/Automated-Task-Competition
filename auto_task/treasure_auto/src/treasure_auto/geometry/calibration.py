"""Independent, versioned calibration evaluation; no command echo as truth."""
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple
from treasure_auto.contracts import core as c, models as m
from .math3d import finite
from .config import GeometryConfig


@dataclass(frozen=True)
class CalibrationBinding:
    """Complete immutable parameters evaluated by the independent samples."""
    camera: m.CameraCalibration
    camera_to_base: m.RigidTransform
    base_to_arm: m.RigidTransform
    tool_transform: m.RigidTransform
    config: GeometryConfig


@dataclass(frozen=True)
class CalibrationSample:
    sample_id: str
    truth: c.ArmPoint
    predicted: c.ArmPoint
    evidence: c.EvidenceRef


@dataclass(frozen=True)
class CalibrationReport:
    version: str
    chain: str
    camera_version: str
    tool_version: str
    domain_mm: Tuple[Tuple[float, ...], ...]
    fit_samples: Tuple[CalibrationSample, ...]
    validation_samples: Tuple[CalibrationSample, ...]
    fit_errors_mm: Tuple[float, ...]
    validation_errors_mm: Tuple[float, ...]
    max_error_mm: float
    rms_error_mm: float
    bias_mm: Tuple[float, ...]
    limit_mm: float
    accepted: bool
    binding: Optional[CalibrationBinding] = None


def make_report(version,chain,camera_version,tool_version,domain_mm,fit_samples,validation_samples,limit_mm,binding=None):
    for x in (version,camera_version,tool_version):c.require(type(x) is str and bool(x.strip()),'version','explicit versions required')
    c.require(chain in ('image','placement'),'chain','separate image and map-placement validation required')
    finite(limit_mm,'limit_mm',0.)
    c.require(binding is None or type(binding) is CalibrationBinding,'binding','typed full calibration binding required')
    c.require(len(domain_mm)==2 and all(len(p)==3 for p in domain_mm),'domain','explicit 3D bounds required')
    for a,b in zip(*domain_mm):
        finite(a,'domain');finite(b,'domain');c.require(a<b,'domain','nonempty ordered domain required')
    c.require(bool(fit_samples) and bool(validation_samples),'samples','fit and independent validation both required')
    all_samples=tuple(fit_samples)+tuple(validation_samples)
    c.require(len({s.sample_id for s in all_samples})==len(all_samples) and
              len({s.evidence.evidence_id for s in all_samples})==len(all_samples),
              'samples','duplicate points/evidence or fit-validation leakage',c.ErrorCode.ID_CONFLICT)
    origin=all_samples[0].evidence.origin;frame=all_samples[0].truth.frame_id
    errors=[];vectors=[]
    for s in all_samples:
        c.require(s.evidence.kind is c.EvidenceKind.PHYSICAL,'truth','independent physical measurement required',c.ErrorCode.EVIDENCE_NOT_PHYSICAL)
        c.require(s.evidence.origin is origin,'evidence.origin','mixed calibration provenance',c.ErrorCode.SIMULATED_EVIDENCE)
        c.require(s.truth.frame_id==s.predicted.frame_id==frame,'frame','truth and prediction must share frame')
        xyz=(s.truth.x,s.truth.y,s.truth.z)
        c.require(all(a<=x<=b for x,a,b in zip(xyz,*domain_mm)),'domain','sample outside declared domain')
        vec=tuple(getattr(s.predicted,k)-getattr(s.truth,k) for k in ('x','y','z'))
        vectors.append(vec);errors.append(math.sqrt(sum(v*v for v in vec))+s.truth.error_mm+s.predicted.error_mm)
    n=len(fit_samples);checks=tuple(errors[n:]);v=vectors[n:]
    maximum=max(checks)
    return CalibrationReport(version,chain,camera_version,tool_version,tuple(tuple(p) for p in domain_mm),
        tuple(fit_samples),tuple(validation_samples),tuple(errors[:n]),checks,maximum,
        math.sqrt(sum(e*e for e in checks)/len(checks)),tuple(sum(p[i] for p in v)/len(v) for i in range(3)),limit_mm,maximum<=limit_mm,binding)


def save_report(report,path):
    # Calibration authoring only; runtime intent/result durability belongs to P10B.
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(asdict(report),f,ensure_ascii=False,allow_nan=False,indent=2)
        f.write('\n')
