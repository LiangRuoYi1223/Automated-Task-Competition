"""Strict private P02 parameter file. Null/unmeasured inputs never enable use."""
import json
from dataclasses import dataclass, fields
from pathlib import Path
from treasure_auto.contracts import core as c, models as m
from .localization import LocalizationPolicy
from .service import FootprintProfile
from .storage import _pairs


@dataclass(frozen=True)
class Configuration:
    policy: LocalizationPolicy
    footprint: FootprintProfile
    base_frame: str
    mapping_source: str
    localization_source: str


def load_parameters(path, origin):
    try:
        data=json.loads(Path(path).read_text(encoding='utf-8'),object_pairs_hook=_pairs,
                        parse_constant=lambda s:c.require(False,s,'non-finite number'))
        expected={'format','contract_version','measurement','evidence','base_frame','mapping_source',
                  'localization_source','localization','footprint'}
        c.require(type(data) is dict and set(data) == expected,'parameters','exact parameter fields required')
        c.require(type(data['format']) is int and data['format']==1 and data['contract_version']==c.CONTRACT_VERSION,
                  'format','parameter or C0 version mismatch',c.ErrorCode.VERSION_MISMATCH)
        measurement=m.Measurement(data['measurement'])
        c.require(measurement is not m.Measurement.UNMEASURED,'measurement','field measurements missing',c.ErrorCode.UNCONFIGURED)
        refs=tuple(c.from_data(c.EvidenceRef,e) for e in data['evidence'])
        if origin is c.Origin.LIVE:
            c.require(measurement is m.Measurement.VERIFIED and refs and
                      all(e.origin is c.Origin.LIVE and e.kind is c.EvidenceKind.PHYSICAL for e in refs),
                      'measurement','verified LIVE parameter evidence required',c.ErrorCode.UNCONFIGURED)
        names={f.name for f in fields(LocalizationPolicy)}-{'measurement','evidence'}
        c.require(type(data['localization']) is dict and set(data['localization']) == names,
                  'localization','all named limits required; no implicit defaults')
        c.require(all(v is not None for v in data['localization'].values()),'localization','unset limit',c.ErrorCode.UNCONFIGURED)
        policy=LocalizationPolicy(measurement=measurement,evidence=refs,**data['localization'])
        footprint=data['footprint']
        c.require(type(footprint) is dict and set(footprint)=={'body','carried','margin_m'},'footprint','exact footprint fields required')
        def polygon(value):
            return None if value is None else tuple(c.from_data(m.Point2,p) for p in value)
        profile=FootprintProfile(polygon(footprint['body']),polygon(footprint['carried']),footprint['margin_m'],measurement,refs)
        profile.validate(origin)
        for key in ('base_frame','mapping_source','localization_source'):
            c.require(type(data[key]) is str and data[key].strip(),key,'unset frame/source',c.ErrorCode.UNCONFIGURED)
        c.require(data['base_frame'] not in ('map','odom') and data['mapping_source']!=data['localization_source'],
                  'sources','frames and conflicting sources must be distinct')
        return Configuration(policy,profile,data['base_frame'],data['mapping_source'],data['localization_source'])
    except OSError as error:
        raise c.ContractError(c.ErrorCode.UNAVAILABLE,'parameters',str(error)) from None
    except (ValueError,TypeError,KeyError) as error:
        if isinstance(error,c.ContractError):
            raise
        raise c.ContractError(c.ErrorCode.INVALID_FIELD,'parameters','malformed configuration') from None
