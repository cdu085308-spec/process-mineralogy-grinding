"""输入数据校验与可读错误提示。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np


@dataclass
class ValidationIssue:
    severity: str
    code: str
    message: str


@dataclass
class ValidationResult:
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[str]:
        return [x.message for x in self.issues if x.severity == "error"]

    @property
    def warnings(self) -> list[str]:
        return [x.message for x in self.issues if x.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def add_error(self, code: str, message: str):
        self.issues.append(ValidationIssue("error", code, message))

    def add_warning(self, code: str, message: str):
        self.issues.append(ValidationIssue("warning", code, message))

    def extend(self, other: "ValidationResult"):
        self.issues.extend(other.issues)


PARAMETER_RANGES = {
    "density": (1.0, 20.0, "密度", "g/cm3"),
    "hardness": (0.0, 10.0, "莫氏硬度", ""),
    "bond_wi": (0.5, 50.0, "Bond 功指数", "kWh/t"),
    "content": (0.0, 100.0, "矿物含量", "%"),
    "grain_size": (0.1, 100000.0, "嵌布粒度", "μm"),
    "f80": (1.0, 100000.0, "给矿 F80", "μm"),
    "target_p80": (0.1, 100000.0, "目标 P80", "μm"),
    "grind_time": (0.0, 1440.0, "磨矿时间", "min"),
}


def _to_float_array(values: Sequence[float], name: str, result: ValidationResult):
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        result.add_error("invalid_number", f"{name}包含无法识别的数值。")
        return None
    if arr.ndim != 1:
        result.add_error("invalid_shape", f"{name}必须是一维数据。")
        return None
    return arr


def validate_sieve_data(
    sieve_sizes: Sequence[float],
    cumulative_passing: Sequence[float],
    min_points: int = 3,
) -> ValidationResult:
    """校验筛析曲线，不合法数据不得进入正式拟合或报告。"""
    result = ValidationResult()
    sizes = _to_float_array(sieve_sizes, "筛孔尺寸", result)
    passing = _to_float_array(cumulative_passing, "累计通过率", result)
    if sizes is None or passing is None:
        return result
    if len(sizes) != len(passing):
        result.add_error("length_mismatch", "筛孔尺寸与累计通过率的数据个数不一致。")
        return result
    if len(sizes) < min_points:
        result.add_error("too_few_points", f"有效筛析点不足，至少需要 {min_points} 组数据。")
    if not np.all(np.isfinite(sizes)) or not np.all(np.isfinite(passing)):
        result.add_error("non_finite", "筛析数据中包含空值、无穷大或无法识别的数值。")
        return result
    if np.any(sizes <= 0):
        result.add_error("non_positive_size", "筛孔尺寸必须大于 0。")
    if np.any((passing < 0) | (passing > 100)):
        result.add_error("passing_range", "累计通过率必须位于 0%—100% 之间。")
    if len(np.unique(sizes)) != len(sizes):
        result.add_error("duplicate_size", "筛孔尺寸存在重复值，请合并后再计算。")
    if result.errors:
        return result

    order = np.argsort(sizes)
    sorted_passing = passing[order]
    diffs = np.diff(sorted_passing)
    if np.any(diffs < -0.5):
        result.add_error(
            "non_monotonic",
            "累计通过率必须随筛孔尺寸增大而单调不减，请检查粒级是否填反。",
        )
    if np.max(sorted_passing) < 95:
        result.add_warning(
            "coverage_upper",
            "最粗筛孔的累计通过率低于95%，P80 结果可能需要外推，建议补充粗粒级数据。",
        )
    if np.min(sorted_passing) > 20:
        result.add_warning(
            "coverage_lower",
            "最细筛孔的累计通过率高于20%，P80 可能依赖外推，建议补充细粒级数据。",
        )
    return result


def validate_grinding_inputs(
    f80: float,
    target_p80: float,
    bond_wi: float | None = None,
    grind_time: float | None = None,
    mill_type: str | None = None,
) -> ValidationResult:
    result = ValidationResult()
    numeric = {"F80": f80, "目标P80": target_p80}
    if bond_wi is not None:
        numeric["Bond Wi"] = bond_wi
    if grind_time is not None:
        numeric["磨矿时间"] = grind_time
    for name, value in numeric.items():
        if value is None or not np.isfinite(float(value)):
            result.add_error("missing_value", f"{name}不能为空。")
    if result.errors:
        return result
    if f80 <= 0 or target_p80 <= 0:
        result.add_error("non_positive", "F80 和目标 P80 必须大于 0。")
    if target_p80 >= f80:
        result.add_error("p80_order", "目标 P80 必须小于给矿 F80。")
    if bond_wi is not None and not 0.5 <= bond_wi <= 50:
        result.add_error("wi_range", "Bond Wi 建议位于 0.5—50 kWh/t；超出范围时请核对单位。")
    if grind_time is not None and not 0 <= grind_time <= 1440:
        result.add_error("time_range", "磨矿时间必须位于 0—1440 min 之间。")
    if mill_type and mill_type not in {"球磨机", "棒磨机", "自磨机", "破碎机"}:
        result.add_error("mill_type", f"未识别的磨机类型：{mill_type}。")
    return result


def validate_mineral_parameters(
    name: str,
    density: float,
    hardness: float,
    bond_wi: float,
    grain_size: float,
    content: float,
) -> ValidationResult:
    result = ValidationResult()
    if not str(name).strip():
        result.add_error("name_required", "矿物名称不能为空。")
    values = {
        "density": density,
        "hardness": hardness,
        "bond_wi": bond_wi,
        "grain_size": grain_size,
        "content": content,
    }
    for key, value in values.items():
        low, high, label, unit = PARAMETER_RANGES[key]
        if value is None or not np.isfinite(float(value)):
            result.add_error(f"{key}_missing", f"{label}不能为空。")
        elif not low <= float(value) <= high:
            result.add_error(
                f"{key}_range",
                f"{label}应位于 {low:g}—{high:g} {unit}，当前值为 {float(value):g}。",
            )
    return result


def validate_mineral_composition(components: Iterable[dict]) -> ValidationResult:
    result = ValidationResult()
    rows = list(components)
    if not rows:
        result.add_error("composition_empty", "尚未添加矿物组成。")
        return result
    total = 0.0
    for row in rows:
        name = row.get("name", "未命名矿物")
        content = float(row.get("content", 0) or 0)
        total += content
        if content < 0:
            result.add_error("negative_content", f"{name}的含量不能为负数。")
        if row.get("bond_wi") is not None and not 0.5 <= float(row["bond_wi"]) <= 50:
            result.add_warning("wi_outlier", f"{name}的 Bond Wi 超出常用教学范围。")
    if total <= 0:
        result.add_error("zero_total", "矿物总含量必须大于 0。")
    elif abs(total - 100) > 5:
        result.add_error(
            "total_out_of_range",
            f"矿物总含量为 {total:.1f}%，偏离100%超过5%，请先归一化或补全。",
        )
    elif abs(total - 100) > 0.5:
        result.add_warning(
            "total_requires_normalization",
            f"矿物总含量为 {total:.1f}%，正式计算前建议归一化到100%。",
        )
    return result


def format_validation_messages(result: ValidationResult, include_warnings: bool = True) -> list[str]:
    messages = list(result.errors)
    if include_warnings:
        messages.extend(f"提示：{x}" for x in result.warnings)
    return messages
