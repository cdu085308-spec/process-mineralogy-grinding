"""闭路磨矿分级效率与循环负荷计算。"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CircuitResult:
    feed_tph: float
    overflow_tph: float
    underflow_tph: float
    circulating_load_ratio: float
    circulating_load_pct: float
    mass_balance_error_pct: float
    classification_efficiency_pct: float | None
    cut_size_um: float | None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "feed_tph": round(self.feed_tph, 4),
            "overflow_tph": round(self.overflow_tph, 4),
            "underflow_tph": round(self.underflow_tph, 4),
            "circulating_load_ratio": round(self.circulating_load_ratio, 6),
            "circulating_load_pct": round(self.circulating_load_pct, 3),
            "mass_balance_error_pct": round(self.mass_balance_error_pct, 3),
            "classification_efficiency_pct": (
                round(self.classification_efficiency_pct, 3)
                if self.classification_efficiency_pct is not None
                else None
            ),
            "cut_size_um": round(self.cut_size_um, 3) if self.cut_size_um is not None else None,
            "warnings": self.warnings,
        }


def calculate_circuit(
    feed_tph: float,
    overflow_tph: float,
    underflow_tph: float | None = None,
    feed_fines_pct: float | None = None,
    overflow_fines_pct: float | None = None,
    cut_size_um: float | None = None,
) -> CircuitResult:
    if feed_tph <= 0 or overflow_tph <= 0:
        raise ValueError("给矿量和溢流量必须大于 0。")
    if overflow_tph >= feed_tph:
        raise ValueError("溢流量必须小于给矿量，否则无法形成有效返砂。")
    calculated_underflow = (
        float(underflow_tph) if underflow_tph is not None else feed_tph - overflow_tph
    )
    if calculated_underflow < 0:
        raise ValueError("沉砂量不能为负数。")
    balance_error = abs(feed_tph - overflow_tph - calculated_underflow) / feed_tph * 100.0
    circulating = calculated_underflow / overflow_tph
    warnings: list[str] = []
    if balance_error > 10:
        raise ValueError(f"质量平衡误差 {balance_error:.1f}% 超过10%，请复核流量数据。")
    if balance_error > 5:
        warnings.append(f"质量平衡误差 {balance_error:.1f}% 超过5%，建议复核实测流量。")

    classification_efficiency = None
    if feed_fines_pct is not None and overflow_fines_pct is not None:
        if not 0 <= feed_fines_pct <= 100 or not 0 <= overflow_fines_pct <= 100:
            raise ValueError("细粒含量必须位于 0%—100% 之间。")
        if feed_fines_pct <= 0:
            raise ValueError("给矿细粒含量必须大于 0。")
        feed_fines = feed_tph * feed_fines_pct / 100.0
        overflow_fines = overflow_tph * overflow_fines_pct / 100.0
        classification_efficiency = min(overflow_fines / feed_fines * 100.0, 100.0)
        if float(overflow_fines) > float(feed_fines) * 1.02:
            warnings.append("溢流细粒量高于给矿细粒量，已按100%显示，请复核粒度数据。")

    return CircuitResult(
        feed_tph=float(feed_tph),
        overflow_tph=float(overflow_tph),
        underflow_tph=float(calculated_underflow),
        circulating_load_ratio=float(circulating),
        circulating_load_pct=float(circulating * 100.0),
        mass_balance_error_pct=float(balance_error),
        classification_efficiency_pct=(
            float(classification_efficiency)
            if classification_efficiency is not None
            else None
        ),
        cut_size_um=float(cut_size_um) if cut_size_um is not None else None,
        warnings=warnings,
    )


def passing_at_size(sizes, cumulative_passing, target_size_um: float) -> float:
    x = np.asarray(sizes, dtype=float)
    y = np.asarray(cumulative_passing, dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    if len(x) < 2:
        raise ValueError("至少需要两组粒度数据。")
    if target_size_um < x.min() or target_size_um > x.max():
        raise ValueError("分级粒径超出粒度曲线覆盖范围。")
    return float(np.interp(np.log(target_size_um), np.log(x), y))


def circuit_from_size_curves(
    feed_tph: float,
    overflow_tph: float,
    feed_sizes,
    feed_passing,
    overflow_sizes,
    overflow_passing,
    cut_size_um: float,
) -> CircuitResult:
    feed_fines = passing_at_size(feed_sizes, feed_passing, cut_size_um)
    overflow_fines = passing_at_size(overflow_sizes, overflow_passing, cut_size_um)
    return calculate_circuit(
        feed_tph=feed_tph,
        overflow_tph=overflow_tph,
        feed_fines_pct=feed_fines,
        overflow_fines_pct=overflow_fines,
        cut_size_um=cut_size_um,
    )
