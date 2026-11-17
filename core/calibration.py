"""实测筛析数据回代与单参数模型校准。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.kinetics import estimate_k_from_wi, fit_rosin_rammler
from core.validation import validate_grinding_inputs, validate_sieve_data


PREDICTION_SIEVES = np.array(
    [1, 5, 10, 20, 38, 45, 74, 100, 150, 200, 300, 400, 500, 1000, 2000, 5000],
    dtype=float,
)


@dataclass
class CalibrationResult:
    correction_factor: float
    observed_p80: float
    observed_rr_n: float
    observed_r_squared: float
    predicted_p80_before: float
    predicted_p80_after: float
    p80_error_before_pct: float
    p80_error_after_pct: float
    curve_rmse_before: float
    curve_rmse_after: float
    p80_improvement_pct: float
    inputs: dict[str, Any]
    observed_curve: dict[str, list[float]]
    predicted_before: dict[str, list[float]]
    predicted_after: dict[str, list[float]]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "correction_factor": round(float(self.correction_factor), 6),
            "observed_p80": round(float(self.observed_p80), 3),
            "observed_rr_n": round(float(self.observed_rr_n), 6),
            "observed_r_squared": round(float(self.observed_r_squared), 6),
            "predicted_p80_before": round(float(self.predicted_p80_before), 3),
            "predicted_p80_after": round(float(self.predicted_p80_after), 3),
            "p80_error_before_pct": round(float(self.p80_error_before_pct), 3),
            "p80_error_after_pct": round(float(self.p80_error_after_pct), 3),
            "curve_rmse_before": round(float(self.curve_rmse_before), 3),
            "curve_rmse_after": round(float(self.curve_rmse_after), 3),
            "p80_improvement_pct": round(float(self.p80_improvement_pct), 3),
            "inputs": self.inputs,
            "observed_curve": self.observed_curve,
            "predicted_before": self.predicted_before,
            "predicted_after": self.predicted_after,
            "warnings": self.warnings,
        }


def predict_grinding_curve(
    bond_wi: float,
    f80: float,
    grind_time: float,
    mill_type: str = "球磨机",
    correction_factor: float = 1.0,
    rr_n: float = 1.2,
) -> dict[str, Any]:
    """按现有教学模型预测曲线，correction_factor 用于校准磨矿速率系数。"""
    factor = max(float(correction_factor), 1e-6)
    n = max(float(rr_n), 0.2)
    base_k = float(estimate_k_from_wi(float(bond_wi), mill_type))
    p80 = float(f80) * np.exp(-base_k * factor * float(grind_time) * 1.8)
    p80 = float(np.clip(p80, 1.0, float(f80)))
    d63 = p80 / (np.log(5.0) ** (1.0 / n))
    passing = 100.0 * (1.0 - np.exp(-((PREDICTION_SIEVES / max(d63, 1e-9)) ** n)))
    return {
        "p80": round(p80, 3),
        "sieve": PREDICTION_SIEVES.astype(float).tolist(),
        "passing": np.round(passing, 3).tolist(),
        "k_base": round(base_k, 6),
        "k_effective": round(base_k * factor, 6),
        "correction_factor": round(factor, 6),
        "rr_n": round(n, 6),
    }


def _prepare_curve(sizes, passing):
    sizes_arr = np.asarray(sizes, dtype=float)
    passing_arr = np.asarray(passing, dtype=float)
    order = np.argsort(sizes_arr)
    sizes_arr = sizes_arr[order]
    passing_arr = passing_arr[order]
    unique_sizes, unique_idx = np.unique(sizes_arr, return_index=True)
    return unique_sizes, passing_arr[unique_idx]


def compare_curves(
    observed_sizes,
    observed_passing,
    predicted_sizes,
    predicted_passing,
) -> dict[str, float]:
    """在两条曲线共同覆盖区间内比较累计通过率误差。"""
    obs_x, obs_y = _prepare_curve(observed_sizes, observed_passing)
    pred_x, pred_y = _prepare_curve(predicted_sizes, predicted_passing)
    low = max(obs_x.min(), pred_x.min())
    high = min(obs_x.max(), pred_x.max())
    if high <= low:
        raise ValueError("实测与预测曲线没有可比较的重叠粒度区间。")
    grid = np.exp(np.linspace(np.log(low), np.log(high), 80))
    obs_interp = np.interp(np.log(grid), np.log(obs_x), obs_y)
    pred_interp = np.interp(np.log(grid), np.log(pred_x), pred_y)
    errors = pred_interp - obs_interp
    return {
        "rmse": round(float(np.sqrt(np.mean(errors**2))), 3),
        "mae": round(float(np.mean(np.abs(errors))), 3),
        "max_abs_error": round(float(np.max(np.abs(errors))), 3),
        "max_error_size_um": round(float(grid[int(np.argmax(np.abs(errors)))]), 3),
    }


def calibrate_grinding_model(
    bond_wi: float,
    f80: float,
    grind_time: float,
    mill_type: str,
    observed_sizes,
    observed_passing,
    factor_bounds: tuple[float, float] = (0.01, 5.0),
) -> CalibrationResult:
    """通过磨矿速率系数反演，使预测 P80 接近实测 P80。

    只调整一个具有工程含义的参数，避免在单组数据上任意拟合多个系数。
    """
    grinding_validation = validate_grinding_inputs(
        f80=f80,
        target_p80=min(float(f80) / 2.0, max(0.1, float(f80) - 1.0)),
        bond_wi=bond_wi,
        grind_time=grind_time,
        mill_type=mill_type,
    )
    if not grinding_validation.ok:
        raise ValueError("；".join(grinding_validation.errors))
    sieve_validation = validate_sieve_data(observed_sizes, observed_passing)
    if not sieve_validation.ok:
        raise ValueError("；".join(sieve_validation.errors))

    fit = fit_rosin_rammler(observed_sizes, observed_passing)
    if not isinstance(fit, dict) or fit.get("p80", 0) <= 0:
        raise ValueError("实测筛析数据无法完成 Rosin-Rammler 拟合。")

    observed_p80 = float(fit["p80"])
    base = predict_grinding_curve(bond_wi, f80, grind_time, mill_type, correction_factor=1.0)
    base_k = float(base["k_base"])
    if base_k <= 0 or grind_time <= 0:
        raise ValueError("磨矿速率系数或磨矿时间无效，无法完成校准。")

    raw_factor = np.log(float(f80) / observed_p80) / (base_k * float(grind_time) * 1.8)
    low, high = factor_bounds
    factor = float(np.clip(raw_factor, low, high))
    calibrated = predict_grinding_curve(
        bond_wi, f80, grind_time, mill_type, correction_factor=factor
    )

    warnings: list[str] = list(sieve_validation.warnings)
    if not np.isfinite(raw_factor):
        raise ValueError("实测 P80 与 F80 的组合无法形成有效校准系数。")
    if raw_factor < low or raw_factor > high:
        warnings.append(
            f"反演系数 {raw_factor:.3f} 超出建议范围 {low:g}—{high:g}，已截断；请优先复核实验条件。"
        )

    observed_curve = {
        "sieve": np.asarray(fit["fitted_curve"]["sieve"], dtype=float).tolist(),
        "passing": np.asarray(fit["fitted_curve"]["passing"], dtype=float).tolist(),
    }
    error_before = compare_curves(
        observed_curve["sieve"],
        observed_curve["passing"],
        base["sieve"],
        base["passing"],
    )
    error_after = compare_curves(
        observed_curve["sieve"],
        observed_curve["passing"],
        calibrated["sieve"],
        calibrated["passing"],
    )
    before_p80 = float(base["p80"])
    after_p80 = float(calibrated["p80"])
    before_error = abs(before_p80 - observed_p80) / observed_p80 * 100.0
    after_error = abs(after_p80 - observed_p80) / observed_p80 * 100.0
    improvement = (
        (before_error - after_error) / before_error * 100.0 if before_error > 1e-9 else 0.0
    )

    return CalibrationResult(
        correction_factor=factor,
        observed_p80=observed_p80,
        observed_rr_n=float(fit["n"]),
        observed_r_squared=float(fit["r_squared"]),
        predicted_p80_before=before_p80,
        predicted_p80_after=after_p80,
        p80_error_before_pct=before_error,
        p80_error_after_pct=after_error,
        curve_rmse_before=error_before["rmse"],
        curve_rmse_after=error_after["rmse"],
        p80_improvement_pct=improvement,
        inputs={
            "bond_wi": round(float(bond_wi), 3),
            "f80": round(float(f80), 3),
            "grind_time": round(float(grind_time), 3),
            "mill_type": mill_type,
            "observed_points": int(len(observed_sizes)),
        },
        observed_curve=observed_curve,
        predicted_before={
            "sieve": np.asarray(base["sieve"]).tolist(),
            "passing": np.asarray(base["passing"]).tolist(),
        },
        predicted_after={
            "sieve": np.asarray(calibrated["sieve"]).tolist(),
            "passing": np.asarray(calibrated["passing"]).tolist(),
        },
        warnings=warnings,
    )


def leave_one_out_calibration(records: list[dict]) -> dict[str, float]:
    """对多组实验记录做留一法评估，返回平均 P80 误差。

    每条记录至少包含 bond_wi、f80、grind_time、mill_type、observed_p80。
    """
    if len(records) < 3:
        raise ValueError("留一法至少需要 3 组实验数据。")
    factors = []
    for record in records:
        base_k = float(estimate_k_from_wi(float(record["bond_wi"]), record["mill_type"]))
        factor = np.log(float(record["f80"]) / float(record["observed_p80"])) / (
            base_k * float(record["grind_time"]) * 1.8
        )
        factors.append(float(factor))

    errors = []
    for idx, record in enumerate(records):
        other_factors = [v for j, v in enumerate(factors) if j != idx]
        factor = float(np.median(other_factors))
        prediction = predict_grinding_curve(
            record["bond_wi"],
            record["f80"],
            record["grind_time"],
            record["mill_type"],
            correction_factor=factor,
        )
        errors.append(
            abs(float(prediction["p80"]) - float(record["observed_p80"]))
            / float(record["observed_p80"])
            * 100.0
        )
    return {
        "mean_abs_p80_error_pct": round(float(np.mean(errors)), 3),
        "max_abs_p80_error_pct": round(float(np.max(errors)), 3),
    }
