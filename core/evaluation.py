"""教学过程量规、辅助初评与评分一致性。"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import numpy as np


DEFAULT_RUBRIC = [
    {
        "id": "scheme",
        "name": "方案完整性",
        "weight": 0.20,
        "description": "目标、给矿条件、磨矿条件和方案论证是否完整。",
        "keywords": ["方案", "目标", "F80", "P80", "能耗", "段数"],
    },
    {
        "id": "data",
        "name": "数据处理",
        "weight": 0.20,
        "description": "筛析数据质量、RR拟合、P80与误差记录是否规范。",
        "keywords": ["筛析", "拟合", "R²", "P80", "误差", "数据"],
    },
    {
        "id": "calibration",
        "name": "偏差归因与校准",
        "weight": 0.20,
        "description": "是否识别预测偏差、选择合理参数并完成回代校准。",
        "keywords": ["校准", "偏差", "归因", "修正", "回代", "验证"],
    },
    {
        "id": "iteration",
        "name": "方案迭代",
        "weight": 0.15,
        "description": "是否根据校准结果再次预测并比较方案。",
        "keywords": ["迭代", "重新预测", "再次", "优化", "对比"],
    },
    {
        "id": "ai",
        "name": "AI批判使用",
        "weight": 0.15,
        "description": "是否检查AI输出、识别错误并说明工具边界。",
        "keywords": ["AI", "验证", "找错", "边界", "错误", "不可替代"],
    },
    {
        "id": "reflection",
        "name": "工程反思",
        "weight": 0.10,
        "description": "是否解释参数工程含义、模型适用条件和改进方向。",
        "keywords": ["反思", "适用条件", "工程含义", "改进", "局限"],
    },
]


@dataclass
class EvaluationResult:
    ai_weighted_score: float
    teacher_weighted_score: float
    score_gap: float
    weighted_kappa: float | None
    rubric_rows: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ai_weighted_score": round(self.ai_weighted_score, 3),
            "teacher_weighted_score": round(self.teacher_weighted_score, 3),
            "score_gap": round(self.score_gap, 3),
            "weighted_kappa": (
                round(self.weighted_kappa, 6)
                if self.weighted_kappa is not None
                else None
            ),
            "rubric_rows": self.rubric_rows,
        }


def _validate_scores(scores: dict[str, float], rubric=DEFAULT_RUBRIC):
    expected = {item["id"] for item in rubric}
    missing = expected - set(scores)
    if missing:
        raise ValueError(f"缺少量规评分项：{', '.join(sorted(missing))}")
    for item in rubric:
        value = float(scores[item["id"]])
        if not 0 <= value <= 4:
            raise ValueError(f"{item['name']}评分必须位于0—4。")


def weighted_score(scores: dict[str, float], rubric=DEFAULT_RUBRIC, max_score=4.0) -> float:
    _validate_scores(scores, rubric)
    total = sum(float(scores[item["id"]]) * float(item["weight"]) for item in rubric)
    return total / max_score * 100.0


def quadratic_weighted_kappa(
    ai_scores: list[float],
    teacher_scores: list[float],
    min_rating: int = 0,
    max_rating: int = 4,
) -> float | None:
    if len(ai_scores) != len(teacher_scores) or not ai_scores:
        raise ValueError("两组评分数量必须一致且非空。")
    a = np.rint(np.asarray(ai_scores, dtype=float)).astype(int)
    b = np.rint(np.asarray(teacher_scores, dtype=float)).astype(int)
    if np.any(a < min_rating) or np.any(a > max_rating):
        raise ValueError("辅助评分超出量规等级范围。")
    if np.any(b < min_rating) or np.any(b > max_rating):
        raise ValueError("教师评分超出量规等级范围。")
    labels = list(range(min_rating, max_rating + 1))
    index = {value: idx for idx, value in enumerate(labels)}
    observed = np.zeros((len(labels), len(labels)), dtype=float)
    for ai_value, teacher_value in zip(a, b):
        observed[index[int(ai_value)], index[int(teacher_value)]] += 1
    n = observed.sum()
    if n == 0:
        return None
    observed /= n
    hist_a = observed.sum(axis=1)
    hist_b = observed.sum(axis=0)
    expected = np.outer(hist_a, hist_b)
    weights = np.zeros_like(observed)
    for i in range(len(labels)):
        for j in range(len(labels)):
            weights[i, j] = ((i - j) ** 2) / ((len(labels) - 1) ** 2)
    denominator = float((weights * expected).sum())
    if denominator <= 1e-12:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(1.0 - (weights * observed).sum() / denominator)


def evaluate_scores(
    ai_scores: dict[str, float],
    teacher_scores: dict[str, float],
    rubric=DEFAULT_RUBRIC,
) -> EvaluationResult:
    _validate_scores(ai_scores, rubric)
    _validate_scores(teacher_scores, rubric)
    rows = []
    for item in rubric:
        ai_value = float(ai_scores[item["id"]])
        teacher_value = float(teacher_scores[item["id"]])
        rows.append(
            {
                "id": item["id"],
                "name": item["name"],
                "weight": item["weight"],
                "ai_score": round(ai_value, 3),
                "teacher_score": round(teacher_value, 3),
                "gap": round(teacher_value - ai_value, 3),
                "description": item["description"],
            }
        )
    ai_weighted = weighted_score(ai_scores, rubric)
    teacher_weighted = weighted_score(teacher_scores, rubric)
    kappa = quadratic_weighted_kappa(
        [float(ai_scores[item["id"]]) for item in rubric],
        [float(teacher_scores[item["id"]]) for item in rubric],
    )
    return EvaluationResult(
        ai_weighted_score=ai_weighted,
        teacher_weighted_score=teacher_weighted,
        score_gap=abs(teacher_weighted - ai_weighted),
        weighted_kappa=kappa,
        rubric_rows=rows,
    )


def heuristic_pre_review(submission_text: str, rubric=DEFAULT_RUBRIC) -> dict[str, Any]:
    """离线规则初评：只作为证据完整性提示，不生成最终成绩。"""
    text = str(submission_text or "").strip()
    normalized = re.sub(r"\s+", "", text).lower()
    scores: dict[str, float] = {}
    rationales: dict[str, str] = {}
    for item in rubric:
        matched = [
            keyword
            for keyword in item["keywords"]
            if keyword.lower().replace(" ", "") in normalized
        ]
        length_bonus = 1 if len(text) >= 300 else 0
        if len(matched) >= 3 and length_bonus:
            score = 3.0
        elif len(matched) >= 2:
            score = 2.0
        elif matched:
            score = 1.0
        else:
            score = 0.0
        scores[item["id"]] = score
        rationales[item["id"]] = (
            f"命中关键词：{'、'.join(matched)}"
            if matched
            else "未检测到该量规所需的可观察证据"
        )
    return {
        "scores": scores,
        "rationales": rationales,
        "note": "本地规则初评仅用于提示证据完整性；教师复核后成绩才生效。",
    }


def build_llm_review_request(
    task_description: str,
    submission_text: str,
    rubric=DEFAULT_RUBRIC,
) -> dict[str, Any]:
    """生成可交给内网大模型的结构化请求，不在本函数中发送数据。"""
    return {
        "task": "依据量规对教学作品进行辅助初评",
        "constraints": [
            "不得直接给出最终成绩",
            "每项必须引用作品证据并说明理由",
            "证据不足时降低置信度并标记人工复核",
        ],
        "task_description": task_description,
        "rubric": rubric,
        "submission_text": submission_text,
        "output_schema": {
            "scores": {item["id"]: "0-4" for item in rubric},
            "rationales": {item["id"]: "引用证据和理由" for item in rubric},
            "confidence": "0-1",
            "manual_review_required": "boolean",
        },
    }
