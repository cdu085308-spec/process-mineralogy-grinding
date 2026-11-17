"""Streamlit pages for the upgraded teaching workflow."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.calibration import calibrate_grinding_model
from core.closed_circuit import calculate_circuit
from core.evaluation import (
    DEFAULT_RUBRIC,
    build_llm_review_request,
    evaluate_scores,
    heuristic_pre_review,
)
from core.kinetics import read_sieve_csv
from core.mineral_data import (
    COMMON_MINERALS,
    LOADED_MINERAL_COUNT,
    mineral_catalog_dataframe,
)
from core.storage import latest_record, save_record
from core.validation import validate_mineral_parameters, validate_sieve_data


EXAMPLE_SIEVE_DATA = """粒度(um),累计通过率(%)
1000,95.2
500,82.1
200,63.5
100,42.3
74,28.6
45,15.2
38,10.8"""


def _show_validation(result):
    for message in result.errors:
        st.error(message)
    for message in result.warnings:
        st.warning(message)


def _curve_text_from_session() -> str:
    raw = st.session_state.get("sieve_raw")
    if not raw:
        return EXAMPLE_SIEVE_DATA
    sizes, passing = raw
    lines = ["粒度(um),累计通过率(%)"]
    lines.extend(f"{float(size):g},{float(value):g}" for size, value in zip(sizes, passing))
    return "\n".join(lines)


def render_parameter_library_page():
    st.markdown("<div class='main-header'>矿物工艺参数库</div>", unsafe_allow_html=True)
    st.caption(
        f"当前载入 {len(COMMON_MINERALS)} 种矿物，其中 {LOADED_MINERAL_COUNT} 条来自外部参数库。"
    )
    st.info(
        "参数库中的 Bond Wi、密度和嵌布粒度为课程教学初值。用于正式仿真或结题报告前，"
        "应按课程指定教材、矿石性质测试或企业数据完成校核。"
    )
    catalog = mineral_catalog_dataframe()
    kind = st.segmented_control(
        "筛选类型",
        options=["全部", "目的矿物", "脉石"],
        default="全部",
    )
    if kind == "目的矿物":
        catalog = catalog[catalog["类型"] == "valuable"]
    elif kind == "脉石":
        catalog = catalog[catalog["类型"] == "gangue"]
    search = st.text_input("搜索矿物或化学式", "")
    if search.strip():
        mask = catalog["矿物"].str.contains(search.strip(), case=False, na=False) | catalog[
            "化学式"
        ].str.contains(search.strip(), case=False, na=False)
        catalog = catalog[mask]
    st.dataframe(catalog, width="stretch", hide_index=True, height=520)
    csv_data = catalog.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "导出当前参数库 CSV",
        data=csv_data,
        file_name="矿物工艺参数库.csv",
        mime="text/csv",
        width="stretch",
    )
    st.caption("参数来源和备注列随 CSV 一同导出，便于教师复核与版本管理。")


def render_calibration_page():
    st.markdown("<div class='main-header'>实测数据回代校准</div>", unsafe_allow_html=True)
    st.caption("用批磨实验筛析数据反演磨矿速率系数，并比较校准前后的 P80 与曲线误差。")

    analysis = st.session_state.get("results") if st.session_state.get("analysis_done") else {}
    params = analysis.get("params", {}) if isinstance(analysis, dict) else {}
    default_f80 = float(params.get("f80", 10000))
    default_wi = float(params.get("avg_wi", 14.2))
    default_time = float(params.get("grind_time", 15))
    default_mill = params.get("mill_type", "球磨机")

    with st.form("calibration_form"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            f80 = st.number_input("F80 (μm)", 1.0, 100000.0, default_f80, 100.0)
        with c2:
            bond_wi = st.number_input("Bond Wi (kWh/t)", 0.5, 50.0, default_wi, 0.5)
        with c3:
            grind_time = st.number_input("批磨时间 (min)", 0.1, 1440.0, default_time, 1.0)
        with c4:
            mill_type = st.selectbox(
                "磨机类型",
                ["球磨机", "棒磨机", "自磨机"],
                index=["球磨机", "棒磨机", "自磨机"].index(default_mill)
                if default_mill in {"球磨机", "棒磨机", "自磨机"}
                else 0,
            )
        data_text = st.text_area(
            "实测筛析数据",
            value=_curve_text_from_session(),
            height=230,
            help="第一列为筛孔尺寸(μm)，第二列为累计通过率(%)。",
        )
        run = st.form_submit_button("执行回代校准", type="primary", width="stretch")

    if run:
        try:
            sizes, passing = read_sieve_csv(data_text)
            validation = validate_sieve_data(sizes, passing)
            if not validation.ok:
                _show_validation(validation)
            else:
                result = calibrate_grinding_model(
                    bond_wi=bond_wi,
                    f80=f80,
                    grind_time=grind_time,
                    mill_type=mill_type,
                    observed_sizes=sizes,
                    observed_passing=passing,
                )
                st.session_state.calibration_result = result
                st.rerun()
        except Exception as exc:
            st.error(f"校准失败：{exc}")

    result = st.session_state.get("calibration_result")
    if result is None:
        st.info("填写实验条件并执行校准后，可查看校准前后误差。")
        return

    for warning in result.warnings:
        st.warning(warning)
    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("反演校准系数", f"{result.correction_factor:.3f}")
    m2.metric("实测 P80", f"{result.observed_p80:.1f} μm")
    m3.metric(
        "校准前 P80 偏差",
        f"{result.p80_error_before_pct:.1f}%",
        delta=f"校准后 {result.p80_error_after_pct:.1f}%",
        delta_color="inverse",
    )
    m4.metric("P80 误差改善", f"{result.p80_improvement_pct:.1f}%")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=result.observed_curve["sieve"],
            y=result.observed_curve["passing"],
            mode="markers",
            name=f"实测 RR 拟合 (P80={result.observed_p80:.1f} μm)",
            marker=dict(size=8, color="#2E7D32"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=result.predicted_before["sieve"],
            y=result.predicted_before["passing"],
            mode="lines",
            name=f"校准前 (P80={result.predicted_p80_before:.1f} μm)",
            line=dict(color="#C62828", width=2, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=result.predicted_after["sieve"],
            y=result.predicted_after["passing"],
            mode="lines",
            name=f"校准后 (P80={result.predicted_p80_after:.1f} μm)",
            line=dict(color="#1565C0", width=3),
        )
    )
    fig.add_hline(y=80, line_dash="dot", line_color="#616161")
    fig.update_layout(
        xaxis_title="粒度 (μm)",
        yaxis_title="累计通过率 (%)",
        xaxis_type="log",
        height=440,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig, width="stretch")

    detail = pd.DataFrame(
        [
            {
                "指标": "P80 相对偏差",
                "校准前": f"{result.p80_error_before_pct:.2f}%",
                "校准后": f"{result.p80_error_after_pct:.2f}%",
            },
            {
                "指标": "粒度曲线 RMSE",
                "校准前": f"{result.curve_rmse_before:.2f}",
                "校准后": f"{result.curve_rmse_after:.2f}",
            },
            {
                "指标": "实测 RR 拟合 R²",
                "校准前": f"{result.observed_r_squared:.4f}",
                "校准后": f"{result.observed_r_squared:.4f}",
            },
        ]
    )
    st.dataframe(detail, width="stretch", hide_index=True)

    b1, b2, b3 = st.columns(3)
    record = {"type": "grinding_calibration", **result.to_dict()}
    with b1:
        if st.button("保存校准记录", width="stretch"):
            saved = save_record("calibrations", record)
            st.success(f"已保存记录 {saved['record_id']}")
    with b2:
        if st.button("应用该系数到后续分析", width="stretch"):
            st.session_state.calibration_factor = result.correction_factor
            st.success("已设为当前教学分析校准系数。")
    with b3:
        st.download_button(
            "导出校准 JSON",
            data=json.dumps(record, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="磨矿回代校准记录.json",
            mime="application/json",
            width="stretch",
        )

    latest = latest_record("calibrations")
    if latest:
        st.caption(
            f"最近保存：{latest.get('created_at', '')}｜记录号 {latest.get('record_id', '')}"
        )


def render_closed_circuit_page():
    st.markdown("<div class='main-header'>闭路磨矿：分级效率与循环负荷</div>", unsafe_allow_html=True)
    st.caption("按质量平衡计算循环负荷，并根据目标粒级细粒回收计算分级效率。")

    with st.form("circuit_form"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            feed = st.number_input("给矿量 F (t/h)", 0.1, 100000.0, 100.0, 5.0)
        with c2:
            overflow = st.number_input("溢流量 O (t/h)", 0.1, 99999.0, 70.0, 5.0)
        with c3:
            underflow = st.number_input("沉砂量 U (t/h)", 0.0, 100000.0, 30.0, 5.0)
        with c4:
            cut_size = st.number_input("分级粒径 d50c (μm)", 1.0, 10000.0, 74.0, 5.0)
        c5, c6 = st.columns(2)
        with c5:
            feed_fines = st.number_input("给矿中目标细粒含量 (%)", 0.0, 100.0, 55.0, 1.0)
        with c6:
            overflow_fines = st.number_input("溢流中目标细粒含量 (%)", 0.0, 100.0, 60.0, 1.0)
        run = st.form_submit_button("计算闭路指标", type="primary", width="stretch")

    if run:
        try:
            result = calculate_circuit(
                feed_tph=feed,
                overflow_tph=overflow,
                underflow_tph=underflow,
                feed_fines_pct=feed_fines,
                overflow_fines_pct=overflow_fines,
                cut_size_um=cut_size,
            )
            st.session_state.circuit_result = result
            st.rerun()
        except Exception as exc:
            st.error(f"计算失败：{exc}")

    result = st.session_state.get("circuit_result")
    if result is None:
        st.info("输入流量和细粒含量后执行计算。没有实测分级数据时，结果仅作为教学仿真。")
        return

    for warning in result.warnings:
        st.warning(warning)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("循环负荷 C", f"{result.circulating_load_pct:.1f}%")
    m2.metric("返砂比 U/O", f"{result.circulating_load_ratio:.2f}")
    m3.metric(
        "分级效率 η",
        f"{result.classification_efficiency_pct:.1f}%"
        if result.classification_efficiency_pct is not None
        else "未计算",
    )
    m4.metric("质量平衡误差", f"{result.mass_balance_error_pct:.2f}%")

    flow_df = pd.DataFrame(
        {
            "物流": ["给矿 F", "溢流 O", "沉砂 U"],
            "流量(t/h)": [result.feed_tph, result.overflow_tph, result.underflow_tph],
        }
    )
    fig = go.Figure(
        go.Bar(
            x=flow_df["物流"],
            y=flow_df["流量(t/h)"],
            marker_color=["#455A64", "#2E7D32", "#C7832B"],
            text=flow_df["流量(t/h)"].round(2),
            textposition="outside",
        )
    )
    fig.update_layout(
        yaxis_title="流量 (t/h)",
        height=360,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig, width="stretch")
    st.latex(r"C=\frac{U}{O},\qquad \eta=\frac{O \times o_f}{F \times f_f}\times 100\%")
    st.caption("C 为循环负荷，η 为目标细粒进入溢流的回收率；质量平衡误差应优先低于5%。")

    record = {"type": "closed_circuit", **result.to_dict()}
    b1, b2 = st.columns(2)
    with b1:
        if st.button("保存闭路计算记录", width="stretch"):
            saved = save_record("circuits", record)
            st.success(f"已保存记录 {saved['record_id']}")
    with b2:
        st.download_button(
            "导出闭路计算 JSON",
            data=json.dumps(record, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="闭路磨矿计算记录.json",
            mime="application/json",
            width="stretch",
        )


def render_evaluation_page():
    st.markdown("<div class='main-header'>过程性评价与教师复核</div>", unsafe_allow_html=True)
    st.caption("量规初评只提供证据提示，教师复核后的评分才作为最终成绩。")

    c1, c2 = st.columns(2)
    with c1:
        student_code = st.text_input("匿名学生代码", value="S001")
    with c2:
        task_name = st.text_input("任务名称", value="碎磨方案设计与回代校准")
    submission_text = st.text_area(
        "学生作品证据文本",
        height=220,
        placeholder="粘贴方案说明、实验数据、校准分析、AI验证记录和反思等内容。",
    )

    if st.button("生成本地规则辅助初评", width="stretch"):
        preview = heuristic_pre_review(submission_text)
        st.session_state.evaluation_ai_scores = preview["scores"]
        st.session_state.evaluation_rationales = preview["rationales"]
        for item in DEFAULT_RUBRIC:
            st.session_state[f"ai_score_{item['id']}"] = float(preview["scores"][item["id"]])
        st.rerun()

    ai_defaults = st.session_state.get(
        "evaluation_ai_scores", {item["id"]: 0.0 for item in DEFAULT_RUBRIC}
    )
    teacher_defaults = st.session_state.get(
        "evaluation_teacher_scores", {item["id"]: 0.0 for item in DEFAULT_RUBRIC}
    )
    rationales = st.session_state.get("evaluation_rationales", {})
    ai_scores = {}
    teacher_scores = {}

    st.markdown("---")
    st.markdown("<div class='sub-header'>量规评分</div>", unsafe_allow_html=True)
    for item in DEFAULT_RUBRIC:
        st.markdown(
            f"**{item['name']}**（权重 {item['weight'] * 100:.0f}%）  \n"
            f"{item['description']}"
        )
        c_ai, c_teacher = st.columns(2)
        with c_ai:
            ai_key = f"ai_score_{item['id']}"
            ai_kwargs = {
                "label": f"辅助初评：{item['name']}",
                "min_value": 0.0,
                "max_value": 4.0,
                "step": 1.0,
                "key": ai_key,
            }
            if ai_key not in st.session_state:
                ai_kwargs["value"] = float(ai_defaults.get(item["id"], 0.0))
            ai_scores[item["id"]] = st.slider(**ai_kwargs)
        with c_teacher:
            teacher_key = f"teacher_score_{item['id']}"
            teacher_kwargs = {
                "label": f"教师复核：{item['name']}",
                "min_value": 0.0,
                "max_value": 4.0,
                "step": 1.0,
                "key": teacher_key,
            }
            if teacher_key not in st.session_state:
                teacher_kwargs["value"] = float(teacher_defaults.get(item["id"], 0.0))
            teacher_scores[item["id"]] = st.slider(**teacher_kwargs)
        if rationales.get(item["id"]):
            st.caption(rationales[item["id"]])

    result = evaluate_scores(ai_scores, teacher_scores)
    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("辅助初评", f"{result.ai_weighted_score:.1f}/100")
    m2.metric("教师复核", f"{result.teacher_weighted_score:.1f}/100")
    m3.metric("分差", f"{result.score_gap:.1f}")
    m4.metric(
        "加权 Kappa",
        f"{result.weighted_kappa:.2f}" if result.weighted_kappa is not None else "不适用",
    )
    st.dataframe(pd.DataFrame(result.rubric_rows), width="stretch", hide_index=True)

    record = {
        "type": "rubric_evaluation",
        "student_code": student_code,
        "task_name": task_name,
        "submission_text": submission_text,
        "ai_scores": ai_scores,
        "teacher_scores": teacher_scores,
        **result.to_dict(),
    }
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("保存教师复核记录", width="stretch"):
            st.session_state.evaluation_teacher_scores = teacher_scores
            saved = save_record("evaluations", record)
            st.success(f"已保存记录 {saved['record_id']}")
    with b2:
        st.download_button(
            "导出评价 JSON",
            data=json.dumps(record, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="过程性评价记录.json",
            mime="application/json",
            width="stretch",
        )
    with b3:
        llm_request = build_llm_review_request(task_name, submission_text)
        st.download_button(
            "导出大模型复核请求",
            data=json.dumps(llm_request, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="大模型辅助初评请求.json",
            mime="application/json",
            width="stretch",
        )

    st.info(
        "本地规则初评不调用外部网络。若使用内网或本地大模型，应先脱敏作品文本，"
        "并保存模型版本、提示词版本和教师复核意见。"
    )
