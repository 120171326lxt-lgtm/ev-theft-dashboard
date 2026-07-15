import streamlit as st
import pandas as pd

from common import inject_base_css, topbar, metric_card
import mock_data as md

st.set_page_config(page_title="模型训练与评估", layout="wide", page_icon="🚲")
inject_base_css()
topbar("模型训练与评估", "RW-STGCN&nbsp;&nbsp;|&nbsp;&nbsp;N=1085路段&nbsp;&nbsp;|&nbsp;&nbsp;T=1992步")

# ---- 侧边栏：模型配置 ----
with st.sidebar:
    st.markdown("#### 模型配置")
    st.selectbox("当前模型", ["RW-STGCN（本文方法）", "STGCN（无道路权重）", "GCN", "GRU", "KDE", "HA"])
    st.selectbox("切比雪夫阶数 K", ["K = 1", "K = 2", "K = 3", "K = 4"], index=2)
    st.selectbox("权重平衡系数 α", ["α = 0.4", "α = 0.5", "α = 0.6（验证集最优）", "α = 0.7"], index=2)

    st.markdown("#### 训练状态")
    st.markdown("**训练轮数**")
    st.markdown("### 200 epoch")
    st.caption("早停于 epoch 187")

    st.markdown("**数据集划分**")
    st.markdown("### 70/15/15")
    st.caption("训练/验证/测试")

    st.markdown("**训练时段**")
    st.markdown("### 03.20—05.16")
    st.caption("共58天")

st.markdown("### 模型训练过程与测试集评估结果")

table = md.model_compare_table()
best = table.iloc[-1]
baseline = table[table["模型"].str.contains("STGCN（无道路权重）")].iloc[0]

c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card(f"{best['MAE']:.3f}", "MAE（测试集）", f"↓ vs STGCN {baseline['MAE']:.3f}", sub_direction="down")
with c2:
    metric_card(f"{best['RMSE']:.3f}", "RMSE（测试集）", f"↓ vs STGCN {baseline['RMSE']:.3f}", sub_direction="down")
with c3:
    metric_card(f"{best['命中率(%)']:.1f}%", "命中率（测试集）", f"↑ vs STGCN {baseline['命中率(%)']:.1f}%",
                value_color="green", sub_direction="up")
with c4:
    metric_card(f"{best['PAI']:.2f}", "PAI（测试集）", f"↑ vs STGCN {baseline['PAI']:.2f}",
                value_color="green", sub_direction="up")

st.write("")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**训练/验证损失曲线（MAE）**")
    curve = md.loss_curve().set_index("epoch")
    st.line_chart(curve, height=280)
with col2:
    st.markdown("**各模型 MAE 对比（测试集）**")
    bar = md.model_bar_mae().set_index("模型简称")[["MAE"]]
    st.bar_chart(bar, height=280)

st.write("")
st.markdown("**各模型测试集性能对比（含消融实验）**")


def style_best(row):
    return ["background-color:#e8f7ee" if row["模型"].startswith("本文方法") else "" for _ in row]


st.dataframe(
    table.style.apply(style_best, axis=1).format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "命中率(%)": "{:.1f}%", "PAI": "{:.2f}"}),
    use_container_width=True,
    hide_index=True,
)
