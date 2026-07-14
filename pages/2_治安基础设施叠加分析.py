import datetime

import pydeck as pdk
import streamlit as st

from common import inject_base_css, topbar, metric_card, risk_badge
import data_loader as dl

st.set_page_config(page_title="治安基础设施叠加分析", layout="wide", page_icon="🚲")
inject_base_css()
topbar("治安基础设施叠加分析", f"明湖派出所辖区&nbsp;&nbsp;|&nbsp;&nbsp;{datetime.date.today()}")

with st.sidebar:
    st.markdown("#### 图层设置")
    st.selectbox("风险时段", ["全天综合（24h）", "白班（07:00-19:00）", "夜班（19:00-07:00）"])
    st.markdown("#### 叠加要素")
    show_camera = st.checkbox("视频监控点位", value=True)
    show_risk = st.checkbox("高风险路段", value=True)
    show_light = st.checkbox("路灯", value=False)
    show_parking = st.checkbox("公共停车场", value=False)
    show_theft = st.checkbox("历史案发点", value=True)
    st.selectbox("薄弱度阈值", ["最近监控点 > 1.5km", "最近监控点 > 800m"])
    top_n = st.slider("展示高风险路段数", 5, 30, 11)
    st.button("更新分析", type="primary", use_container_width=True)

    table = dl.infra_overlap_real(top_n=top_n)
    cov_risk = dl.infra_coverage_rate()
    cov_all = dl.infra_coverage_rate_all()
    st.markdown("#### 分析结果")
    st.markdown(f"**高风险×薄弱叠加路段**\n\n### {int((table['薄弱度'] != '达标').sum())} 条")
    st.caption("需优先补强")
    st.markdown(f"**监控覆盖率（高风险路段）**\n\n### {cov_risk:.1f}%")
    st.caption(f"全路网基准 {cov_all:.1f}%")

st.markdown("### 高风险路段 × 治安基础设施薄弱区域 空间重合分析")

c1, c2, c3 = st.columns(3)
with c1:
    metric_card(str(len(table)), "高风险路段")
with c2:
    metric_card(f"{cov_risk:.1f}%", "监控覆盖率（高风险）",
                f"全路网基准 {cov_all:.1f}%", sub_direction=("down" if cov_risk < cov_all else "up"))
with c3:
    metric_card(str(int((table["薄弱度"] == "严重不足").sum())), "严重薄弱路段", value_color="red")

st.write("")
st.markdown("**高风险路段 × 治安基础设施 空间分布**")

map_df = table.dropna(subset=["lon", "lat"]).copy()
prio_color = {"P1": [229, 52, 43], "P2": [230, 126, 34], "P3": [46, 158, 91]}
map_df["risk_color"] = map_df["优先级"].map(prio_color)

fac = dl.load_facility_points()
theft = dl.load_theft_locations()

layers = []
if show_risk:
    layers.append(
        pdk.Layer(
            "ScatterplotLayer", data=map_df, get_position="[lon, lat]",
            get_fill_color="risk_color", get_radius=80, pickable=True,
        )
    )
if show_camera:
    cams = fac[fac["facility_type"] == "监控摄像头"]
    layers.append(
        pdk.Layer(
            "ScatterplotLayer", data=cams, get_position="[lon, lat]",
            get_fill_color="[70, 130, 220, 160]", get_radius=90, pickable=True,
        )
    )
if show_light:
    lights = fac[fac["facility_type"] == "路灯"]
    layers.append(
        pdk.Layer(
            "ScatterplotLayer", data=lights, get_position="[lon, lat]",
            get_fill_color="[240, 190, 40, 160]", get_radius=60, pickable=True,
        )
    )
if show_parking:
    parking = fac[fac["facility_type"] == "公共停车场"]
    layers.append(
        pdk.Layer(
            "ScatterplotLayer", data=parking, get_position="[lon, lat]",
            get_fill_color="[140, 90, 200, 160]", get_radius=70, pickable=True,
        )
    )
if show_theft:
    layers.append(
        pdk.Layer(
            "ScatterplotLayer", data=theft, get_position="[lon, lat]",
            get_fill_color="[20, 20, 20, 130]", get_radius=40, pickable=True,
        )
    )

view_state = pdk.ViewState(
    latitude=float(map_df["lat"].mean()) if len(map_df) else 34.72,
    longitude=float(map_df["lon"].mean()) if len(map_df) else 113.73,
    zoom=12,
)
st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view_state,
                          tooltip={"text": "{路段名称}{name}\n风险值: {风险值}\n薄弱度: {薄弱度}"}))
st.caption("🔴🟠🟢 高风险路段（按优先级）　🔵 监控摄像头　🟡 路灯　🟣 公共停车场　⚫ 历史案发点")

st.write("")
st.markdown("**重点补强路段清单（高风险 × 监控薄弱 双重叠加）**")

disp = table.copy()
disp["优先级"] = disp["优先级"].apply(risk_badge)
disp["薄弱度"] = disp["薄弱度"].apply(risk_badge)
disp["风险值"] = disp["风险值"].map(lambda v: f"{v:.2f}")
disp = disp.drop(columns=["lon", "lat"])

st.markdown(
    disp.to_html(escape=False, index=False, classes="section-table", border=0),
    unsafe_allow_html=True,
)

st.write("")
st.markdown("**辖区案发地点排行**")
theft_disp = theft.sort_values("count", ascending=False).head(15).rename(
    columns={"name": "地点", "count": "案发次数", "match_level": "定位精度"}
)
st.dataframe(theft_disp[["地点", "案发次数", "定位精度"]], use_container_width=True, hide_index=True)
