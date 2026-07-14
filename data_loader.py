"""
真实数据加载层。

数据来源（本次实际上传、已核对可用）：
  - data/road_risk.csv   : 1311 条路段的风险扩散分数（路段ID 为 0-1310）
  - data/roads.geojson   : OSM 路网，1313 个 feature，其中前 2 个是面要素(广场/步行区)，
                            其余 1311 个 LineString 与 road_risk.csv 的 路段ID 按顺序一一对应
                            （road_risk 路段ID = geojson 中第 (ID+2) 个 feature）
  - data/theft_locations.csv : 案发地点名称 + 次数（无经纬度，暂只能做排行，不能上地图）

仍缺失、未上传成功的文件：geocode_failed.csv、graph.npz、
模型训练 loss / 各模型指标对比 的原始导出文件 —— 因此"模型训练与评估"页仍使用
截图录入的演示数值，其余两页已全部改为基于以上真实文件计算。
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

DATA_DIR = Path(__file__).parent / "data"


def _road_geometries():
    with open(DATA_DIR / "roads.geojson", encoding="utf-8") as f:
        gj = json.load(f)
    lines = [feat for feat in gj["features"] if feat["geometry"]["type"] == "LineString"]
    return lines


def load_road_risk() -> pd.DataFrame:
    """road_risk.csv 关联 roads.geojson 的名称/坐标/道路等级，返回按风险降序的完整路段表"""
    risk = pd.read_csv(DATA_DIR / "road_risk.csv")
    risk.columns = [c.strip().replace("\ufeff", "") for c in risk.columns]

    lines = _road_geometries()

    names, lons, lats, highway = [], [], [], []
    for rid in risk["路段ID"]:
        feat = lines[int(rid)]
        props = feat["properties"]
        coords = feat["geometry"]["coordinates"]
        mid = coords[len(coords) // 2]
        nm = props.get("name")
        if not nm:
            nm = f"{props.get('highway','未命名')}路段(ID:{rid})"
        names.append(nm)
        lons.append(mid[0])
        lats.append(mid[1])
        highway.append(props.get("highway", "未知"))

    risk = risk.copy()
    risk["路段名称"] = names
    risk["lon"] = lons
    risk["lat"] = lats
    risk["道路等级"] = highway

    # z-score 标准化风险值，用于分级（与截图注释一致：风险值 = z-score 标准化后的发案风险预测值）
    scores = risk["风险扩散分数"].values
    z = (scores - scores.mean()) / scores.std()
    risk["风险值"] = (z - z.min()) / (z.max() - z.min()) * 4  # 映射到 0~4 区间，贴近截图数值感
    mu, sigma = risk["风险值"].mean(), risk["风险值"].std()

    def level(v):
        if v > mu + 2 * sigma:
            return "预警激增"
        if v > mu + sigma:
            return "高风险"
        if v > mu:
            return "中高风险"
        return "低风险"

    risk["等级"] = risk["风险值"].apply(level)
    return risk.sort_values("风险扩散分数", ascending=False).reset_index(drop=True)


_ACTION_TEMPLATES = {
    "预警激增": ("就近民警立即到场，调取周边视频，短蹲守≥30分钟；辅警定点值守", "指挥室直调 + 辅警2名", "≥30min",
                 "07:30-09:00 / 17:30-19:00 / 21:00-23:00"),
    "高风险": ("2人专项小组定点盘查，移动终端核查可疑人员，联动相邻路段同步巡防", "2人专项小组", "15-20min",
              "17:00-19:00 / 21:00-23:00"),
    "中高风险": ("机动组穿插巡逻，路段停留5-10分钟观察，重点时段随机抽查", "机动组", "5-10min",
                "07:30-09:00 / 17:30-19:00"),
    "低风险": ("纳入日常巡逻路线，无需专项加派警力", "常规巡逻", "5min", "按日常班次"),
}


def build_patrol_table(top_n: int = 18) -> pd.DataFrame:
    risk = load_road_risk().head(top_n).copy()
    risk = risk.reset_index(drop=True)
    risk.insert(0, "序号", risk.index + 1)
    tmpl = risk["等级"].map(_ACTION_TEMPLATES)
    risk["巡防动作"] = tmpl.apply(lambda t: t[0])
    risk["建议警力"] = tmpl.apply(lambda t: t[1])
    risk["预计用时"] = tmpl.apply(lambda t: t[2])
    risk["建议时段"] = tmpl.apply(lambda t: t[3])
    return risk[["序号", "路段名称", "风险值", "等级", "建议时段", "巡防动作", "建议警力", "预计用时", "lon", "lat"]]


def patrol_overview_real():
    risk = load_road_risk()
    return {
        "预警激增路段": int((risk["等级"] == "预警激增").sum()),
        "高/中高风险段": int(risk["等级"].isin(["高风险", "中高风险"]).sum()),
        "路段总数": int(len(risk)),
        "案发点所在分量路段占比": f"{risk['是否为案发点所在分量'].mean()*100:.1f}%",
    }


def load_theft_locations() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "theft_locations.csv")
    df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]
    return df.sort_values("count", ascending=False).reset_index(drop=True)


def load_facility_points() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "facility_points.csv")
    df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]
    return df


def _haversine_m(lon1, lat1, lon2, lat2):
    """向量化 haversine 距离（米）"""
    r = 6371000.0
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def _weak_level(nearest_dist_m: float) -> str:
    if nearest_dist_m > 3000:
        return "严重不足"
    if nearest_dist_m > 1500:
        return "不足"
    if nearest_dist_m > 800:
        return "临界"
    return "达标"


_MEASURE_TEMPLATES = {
    "严重不足": "增设高清探头×3，申请智能门禁",
    "不足": "补充移动探头，设置规范停车棚",
    "临界": "巡逻频次↑，连通路段同步提级",
    "达标": "维持现有配置，纳入常态巡查",
}


def infra_overlap_real(top_n: int = 10):
    """高风险路段关联 facility_points.csv 中的真实监控摄像头点位，计算最近摄像头距离与 500m/1500m 范围内点位数"""
    risk = load_road_risk().head(top_n).copy().reset_index(drop=True)
    fac = load_facility_points()
    cams = fac[fac["facility_type"] == "监控摄像头"]

    nearest_dist, nearby_count = [], []
    for lon, lat in zip(risk["lon"], risk["lat"]):
        d = _haversine_m(lon, lat, cams["lon"].values, cams["lat"].values)
        nearest_dist.append(float(d.min()))
        nearby_count.append(int((d <= 1500).sum()))

    risk["最近监控点距离(m)"] = [round(d) for d in nearest_dist]
    risk["1.5km范围内监控点数"] = nearby_count
    risk["薄弱度"] = [_weak_level(d) for d in nearest_dist]
    risk["建议措施"] = risk["薄弱度"].map(_MEASURE_TEMPLATES)
    risk["优先级"] = risk.apply(
        lambda r: "P1" if r["薄弱度"] == "严重不足" else ("P2" if r["薄弱度"] in ("不足", "临界") else "P3"), axis=1
    )
    return risk[["路段名称", "风险值", "最近监控点距离(m)", "1.5km范围内监控点数", "薄弱度",
                 "建议措施", "优先级", "lon", "lat"]]


def infra_coverage_rate(sample_n: int = 300) -> float:
    """辖区（抽样）高风险路段的监控覆盖率：1.5km 范围内存在摄像头视为覆盖"""
    risk = load_road_risk().head(sample_n).copy()
    fac = load_facility_points()
    cams = fac[fac["facility_type"] == "监控摄像头"]
    covered = 0
    for lon, lat in zip(risk["lon"], risk["lat"]):
        d = _haversine_m(lon, lat, cams["lon"].values, cams["lat"].values)
        if d.min() <= 1500:
            covered += 1
    return covered / len(risk) * 100


def infra_coverage_rate_all() -> float:
    """全路网（1311条路段）的监控覆盖率，作为高风险路段覆盖率的对比基准"""
    risk = load_road_risk()
    fac = load_facility_points()
    cams = fac[fac["facility_type"] == "监控摄像头"]
    covered = 0
    for lon, lat in zip(risk["lon"], risk["lat"]):
        d = _haversine_m(lon, lat, cams["lon"].values, cams["lat"].values)
        if d.min() <= 1500:
            covered += 1
    return covered / len(risk) * 100


# ---------------------------------------------------------------------------
# 路网图结构（graph_adj_sparse.npz + graph_meta.npz）
# bbox=[113.65, 34.62, 113.85, 34.77]，与 roads.geojson 范围一致
# 5341 个节点（真实路口/端点），11199 条边，drive 路网，弱连通分量数=1
# 注：这里的"节点"是路口级的 OSM graph（osmnx graph_from_bbox），
#     与 road_risk.csv 里"所属连通分量"（基于路段级风险传播模型分组，309个分量）
#     是两套不同粒度的图，不能直接互相替代；本模块用它做路口度数（度中心性）分析，
#     识别真实存在的"多路交汇口"，用于确定巡防定点岗的具体路口位置。
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_graph_nodes() -> pd.DataFrame:
    """返回路网节点表：node_id, lon, lat, degree（路口连接的路段数）, component"""
    adj = sparse.load_npz(DATA_DIR / "graph_adj_sparse.npz")
    meta = np.load(DATA_DIR / "graph_meta.npz", allow_pickle=True)
    node_order = meta["node_order"]
    node_lat = meta["node_lat"].item()
    node_lon = meta["node_lon"].item()
    ncm = meta["node_component_map"].item()

    degree = np.array(adj.sum(axis=1)).flatten()
    df = pd.DataFrame(
        {
            "node_id": node_order,
            "lon": [node_lon[n] for n in node_order],
            "lat": [node_lat[n] for n in node_order],
            "degree": degree.astype(int),
            "component": [ncm[n] for n in node_order],
        }
    )
    return df


def load_graph_edges() -> pd.DataFrame:
    """返回路网边表：source_lon, source_lat, target_lon, target_lat，用于在地图上画路网连线"""
    adj = sparse.load_npz(DATA_DIR / "graph_adj_sparse.npz").tocoo()
    nodes = load_graph_nodes()
    lons = nodes["lon"].values
    lats = nodes["lat"].values
    mask = adj.row < adj.col  # 无向边去重（有向图里 i->j 和 j->i 只画一次）
    rows, cols = adj.row[mask], adj.col[mask]
    return pd.DataFrame(
        {
            "source_lon": lons[rows], "source_lat": lats[rows],
            "target_lon": lons[cols], "target_lat": lats[cols],
        }
    )


def key_intersections(top_n: int = 15) -> pd.DataFrame:
    """全路网度数最高（多路交汇）的真实路口，度数=4/5表示三/四岔以上路口，适合作定点岗位置"""
    nodes = load_graph_nodes()
    return nodes.sort_values("degree", ascending=False).head(top_n).reset_index(drop=True)


def patrol_table_with_junction(top_n: int = 18, junction_radius_m: float = 250) -> pd.DataFrame:
    """在每日巡防推荐表基础上，为每条高风险路段匹配最近的真实路网交叉口（度数>=3视为路口），
    给出更具体的"定点位置"，而不仅是路段中点坐标"""
    table = build_patrol_table(top_n=top_n)
    nodes = load_graph_nodes()
    junctions = nodes[nodes["degree"] >= 3]

    nearest_names, nearest_deg, nearest_dist = [], [], []
    for lon, lat in zip(table["lon"], table["lat"]):
        d = _haversine_m(lon, lat, junctions["lon"].values, junctions["lat"].values)
        idx = int(np.argmin(d))
        nearest_dist.append(round(float(d[idx])))
        nearest_deg.append(int(junctions["degree"].values[idx]))
        nearest_names.append(f"路口(度{junctions['degree'].values[idx]})")

    table = table.copy()
    table["最近路口距离(m)"] = nearest_dist
    table["最近路口连接数"] = nearest_deg
    table["建议定点位置"] = [
        f"路段中点{d}m内的{n}路交汇口" if d <= junction_radius_m else "路段中点（附近无明显多路交汇口）"
        for d, n in zip(nearest_dist, nearest_deg)
    ]
    return table
