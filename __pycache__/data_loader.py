"""
真实数据加载层。

数据来源（2026-07-15 已按 export__1_.geojson 全量重新核验/生成）：
  - data/roads.geojson   : OSM 路网（车行道路，已剔除人行道/步行街/小路/台阶），
                            共 1085 个 LineString feature，无面要素混入；
                            road_risk.csv 的 路段ID(0-1084) 与 roads.geojson 的 feature 顺序 1:1 直接对应，
                            不再有 "+2 偏移"（旧版 1311 条数据里混了2个 Polygon 面要素，本次已去除）。
  - data/road_risk.csv   : 1085 条路段的风险扩散分数。风险分数通过与旧版 1311 条路段数据做
                            路段中点最近邻空间匹配继承而来（中位匹配距离=0m，1085条里1048条落在
                            50m内，19条超过200m——大概率是旧版里被排除的路段附近的替代路段，
                            此类分数继承存在一定误差，仅供参考，不是独立重新建模的结果）。
                            "所属连通分量"、"是否为案发点所在分量"均基于新路网重新计算：
                            路段级图共有 466 个弱连通分量（详见下方图结构说明），比旧版(309个分量)
                            碎片化程度更高——这是过滤掉人行道/小路后，部分路段仅靠这些被删掉的
                            连接道相连所致，如果后续要跑图卷积模型，建议先确认这种碎片化是否符合预期。
  - data/theft_locations.csv : 案发地点名称 + 次数 + 经纬度（未改动，203条真实地理编码点）。
  - data/facility_points.csv : ⚠️ 仍是 gen_facility.py 生成的随机模拟点位（320个，监控/路灯/停车场
                            三类各占一部分），不是真实设施数据。论文提到辖区真实视频监控点位约2293个，
                            与这320个模拟点数量级不符——目前没有拿到真实设施点位文件，这一项无法据实修正，
                            所有依赖 facility_points.csv 的分析（治安基础设施叠加分析页）仍建立在模拟数据上。
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

DATA_DIR = Path(__file__).parent / "data"


def _road_geometries():
    """新版 roads.geojson(1085条)本身已全部是LineString、不含Polygon，这层过滤仍保留作为防御性写法，
    不影响索引对应关系（现在 road_risk 路段ID 直接等于这里返回列表的下标，无偏移）。"""
    with open(DATA_DIR / "roads.geojson", encoding="utf-8") as f:
        gj = json.load(f)
    lines = [feat for feat in gj["features"] if feat["geometry"]["type"] == "LineString"]
    return lines


# OSM highway 标签中英对照，用于给没有真实路名的路段拼一个可读的中文类型名
_HIGHWAY_CN = {
    "motorway": "高速公路", "motorway_link": "高速匝道",
    "trunk": "干线道路", "trunk_link": "干线匝道",
    "primary": "主干道", "primary_link": "主干道匝道",
    "secondary": "次干道", "secondary_link": "次干道连接线",
    "tertiary": "三级道路", "tertiary_link": "三级道路连接线",
    "residential": "居住区道路", "living_street": "生活街道",
    "service": "支路", "unclassified": "普通道路",
    "footway": "人行步道", "pedestrian": "步行街",
    "cycleway": "非机动车道", "path": "小路",
    "track": "小道",
}


def _friendly_names(lines: list, needed_ids: list) -> dict:
    """为 needed_ids 里没有真实 name 的路段生成一个可读中文名：
    有真实名字的直接用；没有的，在全路网里找最近的一条"有名字的路"，
    标成"近XX路 类型"；如果附近300m内也没有任何有名道路，就退化成
    "类型+序号"（比如"支路12号"），比原来的 living_street路段(ID:51) 更像正常地名。"""
    # 收集全路网所有有真实 name 的路段中点，作为最近邻参照
    named_lons, named_lats, named_names = [], [], []
    for feat in lines:
        nm = feat["properties"].get("name")
        if nm:
            coords = feat["geometry"]["coordinates"]
            mid = coords[len(coords) // 2]
            named_lons.append(mid[0])
            named_lats.append(mid[1])
            named_names.append(nm)
    named_lons, named_lats = np.array(named_lons), np.array(named_lats)

    result = {}
    type_counter = {}
    for rid in needed_ids:
        feat = lines[int(rid)]
        props = feat["properties"]
        coords = feat["geometry"]["coordinates"]
        mid = coords[len(coords) // 2]
        nm = props.get("name")
        hwy = props.get("highway", "unclassified")
        hwy_cn = _HIGHWAY_CN.get(hwy, "道路")

        if nm:
            result[rid] = nm
            continue

        if len(named_lons):
            d = _haversine_m(mid[0], mid[1], named_lons, named_lats)
            idx = int(np.argmin(d))
            if d[idx] <= 300:
                result[rid] = f"近{named_names[idx]}{hwy_cn}"
                continue

        type_counter[hwy_cn] = type_counter.get(hwy_cn, 0) + 1
        result[rid] = f"{hwy_cn}{type_counter[hwy_cn]}号"
    return result


def load_road_risk() -> pd.DataFrame:
    """road_risk.csv 关联 roads.geojson 的名称/坐标/道路等级，返回按风险降序的完整路段表"""
    risk = pd.read_csv(DATA_DIR / "road_risk.csv")
    risk.columns = [c.strip().replace("\ufeff", "") for c in risk.columns]

    lines = _road_geometries()
    ids = risk["路段ID"].astype(int).tolist()
    friendly = _friendly_names(lines, ids)

    names, lons, lats, highway = [], [], [], []
    for rid in ids:
        feat = lines[rid]
        props = feat["properties"]
        coords = feat["geometry"]["coordinates"]
        mid = coords[len(coords) // 2]
        names.append(friendly[rid])
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


@lru_cache(maxsize=1)
def _network_camera_distance_baseline() -> np.ndarray:
    """全路网（1085条路段）每条路段到最近摄像头的距离，用作薄弱度分档的基准分布。
    用分位数而不是写死的绝对数值（800/1500/3000m），是因为高风险路段本身可能
    整体偏远、系统性地离监控点较远——如果用固定阈值切，容易出现"全部路段落在
    同一档"、看起来像写死数据的问题；用相对分位数能保证薄弱度有区分度。"""
    risk = load_road_risk()
    fac = load_facility_points()
    cams = fac[fac["facility_type"] == "监控摄像头"]
    dists = np.empty(len(risk))
    for i, (lon, lat) in enumerate(zip(risk["lon"], risk["lat"])):
        d = _haversine_m(lon, lat, cams["lon"].values, cams["lat"].values)
        dists[i] = d.min()
    return dists


def infra_overlap_real(top_n: int = 10):
    """高风险路段关联 facility_points.csv 中的真实监控摄像头点位：
    - 最近摄像头距离 / 1.5km 范围内点位数：直接计算，无改动
    - 薄弱度：按全路网距离分布的25/50/75分位数动态分档（而非固定绝对阈值）
    - 优先级：综合"风险值排名"与"监控薄弱程度排名"两个维度打分后三等分，
      避免因为这批路段风险值/薄弱度普遍偏高，导致优先级全员挤在 P1
    - 建议措施：把该路段的实际距离、探头数嵌入文案，逐行动态生成，不再是一句写死的话
    """
    risk = load_road_risk().head(top_n).copy().reset_index(drop=True)
    fac = load_facility_points()
    cams = fac[fac["facility_type"] == "监控摄像头"]

    baseline = _network_camera_distance_baseline()
    q25, q50, q75 = np.percentile(baseline, [25, 50, 75])

    nearest_dist, nearby_count = [], []
    for lon, lat in zip(risk["lon"], risk["lat"]):
        d = _haversine_m(lon, lat, cams["lon"].values, cams["lat"].values)
        nearest_dist.append(float(d.min()))
        nearby_count.append(int((d <= 1500).sum()))

    risk["最近监控点距离(m)"] = [round(d) for d in nearest_dist]
    risk["1.5km范围内监控点数"] = nearby_count

    def weak_level(d):
        if d >= q75:
            return "严重不足"
        if d >= q50:
            return "不足"
        if d >= q25:
            return "临界"
        return "达标"

    risk["薄弱度"] = [weak_level(d) for d in nearest_dist]

    # 优先级：风险值排名 + 距离远近排名，两个排名相加后三等分
    risk_rank = risk["风险值"].rank(ascending=False, method="min").values
    dist_rank = pd.Series(nearest_dist).rank(ascending=False, method="min").values
    combined = risk_rank + dist_rank
    t1, t2 = np.percentile(combined, [33.3, 66.7])

    def priority(c):
        if c <= t1:
            return "P1"
        if c <= t2:
            return "P2"
        return "P3"

    risk["优先级"] = [priority(c) for c in combined]

    def measure(row):
        d, cnt = row["最近监控点距离(m)"], row["1.5km范围内监控点数"]
        if row["优先级"] == "P1":
            return f"最近监控点{d}m外，1.5km内仅{cnt}个探头，建议优先增设高清探头并申请智能门禁"
        if row["优先级"] == "P2":
            return f"最近监控点{d}m，1.5km内{cnt}个探头，建议补充移动探头或加装规范停车棚位"
        return f"最近监控点{d}m，1.5km内{cnt}个探头，建议提升巡逻频次、纳入联动巡查即可"

    risk["建议措施"] = risk.apply(measure, axis=1)

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
    """全路网（1085条路段）的监控覆盖率，作为高风险路段覆盖率的对比基准"""
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
# 路网图结构（graph_adj_sparse.npz + graph_meta.npz）—— 2026-07-15 基于 export__1_.geojson 重新生成
# bbox=[113.6528, 34.6201, 113.8520, 34.7660]，与 roads.geojson(1085条) 范围一致
#
# 这里存的是"端点级"路网图：节点=路段端点(共1538个)，边=路段本身(共1085条，每条路段就是一条边)。
# 用端点坐标去重（保留6位小数，约0.1m精度）统计后，degree>=2 的端点即为真实交叉口，共 517 个；
# 也就是说 1538 个端点里，517 个是真正的路口，其余 1021 个是断头端点(dead-end，比如施工路段/
# 死路/数据被裁切边界处)。
#
# 另外，论文/模型实际使用的是"路段级"图（以路段为图节点，N=1085，邻接矩阵 1085×1085）：
# 两条路段如果共享同一个端点，就在这个路段级图里连一条边——按此统计，边数为 756 条，
# 弱连通分量数为 466（即整个路网并不是一个连通整体，而是碎成了466块，最大的一块只包含120条路段）。
# 这与旧版基于1311条路段(含步行道)算出的309个分量不同——本次统计口径是纯车行道路网，
# 步行道/小路被过滤掉后连通性明显变差，如果要用于图卷积模型训练，建议先确认这466个分量、
# 尤其是大量仅有1-2条路段的微小分量，是否会影响模型学习效果。
#
# 本模块继续用"端点级"图做路口度数（度中心性）分析，识别真实存在的"多路交汇口"，
# 用于确定巡防定点岗的具体路口位置；"路段级"图的连通分量数据已经写入 road_risk.csv 的
# "所属连通分量"列，两套图不能直接互相替代。
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
