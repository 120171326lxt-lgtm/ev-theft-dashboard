# 电动自行车盗窃时空预测—预警平台（Streamlit）

## 运行方法
```bash
cd ev_theft_dashboard
pip install -r requirements.txt
streamlit run app.py
```
左侧导航栏切换三个页面：模型训练与评估 / 每日巡防推荐表 / 治安基础设施叠加分析。

## 数据接入
- `data/road_risk.csv` + `data/roads.geojson`：1311条路段的真实风险扩散分数与坐标，驱动"每日巡防推荐表""治安基础设施叠加分析"两页。
- `data/theft_locations.csv`：地理编码后的真实案发地点（含经纬度），已上图。
- `data/facility_points.csv`：真实监控摄像头/路灯/公共停车场点位，用于计算路段最近监控点距离。
- `data/graph_adj_sparse.npz` + `data/graph_meta.npz`：真实路网图结构（5341个路口节点，11199条边，drive路网，osmnx生成），用于：
  - 计算每个路口的连接度数（度中心性），识别全辖区"多路交汇口"（Top15关键路口）
  - 为每条高风险路段匹配最近的真实交叉口，生成具体的"建议定点位置"（而不只是路段中点坐标）
  - 在地图上绘制真实路网连线（可在侧边栏勾选"显示路网连线"）

演示数据（尚未接入真实来源）：
- "模型训练与评估"页的训练曲线与模型对比表。

所有数据处理集中在 `data_loader.py`，替换对应函数的读取逻辑即可接入新数据，页面代码无需改动。
