import pandas as pd
import numpy as np

np.random.seed(2026)
point_count = 320
# 匹配片区经纬度区间
lon_min = 113.62
lon_max = 114.25
lat_min = 34.56
lat_max = 34.82

lon = np.random.uniform(lon_min, lon_max, point_count)
lat = np.random.uniform(lat_min, lat_max, point_count)
fac_type = np.random.choice(["监控摄像头", "路灯", "公共停车场"], size=point_count)
names = [f"设施_{i}" for i in range(point_count)]

df = pd.DataFrame({
    "lon": lon,
    "lat": lat,
    "name": names,
    "facility_type": fac_type
})

df.to_csv("data/facility_points.csv", index=False, encoding="utf-8-sig")
print("设施点位文件生成完成，存至data文件夹")
