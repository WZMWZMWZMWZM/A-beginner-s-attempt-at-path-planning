import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import contextily as ctx  # 需提前安装：pip install contextily mapclassify

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# 读取订单数据
file_path = r"C:\Users\13161\Desktop\附件一.xlsx"
try:
    orders = pd.read_excel(file_path)
    print(f"成功读取数据，共{len(orders)}条订单")
except Exception as e:
    print(f"读取文件失败：{e}")
    print("请检查附件一的路径是否正确")
    exit()

# 数据清洗
required_cols = ['经度', '纬度', '总重量（g）']
for col in required_cols:
    if col not in orders.columns:
        print(f"数据缺少必要列：{col}")
        exit()

valid_orders = orders.dropna(subset=required_cols)
valid_orders = valid_orders[
    (valid_orders['经度'].between(73, 135)) &
    (valid_orders['纬度'].between(18, 53)) &
    (valid_orders['总重量（g）'] > 0)
]

if len(valid_orders) == 0:
    print("没有有效订单数据可绘制")
    exit()

# 计算密度（保持分布均衡）
coords = np.vstack([valid_orders['经度'], valid_orders['纬度']])
kde = gaussian_kde(coords, bw_method=0.2)
density = kde(coords)
percentile = np.percentile(density, 95)
density = np.clip(density, 0, percentile)

# 点大小设置（适中且美观）
weights = valid_orders['总重量（g）']
max_size = 200
min_size = 15
sizes = min_size + (np.sqrt(weights / weights.max()) * (max_size - min_size))

# 创建图形
plt.figure(figsize=(16, 12))

# 绘制订单点（高饱和度+清晰边缘）
scatter = plt.scatter(
    valid_orders['经度'],
    valid_orders['纬度'],
    s=sizes,
    c=density,
    cmap='jet',     # 高饱和度配色
    alpha=0.7,      # 适中透明度，避免重叠混乱
    edgecolors='black',
    linewidth=0.5,  # 细边缘增强轮廓
    marker='o'      # 圆形点更美观
)

# 动态计算坐标范围（核心优化：根据数据跨度设置缓冲）
lon_min, lon_max = valid_orders['经度'].min(), valid_orders['经度'].max()
lat_min, lat_max = valid_orders['纬度'].min(), valid_orders['纬度'].max()

lon_range = lon_max - lon_min
lat_range = lat_max - lat_min

buffer_ratio = 0.3  # 缓冲占数据跨度的30%（可调整，建议0.2~0.5）
buffer_lon = lon_range * buffer_ratio
buffer_lat = lat_range * buffer_ratio

xmin, xmax = lon_min - buffer_lon, lon_max + buffer_lon
ymin, ymax = lat_min - buffer_lat, lat_max + buffer_lat

plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

# 添加OpenStreetMap背景（自动适配坐标范围）
ctx.add_basemap(
    plt.gca(),
    crs='EPSG:4326',  # 与经纬度数据匹配
    source=ctx.providers.OpenStreetMap.Mapnik,  # 开源街道地图
    alpha=0.8         # 地图透明度，让订单点更突出
)

# 图例与标注优化
cbar = plt.colorbar(scatter)
cbar.set_label('订单密集程度', fontsize=14, weight='bold')
plt.title('客户订单分布地图（颜色=密集程度，大小=订单重量）', fontsize=18, weight='bold', pad=20)
plt.xlabel('经度', fontsize=16, labelpad=10)
plt.ylabel('纬度', fontsize=16, labelpad=10)
plt.grid(linestyle='--', alpha=0.3, color='white')  # 白色网格在地图上更清晰
plt.gca().set_facecolor('#f8f9fa')  # 浅灰背景提升对比度

plt.tight_layout()
plt.show()