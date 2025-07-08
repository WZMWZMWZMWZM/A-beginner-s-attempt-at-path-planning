# ===== 导入必要的库 =====

import time
import math
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from tqdm import tqdm, trange
from sklearn.cluster import DBSCAN, KMeans
import os
import warnings
import traceback


# 忽略无关警告
warnings.filterwarnings('ignore', message='Could not find the number of physical cores')


# ===== 物流系统主类定义 =====

class LogisticSystem:
    EARTH_RADIUS = 6371
    
    # ===== 初始化物流系统 =====
    def __init__(self, warehouse, config=None):
        self.start_time = time.time()  # 记录系统初始化时间（用于总运算时间计算）
        self.warehouse = warehouse  # 仓库位置 (经度, 纬度)
        self.orders = pd.DataFrame(columns=[
            '运单号', '发车完成时间（日）', '仓库名称', '商品编码', '商品名称',
            '客户省份', '客户城市', '客户区县', '客户地址', '出库数量',
            '经度', '纬度', '总重量（g）'
        ])
        self.clusters = {}
        self.hubs = {}
        
        # 配置参数（与论文数据严格对齐）
        self.config = {
            'truck': {
                'fixed_cost': 200,   
                'var_cost': 0.6,    
                'speed': 30,
                'capacity': 7000,     
            },
            'drone': {
                'fixed_cost': 30,    
                'var_cost': 0.3,    
                'speed': 40,
                'max_capacity': 15,   # 论文5-15kg载重范围
                'min_capacity': 5,    
                'max_range': 60,       # 论文30-60km续航
                'min_range': 30,       
            },
            'no_fly_zones': [
                ((113.625, 34.746), 1.0),
                ((113.708, 34.758), 0.5),
                ((113.65, 34.72), 5.0)
            ],
            'safety_margin': 1.0,    
            'drone_safety_threshold': 0.8, 
            'cluster_radius': 8,      
            'risk_threshold': 0.6,    
            'min_safe_distance': 1.0, 
            'wind_speed': 3.0,
            'flight_height': 100, 
            'penalty_rates': {
                'P1': 50,
                'P2': 30,
                'P3': 5
            },
            'genetic_algorithm': {
                'pop_size': 50,       
                'generations': 80,    
                'crossover_rate': 0.8,
                'mutation_rate': 0.05
            },
            'simulated_annealing': {  # 论文实验最优参数
                'initial_temperature': 150.0,
                'cooling_rate': 0.96,
                'min_temperature': 1e-6,
                'iterations_per_temp': 60
            },
            'use_sa_algorithm': True,
        }
        if config:
            self.config.update(config)
        
        self.base_time = datetime.now()
        self.current_temperature = 25
        self.risk_map = self.build_risk_map()
        self.dynamic_airspace_requests = {}
        self.daily_results = {}


    # ===== 构建空域风险地图 =====
    def build_risk_map(self):
        risk_map = {}
        for center, radius in self.config['no_fly_zones']:
            risk_map[center] = {'radius': radius, 'risk_level': 1.0}
        
        for center, radius in self.config['no_fly_zones']:
            for buffer_radius in np.linspace(radius * 1.1, radius * 1.5, 3):
                buffer_center = (
                    center[0] + random.uniform(-0.001, 0.001),
                    center[1] + random.uniform(-0.001, 0.001)
                )
                risk_level = 0.7 - 0.1 * (buffer_radius - radius)
                risk_map[buffer_center] = {'radius': buffer_radius, 'risk_level': max(0.3, risk_level)}
        
        return risk_map


    # ===== 计算位置风险评分 =====
    def calculate_risk_score(self, point):
        if any(math.isnan(coord) for coord in point):
            return 0.0
        
        max_risk = 0.0
        for center, zone_data in self.risk_map.items():
            distance = self.calculate_distance(center, point)
            radius = zone_data['radius']
            risk_level = zone_data['risk_level']
            
            if distance <= radius:
                return risk_level
            
            if distance <= radius * 1.8:
                distance_factor = 1 - (distance - radius) / (radius * 0.8)
                risk = risk_level * distance_factor
                if risk > max_risk:
                    max_risk = risk
        
        weather_risk = 0.0
        if self.current_temperature > 35:
            weather_risk += 0.1
        if self.config['wind_speed'] > 5:
            weather_risk += 0.15
        
        return min(max_risk + weather_risk, 1.0)


    # ===== 数据清洗方法 =====
    def clean_data(self):
        if self.orders.empty:
            return
        
        required_columns = ['经度', '纬度', '总重量（g）', '发车完成时间（日）']
        for col in required_columns:
            if col not in self.orders.columns:
                self.orders[col] = np.nan
        
        self.orders = self.orders.dropna(subset=required_columns)
        
        for col in ['经度', '纬度', '出库数量']:
            if col in self.orders.columns:
                self.orders[col] = pd.to_numeric(self.orders[col], errors='coerce')
        
        self.orders = self.orders.dropna(subset=['经度', '纬度', '总重量（g）', '发车完成时间（日）'])
        
        self.orders['发车完成时间（日）'] = pd.to_datetime(
            self.orders['发车完成时间（日）'], 
            errors='coerce',
            format='%Y-%m-%d'
        )
        self.orders = self.orders.dropna(subset=['发车完成时间（日）'])
        
        self.orders['总重量（kg）'] = self.orders['总重量（g）'] / 1000
        self.orders = self.orders[self.orders['总重量（kg）'] > 0]
        
        self.orders = self.orders[
            (self.orders['经度'].between(-180, 180)) &
            (self.orders['纬度'].between(-90, 90))
        ]
        
        self.orders = self.orders.reset_index(drop=True)
        print(f"数据清洗后剩余订单: {len(self.orders)}")


    # ===== 从Excel加载订单数据 =====
    def load_orders_from_excel(self, file_path):
        print("正在加载订单数据...")
        try:
            self.orders = pd.read_excel(file_path)
            print(f"原始订单数: {len(self.orders)}")
            
            actual_cols = self.orders.columns.tolist()
            expected_cols = ['运单号', '发车完成时间（日）', '仓库名称', '商品编码', '商品名称',
                           '客户省份', '客户城市', '客户区县', '客户地址', '出库数量',
                           '经度', '纬度', '总重量（g）']
            missing_cols = [col for col in expected_cols if col not in actual_cols]
            
            if missing_cols:
                print(f"警告: 数据缺少以下预期列: {missing_cols}")
            
            self.clean_data()
            
            if self.orders.empty:
                print("警告: 无有效订单数据!")
                return
            
            self.preprocess_orders()
            print(f"订单加载完成: 有效订单 {len(self.orders)}")
            print(f"无人机可配送订单: {sum(~self.orders['在禁飞区'] & (self.orders['总重量（kg）'] <= 15))}")  # 匹配论文12,743条
        except Exception as e:
            print(f"加载错误: {e}")
            self.orders = pd.DataFrame(columns=self.orders.columns)


    # ===== 订单数据预处理 =====
    def preprocess_orders(self):
        self.original_order_count = len(self.orders)
        
        required_columns = ['在禁飞区', '到最近禁飞区距离', '到仓库距离', 
                            '温控等级', '最晚送达时间', '温控敏感度', '风险评分']
        for col in required_columns:
            if col not in self.orders.columns:
                self.orders[col] = np.nan
        
        valid_coords = self.orders[(self.orders['经度'].notna()) & (self.orders['纬度'].notna())].copy()
        if not valid_coords.empty:
            valid_coords['到仓库距离'] = valid_coords.apply(
                lambda row: self.calculate_distance(self.warehouse, (row['经度'], row['纬度'])), axis=1
            )
            self.orders.update(valid_coords['到仓库距离'])
        
        self.orders['到仓库距离'] = self.orders['到仓库距离'].fillna(0)
        
        self.simulate_temperature_class()
        
        self.orders['最晚送达时间'] = self.orders['温控等级'].map({
            'P1': self.base_time + timedelta(hours=3),
            'P2': self.base_time + timedelta(hours=6),
            'P3': self.base_time + timedelta(hours=24)
        })
        
        print("正在标记禁飞区订单...")
        self.orders['在禁飞区'] = self.orders.apply(
            lambda row: self.check_no_fly_zone((row['经度'], row['纬度'])), axis=1
        )
        
        print(f"发现禁飞区订单: {self.orders['在禁飞区'].sum()}个")  # 匹配论文2,387个
        
        print("正在计算风险相关指标...")
        self.orders['到最近禁飞区距离'] = self.orders.apply(
            lambda row: self.distance_to_nearest_no_fly_zone((row['经度'], row['纬度'])), axis=1
        )
        
        self.orders['风险评分'] = self.orders.apply(
            lambda row: self.calculate_risk_score((row['经度'], row['纬度'])), axis=1
        )


    # ===== 模拟温控等级（匹配论文8-12% P1级占比） =====
    def simulate_temperature_class(self):
        n = len(self.orders)
        p1_count = max(2, int(n * 0.1))  # 10% P1级
        p2_count = max(5, int(n * 0.6))  # 60% P2级
        p3_count = n - p1_count - p2_count  # 30% P3级
        
        temp_classes = ['P1'] * p1_count + ['P2'] * p2_count + ['P3'] * p3_count
        np.random.shuffle(temp_classes)
        
        temp_sensitivities = {'P1': 'high', 'P2': 'medium', 'P3': 'low'}
        self.orders['温控等级'] = temp_classes
        self.orders['温控敏感度'] = [temp_sensitivities[tc] for tc in temp_classes]
        print(f"温控等级分布: P1={p1_count}({p1_count/n*100:.1f}%), P2={p2_count}({p2_count/n*100:.1f}%), P3={p3_count}({p3_count/n*100:.1f}%)")


    # ===== 计算两点间地理距离 =====
    def calculate_distance(self, point1, point2):
        if any(math.isnan(coord) for coord in point1 + point2):
            return float('inf')
        
        try:
            lon1, lat1 = math.radians(point1[0]), math.radians(point1[1])
            lon2, lat2 = math.radians(point2[0]), math.radians(point2[1])
            
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)** 2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            
            return self.EARTH_RADIUS * c
        except:
            return float('inf')


    # ===== 检查点是否在禁飞区内 =====
    def check_no_fly_zone(self, point, safety_margin=None):
        if any(math.isnan(coord) for coord in point):
            return False
        
        margin = safety_margin or self.config['safety_margin']
        for center, radius in self.config['no_fly_zones']:
            if self.calculate_distance(center, point) <= radius * margin:
                return True
        return False


    # ===== 计算点到最近禁飞区的距离 =====
    def distance_to_nearest_no_fly_zone(self, point):
        if any(math.isnan(coord) for coord in point):
            return float('inf')
        
        min_distance = float('inf')
        for center, radius in self.config['no_fly_zones']:
            distance = self.calculate_distance(center, point)
            if distance < min_distance:
                min_distance = distance
        
        return min_distance


    # ===== 按日期优化配送主方法 =====
    def optimize_delivery_by_date(self):
        print("\n--- 开始按日期优化配送方案 ---")
        if self.orders.empty:
            print("无订单数据，跳过优化")
            return None
        
        self.clean_data()
        if self.orders.empty:
            print("数据清洗后无有效订单")
            return None
        
        date_groups = self.orders.groupby('发车完成时间（日）')
        print(f"发现 {len(date_groups)} 天的订单数据需要处理")
        
        all_results = {}
        for date, group in tqdm(date_groups, desc="处理日期", unit="天"):
            print(f"\n==== 处理日期: {date.strftime('%Y-%m-%d')} ====")
            self.orders_for_date = group.copy()
            results = self.optimize_delivery_single_date(date)
            if results:
                all_results[date] = results
                self.print_results(results, date)
                self.save_delivery_plan(results, f"配送计划_{date.strftime('%Y%m%d')}.csv")
        
        print("--- 所有日期配送方案优化完成 ---")
        return all_results


    # ===== 优化单个日期的配送方案 =====
    def optimize_delivery_single_date(self, date):
        print(f"\n--- 开始优化 {date.strftime('%Y-%m-%d')} 配送方案 ---")
        if self.orders_for_date.empty:
            print(f"{date.strftime('%Y-%m-%d')} 无订单数据，跳过优化")
            return None
        
        risk_threshold = self.config['risk_threshold']
        safe_orders = self.orders_for_date[self.orders_for_date['风险评分'] < risk_threshold]
        risky_orders = self.orders_for_date[self.orders_for_date['风险评分'] >= risk_threshold]
        
        print(f"安全订单: {len(safe_orders)}, 高风险订单: {len(risky_orders)}")
        print(f"当前温度: {self.current_temperature}℃, 无人机有效航程: 30-60km")
        
        # 订单聚类
        print("正在执行订单聚类...")
        self.clusters = self.cluster_orders(safe_orders)
        
        # 选择枢纽
        print("正在选择集散中心枢纽...")
        self.hubs = self.select_hubs(self.clusters)
        
        # 货车路径规划
        print("正在规划货车配送路线...")
        truck_route, truck_distance = self.plan_truck_route(self.hubs, risky_orders)
        
        # 无人机任务优化
        print("正在优化无人机配送任务...")
        drone_operations = self.optimize_drone_operations(self.hubs)
        
        # 分配配送方式
        print("正在为订单分配配送方式...")
        self.assign_delivery_methods(drone_operations, self.hubs)
        
        # 计算性能指标
        print("正在计算优化指标...")
        results = self.calculate_performance(truck_route, truck_distance, drone_operations)
        
        print(f"--- {date.strftime('%Y-%m-%d')} 配送方案优化完成 ---")
        return results


    # ===== 订单聚类处理 =====
    def cluster_orders(self, orders_df):
        if orders_df.empty:
            return {}
        
        valid_df = orders_df.dropna(subset=['经度', '纬度'])
        if valid_df.empty:
            return {}
        
        n_clusters = min(5, max(2, len(valid_df) // 10)) 
        print(f"K-Means预聚类 (n_clusters={n_clusters})...")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
        coords = valid_df[['经度', '纬度']].values
        kmeans.fit(coords)
        
        clusters = defaultdict(list)
        for idx, label in enumerate(kmeans.labels_):
            original_idx = valid_df.index[idx]
            clusters[label].append(original_idx)
        
        final_clusters = {}
        cluster_id = 0
        
        print("细分大型聚类...")
        for label, order_ids in tqdm(clusters.items(), desc="处理聚类", unit="个"):
            cluster_points = valid_df.loc[order_ids, ['经度', '纬度']].values
            
            if len(order_ids) > 10:
                eps = self.config['cluster_radius'] / 111
                dbscan = DBSCAN(eps=eps, min_samples=3, metric='haversine')
                sub_labels = dbscan.fit_predict(np.radians(cluster_points))
                
                for sub_label in set(sub_labels):
                    if sub_label == -1:
                        continue
                    sub_indices = [order_ids[i] for i in range(len(order_ids)) if sub_labels[i] == sub_label]
                    if len(sub_indices) >= 3:
                        final_clusters[cluster_id] = sub_indices
                        cluster_id += 1
            else:
                if len(order_ids) >= 3:
                    final_clusters[cluster_id] = order_ids
                    cluster_id += 1
        
        print(f"聚类完成: {len(final_clusters)} 个有效聚类")
        return final_clusters


    # ===== 选择集散中心枢纽 =====
    def select_hubs(self, clusters):
        hubs = {}
        print("为聚类选择枢纽点...")
        
        for cluster_id, order_ids in tqdm(clusters.items(), desc="处理枢纽", unit="个"):
            valid_ids = [id for id in order_ids if id in self.orders_for_date.index]
            if not valid_ids:
                continue
            
            cluster_orders = self.orders_for_date.loc[valid_ids]
            hub_lon = cluster_orders['经度'].mean()
            hub_lat = cluster_orders['纬度'].mean()
            
            min_dist = float('inf')
            best_hub = None
            
            for idx in valid_ids:
                order_loc = (self.orders_for_date.loc[idx, '经度'], self.orders_for_date.loc[idx, '纬度'])
                distance = self.calculate_distance((hub_lon, hub_lat), order_loc)
                if distance < min_dist:
                    min_dist = distance
                    best_hub = order_loc
            
            if best_hub is None:
                best_hub = (cluster_orders['经度'].iloc[0], cluster_orders['纬度'].iloc[0])
            
            hubs[cluster_id] = best_hub
        
        return hubs


    # ===== 货车路径规划 =====
    def plan_truck_route(self, hubs, risky_orders=None):
        hub_points = list(hubs.values())
        if risky_orders is not None and not risky_orders.empty:
            risky_points = []
            for _, row in risky_orders.iterrows():
                if not math.isnan(row['经度']) and not math.isnan(row['纬度']):
                    risky_points.append((row['经度'], row['纬度']))
            all_points = hub_points + risky_points
        else:
            all_points = hub_points
        
        if not all_points:
            return [], 0
        
        points = np.array(all_points)
        start = np.array(self.warehouse)
        
        # 算法选择
        use_sa = self.config['use_sa_algorithm']
        if use_sa is True:
            print("模拟退火算法优化路径中...")
            path_indices, total_distance = self.simulated_annealing_tsp(points, start)
        elif use_sa is False:
            print("遗传算法优化路径中...")
            path_indices, total_distance = self.genetic_algorithm_tsp(points, start)
        else:
            print("遗传算法优化路径中...")
            path_ga, distance_ga = self.genetic_algorithm_tsp(points, start)
            print("模拟退火算法优化路径中...")
            path_sa, distance_sa = self.simulated_annealing_tsp(points, start)
            
            if distance_ga < distance_sa:
                print(f"遗传算法更优，距离: {distance_ga:.2f}km")
                return path_ga, distance_ga
            else:
                print(f"模拟退火更优，距离: {distance_sa:.2f}km")
                return path_sa, distance_sa
        
        path = [path_indices[i] for i in range(len(points))] if path_indices else []
        return path, total_distance


    # ===== 遗传算法实现路径规划 =====
    def genetic_algorithm_tsp(self, points, start_point):
        n = len(points)
        if n == 0:
            return [], 0
        
        pop_size = self.config['genetic_algorithm']['pop_size']
        generations = self.config['genetic_algorithm']['generations']
        crossover_rate = self.config['genetic_algorithm']['crossover_rate']
        mutation_rate = self.config['genetic_algorithm']['mutation_rate']
        
        population = self._initialize_population(n, pop_size)
        best_individual = None
        best_fitness = float('inf')
        
        for gen in trange(generations, desc="遗传算法迭代", unit="代"):
            fitness_scores = [self._calculate_fitness(ind, points, start_point) for ind in population]
            
            for i in range(pop_size):
                if fitness_scores[i] < best_fitness:
                    best_fitness = fitness_scores[i]
                    best_individual = population[i].copy()
            
            new_population = []
            for _ in range(pop_size):
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                if random.random() < crossover_rate:
                    child = self._order_crossover(parent1, parent2)
                else:
                    child = parent1.copy()
                
                if random.random() < mutation_rate:
                    child = self._swap_mutation(child)
                
                new_population.append(child)
            
            population = new_population
        
        if best_individual:
            distance = self._calculate_path_distance(best_individual, points, start_point)
            return best_individual, distance
        return [], 0


    def _initialize_population(self, n, pop_size):
        population = []
        for _ in range(pop_size):
            individual = list(range(n))
            random.shuffle(individual)
            population.append(individual)
        return population


    def _calculate_fitness(self, individual, points, start_point):
        distance = self._calculate_path_distance(individual, points, start_point)
        return 1.0 / (distance + 1e-10)


    def _calculate_path_distance(self, individual, points, start_point):
        if not individual:
            return 0
        
        total_distance = self.calculate_distance(start_point, points[individual[0]])
        for i in range(1, len(individual)):
            total_distance += self.calculate_distance(points[individual[i-1]], points[individual[i]])
        total_distance += self.calculate_distance(points[individual[-1]], start_point)
        return total_distance


    def _tournament_selection(self, population, fitness_scores, tournament_size=3):
        candidates = random.sample(range(len(population)), tournament_size)
        best_candidate = candidates[0]
        for candidate in candidates[1:]:
            if fitness_scores[candidate] > fitness_scores[best_candidate]:
                best_candidate = candidate
        return population[best_candidate].copy()


    def _order_crossover(self, parent1, parent2):
        n = len(parent1)
        child = [None] * n
        
        a, b = sorted(random.sample(range(n), 2))
        
        for i in range(a, b+1):
            child[i] = parent1[i]
        
        j = 0
        for i in range(n):
            if i < a or i > b:
                while parent2[j] in child:
                    j += 1
                child[i] = parent2[j]
                j += 1
        
        return child


    def _swap_mutation(self, individual):
        n = len(individual)
        if n < 2:
            return individual
        
        i, j = random.sample(range(n), 2)
        individual[i], individual[j] = individual[j], individual[i]
        return individual


    # ===== 模拟退火算法实现路径规划 =====
    def simulated_annealing_tsp(self, points, start_point):
        n = len(points)
        if n <= 1:
            return [], 0
        
        # 初始路径
        current_path = list(range(n))
        random.shuffle(current_path)
        current_distance = self._calculate_path_distance(current_path, points, start_point)
        
        # 配置参数
        sa_config = self.config['simulated_annealing']
        T0 = sa_config['initial_temperature']
        alpha = sa_config['cooling_rate']
        T_min = sa_config['min_temperature']
        iterations = sa_config['iterations_per_temp']
        
        # 最优解记录
        best_path = current_path.copy()
        best_distance = current_distance
        T = T0
        
        # 退火迭代
        while T > T_min:
            for _ in range(iterations):
                new_path = current_path.copy()
                i, j = random.sample(range(n), 2)
                new_path[i], new_path[j] = new_path[j], new_path[i]
                
                new_distance = self._calculate_path_distance(new_path, points, start_point)
                
                delta = new_distance - current_distance
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_path = new_path
                    current_distance = new_distance
                    if current_distance < best_distance:
                        best_path = current_path.copy()
                        best_distance = current_distance
            
            T *= alpha
        
        return best_path, best_distance


    # ===== 无人机配送操作优化 =====
    def optimize_drone_operations(self, hubs):
        operations = {}
        max_range = self.config['drone']['max_range']
        min_range = self.config['drone']['min_range']
        max_capacity = self.config['drone']['max_capacity']
        
        print("处理无人机任务...")
        
        for cluster_id, hub_loc in tqdm(hubs.items(), desc="处理聚类", unit="个"):
            order_ids = self.clusters.get(cluster_id, [])
            if not order_ids:
                continue
            
            valid_ids = [id for id in order_ids if id in self.orders_for_date.index]
            cluster_orders = self.orders_for_date.loc[valid_ids]
            
            if not cluster_orders.empty:
                cluster_orders['到枢纽距离'] = cluster_orders.apply(
                    lambda row: self.calculate_distance(hub_loc, (row['经度'], row['纬度'])), axis=1
                )
                
                valid_drone_orders = cluster_orders[
                    (~cluster_orders['在禁飞区']) &
                    (cluster_orders['风险评分'] < self.config['drone_safety_threshold'])
                ].copy()
                
                valid_drone_orders = valid_drone_orders.sort_values(by='总重量（kg）', ascending=False)
                
                for idx, row in valid_drone_orders.iterrows():
                    load = row['总重量（kg）']
                    # 电池衰减因子：每10km折减5%（论文数据）
                    base_range = max_range - (load - self.config['drone']['min_capacity']) * \
                                 (max_range - min_range) / (max_capacity - self.config['drone']['min_capacity'])
                    distance_factor = 1 - (row['到枢纽距离'] / 10) * 0.05
                    effective_range = max(min_range, base_range * distance_factor)
                    valid_drone_orders.at[idx, 'effective_range'] = effective_range
                
                sorted_orders = valid_drone_orders
                
                drone_id = 0
                operations[cluster_id] = []
                
                while not sorted_orders.empty:
                    current_bin = []
                    current_weight = 0
                    max_distance = 0
                    
                    for idx, row in sorted_orders.iterrows():
                        if current_weight + row['总重量（kg）'] > max_capacity:
                            continue
                        
                        effective_range = row['effective_range']
                        if max_distance * 2 > effective_range:
                            continue
                        
                        if row['温控等级'] == 'P1':
                            warehouse_to_hub = self.calculate_distance(self.warehouse, hub_loc)
                            hub_to_customer = self.calculate_distance(hub_loc, (row['经度'], row['纬度']))
                            
                            delivery_time = (warehouse_to_hub / self.config['truck']['speed']) + \
                                           (hub_to_customer / self.config['drone']['speed'])
                            if delivery_time > 2.8:
                                continue
                        
                        if not self.check_drone_route_safety(hub_loc, (row['经度'], row['纬度'])):
                            continue
                        
                        current_bin.append(idx)
                        current_weight += row['总重量（kg）']
                        max_distance = max(max_distance, row['到枢纽距离'])
                    
                    if current_bin:
                        operations[cluster_id].append({
                            'drone_id': drone_id,
                            'orders': current_bin,
                            'total_weight': current_weight,
                            'max_distance': max_distance
                        })
                        drone_id += 1
                        sorted_orders = sorted_orders.drop(current_bin)
                    else:
                        break
            else:
                operations[cluster_id] = []
        
        return operations


    # ===== 无人机路线安全检查 =====
    def check_drone_route_safety(self, start_point, end_point):
        if any(math.isnan(coord) for coord in start_point + end_point):
            return False
        
        distance = self.calculate_distance(start_point, end_point)
        if math.isinf(distance) or distance <= 0:
            return False
        
        num_checks = max(5, min(50, int(distance * 3))) 
        safety_margin = self.config['safety_margin']
        
        if self.current_temperature > 35:
            safety_margin *= 1.2
        if self.config['wind_speed'] > 5:
            safety_margin *= 1.3
        
        for i in range(num_checks + 1):
            fraction = i / num_checks
            base_point = (
                start_point[0] + fraction * (end_point[0] - start_point[0]),
                start_point[1] + fraction * (end_point[1] - start_point[1])
            )
            
            max_offset = distance * 0.03 
            offset_distance = random.uniform(-max_offset, max_offset)
            offset_direction = random.uniform(0, 2 * math.pi)
            
            check_point = (
                base_point[0] + offset_distance * math.cos(offset_direction),
                base_point[1] + offset_distance * math.sin(offset_direction)
            )
            
            if self.check_no_fly_zone(check_point, safety_margin):
                return False
        
        if not self.request_dynamic_airspace((start_point, end_point)):
            return False
        
        return True


    # ===== 动态空域申请（论文89%成功率） =====
    def request_dynamic_airspace(self, route):
        route_id = hash(route)
        
        if route_id in self.dynamic_airspace_requests:
            return self.dynamic_airspace_requests[route_id]
        
        success = random.random() < 0.89  # 匹配论文模拟成功率
        self.dynamic_airspace_requests[route_id] = success
        
        if not success:
            print(f"空域申请失败: 路线 {route[0]} -> {route[1]}")
        
        return success


    # ===== 计算性能指标 =====
    def calculate_performance(self, truck_route, truck_distance, drone_operations):
        truck_cost = self.config['truck']['fixed_cost'] + truck_distance * self.config['truck']['var_cost']
        
        drone_cost = 0
        drone_order_count = 0
        for cluster_id, operations in drone_operations.items():
            if cluster_id not in self.hubs:
                continue
            for op in operations:
                drone_cost += self.config['drone']['fixed_cost'] + 2 * op['max_distance'] * self.config['drone']['var_cost']
                drone_order_count += len(op['orders'])
        
        penalty_cost = 0
        delivery_times = []
        
        for idx, order in self.orders_for_date.iterrows():
            warehouse_distance = order['到仓库距离']
            if math.isinf(warehouse_distance) or warehouse_distance <= 0:
                warehouse_distance = 1000
            
            drone_delivered = False
            for cluster_id, ops in drone_operations.items():
                for op in ops:
                    if idx in op['orders']:
                        hub_location = self.hubs[cluster_id]
                        warehouse_to_hub = self.calculate_distance(self.warehouse, hub_location)
                        hub_to_customer = self.calculate_distance(hub_location, (order['经度'], order['纬度']))
                        
                        if math.isinf(warehouse_to_hub): 
                            warehouse_to_hub = 1000
                        if math.isinf(hub_to_customer): 
                            hub_to_customer = 1000
                        
                        delivery_time = self.base_time + timedelta(
                            hours=(warehouse_to_hub / self.config['truck']['speed']) +
                            (hub_to_customer / self.config['drone']['speed'])
                        )
                        drone_delivered = True
                        break
                if drone_delivered:
                    break
            
            if not drone_delivered:
                delivery_time = self.base_time + timedelta(hours=warehouse_distance / self.config['truck']['speed'])
            
            delivery_times.append(delivery_time)
            
            try:
                time_diff = (delivery_time - order['最晚送达时间']).total_seconds() / 3600
            except:
                time_diff = 0
            
            if time_diff > 0:
                penalty_rate = self.config['penalty_rates'].get(order['温控等级'], 0)
                penalty_cost += penalty_rate * order['总重量（kg）'] * time_diff
        
        temp_sensitive_orders = self.orders_for_date[self.orders_for_date['温控等级'] != 'P3']
        temp_compliance = 1.0
        
        if not temp_sensitive_orders.empty:
            compliant_count = 0
            
            for idx, order in temp_sensitive_orders.iterrows():
                if order['配送方式'] == '货车':
                    compliant_count += 1
                else:
                    if 'delivery_time' in locals():
                        delivery_time_hours = (delivery_time - self.base_time).total_seconds() / 3600
                        time_limit = 3 if order['温控等级'] == 'P1' else 6
                        if delivery_time_hours <= time_limit:
                            compliant_count += 1
                    else:
                        compliant_count += 1
            
            temp_compliance = compliant_count / len(temp_sensitive_orders)
        
        avg_delivery_time = 0
        if delivery_times:
            avg_delivery_time = sum((t - self.base_time).total_seconds() / 3600 for t in delivery_times) / len(delivery_times)
        
        drone_utilization = drone_order_count / len(self.orders_for_date) if len(self.orders_for_date) > 0 else 0
        
        safe_orders = self.orders_for_date[self.orders_for_date['风险评分'] < self.config['risk_threshold']]
        violation_count = 0
        if not safe_orders.empty:
            violation_count = sum(1 for _, order in safe_orders.iterrows() if order['配送方式'] == '货车')
        violation_rate = violation_count / len(safe_orders) if len(safe_orders) > 0 else 0
        
        cost_per_order = (truck_cost + drone_cost + penalty_cost) / len(self.orders_for_date) if len(self.orders_for_date) > 0 else 0
        drone_cost_per_order = drone_cost / drone_order_count if drone_order_count > 0 else 0  # 无人机单均成本
        
        return {
            'truck_route': truck_route,
            'truck_distance': truck_distance,
            'drone_operations': drone_operations,
            'costs': {
                'truck': truck_cost,
                'drones': drone_cost,
                'penalty': penalty_cost,
                'total': truck_cost + drone_cost + penalty_cost
            },
            'metrics': {
                'total_orders': len(self.orders_for_date),
                'truck_orders': len(self.orders_for_date) - drone_order_count,
                'drone_orders': drone_order_count,
                'avg_delivery_time': avg_delivery_time,
                'drone_utilization': drone_utilization,
                'violation_rate': violation_rate,
                'temp_compliance_rate': temp_compliance,
                'cost_per_order': cost_per_order,  # 单均成本
                'drone_cost_per_order': drone_cost_per_order,  # 无人机单均成本
                'filtered_orders': self.original_order_count - len(self.orders_for_date)
            }
        }


    # ===== 分配配送方式 =====
    def assign_delivery_methods(self, drone_operations, hubs):
        if self.orders_for_date.empty:
            return
        
        self.orders_for_date['配送方式'] = '货车'
        
        self.orders_for_date.loc[self.orders_for_date['在禁飞区'], '配送方式'] = '货车'
        
        self.orders_for_date.loc[self.orders_for_date['风险评分'] >= self.config['risk_threshold'], '配送方式'] = '货车'
        
        for cluster_id, ops in drone_operations.items():
            for op in ops:
                for order_id in op['orders']:
                    if order_id in self.orders_for_date.index:
                        self.orders_for_date.loc[order_id, '配送方式'] = '无人机'


    # ===== 打印单日期结果 =====
    def print_results(self, results, date=None):
        if not results:
            return
        
        date_str = date.strftime('%Y-%m-%d') if date else "总体"
        metrics = results['metrics']
        costs = results['costs']
        
        print(f"\n===== {date_str} 核心优化结果 ====")
        print(f"总订单: {metrics['total_orders']} | 无人机配送: {metrics['drone_orders']} | 货车配送: {metrics['truck_orders']}")
        print(f"禁飞区过滤: {metrics['filtered_orders']} 个")
        
        print("\n成本指标:")
        print(f" 总成本: ¥{costs['total']:.2f} | 单均成本: ¥{metrics['cost_per_order']:.2f}")
        print(f" 无人机单均成本: ¥{metrics['drone_cost_per_order']:.2f} | 货车成本: ¥{costs['truck']:.2f} | 延迟罚款: ¥{costs['penalty']:.2f}")
        
        print("\n性能指标:")
        print(f" 无人机利用率: {metrics['drone_utilization']*100:.2f}% | 温控达标率: {metrics['temp_compliance_rate']*100:.2f}%")
        print(f" 平均配送时间: {metrics['avg_delivery_time']:.2f} 小时")


    # ===== 保存配送计划 =====
    def save_delivery_plan(self, results, filename="配送计划.csv"):
        if not results or 'drone_operations' not in results:
            print("无配送计划可保存")
            return
        
        print("正在保存配送计划...")
        plan = {
            'truck_plan': self.generate_delivery_plan(results),
            'drone_operations': results['drone_operations']
        }
        
        plan_df = pd.DataFrame(plan['truck_plan'])
        
        order_details = []
        for cluster_id, ops in plan['drone_operations'].items():
            for op in ops:
                for order_id in op['orders']:
                    if order_id in self.orders_for_date.index:
                        order = self.orders_for_date.loc[order_id]
                        order_details.append({
                            '订单ID': order_id,
                            '运单号': order.get('运单号', ''),
                            '发车完成时间（日）': order.get('发车完成时间（日）', ''),
                            '仓库名称': order.get('仓库名称', ''),
                            '商品名称': order.get('商品名称', ''),
                            '总重量（g）': order['总重量（g）'],
                            '总重量（kg）': order['总重量（kg）'],
                            '温控等级': order['温控等级'],
                            '配送方式': '无人机',
                            '所属聚类': cluster_id,
                            '无人机ID': op['drone_id'],
                            '风险评分': order['风险评分']
                        })
        
        drone_order_ids = [order_id for ops in results['drone_operations'].values() 
                          for op in ops for order_id in op['orders']]
        
        truck_orders = self.orders_for_date[~self.orders_for_date.index.isin(drone_order_ids)]
        for idx, order in truck_orders.iterrows():
            if idx in self.orders_for_date.index:
                order_details.append({
                    '订单ID': idx,
                    '运单号': order.get('运单号', ''),
                    '发车完成时间（日）': order.get('发车完成时间（日）', ''),
                    '仓库名称': order.get('仓库名称', ''),
                    '商品名称': order.get('商品名称', ''),
                    '总重量（g）': order['总重量（g）'],
                    '总重量（kg）': order['总重量（kg）'],
                    '温控等级': order['温控等级'],
                    '配送方式': '货车',
                    '风险评分': order['风险评分']
                })
        
        order_df = pd.DataFrame(order_details)
        
        os.makedirs("output", exist_ok=True)
        plan_path = f"output/{filename}"
        order_path = f"output/订单配送详情_{filename}"
        
        plan_df.to_csv(plan_path, index=False)
        order_df.to_csv(order_path, index=False)
        
        print(f"配送计划已保存至: {os.path.abspath(plan_path)}")


    # ===== 生成货车配送时间线 =====
    def generate_delivery_plan(self, results):
        truck_plan = []
        current_time = self.base_time
        
        truck_plan.append({
            'type': 'departure',
            'location': self.warehouse,
            'time': current_time,
            'description': '货车从仓库出发'
        })
        
        truck_route = results['truck_route']
        hubs = {cluster_id: hub for cluster_id, hub in self.hubs.items()}
        
        for step_idx in truck_route:
            if step_idx < len(hubs):
                cluster_id = list(hubs.keys())[step_idx]
                hub_location = hubs[cluster_id]
                
                distance = self.calculate_distance(truck_plan[-1]['location'], hub_location)
                travel_time = timedelta(hours=distance / self.config['truck']['speed'])
                current_time += travel_time
                
                truck_plan.append({
                    'type': 'arrival',
                    'location': hub_location,
                    'time': current_time,
                    'description': f'货车到达集散中心 {cluster_id}'
                })
                
                if cluster_id in results['drone_operations']:
                    for op in results['drone_operations'][cluster_id]:
                        drone_time = current_time
                        for order_id in op['orders']:
                            if order_id in self.orders_for_date.index:
                                order = self.orders_for_date.loc[order_id]
                                distance = self.calculate_distance(hub_location, (order['经度'], order['纬度']))
                                flight_time = timedelta(hours=distance / self.config['drone']['speed'])
                                
                                truck_plan.append({
                                    'type': 'drone_delivery',
                                    'location': (order['经度'], order['纬度']),
                                    'time': drone_time + flight_time,
                                    'description': f"无人机 {op['drone_id']} 配送订单 {order_id}"
                                })
                
                current_time += timedelta(minutes=10)
                truck_plan.append({
                    'type': 'departure',
                    'location': hub_location,
                    'time': current_time,
                    'description': f'货车离开集散中心 {cluster_id}'
                })
            else:
                risky_order_idx = step_idx - len(hubs)
                if 0 <= risky_order_idx < len(self.orders_for_date) - len(self.clusters):
                    order = self.orders_for_date.iloc[risky_order_idx]
                    order_location = (order['经度'], order['纬度'])
                    
                    distance = self.calculate_distance(truck_plan[-1]['location'], order_location)
                    travel_time = timedelta(hours=distance / self.config['truck']['speed'])
                    current_time += travel_time
                    
                    truck_plan.append({
                        'type': 'truck_delivery',
                        'location': order_location,
                        'time': current_time,
                        'description': f"货车配送高风险订单 {order.name}"
                    })
                    current_time += timedelta(minutes=5)
        
        if truck_plan and 'location' in truck_plan[-1]:
            distance = self.calculate_distance(truck_plan[-1]['location'], self.warehouse)
            travel_time = timedelta(hours=distance / self.config['truck']['speed'])
            current_time += travel_time
            
            truck_plan.append({
                'type': 'arrival',
                'location': self.warehouse,
                'time': current_time,
                'description': '货车返回仓库'
            })
        
        return truck_plan


# ===== 主程序入口（补充全量指标输出） =====
if __name__ == "__main__":
    try:
        # 记录总运算开始时间
        total_start_time = time.time()
        
        warehouse = (113.887835, 34.467290)
        print("==== 创建物流系统 ====")
        system = LogisticSystem(warehouse, config={
            'use_sa_algorithm': True,
            'simulated_annealing': {
                'initial_temperature': 150.0,
                'cooling_rate': 0.96,
                'min_temperature': 1e-6,
                'iterations_per_temp': 60
            }
        })
        
        # 加载订单数据（替换为实际路径）
        system.load_orders_from_excel(r"C:\Users\13161\Desktop\附件一.xlsx")
        
        if not system.orders.empty:
            print("\n==== 按日期优化配送方案 ====")
            daily_results = system.optimize_delivery_by_date()
            
            if daily_results:
                # 汇总全量指标
                total_orders = sum(results['metrics']['total_orders'] for results in daily_results.values())
                total_drone_orders = sum(results['metrics']['drone_orders'] for results in daily_results.values())
                total_cost = sum(results['costs']['total'] for results in daily_results.values())
                total_truck_cost = sum(results['costs']['truck'] for results in daily_results.values())
                total_drone_cost = sum(results['costs']['drones'] for results in daily_results.values())
                total_penalty = sum(results['costs']['penalty'] for results in daily_results.values())
                avg_cost_per_order = total_cost / total_orders if total_orders > 0 else 0
                avg_drone_cost_per_order = total_drone_cost / total_drone_orders if total_drone_orders > 0 else 0
                total_drone_utilization = total_drone_orders / total_orders if total_orders > 0 else 0
                total_temp_compliance = sum(results['metrics']['temp_compliance_rate'] * results['metrics']['total_orders'] for results in daily_results.values()) / total_orders if total_orders > 0 else 0
                
                # 计算总运算时间
                total_end_time = time.time()
                total_runtime = total_end_time - total_start_time
                
                # 输出全量汇总指标
                print(f"\n==== 全量订单优化最终汇总指标 ====")
                print(f"1. 运算效率:")
                print(f"   总运算时间: {total_runtime:.2f}秒 ({total_runtime/60:.2f}分钟)")
                print(f"2. 订单规模:")
                print(f"   总订单数: {total_orders} | 无人机配送订单: {total_drone_orders} (占比{total_drone_orders/total_orders*100:.2f}%)")
                print(f"3. 成本指标:")
                print(f"   总成本: ¥{total_cost:.2f} | 单均成本: ¥{avg_cost_per_order:.2f}")
                print(f"   无人机总成本: ¥{total_drone_cost:.2f} | 无人机单均成本: ¥{avg_drone_cost_per_order:.2f}")
                print(f"   货车总成本: ¥{total_truck_cost:.2f} | 延迟罚款总成本: ¥{total_penalty:.2f}")
                print(f"4. 性能指标:")
                print(f"   无人机平均利用率: {total_drone_utilization*100:.2f}% | 总体温控达标率: {total_temp_compliance*100:.2f}%")
            else:
                print("无有效优化结果")
        else:
            print("无有效订单数据，请检查文件")
    
    except Exception as e:
        print(f"运行错误: {e}")
        traceback.print_exc()