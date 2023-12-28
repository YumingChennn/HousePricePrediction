from geopy.distance import geodesic
from pyproj import Proj, Transformer
from collections import OrderedDict
import csv
from rtree import index


# 指定輸入和輸出檔案路徑
input_file_path = r'C:\Users\weili\OneDrive\桌面\Code\30_Private Dataset _Private and Publict Submission Template_v2\updated_PRDistance_dataset.csv'
output_file_path = r'C:\Users\weili\OneDrive\桌面\Code\30_Private Dataset _Private and Publict Submission Template_v2\Uupdated_PRDistance_dataset.csv'

# 定義 TWD97 和 WGS 84 座標系統，使用 <authority>:<code> 語法
twd97 = Proj("epsg:3826")
wgs84 = Proj("epsg:4326")

# 初始化函式呼叫計數器
twd97_to_latlon_counter = 0
load_house_coordinates_counter = 0
load_facility_coordinates_counter = 0
facility_index = index.Index()

# 函式：將 TWD97 座標轉換為緯度和經度
def twd97_to_latlon(x, y):
    global twd97_to_latlon_counter  # 使用全域變數追蹤呼叫次數
    twd97_to_latlon_counter += 1  # 每次呼叫增加計數器

    # 使用 pyproj 中的 Transformer 類別進行坐標轉換
    transformer = Transformer.from_proj(twd97, wgs84, always_xy=True)
    lon, lat = transformer.transform(x, y)  # 將 TWD97 座標轉換為緯度和經度
    return lat, lon  # 回傳緯度和經度


# 函式：載入訓練屋座標資料
# 修改 load_house_coordinates 函式以從特定欄位讀取座標資料
def load_house_coordinates():
    global load_house_coordinates_counter  # 使用全域變數追蹤呼叫次數
    load_house_coordinates_counter += 1  # 每次呼叫增加計數器
    coordinates = []  # 儲存座標資料的列表

    with open(input_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        # 假設橫坐標的欄位名稱為 "橫坐標"，縱坐標的欄位名稱為 "縱坐標"
        lat_index = 15  # 替換成實際的緯度欄位索引
        lon_index = 16  # 替換成實際的經度欄位索引

        for line in lines[1:]:  # 跳過標頭（第一行）
            columns = line.strip().split(',')
            if len(columns) > lat_index and len(columns) > lon_index:
                lat_str = columns[lat_index].strip()
                lon_str = columns[lon_index].strip()
                if lat_str and lon_str:
                    try:
                        x = float(lat_str)
                        y = float(lon_str)
                        lat, lon = twd97_to_latlon(x, y)
                        coordinates.append((lat, lon))
                    except ValueError:
                        # 處理轉換為浮點數失敗的情況
                        pass

    print(f"load_house_coordinates call count: {load_house_coordinates_counter}")
    return coordinates  # 回傳座標資料的列表


# 函式：載入特殊設施座標資料
# 修改 load_facility_coordinates 函式以從特定欄位讀取座標資料
def load_facility_coordinates(facility_files):
    global load_facility_coordinates_counter  # 使用全域變數追蹤呼叫次數
    load_facility_coordinates_counter += 1  # 每次呼叫增加計數器
    facility_data = {}  # 儲存設施座標資料的字典

    for facility, config in facility_files.items():
        coordinates = []  # 儲存特定設施座標的列表
        lat_index = config['lat_index']  # 從設定中取得緯度的欄位索引
        lon_index = config['lon_index']  # 從設定中取得經度的欄位索引

        with open(config['file_path'], 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines[1:]:  # 跳過標頭（第一行）
                columns = line.strip().split(',')
                if len(columns) > lat_index and len(columns) > lon_index:
                    lat_str = columns[lat_index].strip()
                    lon_str = columns[lon_index].strip()
                    if lat_str and lon_str:
                        try:
                            lat = float(lat_str)
                            lon = float(lon_str)
                            coordinates.append((lat, lon))
                        except ValueError:
                            # 處理轉換為浮點數失敗的情況
                            pass

        facility_data[facility] = coordinates  # 將座標資料存入字典

    print(f"load_facility_coordinates call count: {load_facility_coordinates_counter}")
    return facility_data  # 回傳設施座標資料的字典


# 函式：計算每座房屋與每種設施之間最近距離
def calculate_closest_facility_distances(house_coordinates, facility_data):
    # 初始化一個字典來存儲每座房屋與每種設施之間的最近距離
    distances = {house_index: {facility: None for facility in facility_data} for house_index, _ in enumerate(house_coordinates)}

    total_houses = len(house_coordinates)
    total_facilities = len(facility_data)
    total_combinations = total_houses * total_facilities

    # 疊代每座房屋
    for house_index, house_coord in enumerate(house_coordinates):
        house_progress = (house_index + 1) / total_houses * 100  # 計算每座房屋的進度
        print(f"House {house_index + 1}/{total_houses} Calculation Progress: {house_progress:.2f}%")

        # 疊代每種設施
        for facility in facility_data:
            closest_distance = float('inf')  # 初始化最近距離為正無窮大
            if facility_data[facility]:
                # 疊代特定設施的每個座標
                for facility_coord in facility_data[facility]:
                    distance = geodesic(house_coord, facility_coord).kilometers
                    if distance < closest_distance:
                        closest_distance = distance  # 更新最近距離
                distances[house_index][facility] = closest_distance
            else:
                distances[house_index][facility] = None  # 處理當設施資料缺失時的情況

    return distances  # 回傳每座房屋與每種設施之間的最近距離


# 計算每種設施在每座屋子一定範圍內的數量
def count_facilities_around_houses(house_coordinates, facility_data, radius):
    # 初始化一個字典來存儲每種設施在每座房屋一定範圍內的數量
    counts = {facility: [] for facility in facility_data}
    total_houses = len(house_coordinates)
    total_facilities = len(facility_data)
    total_combinations = total_houses * total_facilities

    # 迭代每座房屋
    for i, house_coord in enumerate(house_coordinates):
        # 迭代每種設施
        for facility, facility_coordinates in facility_data.items():
            count = 0
            if facility_coordinates:
                # 迭代特定設施的每個座標
                for facility_coord in facility_coordinates:
                    distance = geodesic(house_coord, facility_coord).kilometers
                    if distance <= radius:
                        count += 1  # 在指定半徑內的設施數量加1

                counts[facility].append(count)  # 將結果添加到字典中

            # 計算當前房屋、設施類別和整體進度的進度百分比
            progress_houses = (i + 1) / total_houses * 100
            progress_facility = (list(facility_data.keys()).index(facility) + 1) / total_facilities * 100
            progress_overall = ((i * total_facilities) + (list(facility_data.keys()).index(facility) + 1)) / total_combinations * 100
        print(f"Progress on house {i+1}/{total_houses}, Overall: {progress_overall:.3f}%")

    return counts  # 回傳每種設施在每座房屋一定範圍內的數量


# 呼叫 load_house_coordinates 函式以載入房屋座標資料
house_coordinates = load_house_coordinates()  # Call the function without any arguments


# 特殊設施的資料檔案路徑以及其對應的緯度和經度的欄位索引，用來計算每座房屋與特殊設施之間最近距離
Dis_facility_files = OrderedDict({
    '國中': {
        'file_path': r'C:\Users\weili\OneDrive\桌面\Code\30_Training Dataset_V2\external_data_TransCoor\國中基本資料.csv',
        'lat_index': 19,
        'lon_index': 20
    },
    '國小': {
        'file_path': r'C:\Users\weili\OneDrive\桌面\Code\30_Training Dataset_V2\external_data_TransCoor\國小基本資料.csv',
        'lat_index': 28,
        'lon_index': 29
    },
    '大學': {
        'file_path': r'C:\Users\weili\OneDrive\桌面\Code\30_Training Dataset_V2\external_data_TransCoor\大學基本資料.csv',
        'lat_index': 25,
        'lon_index': 26
    },
    '捷運站點': {
        'file_path': r'C:\Users\weili\OneDrive\桌面\Code\30_Training Dataset_V2\external_data_TransCoor\捷運站點資料.csv',
        'lat_index': 5,
        'lon_index': 6
    },
    '火車站點': {
        'file_path': r'C:\Users\weili\OneDrive\桌面\Code\30_Training Dataset_V2\external_data_TransCoor\火車站點資料.csv',
        'lat_index': 4,
        'lon_index': 5
    },
    '郵局據點': {
        'file_path': r'C:\Users\weili\OneDrive\桌面\Code\30_Training Dataset_V2\external_data_TransCoor\郵局據點資料.csv',
        'lat_index': 7,
        'lon_index': 8
    },
    '醫療機構': {
        'file_path': r'C:\Users\weili\OneDrive\桌面\Code\30_Training Dataset_V2\external_data_TransCoor\醫療機構基本資料.csv',
        'lat_index': 39,
        'lon_index': 40
    },
    '金融機構': {
        'file_path': r'C:\Users\weili\OneDrive\桌面\Code\30_Training Dataset_V2\external_data_TransCoor\金融機構基本資料.csv',
        'lat_index': 5,
        'lon_index': 6
    },
    '高中': {
        'file_path': r'C:\Users\weili\OneDrive\桌面\Code\30_Training Dataset_V2\external_data_TransCoor\高中基本資料.csv',
        'lat_index': 15,
        'lon_index': 16
    }})

# 每種設施的資料檔案路徑以及其對應的緯度和經度的欄位索引，計算每種設施在每座屋子一定範圍內的數量
Full_acility_files = OrderedDict({
    'ATM': {
        'file_path': r'C:\Users\weili\OneDrive\桌面\Code\30_Training Dataset_V2\external_data_TransCoor\ATM.csv',
        'lat_index': 5,
        'lon_index': 6
    },
    '便利商店': {
        'file_path': r'C:\Users\weili\OneDrive\桌面\Code\30_Training Dataset_V2\external_data_TransCoor\便利商店.csv',
        'lat_index': 3,
        'lon_index': 4
    },
    '公車站點': {
        'file_path': r'C:\Users\weili\OneDrive\桌面\Code\30_Training Dataset_V2\external_data_TransCoor\公車站點資料.csv',
        'lat_index': 4,
        'lon_index': 5
    },
    '國中': {
        'file_path': r'C:\Users\weili\OneDrive\桌面\Code\30_Training Dataset_V2\external_data_TransCoor\國中基本資料.csv',
        'lat_index': 19,
        'lon_index': 20
    },
    '國小': {
        'file_path': r'C:\Users\weili\OneDrive\桌面\Code\30_Training Dataset_V2\external_data_TransCoor\國小基本資料.csv',
        'lat_index': 28,
        'lon_index': 29
    },
    '大學': {
        'file_path': r'C:\Users\weili\OneDrive\桌面\Code\30_Training Dataset_V2\external_data_TransCoor\大學基本資料.csv',
        'lat_index': 25,
        'lon_index': 26
    },
    '捷運站點': {
        'file_path': r'C:\Users\weili\OneDrive\桌面\Code\30_Training Dataset_V2\external_data_TransCoor\捷運站點資料.csv',
        'lat_index': 5,
        'lon_index': 6
    },
    '火車站點': {
        'file_path': r'C:\Users\weili\OneDrive\桌面\Code\30_Training Dataset_V2\external_data_TransCoor\火車站點資料.csv',
        'lat_index': 4,
        'lon_index': 5
    },
    '腳踏車站點': {
        'file_path': r'C:\Users\weili\OneDrive\桌面\Code\30_Training Dataset_V2\external_data_TransCoor\腳踏車站點資料.csv',
        'lat_index': 6,
        'lon_index': 7
    },
    '郵局據點': {
        'file_path': r'C:\Users\weili\OneDrive\桌面\Code\30_Training Dataset_V2\external_data_TransCoor\郵局據點資料.csv',
        'lat_index': 7,
        'lon_index': 8
    },
    '醫療機構': {
        'file_path': r'C:\Users\weili\OneDrive\桌面\Code\30_Training Dataset_V2\external_data_TransCoor\醫療機構基本資料.csv',
        'lat_index': 39,
        'lon_index': 40
    },
    '金融機構': {
        'file_path': r'C:\Users\weili\OneDrive\桌面\Code\30_Training Dataset_V2\external_data_TransCoor\金融機構基本資料.csv',
        'lat_index': 5,
        'lon_index': 6
    },
    '高中': {
        'file_path': r'C:\Users\weili\OneDrive\桌面\Code\30_Training Dataset_V2\external_data_TransCoor\高中基本資料.csv',
        'lat_index': 15,
        'lon_index': 16
    }})


# 指定搜尋範圍的半徑（單位：公里）
search_radius = 1



# 載入每種特殊設施的座標資料
facility_data = load_facility_coordinates(Full_acility_files)
# 使用 r-tree 索引結構加速特殊設施的空間查找
for facility, facility_coordinates in facility_data.items():
    for i, facility_coord in enumerate(facility_coordinates):
        # 在 r-tree 中插入每個設施的座標
        facility_index.insert(i, facility_coord + facility_coord)
# 計算每種特殊設施在每座房屋一定範圍內的數量
facility_counts = count_facilities_around_houses(house_coordinates, facility_data, search_radius)




# 載入每種特殊設施的座標資料
facility_data = load_facility_coordinates(Dis_facility_files)
# 使用 r-tree 索引結構加速特殊設施的空間查找
for facility, facility_coordinates in facility_data.items():
    for i, facility_coord in enumerate(facility_coordinates):
        # 在 r-tree 中插入每個設施的座標
        facility_index.insert(i, facility_coord + facility_coord)
# 計算每種特殊設施在每座房屋一定範圍內的數量
facility_counts = count_facilities_around_houses(house_coordinates, facility_data, search_radius)
# 計算每座房屋到每種特殊設施的最近距離
facility_distances = calculate_closest_facility_distances(house_coordinates, facility_data)



house_dataset = []

# 開啟 CSV 檔案進行讀取
with open(input_file_path, 'r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    header = next(csv_reader)  # 讀取並儲存表頭
    # 加入新的欄位名稱，表示每種設施到最近房屋的距離
    header += [f"{facility}距離" for facility in facility_files.keys()]
    for i, row in enumerate(csv_reader):
        facility_distances_row = []
        # 將每種設施到最近房屋的距離加入到該列的資料中
        for facility_type in facility_files.keys():
            if i in facility_distances and facility_type in facility_distances[i]:
                facility_distances_row.append(str(facility_distances[i][facility_type]))
            else:
                facility_distances_row.append("None")
        # 將每種設施到最近房屋的距離資料加入到該列中
        row += facility_distances_row
        house_dataset.append(row)


# 開啟 CSV 檔案進行讀取
with open(input_file_path, 'r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    header = next(csv_reader)  # 讀取並儲存表頭
    # 將特殊設施的種類加入到表頭中
    header += list(facility_files.keys())
    for i, row in enumerate(csv_reader):
        try:
            # 取得特殊設施在當前列的數量，並轉換為整數
            facility_count = [facility_counts[facility_type][i] for facility_type in facility_files.keys()]
            facility_count = list(map(int, facility_count))
            # 將特殊設施的數量加入到該列的資料中
            row += list(map(str, facility_count))
            house_dataset.append(row)
            print(row)
        except Exception as e:
            # 記錄錯誤並繼續處理下一列
            print(f"Error processing row {i + 2}: {e}")  # 加 2 是因為表頭佔了第一列，索引從 0 開始
            continue


# 儲存更新後的資料集至新的 CSV 檔案，並進行錯誤處理
try:
    with open(output_file_path, 'w', newline='', encoding='utf-8') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(header)  # 寫入表頭
        csv_writer.writerows(house_dataset)  # 寫入更新的資料
except Exception as e:
    print(f"Error saving the updated dataset: {e}")
