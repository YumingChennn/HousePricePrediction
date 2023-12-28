import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import joblib
import os

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from datetime import datetime
from scipy.stats import skew  # for some statistics
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score,train_test_split
from sklearn.metrics import mean_squared_error
from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.linear_model as linear_model
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from scipy.stats import yeojohnson
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore') # 忽略所有警告
import lightgbm as lgb


# 設定中文字型，這裡以 'MingLiU' 為例，你也可以根據需求更改為其他中文字型
plt.rcParams['font.sans-serif'] = ['MingLiU']  # Example: 'Microsoft JhengHei', 'SimHei', 'MingLiU'
# 確保負號正確顯示
plt.rcParams['axes.unicode_minus'] = False  # Ensure that minus signs are shown correctly

# 讀取訓練資料集檔案路徑
df = pd.read_csv(r"Code\30_Training Dataset_V2\updated_Distance_data.csv")

# 指定模型存處資料夾路徑
folder_path = r"Code\Models"
# 檢查資料夾是否存在，如果不存在則創建資料夾
if not os.path.exists(folder_path):
    os.makedirs(folder_path)


# 計算每個欄位的缺失值數量
nan_values = df.isnull().sum()

# 輸出結果標題
print("Missing Values by Column")
print("-" * 30)

# 輸出每個欄位的缺失值數量
print(nan_values)

# 輸出分隔線
print("-" * 30)

# 輸出總缺失值數量
print("TOTAL MISSING VALUES:", nan_values.sum())


# 選擇僅包含數值的欄位，以檢查無窮大（infinite）的數值
numeric_columns = df.select_dtypes(include=[np.number]).columns

# 計算DataFrame中的無窮大數值的總數
inf_values = np.isinf(df[numeric_columns].values).sum()

# 輸出DataFrame中的無窮大數值總數
print("Infinite Values in the DataFrame:", inf_values)

# 再次選擇僅包含浮點數和整數的欄位
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

# 計算相關矩陣
correlation_matrix = df[numeric_columns].corr()

# 繪製相關矩陣的熱圖
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, cmap="RdBu")
plt.title("Correlations Between Variables", size=15)
plt.show()


# 選擇相對於"單價"欄位具有重要相關性的數值型欄位
important_num_cols = list(correlation_matrix["單價"][(correlation_matrix["單價"] > 0.03) | (correlation_matrix["單價"] < -0.03)].index)

# 選擇進行特殊處理的其他特定數值型欄位
Separate_cols = ["移轉層次", "總樓層數", "屋齡", "公車站點", "捷運站點", "金融機構", "高中", "腳踏車站點", "橫坐標", "縱坐標"]

# 選擇類別型欄位
cat_cols = ["縣市", "主要用途", "主要建材", "建物型態"]

# 將所有重要的數值型和類別型欄位組合在一起
important_cols = important_num_cols + Separate_cols + cat_cols

# 輸出提示
print(1)

# 從原始DataFrame中選擇僅包含上述欄位的子集
df = df[important_cols]



# 輸出缺失值的統計資訊
print("Missing Values by Column")
print("-" * 30)

# 計算每個欄位的缺失值數量並輸出
print(df.isna().sum())

# 輸出分隔線
print("-" * 30)

# 計算總缺失值數量並輸出
print("TOTAL MISSING VALUES:", df.isna().sum().sum())

# 從DataFrame中移除目標變數"單價"，得到特徵變數X
X = df.drop("單價", axis=1)

# 輸出特徵變數X的形狀（行數、列數）
print(X.shape[0])

# 從DataFrame中選取目標變數"單價"，得到目標變數y
y = df["單價"]

# 輸出提示 "2"
print(2)






# One Hot encoding

# 對類別變量進行One Hot encoding
X = pd.get_dummies(X, columns=cat_cols)
# 將編碼後的資料索引重新設置為原始索引
print(X.shape[0])

# 從編碼後的資料中選擇特定的子集
X_Separate = X[Separate_cols]




# 使用 PolynomialFeatures 進行多項式特徵轉換，將次數設置為2
poly = PolynomialFeatures(degree=2)

# 對 X_Separate 進行多項式特徵轉換
X_poly = poly.fit_transform(X_Separate)

# 將轉換後的多項式特徵轉換為 DataFrame，並設置列名為原始欄位名稱
X_poly = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(X_Separate.columns))

# 移除原始被轉換的欄位
X = X.drop(Separate_cols, axis=1)

# 將轉換後的多項式特徵與原始 DataFrame 進行合併
X = pd.concat([X, X_poly], axis=1)

# 移除目標變數 "單價"，以及已經在特徵中的數值型變數
important_num_cols.remove("單價")


# 使用 StandardScaler 對數值型變數進行標準化
scaler = StandardScaler()
X[important_num_cols] = scaler.fit_transform(X[important_num_cols])

# 移除指定的欄位
cols_to_drop = ["車位面積", "附屬建物面積", "大學", "火車站點", "捷運站點距離"]
X = X.drop(cols_to_drop, axis=1)



# 輸出缺失值的統計資訊
print("Missing Values by Column")
print("-" * 30)

# 計算每個欄位的缺失值數量並輸出
print(df.isna().sum())

# 輸出分隔線
print("-" * 30)

# 計算總缺失值數量並輸出
print("TOTAL MISSING VALUES:", df.isna().sum().sum())

# 輸出特徵矩陣的形狀（行數、列數）
print(X.shape)


# 將資料集分割成訓練集和測試集，其中 test_size=0.1 表示將 10% 的資料用於測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)





# 使用 KFold 進行十折交叉驗證，設定隨機種子為 42，並進行資料洗牌
kfolds = KFold(n_splits=10, shuffle=True, random_state=42)

# 建立模型評估函式，計算模型的均方根誤差（RMSE）
def cv_rmse(model, X=X):
    # 使用交叉驗證計算負的均方根誤差（Negative Mean Squared Error）
    rmse = np.sqrt(-cross_val_score(model, X, y,
                                    scoring="neg_mean_squared_error",
                                    cv=kfolds))
    return (rmse)



# 輸出一條訊息，表示機器學習的過程開始運行
print('START ML', datetime.now())

# 創建隨機森林回歸模型
random_forest = RandomForestRegressor(n_estimators=100)

# 創建梯度提升回歸模型
gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =42)

score = cv_rmse(random_forest)
print("RandomForestRegressor score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )
                                
score = cv_rmse(gbr)
print("GradientBoosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



# 輸出訓練開始的提示訊息
print('START Fit')

# 隨機森林模型的訓練
print(datetime.now(), 'random_forest')
rdf_model_full_data = random_forest.fit(X_train, y_train)
joblib.dump(rdf_model_full_data, os.path.join(folder_path, "rdf_regression_model_external.joblib"))

# 將訓練後的模型保存到指定資料夾中
print(datetime.now(), 'GradientBoosting')
gbr_model_full_data = gbr.fit(X_train, y_train)
joblib.dump(gbr_model_full_data, os.path.join(folder_path, "gbr_regression_model_external.joblib"))



def calculate_mape(actual_values, predicted_values):
    # 檢查輸入數組的長度是否相等
    if len(actual_values) != len(predicted_values):
        raise ValueError("actual_values 和 predicted_values 的長度必須相等")

    # 計算 MAPE
    mape = (1 / len(actual_values)) * np.sum(np.abs((actual_values - predicted_values) / actual_values)) * 100
    return mape



def Combine_models_predict_test():
    # 返回兩個模型預測結果的平均值（平均加權）
    return ((0.5 * gbr_model_full_data.predict(X_test))+ \
            (0.5 * rdf_model_full_data.predict(X_test)))


def Combine_models_predict_train():
    # 返回兩個模型預測結果的平均值（平均加權）
    return ((0.5 * gbr_model_full_data.predict(X_train))+ \
            (0.5 * rdf_model_full_data.predict(X_train)))



# 輸出當前日期和時間
print(datetime.now())

# 輸出梯度提升模型在測試集上的 MAPE 分數
print('MAPE score on gbr test data:')
print(calculate_mape(y_test, gbr_model_full_data.predict(X_test)))

# 輸出梯度提升模型在訓練集上的 MAPE 分數
print('MAPE score on gbr train data:')
print(calculate_mape(y_train, gbr_model_full_data.predict(X_train)))

# 輸出隨機森林模型在測試集上的 MAPE 分數
print('MAPE score on rdf test data:')
print(calculate_mape(y_test, rdf_model_full_data.predict(X_test)))

# 輸出隨機森林模型在訓練集上的 MAPE 分數
print('MAPE score on rdf train data:')
print(calculate_mape(y_train, rdf_model_full_data.predict(X_train)))

# 輸出結合模型在測試集上的 MAPE 分數
print('MAPE score on Combine test data:')
print(calculate_mape(y_test, Combine_models_predict_test()))

# 輸出結合模型在訓練集上的 MAPE 分數
print('MAPE score on Combine train data:')
print(calculate_mape(y_train, Combine_models_predict_train()))

# 輸出當前日期和時間
print(datetime.now())
