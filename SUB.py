import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from scipy.stats import skew
from scipy.stats import yeojohnson
from sklearn.preprocessing import MinMaxScaler
import joblib


# 讀取公開資料集
df = pd.read_csv(r"Code\30_Public Dataset_Public Sumission Template_v2\updated_PDistance_data.csv")

# 讀取需預測的資料集
df_pr = pd.read_csv(r"Code\30_Private Dataset _Private and Publict Submission Template_v2\Uupdated_PRDistance_dataset.csv")

# 載入模型路徑
gbr_model_full_data = joblib.load(r"Code\Models\gbr_regression_model_external.joblib")
rdf_model_full_data = joblib.load(r"CCode\Models\rdf_regression_model_external.joblib")






# 選取數值型變數欄位
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

# 計算相關性矩陣
correlation_matrix = df[numeric_columns].corr()

# 繪製熱圖顯示相關性
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, cmap="RdBu")
plt.title("Correlations Between Variables", size=15)
plt.show()

# 根據訓練所帶入的特徵定義重要的數值型變數
important_num_cols = ["土地面積","移轉層次","總樓層數","屋齡","建物面積","車位面積","橫坐標","縱坐標","主建物面積","附屬建物面積",
                        "ATM","便利商店","公車站點","國中","國小","大學","捷運站點","火車站點","腳踏車站點","郵局據點","醫療機構","金融機構","高中",
                        "國中距離","國小距離","大學距離","捷運站點距離","火車站點距離","郵局據點距離","醫療機構距離","金融機構距離","高中距離"]

# 定義類別型變數                             
cat_cols = ["縣市","主要用途","主要建材","建物型態"]
# 結合數值型和類別型的所有重要變數
important_cols = important_num_cols + cat_cols
# 定義子集的變數
Separate_cols = ["移轉層次","總樓層數","屋齡","公車站點","捷運站點","金融機構","高中","腳踏車站點","橫坐標","縱坐標"]

# 截取只包含重要變數的資料框
df = df[important_cols]
df_pr = df_pr[important_cols]

# 將 X 設為 df 的子集
X = df
# 將 X_pr 設為 df_pr 的子集
X_pr = df_pr

# 對類別型變數進行獨熱編碼
X = pd.get_dummies(X, columns=cat_cols)
X_pr = pd.get_dummies(X_pr, columns=cat_cols)

# 選取子集 X_Separate
X_Separate = X[Separate_cols]

# 使用 PolynomialFeatures 進行多項式特徵擴展
poly = PolynomialFeatures(degree=2)  # 在這個示例中，我們設置多項式的次數為2
X_poly = poly.fit_transform(X_Separate)

# 將多項式特徵擴展後的結果轉換為 DataFrame
X_poly = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(X_Separate.columns))

# 刪除已經被多項式特徵擴展的原始特徵列
X = X.drop(Separate_cols, axis=1)

# 將多項式特徵擴展後的結果與原始資料框進行串聯
X = pd.concat([X, X_poly], axis=1)

# 使用 StandardScaler 對數值型變數進行標準化
scaler = StandardScaler()
X[important_num_cols] = scaler.fit_transform(X[important_num_cols])

# 刪除指定的欄位
cols_to_drop = ["車位面積", "附屬建物面積", "大學", "火車站點", "捷運站點距離"]
X = X.drop(cols_to_drop, axis=1)






# 選取子集 X_pr_Separate
X_pr_Separate = X_pr[Separate_cols]

# 使用 PolynomialFeatures 進行多項式特徵擴展
poly = PolynomialFeatures(degree=2)
X_pr_poly = poly.fit_transform(X_pr_Separate)

# 將多項式特徵擴展後的結果轉換為 DataFrame
X_pr_poly = pd.DataFrame(X_pr_poly, columns=poly.get_feature_names_out(X_pr_Separate.columns))

# 刪除已經被多項式特徵擴展的原始特徵列
X_pr = X_pr.drop(Separate_cols, axis=1)

# 將多項式特徵擴展後的結果與原始資料框進行串聯
X_pr = pd.concat([X_pr, X_pr_poly], axis=1)

# 使用 StandardScaler 對數值型變數進行標準化
scaler = StandardScaler()
X_pr[important_num_cols] = scaler.fit_transform(X_pr[important_num_cols])

# 刪除指定的欄位
X_pr = X_pr.drop(cols_to_drop, axis=1)




# 結合模型對公開資料集進行預測
def Combine_models_predict_PU():
    return ((0.5 * gbr_model_full_data.predict(X)) + (0.5 * rdf_model_full_data.predict(X)))

# 結合模型對私人資料集進行預測
def Combine_models_predict_PR():
    return ((0.5 * gbr_model_full_data.predict(X_pr)) + (0.5 * rdf_model_full_data.predict(X_pr)))



# 使用載入的模型進行預測
new_predictions = Combine_models_predict_PU()
pr_predictions = Combine_models_predict_PR()

# 將 NumPy 陣列轉換為 DataFrame
new_predictions_df = pd.DataFrame(new_predictions)
pr_predictions_df = pd.DataFrame(pr_predictions)

# 刪除 pr_predictions_df 的最後一行
pr_predictions_df = pr_predictions_df.drop(pr_predictions_df.tail(1).index)

# 在 pr_predictions_df 後添加一個空行，然後再添加 new_predictions_df
combined_predictions = pd.concat([new_predictions_df, pd.DataFrame([[]]), pr_predictions_df])

# 將合併的預測結果保存到 CSV 檔案
combined_predictions.to_csv('submission.csv', index=False)