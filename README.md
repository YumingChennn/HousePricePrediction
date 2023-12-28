# 永豐AI GO競賽-攻房戰

## 隊伍
- 維克多

## 競賽說明
詳細競賽說明請參閱 [競賽網頁](https://tbrain.trendmicro.com.tw/Competitions/Details/30)

## 參賽目標
- 了解 Machine Learning 的應用方法
- 透過 Machine Learning 進行房價預測
- 嘗試透過自己所學的能力，將準確度提高至最高水平。

## 整體架構
該專案主要由三大程式碼組成，包括：EDC、HPP與SUB。
- EDC全名為External Data Calculation，主要功能為將重要設施的經緯度與房屋的經緯度進行計算，並從中取得房屋附近重要設施的個數與距離。
- HPP全名為House Price Prediction，主要功能為將房屋的資訊，透過機器學習預測出其房價。
- SUB全名為Submission，主要功能為將預測結果轉換成主辦方要求的格式。

## 資料前處理

## 模型訓練與預測
- 剛開始我們嘗試過各種不同的模型，嘗試透過trial and error，選擇對於房價預測最佳的模型，最後我們選擇使用RandomForestRegressor(RDF)和GradientBoostingRegressor(GBR)模型進行預測，其中RDF與GBR各佔我們最終預測結果的一半，如圖一所示。

![image](https://github.com/YumingChennn/HousePricePrediction/assets/126893165/400dec84-faca-4f98-a023-e8fe300a2035)  
圖一 模型架構圖


## 創新想法
- 重要設施個數計算

利用提供房屋資訊的主要資料集，包括縣市、鄉鎮市區、路名、土地面積、使用分區、移轉層次、總樓層數、主要用途等，結合外部資料集如便利超商、學校、捷運站等，透過經緯度換算，計算每一個房子500公尺內的重要設施個數。

- 重要設施距離計算

在取得重要設施個數後，進一步計算每一個房子與最近的重要設施之間的距離。透過經緯度換算，對比房子的位置與重要設施的位置，獲得房子離最近的重要設施的距離。

P.S.  
資料集(由主辦方提供)
- 主要資料集：包括縣市、鄉鎮市區、路名、土地面積、使用分區、移轉層次、總樓層數、主要用途等。
- 外部資料集：包括便利超商、學校、捷運站等地點等

## 總結
- MAPE scor on
- Final MAPE score:8.97
- 比賽名次：145/972

![image](https://github.com/YumingChennn/HousePricePrediction/assets/126893165/5fd45c30-866a-4261-81e6-b2e5e6bc0eca)


## 使用說明書
