# data/

本資料夾**不包含任何資料**。依中央研究院學術調查研究資料庫（SRDA）的使用規範，TIGPS 資料僅供申請者本人使用，不得複製或轉贈他人，因此原始資料與分析過程中產生的個人層級資料（`.dta`）皆不會上傳至此專案。

## 如何取得資料

請自行向 [SRDA](https://srda.sinica.edu.tw/) 申請「臺灣數位世代青少年的成長歷程追蹤研究」（TIGPS）第一波與第二波的**學生問卷**資料，並將 Stata 格式的檔案以下列檔名放在本資料夾：

| 檔名 | 內容 |
|---|---|
| `TIGPSw1_s.dta` | 第一波學生問卷 |
| `TIGPSw2_s.dta` | 第二波學生問卷 |

放好之後，依專案首頁 README 的順序執行 `scripts/` 中的程式即可。執行過程中產生的 `TIGPSw1w2_merged_raw.dta`、`TIGPS_clean.dta`、`TIGPS_quadrant_clean.dta` 也會存放在這裡，同樣不會上傳。
