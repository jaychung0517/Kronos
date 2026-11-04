Kronos-base v2 | 台灣加權指數 + 台指期 TX 全日盤對照實驗
功能
直接延續你在 GitHub 已成功的原始方案。Kronos-base 只載入一次，每天對 TAIEX (^TWII) 和 TX 全日盤做四組敏感度實驗：
組別	歷史 K 線	sample_count
A	200	1
B	200	5
C	400	1
D	400	5


每組產生未來 5 個交易日的 OHLC，報告列出第 1 / 3 / 5 日收盤價與累積報酬。共 8 次模型呼叫，但只載入一次權重。

不要將單日四組比較當作歷史回測，它只比較當前截面對不同設定的敏感度。Kronos 官方 sample_count=5 內部會將五條隨機抽樣路徑平均，程式並沒有保存五條個別路徑或估計其信賴區間。
在 GitHub 網頁安裝（不用本機 Python）
1. 開啟你 Fork 的 jaychung0517/Kronos 儲存庫。
2. 上傳並覆蓋 scripts/taiwan_forecast.py。
3. 上傳新增檔案 tests/test_taiwan_experiments.py。
4. 覆蓋 .github/workflows/taiwan_kronos.yml（不要另外新增同名之外的 workflow，避免每天重複跑）。
5. 保留 Settings → Secrets and variables → Actions 中已使用的 FINMIND_TOKEN。
6. Actions → Kronos Base - Taiwan Markets → Run workflow。
7. 從執行 Summary 的 Artifacts 下載 kronos-taiwan-v2-results。
workflow 改成台灣時間週一至週五 17:10（UTC 09:10）預定啟動，以增加 FinMind 收盤後更新的緩衝。GitHub 不保證排程準時，休市日仍可能排到執行；行情過舊／不同步時程序會拒絕或停止，不應把過期輸出當成新行情。
檔案內容
- experiment_summary.csv：12 筆市場差距與 1 / 3 / 5 日對照資料，欄位有 review_status / gap_flag / tx_tail_flag。
- experiment_report.md：便於直接閱讀的繁中 Markdown 報告。
- historical_reference.csv：依兩市場歷史同日收盤，估計 1/3/5 日 TX 絕對報酬、期現貨報酬差的 99 百分位，用來「警示」異常。
- TAIEX_scenario_comparison.png、`TX_scenario_comparison.png`：四組預測漲跌幅曲線。
- TAIEX_A_forecast.csv … TAIEX_D_forecast.csv、`TX_A_forecast.csv` … TX_D_forecast.csv：各組五日預測 K 線。
- TAIEX_forecast.csv、`TX_forecast.csv` 及其 PNG：延續舊檔名，內容固定是 D 組 (400 / 5)，不是經過歷史驗證的最佳模型。
- taiex_history.csv、`tx_contract_bars.csv`、`tx_continuous_adjusted.csv`、`tx_roll_log.csv`：來源行情與換月檢驗紀錄。
- status.json：設定、完成狀態與完整例外資訊。
讀取結果
CSV 中 return_pct 為百分比，例如 -2.5 代表 -2.5%。`return_gap_pp` 為 TX 變動率減 TAIEX 變動率，單位是「百分點」。若 gap_flag 或 tx_tail_flag 為 True，代表超出最近約 400 根同日資料的第 99 百分位絕對值，顯示 REVIEW。這不是機率、勝率、準確率，亦不是交易訊號。
風險與尚待完成
- TX 使用 FinMind TaiwanFuturesDaily，前一夜盤與當日日盤合併；採舊版最近可用合約與 backward additive adjustment，轉倉規則仍需另行驗證。後調整歷史點位不等於可成交報價。
- 無 data/trading_calendar.csv 時，會以工作日推估未來日期，台灣休市日可能錯誤。正式模式先維護日曆，才將 STRICT_TRADING_CALENDAR 改為 1。
- 目前不使用交易量作為模型特徵，保留與前版一致的 OHLC 輸入，以便隔離「歷史長度」與「抽樣次數」的影響。
- 使用固定亂數種子；跨套件、CPU 或模型版本不保證位元完全一致。
- GitHub Actions 的 Artifacts 預設只保留 30 天；要做可靠樣本外回測，應將每日結果連同原始資料截止日存到長期資料庫。
- 目前沒有自動下單或買賣建議。
失敗時怎麼做
開啟 Actions → 失敗的 Run → Run Taiwan 4-scenario forecast，檢查 [FAILED]、`Traceback`。預測步驟即使失敗，仍會嘗試上傳 outputs/status.json。

執行序：`Validate Python syntax → Install → Check imports → Unit tests → Fetch market data → 8 forecasts → Upload`。
不用更換任何本機 Python
全部都由 GitHub 的 Ubuntu + Python 3.11 執行。先不要把 v2 當成精度提升已經成立；下一階段應建立 100～500 個歷史交易日的時間序列樣本外回測，評估方向正確率、MAE、極端預測發生率及相對基準模型的表現。
