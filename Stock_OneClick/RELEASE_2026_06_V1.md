# Stock OneClick - 2026年6月正式版 v1

定版时间：2026-06-02

## 版本名称

2026 6月正式版 v1

## 当前确定规则

- `00 大环境` 和 `01 市场环境` 保留在 watchlist 中，但不进入扫描触发样本、日期 sheet、RawSignals 样本输出、TV 导入 txt。
- `观海买点分` 已加入买入样本表，范围 0-100，并按分数从高到低排序。
- 同一天同时出现多个 BUY rule，例如 `正式买入 + 预警买入`，会重点标注。
- 14 个交易日内重复触发：
  - 原始 D0 行继续记录 `retrigger_dates`。
  - 重复触发当天也单独出现在当天日期 sheet。
  - `prior_14d_signal_dates` 显示之前 14 天内触发日期和当时分数，例如 `2026-05-22 (98)`。
- Excel 输出前 6 个 sheet 固定：
  1. `Summary`
  2. `RawSignals`
  3. `买入观察列表`
  4. `买入历史记录`
  5. `卖出观察列表`
  6. `卖出历史记录`
- 日期 sheet 从第 7 个开始，按日期倒序排列，最新日期紧跟 `卖出历史记录`。
- 每日 BUY TradingView txt 输出保留：
  - 纯导入版
  - 备注版
- 支持从生命周期起点强制重扫：

```bash
STOCK_ONECLICK_NO_OPEN=1 STOCK_ONECLICK_RESCAN_FROM=2026-05-22 /usr/bin/env python3 scan_stocks.py
```

## 当前生命周期起点

`2026-05-22`

## 当前 TradingView Watchlist

https://www.tradingview.com/watchlists/323650703/

当前本地记录：

- Watchlist 名称：`全板块060126`
- 标的总数：153
- 实际扫描标的：131
- 跳过分组：`00 大环境`、`01 市场环境`

