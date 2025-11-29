# 交易信号前端

一个使用 Vue 3 + Vite 构建的简易界面，用于录入四个因子并展示交易信号结果。

## 快速开始

```bash
cd frontend
npm install
npm run dev
```

默认服务地址为 `http://127.0.0.1:8000/predict`，可通过在根目录创建 `.env` 或 `.env.local` 覆盖：

```bash
VITE_SIGNAL_ENDPOINT=https://your-server.example.com/predict
```

## 请求格式

前端会向后端发送如下结构的请求：

```json
{
  "factors": [
    { "name": "w_a50f_1500t1_0400", "value": 0.006287 },
    { "name": "w_usdrmb_1500t1_0400_raw", "value": -0.000351 },
    { "name": "w_CSI_1445t1_1500", "value": -0.001485 },
    { "name": "w_CSI_1500t2_1500t1", "value": 0.001853 }
  ]
}
```

后端应返回类似结构：

```json
{
  "decision": 1.5,
  "prob_long_lgbm": 0.71,
  "prob_short_lgbm": 0.12,
  "prob_long_ksvm": 0.68,
  "prob_short_ksvm": 0.15
}
```

- `decision`：建议仓位，1.5 表示多头加杠杆，1 表示普通多头，-1.5/-1 表示空头，0 表示观望。
- 其余字段会被展示在结果面板里，可根据实际返回字段增减。

