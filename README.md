# 量化交易信号预测系统

这是一个量化交易信号预测系统，包含前端界面和后端预测服务。

## 功能特性

1. **模型训练**：上传Excel数据文件，训练LightGBM和KSVM模型
2. **信号预测**：输入4个关键因子，获取交易信号（做多/做空/观望）

## 项目结构

```
量化新项目/
├── frontend/          # Vue 3 前端界面
├── train_model.py     # 模型训练模块
├── server.py         # Flask 后端服务器
├── requirements.txt  # Python 依赖
└── README.md         # 本文件
```

## 快速开始

### 1. 安装后端依赖

```bash
pip install -r requirements.txt
```

### 2. 启动后端服务器

```bash
python server.py
```

服务器将在 `http://127.0.0.1:8000` 启动。

**注意**：首次使用时，需要通过前端上传Excel文件训练模型。

### 3. 启动前端

在另一个终端窗口中：

```bash
cd frontend
npm install
npm run dev
```

前端将在 `http://localhost:5173` 启动。

### 4. 使用流程

1. **训练模型**：
   - 在网页上点击"选择Excel文件"，选择包含交易数据的Excel文件
   - 点击"开始训练模型"按钮
   - 等待训练完成（可能需要几分钟）

2. **预测信号**：
   - 训练完成后，在"信号预测"区域输入4个因子值
   - 点击"计算信号"按钮
   - 查看预测结果

## API 接口

### POST /train

上传Excel文件训练模型。

**请求格式：**
- Content-Type: `multipart/form-data`
- 参数：
  - `file`: Excel文件（.xlsx 或 .xls）
  - `startTime`: 开始时间（可选，默认'0400'）
  - `endTime`: 结束时间（可选，默认'0929'）
  - `unbalance_ratio`: 不平衡比例（可选，默认5）

**响应格式：**
```json
{
  "success": true,
  "message": "模型训练完成",
  "stats": {
    "data_shape": [1000, 20],
    "date_range": {
      "start": "2023-01-01",
      "end": "2024-12-31",
      "total_days": 365
    },
    "lgbm_long": {...},
    "lgbm_short": {...},
    "ksvm_long": {...},
    "ksvm_short": {...}
  }
}
```

### POST /predict

预测交易信号接口。

**请求格式：**
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

**响应格式：**
```json
{
  "decision": 1.5,
  "prob_long_lgbm": 0.71,
  "prob_short_lgbm": 0.12,
  "prob_long_ksvm": 0.68,
  "prob_short_ksvm": 0.15,
  "long_decision_lgbm": 1,
  "short_decision_lgbm": 0,
  "long_decision_ksvm": 1,
  "short_decision_ksvm": 0
}
```

- `decision`: 建议仓位
  - `1.5`: 双多头信号（两个模型都看多）
  - `1`: 多头信号（一个模型看多）
  - `0.5`: 弱多头信号
  - `-0.5`: 弱空头信号
  - `-1`: 空头信号（一个模型看空）
  - `-1.5`: 双空头信号（两个模型都看空）
  - `0`: 观望

### GET /health

健康检查接口，返回服务器状态和模型加载状态。

## 故障排除

### 问题：无法连接到后端服务

1. **检查后端是否启动**
   - 确认 `python server.py` 已运行
   - 检查终端是否有错误信息

2. **检查端口是否被占用**
   - 后端默认使用 8000 端口
   - 如果端口被占用，可以修改 `server.py` 中的端口号

3. **检查数据文件是否存在**
   - 确认 `a50futures_filter_20mins_stop_20250615.csv` 文件在项目根目录

4. **检查依赖是否安装**
   - 运行 `pip install -r requirements.txt` 安装所有依赖

5. **查看浏览器控制台**
   - 打开浏览器开发者工具（F12）
   - 查看 Console 和 Network 标签页的错误信息

### 问题：模型训练失败

- 检查数据文件格式是否正确
- 确认所有 Python 依赖都已正确安装
- 查看服务器终端的错误信息

## 开发说明

- 后端使用 Flask 框架
- 前端使用 Vue 3 + Vite
- 模型使用 LightGBM 和 KSVM 进行集成预测

