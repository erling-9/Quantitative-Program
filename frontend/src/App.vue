<template>
  <div class="container">
    <header class="header">
      <h1>交易信号助手 (Trading Signal Assistant)</h1>
      <p>训练模型并预测交易信号 (Train models and predict signals)</p>
    </header>

    <!-- 模型训练区域 -->
    <section class="card">
      <h2>1. 模型训练 (Model Training)</h2>
      <p class="section-desc">上传Excel数据文件，训练交易模型 (Upload Excel to train)</p>
      
      <form @submit.prevent="handleTrain" class="form">
        <div class="form-field">
          <label class="label">
            选择Excel文件 (Choose Excel)
            <small>(.xlsx 或 .xls)</small>
          </label>
          <input
            type="file"
            ref="fileInput"
            @change="handleFileSelect"
            accept=".xlsx,.xls"
            required
            :disabled="isTraining"
          />
          <small v-if="selectedFile" class="file-info">
            已选择 (Selected): {{ selectedFile.name }}
          </small>
        </div>

        <div class="actions">
          <button type="submit" :disabled="!selectedFile || isTraining">
            <span v-if="isTraining" class="loader"></span>
            <span v-else>开始训练模型 (Start Training)</span>
          </button>
        </div>
      </form>

      <!-- 训练结果 -->
      <div v-if="trainingResult" class="training-result">
        <h3>训练结果 (Training Result)</h3>
        <div class="result-success">
          <p><strong>✓ 模型训练成功！(Training succeeded)</strong></p>
          <div v-if="trainingResult.stats" class="stats">
            <div class="stat-item">
              <span class="stat-label">数据期间 (Date range):</span>
              <span>{{ trainingResult.stats.date_range?.start }} 至 (to) {{ trainingResult.stats.date_range?.end }}</span>
            </div>
            <div class="stat-item">
              <span class="stat-label">总天数 (Total days):</span>
              <span>{{ trainingResult.stats.date_range?.total_days }}</span>
            </div>
            <div class="stat-item">
              <span class="stat-label">数据行数 (Row count):</span>
              <span>{{ trainingResult.stats.data_shape?.[0] }}</span>
            </div>
            <div class="stat-item" v-if="trainingResult.stats.rows">
              <span class="stat-label">有效行数 (Rows kept):</span>
              <span>{{ trainingResult.stats.rows.after }} / {{ trainingResult.stats.rows.before }}</span>
            </div>
            <div class="stat-item" v-if="trainingResult.stats.rows">
              <span class="stat-label">过滤行数 (Rows dropped):</span>
              <span>{{ trainingResult.stats.rows.dropped }}</span>
            </div>
            <div class="stat-item" v-if="trainingResult.stats.drop_info && trainingResult.stats.drop_info.length">
              <span class="stat-label">过滤原因 (Drop reasons):</span>
              <span class="drop-list">
                <div v-for="(item, idx) in trainingResult.stats.drop_info" :key="idx" class="drop-item">
                  {{ item.field }} - {{ item.reason }}: {{ item.dropped }}
                </div>
              </span>
            </div>
          </div>
        </div>
      </div>

      <div v-if="trainingError" class="error">
        <p><strong>训练失败 (Training failed):</strong> {{ trainingError }}</p>
      </div>
    </section>

    <!-- 信号预测区域 -->
    <section class="card">
      <h2>2. 信号预测 (Signal Prediction)</h2>
      <p class="section-desc">输入四个关键因子，获取交易信号 (Input 4 factors to get signal)</p>
      
      <form @submit.prevent="handleSubmit" class="form">
        <label
          v-for="field in fields"
          :key="field.key"
          class="form-field"
        >
          <span class="label">
            {{ field.label }}
            <small>({{ field.key }})</small>
          </span>
          <input
            v-model.number="factors[field.key]"
            type="number"
            step="0.0000000001"
            required
            :disabled="!models_loaded"
          />
        </label>

        <div v-if="!models_loaded" class="warning">
          <p>⚠️ 模型未加载，请先训练模型 (Model not loaded, please train first)</p>
        </div>

        <div class="actions">
          <button type="button" class="ghost" @click="fillExample" :disabled="!models_loaded">
            使用示例数据 (Use sample)
          </button>
          <button type="submit" :disabled="isLoading || !models_loaded">
            <span v-if="isLoading" class="loader"></span>
            <span v-else>计算信号 (Predict)</span>
          </button>
        </div>
      </form>
    </section>

    <!-- 预测结果 -->
    <section v-if="result" class="card result">
      <h2>预测结果 (Prediction)</h2>
      <p class="signal">
        <span :class="['badge', badgeClass]">{{ resultLabel }}</span>
      </p>
      <ul class="details">
        <li v-for="(value, key) in resultDetails" :key="key">
          <strong>{{ key }}：</strong>{{ value }}
        </li>
      </ul>
    </section>

    <section v-if="error" class="card error">
      <h2>出错了 (Error)</h2>
      <p>{{ error }}</p>
      <p class="hint">
        请确认后端已启动并暴露相应接口，或查看浏览器控制台。
        (Please ensure backend is running and check browser console.)
      </p>
    </section>

    <!-- 预测日志 -->
    <section class="card">
      <h2>3. 预测日志 (Prediction Logs)</h2>
      <p class="section-desc">默认展示当日调用记录，便于回溯与监督 (Show today calls for audit)</p>

      <div class="log-controls">
        <div class="log-date">
          <label>选择日期 (Pick date)</label>
          <input type="date" v-model="logDate" @change="fetchLogs" />
        </div>
        <button type="button" class="ghost" @click="fetchLogs" :disabled="logLoading">
          <span v-if="logLoading" class="loader"></span>
          <span v-else>刷新 (Refresh)</span>
        </button>
      </div>

      <div v-if="logError" class="error">
        <p><strong>日志获取失败 (Load logs failed)：</strong>{{ logError }}</p>
      </div>
      <div v-else-if="logLoading" class="log-loading">正在加载日志... (Loading logs...)</div>
      <div v-else>
        <div v-if="!logs.length" class="empty-log">当日暂无日志 (No logs today)</div>
        <table v-else class="log-table">
          <thead>
            <tr>
              <th>时间 (Time)</th>
              <th>信号 (Signal)</th>
              <th>因子输入 (Factors)</th>
              <th>客户端 (Client)</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="item in logs" :key="item.id">
              <td>{{ formatTime(item.timestamp) }}</td>
              <td>{{ decisionLabel(item.result?.decision) }}</td>
              <td class="factor-cell">{{ formatFactors(item.factors) }}</td>
              <td>{{ item.client_ip || '-' }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>
  </div>
</template>

<script setup>
import { reactive, ref, computed, onMounted } from 'vue';
import axios from 'axios';

const endpoint = import.meta.env.VITE_SIGNAL_ENDPOINT || 'http://127.0.0.1:8000';
const trainEndpoint = `${endpoint}/train`;
const predictEndpoint = `${endpoint}/predict`;
const healthEndpoint = `${endpoint}/health`;
const logEndpoint = `${endpoint}/logs`;

const fields = [
  { key: 'w_a50f_1500t1_0400', label: 'A50 指数动量 (A50 momentum 04:00-09:29)' },
  { key: 'w_usdrmb_1500t1_0400_raw', label: '美元人民币动量 (USD/CNY momentum)' },
  { key: 'w_CSI_1445t1_1500', label: '沪深300 (CSI300 14:45-15:00)' },
  { key: 'w_CSI_1500t2_1500t1', label: '沪深300 高频差分 (CSI300 high-frequency diff)' }
];

const factors = reactive(fields.reduce((acc, field) => {
  acc[field.key] = null;
  return acc;
}, {}));

const formatTime = (ts) => {
  try {
    return new Date(ts).toLocaleString();
  } catch {
    return ts || '-';
  }
};

const formatFactors = (items = []) => {
  if (!Array.isArray(items)) return '';
  return items.map((f) => {
    const val = Number(f.value);
    const safeVal = Number.isFinite(val) ? val.toFixed(10) : f.value;
    return `${f.name || ''}: ${safeVal}`;
  }).join(' | ');
};

const isLoading = ref(false);
const isTraining = ref(false);
const error = ref('');
const trainingError = ref('');
const result = ref(null);
const trainingResult = ref(null);
const selectedFile = ref(null);
const fileInput = ref(null);
const models_loaded = ref(false);
const logs = ref([]);
const logError = ref('');
const logLoading = ref(false);
const logDate = ref(new Date().toISOString().slice(0, 10));

// 检查模型状态
const checkModelStatus = async () => {
  try {
    const response = await axios.get(healthEndpoint);
    models_loaded.value = response.data.models_loaded || false;
  } catch (err) {
    console.error('检查模型状态失败 / Failed to check model status:', err);
    models_loaded.value = false;
  }
};

onMounted(() => {
  checkModelStatus();
  fetchLogs();
});

const handleFileSelect = (event) => {
  const file = event.target.files[0];
  if (file) {
    selectedFile.value = file;
    trainingError.value = '';
    trainingResult.value = null;
  }
};

const handleTrain = async () => {
  if (!selectedFile.value) {
    trainingError.value = '请选择文件 / Please choose a file';
    return;
  }

  isTraining.value = true;
  trainingError.value = '';
  trainingResult.value = null;

  try {
    const formData = new FormData();
    formData.append('file', selectedFile.value);
    formData.append('startTime', '0400');
    formData.append('endTime', '0929');
    formData.append('unbalance_ratio', '5');

    const response = await axios.post(trainEndpoint, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      },
      timeout: 300000 // 5分钟超时 / 5min timeout
    });

    trainingResult.value = response.data;
    models_loaded.value = true;
    
    // 清空文件选择
    if (fileInput.value) {
      fileInput.value.value = '';
    }
    selectedFile.value = null;

  } catch (err) {
    if (err.response) {
      trainingError.value = `服务端返回错误 (Server error)：${err.response.status} ${err.response.statusText} - ${err.response.data?.error || ''}`;
    } else if (err.request) {
      trainingError.value = '无法连接到后端服务，请确认服务已启动 / Cannot reach backend, please ensure it is running.';
    } else {
      trainingError.value = `请求发生异常 (Request exception)：${err.message}`;
    }
    console.error(err);
  } finally {
    isTraining.value = false;
  }
};

const badgeClass = computed(() => {
  if (!result.value) return '';
  const signal = result.value.decision;
  if (signal > 0) return 'long';
  if (signal < 0) return 'short';
  return 'flat';
});

const decisionLabel = (signal) => {
  if (signal === 1.5) return '双多头信号 1.5 (Double Long)';
  if (signal === 1) return '多头信号 1 (Long)';
  if (signal === 0.5) return '弱多头信号 0.5 (Weak Long)';
  if (signal === -0.5) return '弱空头信号 -0.5 (Weak Short)';
  if (signal === -1.5) return '双空头信号 -1.5 (Double Short)';
  if (signal === -1) return '空头信号 -1 (Short)';
  if (signal === 0) return '观望 0 (Flat)';
  return `${signal} (Signal)`;
};

const resultLabel = computed(() => {
  if (!result.value) return '';
  return decisionLabel(result.value.decision);
});

const resultDetails = computed(() => {
  if (!result.value) return {};
  const { decision, ...rest } = result.value;
  return {
    '决策权重 (Decision weight)': decision,
    'LGBM看多概率 (Long prob)': (rest.prob_long_lgbm * 100).toFixed(2) + '%',
    'LGBM阈值 (Long threshold)': Number(rest.threshold_long_lgbm)?.toFixed(10),
    'LGBM看空概率 (Short prob)': (rest.prob_short_lgbm * 100).toFixed(2) + '%',
    'LGBM阈值 (Short threshold)': Number(rest.threshold_short_lgbm)?.toFixed(10),
    'KSVM看多概率 (Long prob)': (rest.prob_long_ksvm * 100).toFixed(2) + '%',
    'KSVM阈值 (Long threshold)': Number(rest.threshold_long_ksvm)?.toFixed(10),
    'KSVM看空概率 (Short prob)': (rest.prob_short_ksvm * 100).toFixed(2) + '%',
    'KSVM阈值 (Short threshold)': Number(rest.threshold_short_ksvm)?.toFixed(10),
  };
});

const payload = () => ({
  factors: fields.map(field => ({
    name: field.key,
    value: Number(factors[field.key])
  }))
});

const fillExample = () => {
  const example = [0.0062871234, -0.0003519876, -0.0014854567, 0.0018532468];
  fields.forEach((field, idx) => {
    factors[field.key] = Number(example[idx].toFixed(10));
  });
};

const validate = () => {
  for (const field of fields) {
    const value = factors[field.key];
    if (value === null || Number.isNaN(value)) {
      return `请输入 ${field.label} / Please enter ${field.label}`;
    }
  }
  return '';
};

const handleSubmit = async () => {
  error.value = '';
  result.value = null;
  const validationError = validate();
  if (validationError) {
    error.value = validationError;
    return;
  }

  isLoading.value = true;
  try {
    const response = await axios.post(predictEndpoint, payload());
    result.value = response.data;
    // 刷新当日日志
    fetchLogs();
  } catch (err) {
    if (err.response) {
      error.value = `服务端返回错误 (Server error)：${err.response.status} ${err.response.statusText} - ${err.response.data?.error || ''}`;
    } else if (err.request) {
      error.value = '无法连接到后端服务，请确认服务已启动 / Cannot reach backend, please ensure it is running.';
    } else {
      error.value = `请求发生异常 (Request exception)：${err.message}`;
    }
    console.error(err);
  } finally {
    isLoading.value = false;
  }
};

const fetchLogs = async () => {
  logLoading.value = true;
  logError.value = '';
  try {
    const resp = await axios.get(`${logEndpoint}?date=${logDate.value}`);
    logs.value = resp.data.logs || [];
  } catch (err) {
    if (err.response) {
      logError.value = `读取日志失败 (Load logs failed)：${err.response.status} ${err.response.statusText} - ${err.response.data?.error || ''}`;
    } else if (err.request) {
      logError.value = '无法连接到后端日志接口 / Cannot reach log API.';
    } else {
      logError.value = `请求发生异常 (Request exception)：${err.message}`;
    }
    console.error(err);
  } finally {
    logLoading.value = false;
  }
};
</script>

<style scoped>
.container {
  width: min(1400px, 100%);
  display: grid;
  gap: 24px;
}

.header {
  text-align: center;
}

.header h1 {
  margin: 0;
  font-size: 2rem;
  color: #1f2933;
}

.header p {
  margin-top: 8px;
  color: #52606d;
}

.card {
  background: #ffffff;
  border-radius: 16px;
  padding: 24px;
  box-shadow: 0 12px 32px rgba(15, 23, 42, 0.08);
}

.card h2 {
  margin: 0 0 8px;
  color: #1f2933;
  font-size: 1.5rem;
}

.section-desc {
  margin: 0 0 20px;
  color: #64748b;
  font-size: 0.9rem;
}

.form {
  display: grid;
  gap: 16px;
}

.form-field {
  display: grid;
  gap: 8px;
}

.label {
  font-weight: 600;
  color: #364152;
  display: flex;
  align-items: baseline;
  justify-content: space-between;
}

.label small {
  font-weight: 400;
  color: #64748b;
  font-size: 0.85rem;
}

input[type="file"] {
  padding: 8px;
  border: 1px solid #d9e2ec;
  border-radius: 12px;
  font-size: 0.95rem;
}

input[type="number"] {
  appearance: none;
  border: 1px solid #d9e2ec;
  border-radius: 12px;
  padding: 12px 16px;
  font-size: 1rem;
  transition: border-color 0.2s, box-shadow 0.2s;
}

input:focus {
  outline: none;
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.25);
}

input:disabled {
  background-color: #f1f5f9;
  cursor: not-allowed;
}

.file-info {
  color: #64748b;
  font-size: 0.9rem;
}

.warning {
  padding: 12px;
  background: rgba(251, 191, 36, 0.1);
  border: 1px solid rgba(251, 191, 36, 0.3);
  border-radius: 8px;
  color: #92400e;
}

.warning p {
  margin: 0;
  font-size: 0.9rem;
}

.actions {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
  margin-top: 8px;
}

button {
  border: none;
  border-radius: 999px;
  padding: 10px 24px;
  font-size: 0.95rem;
  cursor: pointer;
  transition: transform 0.15s ease, box-shadow 0.15s ease;
  background: linear-gradient(135deg, #2563eb, #1d4ed8);
  color: #fff;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
}

button:hover:not(:disabled) {
  transform: translateY(-1px);
  box-shadow: 0 10px 20px rgba(37, 99, 235, 0.25);
}

button:disabled {
  opacity: 0.65;
  cursor: not-allowed;
}

.ghost {
  background: transparent;
  color: #2563eb;
  border: 1px solid #93c5fd;
  box-shadow: none;
}

.ghost:hover:not(:disabled) {
  background: rgba(37, 99, 235, 0.08);
}

.loader {
  width: 16px;
  height: 16px;
  border-radius: 50%;
  border: 2px solid rgba(255, 255, 255, 0.4);
  border-top-color: #fff;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

.training-result {
  margin-top: 20px;
  padding: 16px;
  background: rgba(34, 197, 94, 0.05);
  border: 1px solid rgba(34, 197, 94, 0.2);
  border-radius: 12px;
}

.training-result h3 {
  margin: 0 0 12px;
  color: #166534;
}

.result-success {
  color: #166534;
}

.stats {
  margin-top: 12px;
  display: grid;
  gap: 8px;
}

.stat-item {
  display: flex;
  justify-content: space-between;
  padding: 8px 0;
  border-bottom: 1px solid rgba(34, 197, 94, 0.1);
}

.stat-item:last-child {
  border-bottom: none;
}

.stat-label {
  font-weight: 600;
}

.drop-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.drop-item {
  color: #334155;
  font-size: 0.95rem;
}

.result h2,
.error h2 {
  margin: 0 0 12px;
  color: #1f2933;
}

.signal {
  margin: 0 0 12px;
}

.badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 6px 16px;
  border-radius: 999px;
  font-weight: 600;
  font-size: 1rem;
}

.badge.long {
  background: rgba(34, 197, 94, 0.16);
  color: #166534;
}

.badge.short {
  background: rgba(239, 68, 68, 0.16);
  color: #b91c1c;
}

.badge.flat {
  background: rgba(148, 163, 184, 0.16);
  color: #475569;
}

.details {
  margin: 0;
  padding-left: 20px;
  color: #475569;
}

.error {
  border: 1px solid rgba(239, 68, 68, 0.2);
}

.hint {
  margin-top: 8px;
  color: #b91c1c;
}

.log-controls {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  gap: 12px;
  margin-top: 12px;
  margin-bottom: 12px;
  flex-wrap: wrap;
}

.log-date {
  display: flex;
  flex-direction: column;
  gap: 6px;
  color: #364152;
}

.log-date input[type="date"] {
  border: 1px solid #d9e2ec;
  border-radius: 10px;
  padding: 10px 12px;
}

.log-table {
  width: 100%;
  border-collapse: collapse;
}

.log-table th,
.log-table td {
  padding: 10px 12px;
  border-bottom: 1px solid #e5e7eb;
  text-align: left;
  vertical-align: top;
}

.log-table th {
  background: #f8fafc;
  color: #475569;
}

.factor-cell {
  word-break: break-word;
  color: #334155;
}

.empty-log,
.log-loading {
  color: #475569;
  padding: 8px 0;
}

@media (max-width: 768px) {
  .container {
    gap: 16px;
  }

  .card {
    padding: 20px;
  }

  .actions {
    flex-direction: column;
    align-items: stretch;
  }
}
</style>
