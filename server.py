"""
量化交易信号预测服务器
提供 POST /train 接口用于训练模型
提供 POST /predict 接口用于预测交易信号
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from datetime import datetime
import os
import sys
import pickle
from werkzeug.utils import secure_filename
import json
import uuid
import time

# 导入训练模块
from train_model import train_models_from_excel

# 导入数据预测模块（从 new file 目录）
import sys
import importlib.util
new_file_path = os.path.join(os.path.dirname(__file__), 'new file', 'data predict.py')
spec = importlib.util.spec_from_file_location("data_predict", new_file_path)
data_predict_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_predict_module)

# 导入需要的函数和类
DataCollector = data_predict_module.DataCollector
get_usdcnh_data = data_predict_module.get_usdcnh_data
get_xina50_data = data_predict_module.get_xina50_data
get_hs300_data = data_predict_module.get_hs300_data
calculate_parameters = data_predict_module.calculate_parameters
load_trading_model = data_predict_module.load_trading_model
predict_signal_with_debug = data_predict_module.predict_signal_with_debug

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 配置上传文件
UPLOAD_FOLDER = 'uploads'
MODEL_FILE = 'trading_models.pkl'
ALLOWED_EXTENSIONS = {'xlsx', 'xls'}
LOG_DIR = 'logs'
LOG_FILE = os.path.join(LOG_DIR, 'prediction_logs.jsonl')

# 确保上传文件夹和日志目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 全局变量存储模型
models_loaded = False
model_dict = None
_model_file_mtime = None  # 记录模型文件的最后修改时间

def load_models(force_reload=False):
    """从pkl文件加载模型
    
    Args:
        force_reload: 是否强制重新加载，忽略缓存
    """
    global models_loaded, model_dict, _model_file_mtime
    
    # 获取进程 ID，用于调试多进程问题
    import os as os_module
    pid = os_module.getpid()
    
    try:
        if not os.path.exists(MODEL_FILE):
            print(f"[DEBUG PID={pid}] 模型文件 {MODEL_FILE} 不存在，请先训练模型")
            models_loaded = False
            model_dict = None
            return False
        
        # 检查文件修改时间，如果文件更新了，强制重新加载
        current_mtime = os.path.getmtime(MODEL_FILE)
        current_mtime_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_mtime))
        
        print(f"[DEBUG PID={pid}] load_models 调用: force_reload={force_reload}, models_loaded={models_loaded}, _model_file_mtime={_model_file_mtime}")
        print(f"[DEBUG PID={pid}] 当前模型文件修改时间: {current_mtime} ({current_mtime_str})")
        
        if not force_reload and models_loaded and model_dict is not None:
            if _model_file_mtime is not None and current_mtime == _model_file_mtime:
                # 文件未更新，且模型已加载，直接返回
                print(f"[DEBUG PID={pid}] 模型文件未更新，使用内存中的模型 (mtime={current_mtime})")
                return True
            elif _model_file_mtime is not None and current_mtime != _model_file_mtime:
                # 文件已更新，需要重新加载
                old_mtime_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(_model_file_mtime))
                print(f"[DEBUG PID={pid}] ⚠️ 检测到模型文件已更新！")
                print(f"[DEBUG PID={pid}]   旧时间: {_model_file_mtime} ({old_mtime_str})")
                print(f"[DEBUG PID={pid}]   新时间: {current_mtime} ({current_mtime_str})")
                print(f"[DEBUG PID={pid}]   强制重新加载模型...")
                force_reload = True
        
        # 加载模型
        print(f"[DEBUG PID={pid}] 正在加载模型...")
        with open(MODEL_FILE, 'rb') as f:
            model_dict = pickle.load(f)
        
        models_loaded = True
        _model_file_mtime = current_mtime
        print(f"[DEBUG PID={pid}] ✅ 模型加载完成！mtime={current_mtime} ({current_mtime_str})")
        
        # 记录模型的一些关键信息用于调试
        if model_dict and 'cutoff_long_lgbm' in model_dict:
            print(f"[DEBUG PID={pid}] 模型信息: cutoff_long_lgbm={model_dict.get('cutoff_long_lgbm', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"[DEBUG PID={pid}] ❌ 加载模型时出错: {e}")
        import traceback
        traceback.print_exc()
        models_loaded = False
        model_dict = None
        return False


def append_predict_log(factors, result, request_obj):
    """将预测调用记录到本地 jsonl 日志文件，便于后续前端展示"""
    record = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "date": datetime.utcnow().strftime("%Y-%m-%d"),
        "client_ip": request_obj.remote_addr,
        "user_agent": request_obj.headers.get("User-Agent"),
        "factors": factors,
        "result": result,
    }
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        # 不影响主流程，打印到 stderr 便于排查
        print(f"写入预测日志失败: {e}", file=sys.stderr)

def predict_signal(features):
    """使用训练好的模型进行预测"""
    global model_dict
    
    if not models_loaded or model_dict is None:
        raise ValueError("模型尚未加载")
    
    # 检查输入特征
    features_array = np.array(features)
    if np.isnan(features_array).any() or np.isinf(features_array).any():
        raise ValueError("输入特征包含 NaN 或无穷大值")
    
    # 使用训练好的模型进行预测
    prob_long_lgbm = model_dict['model_long_lgbm'].predict_proba(features_array.reshape(1, -1))[0, 1]
    prob_short_lgbm = model_dict['model_short_lgbm'].predict_proba(features_array.reshape(1, -1))[0, 1]
    prob_long_ksvm = model_dict['model_long_ksvm'].predict_proba(features_array.reshape(1, -1))[0, 1]
    prob_short_ksvm = model_dict['model_short_ksvm'].predict_proba(features_array.reshape(1, -1))[0, 1]
    
    # 决策逻辑（匹配 predict.py）
    long_decision_lgbm = 1 if prob_long_lgbm >= model_dict['cutoff_long_lgbm'] else 0
    short_decision_lgbm = -1 if prob_short_lgbm >= model_dict['cutoff_short_lgbm'] else 0
    long_decision_ksvm = 1 if prob_long_ksvm >= model_dict['cutoff_long_ksvm'] else 0
    short_decision_ksvm = -1 if prob_short_ksvm >= model_dict['cutoff_short_ksvm'] else 0
    
    # ensembler 决策逻辑（参考 predict.py）
    lgbm_votes = long_decision_lgbm + short_decision_lgbm 
    ksvm_votes = long_decision_ksvm + short_decision_ksvm

    if lgbm_votes + ksvm_votes == 2:
        decision = 1.5
    elif lgbm_votes + ksvm_votes == -2:
        decision = -1.5
    elif lgbm_votes + ksvm_votes == -1:
        decision = -1
    elif lgbm_votes + ksvm_votes == 1:
        decision = 1
    else:
        decision = 0
    
    return {
        'decision': decision,
        'prob_long_lgbm': float(prob_long_lgbm),
        'prob_short_lgbm': float(prob_short_lgbm),
        'prob_long_ksvm': float(prob_long_ksvm),
        'prob_short_ksvm': float(prob_short_ksvm),
        'threshold_long_lgbm': float(model_dict['cutoff_long_lgbm']),
        'threshold_short_lgbm': float(model_dict['cutoff_short_lgbm']),
        'threshold_long_ksvm': float(model_dict['cutoff_long_ksvm']),
        'threshold_short_ksvm': float(model_dict['cutoff_short_ksvm']),
        'long_decision_lgbm': int(long_decision_lgbm),
        'short_decision_lgbm': int(short_decision_lgbm),
        'long_decision_ksvm': int(long_decision_ksvm),
        'short_decision_ksvm': int(short_decision_ksvm)
    }

@app.route('/train', methods=['POST'])
def train():
    """训练模型接口"""
    global models_loaded, model_dict
    
    try:
        # 检查是否有文件上传
        if 'file' not in request.files:
            return jsonify({'error': '没有上传文件'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': '文件名为空'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': '文件格式不支持，请上传 Excel 文件 (.xlsx 或 .xls)'}), 400
        
        # 保存上传的文件
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        print(f"收到训练请求，文件: {filename}")
        
        # 获取训练参数（强制固定为 0400-mid，防止被前端参数覆盖）
        startTime = '0400'
        endTime = 'mid'
        unbalance_ratio = 5.0  # 保留占位，不再从外部读取
        
        # 训练模型
        print("开始训练模型...")
        model_dict, training_stats = train_models_from_excel(
            filepath, 
            startTime=startTime, 
            endTime=endTime, 
            unbalance_ratio=unbalance_ratio
        )
        
        # 保存模型
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(model_dict, f)
        
        # 更新全局变量（当前 worker 进程）
        models_loaded = True
        global _model_file_mtime
        _model_file_mtime = os.path.getmtime(MODEL_FILE)
        mtime_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(_model_file_mtime))
        pid = os.getpid()
        print(f"[DEBUG PID={pid}] ✅ 模型训练完成并已保存")
        print(f"[DEBUG PID={pid}]   文件路径: {MODEL_FILE}")
        print(f"[DEBUG PID={pid}]   文件修改时间: {_model_file_mtime} ({mtime_str})")
        print(f"[DEBUG PID={pid}]   当前进程已更新模型，其他 worker 进程会在下次预测时自动检测并重新加载")
        
        # 清理上传的文件
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify({
            'success': True,
            'message': '模型训练完成',
            'stats': training_stats
        }), 200
        
    except Exception as e:
        print(f"训练模型时出错: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'训练失败: {str(e)}'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """预测接口"""
    try:
        data = request.get_json()
        
        if not data or 'factors' not in data:
            return jsonify({'error': '请求格式错误，需要 factors 字段'}), 400
        
        factors = data['factors']
        
        # 验证输入
        if not isinstance(factors, list) or len(factors) != 4:
            return jsonify({'error': '需要提供4个因子'}), 400
        
        # 提取因子值，按照前端发送的顺序
        # 前端发送的顺序是：
        # w_a50f_1500t1_0400, w_usdrmb_1500t1_0400_raw, w_CSI_1445t1_1500, w_CSI_1500t2_1500t1
        feature_values = []
        for factor in factors:
            if 'name' not in factor or 'value' not in factor:
                return jsonify({'error': '每个因子需要包含 name 和 value 字段'}), 400
            feature_values.append(float(factor['value']))
        
        # 确保模型已加载（每次预测前都检查模型文件是否更新）
        pid = os.getpid()
        print(f"[DEBUG PID={pid}] /predict 请求: 开始预测")
        print(f"[DEBUG PID={pid}]   输入特征: {feature_values}")
        
        if not load_models(force_reload=False):
            return jsonify({'error': '模型未加载，请先训练模型'}), 500
        
        # 记录当前使用的模型信息
        if model_dict and 'cutoff_long_lgbm' in model_dict:
            print(f"[DEBUG PID={pid}] 预测时使用的模型: cutoff_long_lgbm={model_dict.get('cutoff_long_lgbm', 'N/A')}")
            print(f"[DEBUG PID={pid}] 模型文件修改时间: {_model_file_mtime}")
        
        # 进行预测
        result = predict_signal(feature_values)
        print(f"[DEBUG PID={pid}] 预测结果: decision={result.get('decision', 'N/A')}")
        append_predict_log(factors=data.get("factors", []), result=result, request_obj=request)
        
        return jsonify(result), 200
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print(f"预测时出错: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'服务器内部错误: {str(e)}'}), 500


@app.route('/logs', methods=['GET'])
def get_logs():
    """获取预测调用日志，默认返回当天的记录"""
    date_str = request.args.get(
        'date', datetime.utcnow().strftime("%Y-%m-%d"))
    limit = int(request.args.get('limit', 200))
    records = []

    if not os.path.exists(LOG_FILE):
        return jsonify({"logs": [], "date": date_str}), 200

    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if item.get("date") == date_str:
                        records.append(item)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        return jsonify({"error": f"读取日志失败: {str(e)}"}), 500

    records = sorted(records, key=lambda x: x.get("timestamp", ""), reverse=True)
    if limit > 0:
        records = records[:limit]
    return jsonify({"logs": records, "date": date_str}), 200

@app.route('/predict_auto', methods=['POST'])
def predict_auto():
    """自动预测接口：接收日期和合约月份，自动下载数据、计算参数并预测"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': '请求格式错误'}), 400
        
        # 验证必需参数
        day1_date = data.get('day1_date', '').strip()
        day2_date = data.get('day2_date', '').strip()
        day3_date = data.get('day3_date', '').strip()
        contract_month = data.get('contract_month', '').strip()
        
        if not all([day1_date, day2_date, day3_date, contract_month]):
            return jsonify({'error': '缺少必需参数：day1_date, day2_date, day3_date, contract_month'}), 400
        
        # 验证日期格式
        try:
            from datetime import datetime
            datetime.strptime(day1_date, "%Y%m%d")
            datetime.strptime(day2_date, "%Y%m%d")
            datetime.strptime(day3_date, "%Y%m%d")
            datetime.strptime(contract_month, "%Y%m")
        except ValueError as e:
            return jsonify({'error': f'日期格式错误，应为 YYYYMMDD 或 YYYYMM: {str(e)}'}), 400
        
        print(f"[预测自动接口] 收到请求: day1={day1_date}, day2={day2_date}, day3={day3_date}, contract={contract_month}")
        
        # 创建数据收集器
        collector = DataCollector()
        
        # 获取USD/CNH数据
        print("[预测自动接口] 获取USD/CNH数据...")
        get_usdcnh_data(collector, day2_date, '15:00:00')
        get_usdcnh_data(collector, day3_date, '04:00:00')
        
        # 获取XINA50数据
        print("[预测自动接口] 获取XINA50数据...")
        get_xina50_data(collector, day2_date, '15:00:00', contract_month)
        get_xina50_data(collector, day3_date, '04:00:00', contract_month)
        
        # 获取HS300数据
        print("[预测自动接口] 获取HS300数据...")
        get_hs300_data(collector, day1_date, day2_date)
        
        # 计算参数
        print("[预测自动接口] 计算参数...")
        parameters = calculate_parameters(collector, day1_date, day2_date, day3_date)
        
        # 检查参数是否完整
        if not all(key in parameters for key in ['W1', 'W2', 'W3', 'W4']):
            missing_params = [key for key in ['W1', 'W2', 'W3', 'W4'] if key not in parameters]
            return jsonify({
                'error': f'无法计算完整特征参数，缺失: {missing_params}',
                'collected_data': collector.get_all_data(),
                'parameters': parameters
            }), 400
        
        # 准备特征
        features = [float(parameters['W1']), float(parameters['W2']), float(parameters['W3']), float(parameters['W4'])]
        
        # 加载模型
        model_dict_local = load_trading_model()
        if not model_dict_local:
            return jsonify({'error': '模型未加载，请先训练模型'}), 500
        
        # 进行预测（使用 predict_signal_with_debug 但需要修改返回格式）
        # 先使用现有的 predict_signal 函数
        if not load_models(force_reload=False):
            return jsonify({'error': '模型未加载，请先训练模型'}), 500
        
        # 使用全局 model_dict 进行预测
        result = predict_signal(features)
        
        # 记录日志（使用计算出的参数作为 factors）
        factors_for_log = [
            {'name': 'W1', 'value': str(parameters['W1'])},
            {'name': 'W2', 'value': str(parameters['W2'])},
            {'name': 'W3', 'value': str(parameters['W3'])},
            {'name': 'W4', 'value': str(parameters['W4'])}
        ]
        append_predict_log(factors=factors_for_log, result=result, request_obj=request)
        
        # 返回结果
        return jsonify({
            'decision': result['decision'],
            'prob_long_lgbm': result['prob_long_lgbm'],
            'prob_short_lgbm': result['prob_short_lgbm'],
            'prob_long_ksvm': result['prob_long_ksvm'],
            'prob_short_ksvm': result['prob_short_ksvm'],
            'threshold_long_lgbm': result['threshold_long_lgbm'],
            'threshold_short_lgbm': result['threshold_short_lgbm'],
            'threshold_long_ksvm': result['threshold_long_ksvm'],
            'threshold_short_ksvm': result['threshold_short_ksvm'],
            'long_decision_lgbm': result['long_decision_lgbm'],
            'short_decision_lgbm': result['short_decision_lgbm'],
            'long_decision_ksvm': result['long_decision_ksvm'],
            'short_decision_ksvm': result['short_decision_ksvm'],
            'parameters': {
                'W1': str(parameters['W1']),
                'W2': str(parameters['W2']),
                'W3': str(parameters['W3']),
                'W4': str(parameters['W4'])
            },
            'collected_data': collector.get_all_data()
        }), 200
        
    except Exception as e:
        print(f"自动预测时出错: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'服务器内部错误: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health():
    """健康检查接口"""
    return jsonify({
        'status': 'ok',
        'models_loaded': models_loaded
    }), 200

if __name__ == '__main__':
    print("=" * 60)
    print("量化交易信号预测服务器")
    print("=" * 60)
    
    # 尝试加载已存在的模型
    if os.path.exists(MODEL_FILE):
        print("发现已存在的模型文件，正在加载...")
        load_models()
    else:
        print("未找到模型文件，请先通过 /train 接口训练模型")
    
    print("=" * 60)
    print("服务器启动在 http://127.0.0.1:8000")
    print("API 接口:")
    print("  - POST /train        : 上传Excel文件训练模型")
    print("  - POST /predict      : 预测交易信号（手动输入参数）")
    print("  - POST /predict_auto : 自动预测（输入日期和合约月份）")
    print("  - GET  /health       : 健康检查")
    print("  - GET  /logs         : 获取预测日志")
    print("=" * 60)
    
    app.run(host='127.0.0.1', port=8000, debug=True)

