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

# 导入训练模块
from train_model import train_models_from_excel

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 配置上传文件
UPLOAD_FOLDER = 'uploads'
MODEL_FILE = 'trading_models.pkl'
ALLOWED_EXTENSIONS = {'xlsx', 'xls'}

# 确保上传文件夹存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 全局变量存储模型
models_loaded = False
model_dict = None

def load_models():
    """从pkl文件加载模型"""
    global models_loaded, model_dict
    
    if models_loaded and model_dict is not None:
        return True
    
    try:
        if not os.path.exists(MODEL_FILE):
            print(f"模型文件 {MODEL_FILE} 不存在，请先训练模型")
            return False
        
        print("正在加载模型...")
        with open(MODEL_FILE, 'rb') as f:
            model_dict = pickle.load(f)
        
        models_loaded = True
        print("模型加载完成！")
        return True
        
    except Exception as e:
        print(f"加载模型时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

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
        
        # 获取训练参数（可选）
        startTime = request.form.get('startTime', '0400')
        endTime = request.form.get('endTime', '0929')
        unbalance_ratio = float(request.form.get('unbalance_ratio', 5))
        
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
        
        models_loaded = True
        print("模型训练完成并已保存")
        
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
        
        # 确保模型已加载
        if not models_loaded:
            if not load_models():
                return jsonify({'error': '模型未加载，请先训练模型'}), 500
        
        # 进行预测
        result = predict_signal(feature_values)
        
        return jsonify(result), 200
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print(f"预测时出错: {e}")
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
    print("  - POST /train    : 上传Excel文件训练模型")
    print("  - POST /predict  : 预测交易信号")
    print("  - GET  /health   : 健康检查")
    print("=" * 60)
    
    app.run(host='127.0.0.1', port=8000, debug=True)

