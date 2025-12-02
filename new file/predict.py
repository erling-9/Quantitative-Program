import pickle
import numpy as np

# 加载模型
with open('trading_models.pkl', 'rb') as f:
    model_dict = pickle.load(f)

# 预测函数
def predict_signal_with_debug(features):
    """
    预测信号并输出详细的调试信息
    """
    if (model_dict['model_long_lgbm'] is None or model_dict['model_short_lgbm'] is None or 
        model_dict['model_long_ksvm'] is None or model_dict['model_short_ksvm'] is None):
        print("错误：模型未训练完成，请先运行完整的训练流程")
        return 0
    
    # LightGBM 预测
    prob_long_lgbm = model_dict['model_long_lgbm'].predict_proba(np.array(features).reshape(1, -1))[0, 1]
    prob_short_lgbm = model_dict['model_short_lgbm'].predict_proba(np.array(features).reshape(1, -1))[0, 1]
    
    # KSVM 预测
    prob_long_ksvm = model_dict['model_long_ksvm'].predict_proba(np.array(features).reshape(1, -1))[0, 1]
    prob_short_ksvm = model_dict['model_short_ksvm'].predict_proba(np.array(features).reshape(1, -1))[0, 1]

    # 决策
    long_decision_lgbm  = 1 if prob_long_lgbm >= model_dict['cutoff_long_lgbm'] else 0
    short_decision_lgbm = -1 if prob_short_lgbm >= model_dict['cutoff_short_lgbm'] else 0
    long_decision_ksvm  = 1 if prob_long_ksvm >= model_dict['cutoff_long_ksvm'] else 0
    short_decision_ksvm = -1 if prob_short_ksvm >= model_dict['cutoff_short_ksvm'] else 0

    # 添加调试信息
    print("=" * 60)
    print("模型预测详细调试信息:")
    print("=" * 60)
    print(f"LGBM - 看多概率: {prob_long_lgbm:.4f} (阈值: {model_dict['cutoff_long_lgbm']:.4f}), 决策: {'看多' if long_decision_lgbm else '不看多'}")
    print(f"LGBM - 看空概率: {prob_short_lgbm:.4f} (阈值: {model_dict['cutoff_short_lgbm']:.4f}), 决策: {'看空' if short_decision_lgbm else '不看空'}")
    print(f"KSVM - 看多概率: {prob_long_ksvm:.4f} (阈值: {model_dict['cutoff_long_ksvm']:.4f}), 决策: {'看多' if long_decision_ksvm else '不看多'}")
    print(f"KSVM - 看空概率: {prob_short_ksvm:.4f} (阈值: {model_dict['cutoff_short_ksvm']:.4f}), 决策: {'看空' if short_decision_ksvm else '不看空'}")
    print("-" * 60)
    
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

    return decision

# 使用示例
features_today = [0.004585,-0.000418,-0.000425,-0.001264]
signal_today = predict_signal_with_debug
signal_today = predict_signal_with_debug(features_today)  # 调用函数并传入参数
print(signal_today)  # 打印函数的返回值