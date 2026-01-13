"""
模型训练模块
从 buildmodel.py 提取的训练逻辑，供后端调用
"""
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '12'

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", None)
pd.set_option('display.max_colwidth', None)
pd.options.display.float_format = '{:,.4f}'.format

class XYZ_Meta():
    def __init__(self):
        super(XYZ_Meta,self).__init__()
        self.transaction_cost_Y = 0.0002
        self.transaction_cost_X = 0
        self.kernel = "rbf"
        self.Lambda = [50, 5]
        
        self.transaction_cost_lgbm_filter = 0.002
        self.lgbm_learning_rate = 0.005
        self.lgbm_min_child_samples = 50
        self.lgbm_reg_lambda = 0.02
        self.lgbm_unbalance_ratio = 5
        
        self.transaction_cost_ksvm_filter = 0.002
        self.ksvm_gamma = 0.4
        self.ksvm_unbalance_ratio = 5

    def utility_function(self, sequence, transaction_cost):
        sequence = np.array(sequence)
        n = len(sequence)
        n1 = np.sum(sequence != 0)
        ave_ann_return = (np.mean(sequence) - transaction_cost * n1 / n) * 252
        ave_daily_return_per_trade = (np.sum(sequence) / (n1 + 0.001) - transaction_cost)
        ave_ann_return_per_trade = ave_daily_return_per_trade * 252
        sequence_non_null = [i for i in sequence if i != 0]
        
        if len(sequence_non_null) == 0:
            sharpe_ratio_trade = 0
        else:
            sharpe_ratio_trade = ave_daily_return_per_trade / (np.std(sequence_non_null) + 0.0001) * np.sqrt(252)
            
        cut = 3
        cut_2 = 5
        if sharpe_ratio_trade < cut:
            uti = sharpe_ratio_trade + self.Lambda[0] * ave_ann_return + self.Lambda[1] * ave_daily_return_per_trade * 252
        elif sharpe_ratio_trade < cut_2:
            uti = cut + (sharpe_ratio_trade - cut) ** 0.6 + self.Lambda[0] * ave_ann_return + self.Lambda[1] * ave_daily_return_per_trade * 252
        else:
            uti = cut + (cut_2 - cut) ** 0.6 + (sharpe_ratio_trade - cut_2) ** 0.3 + self.Lambda[0] * ave_ann_return + self.Lambda[1] * ave_daily_return_per_trade * 252

        return uti, sharpe_ratio_trade, ave_ann_return, ave_ann_return_per_trade
    
    def one_dimensional_filter(self, W, Y):
        nquantiles = 100
        grid_up = pd.Series(list(range(int(0.1 * nquantiles), int(0.9 * nquantiles)))) / nquantiles
        grid_down = pd.Series(list(range(int(0.1 * nquantiles), int(0.9 * nquantiles)))) / nquantiles
        c_up = np.quantile(W, grid_up)
        c_down = np.quantile(W, grid_down)
        utility_up = []
        utility_down = []
        
        for x in c_up:
            Y_filter = Y.copy()
            Y_filter.loc[W < x] = 0
            uti_temp, sharpe_ratio_trade, ave_ann_return, ave_ann_return_per_trade = self.utility_function(Y_filter, self.transaction_cost_Y)
            utility_up.append(uti_temp)
  
        for x in c_down:
            Y_filter = Y.copy()
            Y_filter.loc[W > x] = 0
            uti_temp, sharpe_ratio_trade, ave_ann_return, ave_ann_return_per_trade = self.utility_function(Y_filter, self.transaction_cost_Y)
            utility_down.append(uti_temp)

        i_up = np.argmax(utility_up)
        i_down = np.argmax(utility_down)
        
        if np.max(utility_up) > np.max(utility_down):
            Y_filter = Y.copy()
            Y_filter.loc[W < c_up[i_up]] = 0
            uti_temp, sharpe_ratio_trade, ave_ann_return, ave_ann_return_per_trade = self.utility_function(Y_filter, self.transaction_cost_Y)
            return c_up[i_up], grid_up[i_up], 1, sharpe_ratio_trade, ave_ann_return, ave_ann_return_per_trade, Y_filter
        else:
            Y_filter = Y.copy()
            Y_filter.loc[W > c_down[i_down]] = 0
            uti_temp, sharpe_ratio_trade, ave_ann_return, ave_ann_return_per_trade = self.utility_function(Y_filter, self.transaction_cost_Y)
            return c_down[i_down], grid_down[i_down], -1, sharpe_ratio_trade, ave_ann_return, ave_ann_return_per_trade, Y_filter

    def LGBM_filter_Weighted(self, Y, W):
        lgbm = LGBMClassifier(
            learning_rate=self.lgbm_learning_rate, 
            min_child_samples=self.lgbm_min_child_samples, 
            reg_lambda=self.lgbm_reg_lambda, 
            verbose=-1
        )
        label = np.sign(Y - self.transaction_cost_Y)
        weight = abs(np.multiply(Y - self.transaction_cost_Y, (label - (self.lgbm_unbalance_ratio - 1) / (1 + self.lgbm_unbalance_ratio))))
        lgbm.fit(W, label, sample_weight=weight.values.reshape(-1))
        return lgbm
    
    def Kernel_SVM_Weighted(self, Y, W):
        clf = svm.SVC(kernel=self.kernel, gamma=self.ksvm_gamma, probability=True, random_state=42)
        label = np.sign(Y-self.transaction_cost_Y) # compared with NO-trading
        unbalance_ratio = self.ksvm_unbalance_ratio
        weight = abs(np.multiply(Y-self.transaction_cost_Y, (label - (unbalance_ratio-1)/(1+unbalance_ratio))))
        clf.fit(W, label, sample_weight= weight)
        return clf

    def summarizePrint(self, sequence):
        sequence1 = sequence.loc[sequence.values.reshape(-1) != 0]
        annual_mean_r = (np.sum(sequence) - self.transaction_cost_Y * len(sequence1)) / len(sequence) * 252
        if len(sequence1) > 0:
            mean_r_trade = np.mean(sequence1)
            sharpe_per_trade = (np.mean(sequence1) - self.transaction_cost_Y) / (np.std(sequence1) + 0.0001) * np.sqrt(252)
            win_rate = np.sum(sequence1 > self.transaction_cost_Y) / len(sequence1)
        else:
            mean_r_trade = 0
            sharpe_per_trade = 0
            win_rate = 0
            
        percent_of_trade = len(sequence1) / len(sequence) 
        
        # 返回字典以保持向后兼容（供server.py使用）
        return {
            'annual_mean_r': annual_mean_r,
            'mean_r_trade': mean_r_trade,
            'sharpe_per_trade': sharpe_per_trade,
            'win_rate': win_rate,
            'percent_of_trade': percent_of_trade
        }

def train_final_model(W, Y, model_type, XYZ_Model):
    """训练单个模型"""
    if model_type == "LGBM":
        model_temp = XYZ_Model.LGBM_filter_Weighted(Y, W)
    elif model_type == "KSVM":
        model_temp = XYZ_Model.Kernel_SVM_Weighted(Y, W)
    else:
        raise ValueError("Model type must be 'LGBM' or 'KSVM'")
    
    pred_proba = model_temp.predict_proba(W)
    pos_proba = pred_proba[:, 1]
    
    cut_off, trade_percent, sig, sharpe_ratio_trade, ave_ann_return, ave_ann_return_per_trade, Y_filter = XYZ_Model.one_dimensional_filter(pos_proba, Y)
    
    summary = XYZ_Model.summarizePrint(Y_filter)
    
    return model_temp, cut_off, sig, summary

def train_final_two_sides_models(W, Y, model_type, XYZ_Model):
    """训练双向模型（做多和做空）"""
    model_long, cutoff_long, sig_long, summary_long = train_final_model(W, Y, model_type, XYZ_Model)
    model_short, cutoff_short, sig_short, summary_short = train_final_model(W, -Y, model_type, XYZ_Model)
    
    return model_long, model_short, cutoff_long, cutoff_short, summary_long, summary_short

def train_models_from_excel(excel_path, startTime='0400', endTime='mid', unbalance_ratio=5):
    """
    从Excel文件训练模型
    
    参数:
        excel_path: Excel文件路径
        startTime: 开始时间，默认'0400'
        endTime: 结束时间，默认'mid'（与buildmodel.py保持一致）
        unbalance_ratio: 不平衡比例，默认5（已弃用，保留以保持向后兼容，实际使用类内部参数）
    
    返回:
        dict: 包含模型和训练结果的字典
    """
    print("开始训练交易模型...")
    
    # 读取数据并记录初始规模
    xyzReturnsFinal = pd.read_excel(excel_path)
    xyzReturnsFinal = xyzReturnsFinal.sort_values(by='date_day', ascending=True)
    total_rows_before = len(xyzReturnsFinal)
    numericalVars = [x for x in xyzReturnsFinal.columns if ('date' not in x) and ('Date' not in x)]
    xyzReturnsFinal[numericalVars] = xyzReturnsFinal[numericalVars].astype(float)
    
    print('数据形状: %s' % str(xyzReturnsFinal.shape))
    print('数据期间: %s 到 %s' % (xyzReturnsFinal['date_day'].min(), xyzReturnsFinal['date_day'].max()))
    print('总天数: %s' % len(xyzReturnsFinal['date_day'].unique()))
    print('开始时间: %s, 结束时间: %s' % (startTime, endTime))
    
    varY = 'y_a50f_' + startTime + '_' + endTime
    varW = ['w_a50f_1500t1_' + startTime, 
            'w_usdrmb_1500t1_' + startTime + '_raw',
            'w_CSI_1445t1_1500', 
            'w_CSI_1500t2_1500t1']
    
    # 准备数据并统计过滤
    drop_info = []
    # 过滤因变量缺失
    missing_y = xyzReturnsFinal[varY].isna().sum()
    if missing_y > 0:
        drop_info.append({"field": varY, "reason": "缺失值 / missing", "dropped": int(missing_y)})
    xyzReturnsFinal = xyzReturnsFinal[(~xyzReturnsFinal[varY].isna())]
    # 过滤自变量缺失
    for w_field in ['w_CSI_1445t1_1500', 'w_CSI_1500t2_1500t1']:
        if w_field in varW:
            missing_w = xyzReturnsFinal[w_field].isna().sum()
            if missing_w > 0:
                drop_info.append({"field": w_field, "reason": "缺失值 / missing", "dropped": int(missing_w)})
            xyzReturnsFinal = xyzReturnsFinal[(~xyzReturnsFinal[w_field].isna())]
    
    total_rows_after = len(xyzReturnsFinal)
    total_rows_dropped = total_rows_before - total_rows_after
    
    W_all = xyzReturnsFinal[varW]
    Y_all = xyzReturnsFinal[varY]
    
    # 训练LightGBM模型
    print("\n1. 训练LightGBM模型...")
    XYZ_Model_lgbm = XYZ_Meta()
    model_long_lgbm, model_short_lgbm, cutoff_long_lgbm, cutoff_short_lgbm, summary_long_lgbm, summary_short_lgbm = train_final_two_sides_models(
        W_all, Y_all, 'LGBM', XYZ_Model_lgbm)
    
    # 训练KSVM模型  
    print("\n2. 训练KSVM模型...")
    XYZ_Model_ksvm = XYZ_Meta()
    model_long_ksvm, model_short_ksvm, cutoff_long_ksvm, cutoff_short_ksvm, summary_long_ksvm, summary_short_ksvm = train_final_two_sides_models(
        W_all, Y_all, 'KSVM', XYZ_Model_ksvm)
    
    # 构建返回结果
    model_dict = {
        'model_long_lgbm': model_long_lgbm,
        'model_short_lgbm': model_short_lgbm,
        'model_long_ksvm': model_long_ksvm,
        'model_short_ksvm': model_short_ksvm,
        'cutoff_long_lgbm': cutoff_long_lgbm,
        'cutoff_short_lgbm': cutoff_short_lgbm,
        'cutoff_long_ksvm': cutoff_long_ksvm,
        'cutoff_short_ksvm': cutoff_short_ksvm,
        'varW': varW,
        'varY': varY
    }
    
    # 训练统计信息
    training_stats = {
        'data_shape': xyzReturnsFinal.shape,
        'rows': {
            'before': int(total_rows_before),
            'after': int(total_rows_after),
            'dropped': int(total_rows_dropped)
        },
        'drop_info': drop_info,
        'date_range': {
            'start': str(xyzReturnsFinal['date_day'].min()),
            'end': str(xyzReturnsFinal['date_day'].max()),
            'total_days': len(xyzReturnsFinal['date_day'].unique())
        },
        'lgbm_long': summary_long_lgbm,
        'lgbm_short': summary_short_lgbm,
        'ksvm_long': summary_long_ksvm,
        'ksvm_short': summary_short_ksvm
    }
    
    print("\n模型训练完成!")
    
    return model_dict, training_stats

