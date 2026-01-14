import os
os.environ['LOKY_MAX_CPU_COUNT'] = '12'

import pandas as pd
import datetime
import time
import threading
import pickle
import numpy as np
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ib_insync import IB, Contract as IBContract
from jqdatasdk import auth, get_price
import warnings
warnings.filterwarnings('ignore')

class DataCollector:
    def __init__(self):
        self.price_data = {}  # 存储价格数据
        self.param_data = {}  # 存储参数数据
    
    def add_data(self, source, date, time_str, price):
        """添加价格数据到字典"""
        key = f"{source}_{date}_{time_str}"
        self.price_data[key] = price
    
    def add_param(self, name, value):
        """添加参数到字典"""
        key = f"PARAM_{name}"
        self.param_data[key] = value
    
    def get_all_data(self):
        """获取所有数据（价格数据在前，参数数据在后）"""
        all_data = {**self.price_data, **self.param_data}
        return all_data
    
    def save_to_excel(self, base_date):
        """保存结果到Excel（参数列在最后）"""
        if not self.price_data and not self.param_data:
            print("无数据可保存")
            return None
        
        # 确保价格数据在前，参数数据在后
        sorted_price_keys = sorted(self.price_data.keys())
        sorted_param_keys = sorted(self.param_data.keys())
        
        # 创建一行数据的DataFrame
        row_data = {}
        
        # 添加价格数据
        for key in sorted_price_keys:
            row_data[key] = self.price_data[key]
        
        # 添加参数数据
        for key in sorted_param_keys:
            row_data[key] = self.param_data[key]
        
        df = pd.DataFrame([row_data])
        
        # 保存到Excel
        filename = f'数据结果_{base_date}.xlsx'
        df.to_excel(filename, sheet_name='汇总数据', index=False)
        print(f"\n数据已保存到: {filename}")
        return filename

class ForexApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.data_ready = threading.Event()
        self.all_data = []
        self.connection_error = None

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        """捕获 IB API 错误"""
        if errorCode == 502:  # 连接错误
            self.connection_error = f"无法连接到 TWS/Gateway: {errorString}"
        elif errorCode == 504:  # 未连接
            self.connection_error = f"未连接到 TWS/Gateway: {errorString}"
        # 不打印错误，由调用者处理

    def historicalData(self, reqId, bar):
        self.all_data.append(bar)

    def historicalDataEnd(self, reqId, start, end):
        self.data_ready.set()

def get_usdcnh_data(collector, target_date, target_time):
    """获取USD/CNH数据（修复版）"""
    print(f"获取USD/CNH {target_date} {target_time}...")
    
    # 创建连接
    app = ForexApp()
    try:
        try:
            app.connect('127.0.0.1', 7496, clientId=1)
        except Exception as conn_err:
            print(f"  ✗ 无法连接到 IB TWS/Gateway: {conn_err}")
            print(f"  请确保:")
            print(f"    1. IB TWS 或 IB Gateway 已启动")
            print(f"    2. 在 TWS 中已启用 API 设置 (Edit → Global Configuration → API → Settings)")
            print(f"    3. Socket port 设置为 7496 (实盘) 或 7497 (模拟)")
            return None
        threading.Thread(target=app.run, daemon=True).start()
        time.sleep(3)  # 增加等待时间，确保连接建立
        
        # 检查连接错误和连接状态
        if app.connection_error:
            print(f"  ✗ {app.connection_error}")
            print(f"  请确保:")
            print(f"    1. IB TWS 或 IB Gateway 已启动")
            print(f"    2. 在 TWS 中已启用 API 设置 (Edit → Global Configuration → API → Settings)")
            print(f"    3. Socket port 设置为 7496 (实盘) 或 7497 (模拟)")
            try:
                app.disconnect()
            except:
                pass
            return None
        
        # 检查是否真的连接成功
        if not app.isConnected():
            print(f"  ✗ 连接失败: 无法连接到 IB TWS/Gateway")
            print(f"  请确保:")
            print(f"    1. IB TWS 或 IB Gateway 已启动")
            print(f"    2. 在 TWS 中已启用 API 设置 (Edit → Global Configuration → API → Settings)")
            print(f"    3. Socket port 设置为 7496 (实盘) 或 7497 (模拟)")
            try:
                app.disconnect()
            except:
                pass
            return None
        
        # 设置合约
        contract = Contract()
        contract.symbol = "USD"
        contract.secType = "CASH"
        contract.exchange = "IDEALPRO"
        contract.currency = "CNH"
        
        # 多日期请求策略
        def get_previous_date(date_str):
            dt = datetime.datetime.strptime(date_str, "%Y%m%d")
            return (dt - datetime.timedelta(days=1)).strftime("%Y%m%d")
        
        def get_next_date(date_str):
            dt = datetime.datetime.strptime(date_str, "%Y%m%d")
            return (dt + datetime.timedelta(days=1)).strftime("%Y%m%d")
        
        request_dates = [
            target_date,
            get_previous_date(target_date),
            get_next_date(target_date)
        ]
        
        result_price = None
        
        for request_date in request_dates:
            app.data_ready.clear()
            app.all_data = []  # 清空之前的数据
            
            app.reqHistoricalData(
                reqId=1,
                contract=contract,
                endDateTime=f"{request_date}-23:59:59",
                durationStr="2 D",
                barSizeSetting="1 min",
                whatToShow="MIDPOINT",
                useRTH=0,
                formatDate=1,
                keepUpToDate=False,
                chartOptions=[]
            )
            
            if app.data_ready.wait(timeout=15):
                for bar in app.all_data[-1440:]:
                    if ' ' in bar.date and 'Asia/Hong_Kong' in bar.date:
                        date_part, time_part, _ = bar.date.split()
                        if date_part == target_date and time_part == target_time:
                            result_price = float(f"{bar.close:.5f}")
                            collector.add_data('USDCNH', date_part, time_part, result_price)
                            print(f"  ✓ 成功: {result_price}")
                            return result_price
            
            time.sleep(1)
        
        if result_price is None:
            print("  ✗ 未找到数据")
            
    except Exception as e:
        print(f"  ✗ 错误: {e}")
    finally:
        try:
            app.disconnect()
        except:
            pass
    
    return None

def get_xina50_data(collector, target_date, target_time, contract_month):
    """获取富时中国A50指数期货数据"""
    print(f"获取XINA50 {target_date} {target_time}...")
    
    import asyncio
    import nest_asyncio
    
    # 在Flask多线程环境中，允许嵌套事件循环
    try:
        nest_asyncio.apply()
    except ImportError:
        # 如果nest_asyncio未安装，尝试其他方法
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except:
        pass
    
    ib = IB()
    try:
        try:
            # ib_insync会自动处理事件循环
            ib.connect('127.0.0.1', 7496, clientId=2)
        except Exception as conn_err:
            error_msg = str(conn_err)
            if "event loop" in error_msg.lower() or "no current event loop" in error_msg.lower():
                # 如果是事件循环错误，尝试创建新的事件循环
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    ib.connect('127.0.0.1', 7496, clientId=2)
                except Exception as e2:
                    print(f"  ✗ 无法连接到 IB TWS/Gateway: {e2}")
                    print(f"  请确保:")
                    print(f"    1. IB TWS 或 IB Gateway 已启动")
                    print(f"    2. 在 TWS 中已启用 API 设置 (Edit → Global Configuration → API → Settings)")
                    print(f"    3. Socket port 设置为 7496 (实盘) 或 7497 (模拟)")
                    return None
            else:
                print(f"  ✗ 无法连接到 IB TWS/Gateway: {conn_err}")
                print(f"  请确保:")
                print(f"    1. IB TWS 或 IB Gateway 已启动")
                print(f"    2. 在 TWS 中已启用 API 设置 (Edit → Global Configuration → API → Settings)")
                print(f"    3. Socket port 设置为 7496 (实盘) 或 7497 (模拟)")
                return None
    except Exception as e:
        print(f"  ✗ 连接失败: {e}")
        return None
    
    try:
        # 设置合约
        contract = IBContract()
        contract.symbol = 'XINA50'
        contract.secType = 'FUT'
        contract.exchange = 'SGX'
        contract.currency = 'USD'
        contract.lastTradeDateOrContractMonth = contract_month
        
        # 解析目标时间
        date_obj = datetime.datetime.strptime(target_date, "%Y%m%d")
        time_obj = datetime.datetime.strptime(target_time, "%H:%M:%S")
        target_datetime = datetime.datetime(
            date_obj.year, date_obj.month, date_obj.day,
            time_obj.hour, time_obj.minute, time_obj.second
        )
        
        # 请求数据
        bars = ib.reqHistoricalData(
            contract,
            endDateTime=target_datetime.strftime('%Y%m%d %H:%M:%S'),
            durationStr='2 D',
            barSizeSetting='1 min',
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1
        )
        
        # 处理数据
        if bars:
            target_bar = None
            for bar in bars:
                bar_time = bar.date.replace(tzinfo=None)
                if bar_time <= target_datetime:
                    target_bar = bar
            
            if target_bar:
                price = float(f"{target_bar.close:.2f}")
                collector.add_data('XINA50', target_date, target_time, price)
                print(f"  ✓ 成功: {price}")
                return price
            else:
                print("  ✗ 未找到数据")
        else:
            print("  ✗ 无数据")
            
    except Exception as e:
        print(f"  ✗ 错误: {e}")
    finally:
        try:
            ib.disconnect()
        except:
            pass
    
    return None

def get_hs300_data(collector, day1_date, day2_date):
    """获取沪深300指数数据"""
    print(f"获取HS300数据...")
    print(f"  上一交易日: {day1_date} 15:00")
    print(f"  当日: {day2_date} 14:45 和 15:00")
    
    try:
        auth('13242834459', 'CXWcxw@12345')
        
        # 获取当日数据 (day2)
        date_str = f"{day2_date[:4]}-{day2_date[4:6]}-{day2_date[6:8]}"
        minute_data = get_price(
            '000300.XSHG',
            start_date=date_str,
            end_date=f'{date_str} 15:00:00',
            frequency='1m',
            fields=['close'],
            skip_paused=False
        )
        
        if not minute_data.empty:
            # 14:45:00
            time_1445 = f'{date_str} 14:45:00'
            if time_1445 in minute_data.index:
                price_1445 = float(f"{minute_data.loc[time_1445, 'close']:.2f}")
                collector.add_data('HS300', day2_date, "14:45:00", price_1445)
                print(f"  ✓ 当日14:45: {price_1445}")
            else:
                print(f"  ✗ 当日14:45: 无数据")
            
            # 15:00:00
            time_1500 = f'{date_str} 15:00:00'
            if time_1500 in minute_data.index:
                price_1500 = float(f"{minute_data.loc[time_1500, 'close']:.2f}")
                collector.add_data('HS300', day2_date, "15:00:00", price_1500)
                print(f"  ✓ 当日15:00: {price_1500}")
            else:
                last_price = float(f"{minute_data.iloc[-1]['close']:.2f}")
                collector.add_data('HS300', day2_date, "15:00:00", last_price)
                print(f"  ✓ 当日收盘价: {last_price}")
        else:
            print(f"  ✗ 当日无交易数据")
            return
        
        # 获取上一交易日数据 (day1)
        prev_date_str = f"{day1_date[:4]}-{day1_date[4:6]}-{day1_date[6:8]}"
        
        prev_day_data = get_price(
            '000300.XSHG',
            start_date=prev_date_str,
            end_date=f'{prev_date_str} 15:00:00',
            frequency='1m',
            fields=['close'],
            skip_paused=False
        )
        
        if not prev_day_data.empty:
            prev_time_1500 = f'{prev_date_str} 15:00:00'
            if prev_time_1500 in prev_day_data.index:
                prev_price_1500 = float(f"{prev_day_data.loc[prev_time_1500, 'close']:.2f}")
                collector.add_data('HS300', day1_date, "15:00:00", prev_price_1500)
                print(f"  ✓ 上一交易日15:00: {prev_price_1500}")
            else:
                prev_last_price = float(f"{prev_day_data.iloc[-1]['close']:.2f}")
                collector.add_data('HS300', day1_date, "15:00:00", prev_last_price)
                print(f"  ✓ 上一交易日收盘价: {prev_last_price}")
        else:
            print(f"  ✗ 上一交易日无数据")
                
    except Exception as e:
        print(f"  ✗ 错误: {e}")

def calculate_parameters(collector, day1_date, day2_date, day3_date):
    """计算W1-W4四个参数（精度8位小数）"""
    print(f"\n{'='*50}")
    print("开始计算参数")
    print(f"{'='*50}")
    
    # 辅助函数：从数据字典获取价格
    def get_price(source, date, time_str):
        key = f"{source}_{date}_{time_str}"
        return collector.price_data.get(key, None)
    
    parameters = {}
    
    # 计算W1: (A50次日4点 - A50当日3点) / A50当日3点
    a50_day2 = get_price("XINA50", day2_date, "15:00:00")
    a50_day3 = get_price("XINA50", day3_date, "04:00:00")
    if (a50_day2 is not None) and (a50_day3 is not None):
        w1 = (a50_day3 - a50_day2) / a50_day2
        parameters['W1'] = w1
        collector.add_param("W1", f"{w1:.8f}")
        print(f"W1 = ({a50_day3} - {a50_day2}) / {a50_day2} = {w1:.8f}")
        print(f"  使用: A50({day2_date} 15:00)={a50_day2:.2f}, A50({day3_date} 04:00)={a50_day3:.2f}")
    else:
        print(f"✗ W1 计算失败: 缺少A50数据")
    
    # 计算W2: (USDCNH次日4点 - USDCNH当日3点) / USDCNH当日3点
    usd_day2 = get_price("USDCNH", day2_date, "15:00:00")
    usd_day3 = get_price("USDCNH", day3_date, "04:00:00")
    if (usd_day2 is not None) and (usd_day3 is not None):
        w2 = (usd_day3 - usd_day2) / usd_day2
        parameters['W2'] = w2
        collector.add_param("W2", f"{w2:.8f}")
        print(f"W2 = ({usd_day3} - {usd_day2}) / {usd_day2} = {w2:.8f}")
        print(f"  使用: USDCNH({day2_date} 15:00)={usd_day2:.5f}, USDCNH({day3_date} 04:00)={usd_day3:.5f}")
    else:
        print(f"✗ W2 计算失败: 缺少USDCNH数据")
    
    # 计算W3: (CSI300当日3点 - CSI300当日2点45) / CSI300当日2点45
    hs300_1445 = get_price("HS300", day2_date, "14:45:00")
    hs300_1500 = get_price("HS300", day2_date, "15:00:00")
    if (hs300_1445 is not None) and (hs300_1500 is not None):
        w3 = (hs300_1500 - hs300_1445) / hs300_1445
        parameters['W3'] = w3
        collector.add_param("W3", f"{w3:.8f}")
        print(f"W3 = ({hs300_1500} - {hs300_1445}) / {hs300_1445} = {w3:.8f}")
        print(f"  使用: HS300({day2_date} 14:45)={hs300_1445:.2f}, HS300({day2_date} 15:00)={hs300_1500:.2f}")
    else:
        print(f"✗ W3 计算失败: 缺少HS300数据")
    
    # 计算W4: (CSI300当日3点 - CSI300前日3点) / CSI300前日3点
    hs300_prev = get_price("HS300", day1_date, "15:00:00")
    if (hs300_1500 is not None) and (hs300_prev is not None):
        w4 = (hs300_1500 - hs300_prev) / hs300_prev
        parameters['W4'] = w4
        collector.add_param("W4", f"{w4:.8f}")
        print(f"W4 = ({hs300_1500} - {hs300_prev}) / {hs300_prev} = {w4:.8f}")
        print(f"  使用: HS300({day1_date} 15:00)={hs300_prev:.2f}, HS300({day2_date} 15:00)={hs300_1500:.2f}")
    else:
        print(f"✗ W4 计算失败: 缺少HS300数据")
    
    print("参数计算完成")
    return parameters

def load_trading_model(model_path='trading_models.pkl'):
    """加载交易模型"""
    try:
        with open(model_path, 'rb') as f:
            model_dict = pickle.load(f)
        print(f"✓ 交易模型已加载: {model_path}")
        return model_dict
    except FileNotFoundError:
        print(f"✗ 模型文件未找到: {model_path}")
        return None
    except Exception as e:
        print(f"✗ 加载模型时出错: {e}")
        return None

def predict_signal_with_debug(features, model_dict):
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

    # 调试信息
    print("=" * 60)
    print("模型预测详细调试信息:")
    print("=" * 60)
    print(f"LGBM - 看多概率: {prob_long_lgbm:.4f} (阈值: {model_dict['cutoff_long_lgbm']:.4f}), 决策: {'看多' if long_decision_lgbm else '不看多'}")
    print(f"LGBM - 看空概率: {prob_short_lgbm:.4f} (阈值: {model_dict['cutoff_short_lgbm']:.4f}), 决策: {'看空' if short_decision_lgbm else '不看空'}")
    print(f"KSVM - 看多概率: {prob_long_ksvm:.4f} (阈值: {model_dict['cutoff_long_ksvm']:.4f}), 决策: {'看多' if long_decision_ksvm else '不看多'}")
    print(f"KSVM - 看空概率: {prob_short_ksvm:.4f} (阈值: {model_dict['cutoff_short_ksvm']:.4f}), 决策: {'看空' if short_decision_ksvm else '不看空'}")
    print("-" * 60)
    
    # 投票决策
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

def main(day1_date, day2_date, day3_date, contract_month):
    """主程序 - 手动指定三个日期和合约月份"""
    print(f"{'='*50}")
    print("数据收集 - 手动指定日期")
    print(f"{'='*50}")
    print(f"前一日 (day1): {day1_date}")
    print(f"当日 (day2): {day2_date}")
    print(f"下一日 (day3): {day3_date}")
    print(f"合约月份: {contract_month}")
    print(f"{'='*50}")
    
    # 创建数据收集器
    collector = DataCollector()
    
    # 获取USD/CNH数据
    print(f"\n1. USD/CNH数据:")
    get_usdcnh_data(collector, day2_date, '15:00:00')  # 当日15:00
    get_usdcnh_data(collector, day3_date, '04:00:00')  # 次日04:00
    
    # 获取XINA50数据
    print(f"\n2. XINA50数据:")
    get_xina50_data(collector, day2_date, '15:00:00', contract_month)  # 当日15:00
    get_xina50_data(collector, day3_date, '04:00:00', contract_month)  # 次日04:00
    
    # 获取HS300数据
    print(f"\n3. HS300数据:")
    get_hs300_data(collector, day1_date, day2_date)  # 前一日15:00, 当日14:45和15:00
    
    # 计算参数
    parameters = calculate_parameters(collector, day1_date, day2_date, day3_date)
    
    # 加载模型并进行预测
    if all(key in parameters for key in ['W1', 'W2', 'W3', 'W4']):
        # 准备特征
        features_today = [parameters['W1'], parameters['W2'], parameters['W3'], parameters['W4']]
        print(f"\n特征向量: {[f'{f:.8f}' for f in features_today]}")
        
        # 加载模型
        print(f"\n{'='*50}")
        print("加载交易模型")
        print(f"{'='*50}")
        model_dict = load_trading_model()
        
        if model_dict:
            # 进行预测
            print(f"\n进行交易信号预测...")
            signal_today = predict_signal_with_debug(features_today, model_dict)
            print(f"\n预测结果: {signal_today}")
            
            # 显示交易建议
            signal_descriptions = {
                1.5: "强烈看多信号",
                1: "看多信号",
                0: "中性信号",
                -1: "看空信号",
                -1.5: "强烈看空信号"
            }
            print(f"交易建议: {signal_descriptions[signal_today]}")
    else:
        print(f"\n✗ 无法计算完整特征参数，跳过交易信号预测")
        missing_params = [key for key in ['W1', 'W2', 'W3', 'W4'] if key not in parameters]
        print(f"  缺失的参数: {missing_params}")
    
    # 显示结果
    print(f"\n{'='*50}")
    print("数据收集完成")
    print(f"{'='*50}")
    
    if collector.price_data or collector.param_data:
        print("\n收集到的数据:")
        
        # 先显示价格数据（排序）
        price_keys = sorted(collector.price_data.keys())
        for key in price_keys:
            print(f"  {key}: {collector.price_data[key]}")
        
        # 再显示参数数据（排序）
        param_keys = sorted(collector.param_data.keys())
        for key in param_keys:
            print(f"  {key}: {collector.param_data[key]}")
    else:
        print("\n无数据")
    
    # 保存到Excel（使用 day2 作为基准日期）
    collector.save_to_excel(day2_date)
    
    print(f"\n程序执行完成!")

def get_user_input():
    """获取用户输入的日期和合约月份"""
    print("=" * 60)
    print("请输入预测参数")
    print("=" * 60)
    
    # 输入三个日期
    while True:
        day1_date = input("请输入前一日 (day1) 日期 (格式: YYYYMMDD，例如: 20251231): ").strip()
        if len(day1_date) == 8 and day1_date.isdigit():
            try:
                datetime.datetime.strptime(day1_date, "%Y%m%d")
                break
            except ValueError:
                print("✗ 日期格式错误，请重新输入")
        else:
            print("✗ 日期格式错误，请输入8位数字 (YYYYMMDD)")
    
    while True:
        day2_date = input("请输入当日 (day2) 日期 (格式: YYYYMMDD，例如: 20260105): ").strip()
        if len(day2_date) == 8 and day2_date.isdigit():
            try:
                datetime.datetime.strptime(day2_date, "%Y%m%d")
                break
            except ValueError:
                print("✗ 日期格式错误，请重新输入")
        else:
            print("✗ 日期格式错误，请输入8位数字 (YYYYMMDD)")
    
    while True:
        day3_date = input("请输入下一日 (day3) 日期 (格式: YYYYMMDD，例如: 20260106): ").strip()
        if len(day3_date) == 8 and day3_date.isdigit():
            try:
                datetime.datetime.strptime(day3_date, "%Y%m%d")
                break
            except ValueError:
                print("✗ 日期格式错误，请重新输入")
        else:
            print("✗ 日期格式错误，请输入8位数字 (YYYYMMDD)")
    
    # 输入合约月份
    while True:
        contract_month = input("请输入XINA50合约月份 (格式: YYYYMM，例如: 202601): ").strip()
        if len(contract_month) == 6 and contract_month.isdigit():
            try:
                datetime.datetime.strptime(contract_month, "%Y%m")
                break
            except ValueError:
                print("✗ 合约月份格式错误，请重新输入")
        else:
            print("✗ 合约月份格式错误，请输入6位数字 (YYYYMM)")
    
    print("\n" + "=" * 60)
    print("输入确认:")
    print(f"  前一日 (day1): {day1_date}")
    print(f"  当日 (day2): {day2_date}")
    print(f"  下一日 (day3): {day3_date}")
    print(f"  合约月份: {contract_month}")
    print("=" * 60)
    
    return day1_date, day2_date, day3_date, contract_month


if __name__ == "__main__":
    # 获取用户输入
    day1_date, day2_date, day3_date, contract_month = get_user_input()
    
    # 运行主程序
    main(day1_date, day2_date, day3_date, contract_month)