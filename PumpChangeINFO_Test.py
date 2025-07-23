'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2025-07-03 15:03:38
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2025-07-10 16:58:25
FilePath: \RunStateCheck\PumpChangeINFO.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
'''
    针对采用泵状态判断启停泵的情况
    1. 采用泵状态判断启停泵的情况
    2. 采用流量判断调泵的情况
'''
import pandas as pd
import bisect
from datetime import datetime
import time
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def process_pump_data_new(df):
    """
    处理DataFrame中的PUMP列数据，将其拆分为多个列并处理异常值
    
    参数:
    df (pd.DataFrame): 包含Recordtime和PUMP列的DataFrame
    
    返回:
    pd.DataFrame: 处理后的DataFrame，包含Recordtime和拆分后的泵状态、泵电流、泵电压、泵频率等列
    """
    # 复制原始DataFrame以避免修改原数据
    processed_df = df.copy()    
    # 定义列名映射
    column_mapping = {
        0: ['Pump1State', 'Pump2State'],
        1: ['Pump1EleFlow', 'Pump2EleFlow'],
        2: ['Pump1ElePres', 'Pump2ElePres'],
        3: ['Pump1Hz', 'Pump2Hz']
    }    
    # 定义默认值映射（根据业务需求调整）
    default_values = {
        'Pump1State': 'False',
        'Pump2State': 'False',
        'Pump1EleFlow': 0.0,
        'Pump2EleFlow': 0.0,
        'Pump1ElePres': 0.0,
        'Pump2ElePres': 0.0,
        'Pump1Hz': 0.0,
        'Pump2Hz': 0.0
    }    
    # 处理PUMP列
    def process_pump_value(valueStr):
        value = valueStr.replace('{','').replace('}','')
        # 如果不是字符串，返回默认值字典
        if not isinstance(value, str):
            return default_values.copy()        
        parts = value.split('、')
        # 确保有4个部分，不足则补空字符串
        while len(parts) < 4:
            parts.append('')        
        result = {}
        for i, part in enumerate(parts):
            if i in column_mapping:
                col_names = column_mapping[i]
                # 拆分当前部分
                sub_parts = part.split('&') if part else []                
                # 处理每个子部分
                for j, col_name in enumerate(col_names):
                    if j < len(sub_parts):
                        result[col_name] = sub_parts[j]
                    else:
                        # 使用默认值
                        result[col_name] = default_values[col_name]
        return result
    # 应用处理函数并转换为DataFrame
    pump_data = processed_df['PUMP'].apply(process_pump_value).apply(pd.Series)
    # 合并处理后的PUMP数据
    processed_df = pd.concat([processed_df, pump_data], axis=1)
    # 转换数值列
    for col in pump_data.columns:
        if col not in ['Pump1State', 'Pump2State']:
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce').fillna(default_values[col])
    # 选择需要的列
    result_df = processed_df[['RECORDTIME'] + list(pump_data.columns)]
    return result_df


def replace_values(df):
    # 定义替换映射
    pump1_mapping = {'False': 0, 'True': 2}
    pump2_mapping = {'False': 0, 'True': 3}    
    # 替换Pump1State列的值，并将其他值替换为NaN
    df['Pump1State'] = df['Pump1State'].map(pump1_mapping)    
    # 替换Pump2State列的值，并将其他值替换为NaN
    df['Pump2State'] = df['Pump2State'].map(pump2_mapping)    
    # 统计泵状态NaN的数量
    nan_pump1 = df['Pump1State'].isna().sum()
    nan_pump2 = df['Pump2State'].isna().sum()
    total_nan = nan_pump1 + nan_pump2
    # 将两列转换为数值类型，无法转换的值设为NaN
    df['Pump1Hz'] = pd.to_numeric(df['Pump1Hz'], errors='coerce')
    df['Pump2Hz'] = pd.to_numeric(df['Pump2Hz'], errors='coerce')
    # 检查频率NaN的数据
    nan_Hz1 = df['Pump1Hz'].isna().sum()
    nan_Hz2 = df['Pump2Hz'].isna().sum()
    Hz_nan = nan_Hz1 + nan_Hz2
    if total_nan < 15:        
        if Hz_nan < 15:
            df.fillna(method='ffill', inplace=True)
            df.fillna(method='bfill', inplace=True)
            # 构建PumpStateCode列与PumpHz列
            df['PumpStateCode'] = df['Pump1State'] + df['Pump2State']
            df['PumpHz'] = df['Pump1Hz'] + df['Pump2Hz']
            # 统计频率和状态码掉零的长度
            PumpStateCode_zero_count = (df['PumpStateCode'] == 0).sum()
            PumpHz_zero_count = (df['PumpHz'] == 0).sum()
            # 如果0的数量少于10，则将这些0替换为NaN
            if PumpStateCode_zero_count < 15:
                df.loc[df['PumpStateCode'] == 0, 'PumpStateCode'] = np.nan  # 或使用np.nan
            if PumpHz_zero_count < 15:
                df.loc[df['PumpHz'] == 0, 'PumpHz'] = np.nan  # 或使用np.nan
            df.fillna(method='ffill', inplace=True)
            df.fillna(method='bfill', inplace=True)
            return df
        else:
            return "采集错误&&泵频率采集错误"
    elif total_nan >= 10:
        # print("泵状态采集错误")
        return "采集错误&&泵状态采集错误"

def SplittingTheList(OList, Num, TypeStr):
    """
    @description  : 将list分为Num段，计算每段的平均值
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    ListLen = int(len(OList)/Num)
    if TypeStr == 'before':
        OList_1 = OList[0: ListLen * Num]
    elif TypeStr == 'after':
        OList_1 = OList[-(ListLen * Num):]
    else:
        OList_1 = OList
    n = 0
    NList = []
    while n < len(OList_1):
        NList.append(sum(OList_1[n:n+ListLen])/ListLen)
        n = n + ListLen
    return NList


def group_and_calculate_mean(data, group_size):
    """
    @description  : 将数据分组并计算每组均值
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    # 确保数据长度能被分组大小整除
    if len(data) % group_size != 0:
        # 补全数据（用最后一个值填充）
        data = np.append(data, [data[-1]] * (group_size - len(data) % group_size))
    # 重塑数组并计算每组均值
    groups = data.reshape(-1, group_size)
    mean_groups = np.mean(groups, axis=1)

    # print(f"数据已分为{len(mean_groups)}组，每组{group_size}个数据")
    return mean_groups


def find_AbsMax_change(mean_groups):
    """
    @description  :
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    # 计算相邻均值的变化量（绝对值）
    Abs_changes = np.abs(np.diff(mean_groups))
    changes_values = np.diff(mean_groups)
    # 找出最大变化量的位置（索引+1，因为变化量比均值少一个）
    if len(Abs_changes) == 0:
        return 0, 0
    
    max_change_idx = np.argmax(Abs_changes)
    # 转换为组间位置
    max_change_pos = max_change_idx + 1
    max_change_abs_value = Abs_changes[max_change_idx]
    max_change_value = changes_values[max_change_idx]
    # print(f"最大变化发生在第{max_change_pos}组和第{max_change_pos + 1}组之间")
    # print(f"变化量为:{max_change_abs_value}")
    
    # 最大上升
    max_increase = np.max(changes_values)
    increase_indices = np.where(changes_values == max_increase)[0]  # 可能有多个相同的最大值
    # 最大下降
    max_decrease = np.min(changes_values)
    decrease_indices = np.where(changes_values == max_decrease)[0]  # 可能有多个相同的最小值
    # 可能有多个最大上升或下降值
    max_ChangeINFO = {
        'max_increase': {
            'positions': increase_indices[-1],  # 所有最大上升的位置
            'value': max_increase
        },
        'max_decrease': {
            'positions': decrease_indices[-1],  # 所有最大下降的位置
            'value': max_decrease
        }
    }
    return max_change_pos, max_change_abs_value, max_change_value, max_ChangeINFO

def find_max_change(mean_groups):
    """
    @description  : 根据数据间隔找到变化最大的位置
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    # 计算间隔值的变化量
    Mn = 1
    changes_values_lst = []
    changes_AbsValues_lst = []
    while Mn < len(mean_groups) - 1:
        changes_values_lst.append(mean_groups[Mn + 1] - mean_groups[Mn - 1])
        changes_AbsValues_lst.append(abs(mean_groups[Mn + 1] - mean_groups[Mn - 1]))
        Mn += 1
    # 找出最大变化量的位置（索引+1，因为变化量比均值少一个）
    if len(changes_AbsValues_lst) == 0:
        return 0, 0
    # 变化最大的位置
    max_Abschange_idx = np.argmax(changes_AbsValues_lst)
    max_change_abs_value = changes_AbsValues_lst[max_Abschange_idx]
    # 上升变化最大的位置
    max_increase_idx = np.argmax(changes_values_lst)
    max_decrease_idx = np.argmin(changes_values_lst)
    # 上升/下降变化最大的值
    max_increase_value = changes_values_lst[max_increase_idx]
    max_decrease_value = changes_values_lst[max_decrease_idx]
    # 转换为组间位置
    increase_indices = max_increase_idx + 1 
    decrease_indices = max_decrease_idx + 1
    # print(f"最大变化发生在第{max_change_pos}组和第{max_change_pos + 1}组之间")
    # print(f"变化量为:{max_change_abs_value}")
    # 可能有多个最大上升或下降值
    max_ChangeINFO = {
        'max_increase': {
            'positions': increase_indices,  # 所有最大上升的位置
            'value': max_increase_value
        },
        'max_decrease': {
            'positions': decrease_indices,  # 所有最大下降的位置
            'value': max_decrease_value
        }
    }
    max_change_pos = max_Abschange_idx + 1
    max_change_value = np.max([max_increase_value, max_decrease_value])
    return max_change_pos, max_change_abs_value, max_change_value, max_ChangeINFO


def process_data(data_list):
    """
    @description  : 处理数据列表：清洗异常值、分组求平均值、找出变化最大的位置
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    # 转换为numpy数组便于处理
    data = np.array(data_list)
    # 异常值清洗（使用IQR方法）
    cleaned_data = clean_outliers_iqr(data)
    # 分组求取均值（6个一组）
    mean_groups = group_and_calculate_mean(cleaned_data, group_size=6)
    # 找出变化最大的位置
    max_change_pos, max_change_abs_value, max_change_values, max_ChangeINFO = find_max_change(mean_groups)

    return{
        "cleaned_data": cleaned_data,
        "mean_groups": mean_groups,
        "max_change_pos": max_change_pos,
        "max_change_abs_value": max_change_abs_value,
        "max_change_value": max_change_values,
        "max_ChangeINFO": max_ChangeINFO
    }
    
def clean_outliers_iqr(data):
    """
    @description  : 使用IQR方法清洗异常值
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    # 标记并替换异常值
    is_outlier = (data < lower_bound) | (data > upper_bound)
    cleaned_data = data.copy()
    cleaned_data[is_outlier] = np.nan
    cleaned_data = np.interp(np.arange(len(data)), np.where(~is_outlier)[0], data[~is_outlier])
    
    # print(f"清洗前数据长度:{len(data)}")
    # print(f"检测到{sum(is_outlier)}个异常值，已清洗")
    return cleaned_data




def visualize_results(original_data, cleaned_data, mean_groups, max_change_pos):
    """
    @description  : 可视化处理结果
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    plt.figure(figsize=(12, 8))
    # 原始数据与清洗后数据对比
    plt.subplot(2, 1, 1)
    plt.plot(original_data, 'b-', label='原始数据')
    plt.plot(cleaned_data, 'r-', label='清洗后数据')
    plt.title('原始数据与清洗后数据对比')
    plt.legend()
    plt.grid(True)

    # 分组均值与最大变化位置
    plt.subplot(2, 1, 2)
    plt.plot(mean_groups, 'g-', linewidth=2)

    # 标记最大变化位置
    '''
    if max_change_pos > 0 and max_change_pos < len(mean_groups):
        plt.axvline(x=max_change_pos, color='red', linestyle='--', label=f"最大变化位置:{max_change_pos}")
        print(f"最大变化位置:{max_change_pos}，数列中的位置为:{max_change_pos * 6}。")
    '''

    if max_change_pos > 0 and max_change_pos < len(mean_groups):
        plt.axvline(x=max_change_pos, color='red', linestyle='--', label=f"最大变化位置:{max_change_pos}")
        print(f"最大变化位置:{max_change_pos}，数列中的位置为:{max_change_pos * 6}。")    

    plt.title('分组均值与最大变化位置')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

'''
if __name__ == "__main__":
    # 生成包含异常值得示例数据
    np.random.seed(42)
    original_data = np.random.normal(loc=50, scale=10, size=180)
    # 添加一些异常值
    outliers = np.random.normal(loc=100, scale=20, size=10)
    original_data[np.random.choice(180, 10, replace=False)] = outliers
    # 处理数据
    result = process_data(original_data)
    # 清洗后的数据
    cleaned_data = result["cleaned_data"]
    print(f"清洗后的数据cleaned_data:{cleaned_data}")
    # 分组数据的平均值
    mean_groups = result["mean_groups"]
    print(f"分组数据平均值mean_groups:{mean_groups}")
    # 变化最大位置
    max_change_pos = result["max_change_pos"]
    print(f"最大变化位置max_change_pos:{max_change_pos}")
    # 可视化结果
    visualize_results(original_data, result["cleaned_data"], result["mean_groups"], result["max_change_pos"])
'''

def Get_ChangeMaxPosition(SzFlow_lst):
    """
    @description  : 找到数据变化最大的位置
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    # 对始站排量进行处理提取数据变化最大位置
    SzFlow_Result = process_data(SzFlow_lst)
    # 数据变化最大的位置
    max_change_pos = SzFlow_Result["max_change_pos"]
    if max_change_pos > 6 and max_change_pos < 24:
        return max_change_pos
    else:
        return 0


def Get_Pump_UOD(re_list, Pumpspread):
    """
    @description  : 根据数据波动判断调泵情况
    ---------
    @param  : re_list：监测数据list型
    -------
    @Returns  : 返回上升还是下降
    -------
    """
    alen = int(len(re_list) / 7)
    # 将数据分为3组 每组alen个数据，以每组180个数据为例，什解
    a0 = SplittingTheList(re_list[0:alen], 3, 'before')
    a1 = SplittingTheList(re_list[alen:alen + alen], 3, 'normal')
    a2 = SplittingTheList(re_list[-alen:], 3, 'after')
    a0max = np.max(a0)
    a2max = np.max(a2)
    a0min = np.min(a0)
    a2min = np.min(a2)
    a0avg = np.mean(a0)
    a2avg = np.mean(a2)
    if a0min > a2max:
        if a0avg - a2avg > Pumpspread:
            alarm = '下降'  # 下降
        else:
            alarm = '运行'
    elif a0max < a2min:
        if a2avg - a0avg > Pumpspread:
            alarm = '上升'  # 下降
        else:
            alarm = '运行'
    else:
        alarm = '运行'
    return alarm


def WritingPumpChange2CSV(PumpChangeInfo_df, PumpChangeInfo_Path, PumpStatChange):
    """
    @description  :
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    # 截取启停泵状态信息只保存一个月数据
    PumpChangeTime_lst = PumpChangeInfo_df['PUMPCHANGETIME'].values
    StartTime = int(time.time()) - 2592000000 # (30 * 24 * 3600 * 1000)
    StartPosition = bisect.bisect_left(PumpChangeTime_lst, StartTime)
    # 组建新的启停泵DataFrame
    PumpChangeInfo_df = pd.concat([PumpChangeInfo_df.iloc[StartPosition:].reset_index(drop=True), pd.DataFrame(PumpStatChange)], ignore_index=True)
    # 写入CSV文件
    PumpChangeInfo_df.to_csv(PumpChangeInfo_Path, index=False, encoding='utf-8-sig')


def Check_Pump_Regulate(PipeConfig, HighFrequency_df, PumpChangeInfo_Path, HighFrequencyPump_df):
    """
    @description  : 检查是否有调泵的情况
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    # 根据排量数据检查最近是否有状态变化
    # 泵的有效变化范围，超出该范围则认为存在调泵或启停泵的情况
    pumpspread = float(PipeConfig['PUMPSPREAD'])
    # 提取180秒的数据
    Rundata_df = HighFrequency_df.iloc[-180:].reset_index(drop=True)
    SzFlow_lst = Rundata_df['SZPL'].to_list()
    # 对始站排量进行处理提取数据变化最大位置
    SzFlow_Result = process_data(SzFlow_lst)
    # 提取清洗后的数据
    cleaned_data = SzFlow_Result["cleaned_data"]
    # 数据变化最大的位置
    max_change_pos = SzFlow_Result["max_change_pos"]
    # 数据变化最大值
    max_change_value = SzFlow_Result["max_change_value"]
    # 最大上升或下降信息记录位置
    max_ChangeINFO = SzFlow_Result["max_ChangeINFO"]
    # 根据清洗后的数据检查波动情况
    StartFlow_mean = np.mean(cleaned_data[0:30])
    EndFlow_mean = np.mean(cleaned_data[-30:])
    # 根据数据前后的变化判断是否有调泵情况
    if StartFlow_mean - EndFlow_mean > pumpspread and EndFlow_mean > 0.1:
        crease_value_max = float(max_ChangeINFO['max_decrease']['value'])
        if np.abs(crease_value_max) > pumpspread:
            PumpState = '下降'
            crease_pos = int(max_ChangeINFO['max_decrease']['positions']) + 1
        else:
            PumpState = '运行'
            crease_pos = -10000
    elif EndFlow_mean - StartFlow_mean > pumpspread and StartFlow_mean > 0.1:
        crease_value_max = float(max_ChangeINFO['max_increase']['value'])
        if np.abs(crease_value_max) > pumpspread:
            PumpState = '上升'
            crease_pos = int(max_ChangeINFO['max_increase']['positions']) + 1
        else:
            PumpState = '运行'
            crease_pos = -10000
    else:
        PumpState = '运行'
        crease_pos = -10000
    


    if crease_pos > 9 and crease_pos < 20:
        # 检查改变化是否已经记录
        New_PumpChangeInfo_df = pd.read_csv(PumpChangeInfo_Path, encoding='utf-8-sig')
        # 本次变化的时间点
        RunTime_lst = Rundata_df['RECORDTIME'].to_list()
        # RunTime_str = datetime.fromtimestamp(int(RunTime_lst[-1]/1000))
        NowChangeTimePoint = RunTime_lst[crease_pos * 6]
        # PumpChangeTime = RunTime_lst[NowChangeTimePoint]
        PumpChangeTime_str = datetime.fromtimestamp(int(NowChangeTimePoint/1000))
        # 提取泵的状态信息
        lastPumpCode = int(HighFrequencyPump_df.iloc[crease_pos * 6]['PumpStateCode'])
        lastPumpHz = int(HighFrequencyPump_df.iloc[crease_pos * 6]['PumpHz'])
        # 对比该时间点前后是否有泵的变化信息
        # PumpChangeTime_lst = New_PumpChangeInfo_df['ChangeTime'].to_list()
        # 检查是否有启泵信息存在
        # CheckChangeTimePoint = NowChangeTimePoint - 30000
        # CheckChangeTimePosition = bisect.bisect_left(PumpChangeTime_lst, CheckChangeTimePoint)
        # CheckPumpChangeINFO_df = New_PumpChangeInfo_df.iloc[CheckChangeTimePosition: ].reset_index(drop=True)
        # 检查最后的调泵或启停泵信息
        lastPumpChangeTime = int(New_PumpChangeInfo_df.iloc[-1]['PUMPCHANGETIME'])
        lastPumpChangeState = New_PumpChangeInfo_df.iloc[-1]['PUMPCHANGETYPE']
        if NowChangeTimePoint > lastPumpChangeTime:
            PumpStatChange = [{'PIPENAME':PipeConfig['PIPENAME'], 'PumpStateCode': lastPumpCode, 'PumpHz': lastPumpHz, 'PUMPCHANGETIME': NowChangeTimePoint, 'PUMPCHANGETIMESTR': PumpChangeTime_str, 'PUMPCHANGETYPE': PumpState}]
            WritingPumpChange2CSV(New_PumpChangeInfo_df, PumpChangeInfo_Path, PumpStatChange)
            print(PumpStatChange)

def PumpStateChangeAnalysis(HighFrequency_df, PumpRun_df, PumpChangeInfo_Path, PipeConfig):
    """
    @description  : 通过对泵的状态进行数据变换后获取两泵相加的数值，从而判断泵的启停变化
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    
    # 泵的运行状态码
    PumpStateCode_lst = PumpRun_df['PumpStateCode'].values
    PumpRunHz_lst = PumpRun_df['PumpHz'].values
    # 时间序列
    Recordtime_lst = PumpRun_df['RECORDTIME'].values
    # 检查泵变化信息文件是否存在
    if os.path.exists(PumpChangeInfo_Path):
        PumpChangeInfo_df = pd.read_csv(PumpChangeInfo_Path, encoding='utf-8-sig')
        if PumpChangeInfo_df.shape[0] > 0:
            lastPumpCode = int(PumpChangeInfo_df.iloc[-1]['PumpStateCode'])
            lastPumpHz = int(PumpChangeInfo_df.iloc[-1]['PumpHz'])
        else:
            lastPumpCode = PumpStateCode_lst[0]
            lastPumpHz = PumpRunHz_lst[0]
            PumpChangeTime = int(PumpRun_df.iloc[0]['RECORDTIME'])
            PumpChangeTime_str = datetime.fromtimestamp(int(PumpChangeTime/1000))
            if lastPumpCode == 0: 
                PumpStateChange = '停泵'
            else:
                PumpStateChange = '启泵'
            PumpChangeInfo_df = pd.DataFrame([{'PIPENAME':PipeConfig['PIPENAME'], 'PumpStateCode':lastPumpCode,'PumpHz': lastPumpHz, 'PUMPCHANGETIME':PumpChangeTime, 'PUMPCHANGETIMESTR':PumpChangeTime_str, 'PUMPCHANGETYPE':PumpStateChange}])
    else:
        lastPumpCode = PumpStateCode_lst[0]
        lastPumpHz = PumpRunHz_lst[0]
        PumpChangeTime = int(PumpRun_df.iloc[0]['RECORDTIME'])
        PumpChangeTime_str = datetime.fromtimestamp(int(PumpChangeTime/1000))
        if lastPumpCode == 0: 
            PumpStateChange = '停泵'
        else:
            PumpStateChange = '启泵'
        PumpChangeInfo_df = pd.DataFrame([{'PIPENAME':PipeConfig['PIPENAME'], 'PumpStateCode':lastPumpCode, 'PumpHz':lastPumpHz, 'PUMPCHANGETIME':PumpChangeTime, 'PUMPCHANGETIMESTR':PumpChangeTime_str, 'PUMPCHANGETYPE':PumpStateChange}])

    for i in range(1, len(PumpStateCode_lst)):
        PumpStatChange = []
        if int(PumpStateCode_lst[i]) != lastPumpCode:
            # 泵的状态码发生了变化，说明有操控泵的情况
            if lastPumpCode == 0 and int(PumpStateCode_lst[i]) != 0:
                NewPumpState = '启泵'
            elif lastPumpCode != 0 and int(PumpStateCode_lst[i]) == 0:
                NewPumpState = '停泵'
            elif lastPumpCode != 0 and int(PumpStateCode_lst[i]) != 0:
                NewPumpState = '倒泵'
            else:                
                NewPumpState = '其他'
        else: 
            NewPumpState = '无变化'
        if NewPumpState == '倒泵' or NewPumpState == '停泵' or NewPumpState == '启泵':
            # 记录状态变化信息        
            ChangeTime_str = datetime.fromtimestamp(int(int(Recordtime_lst[i])/1000))
            if int(Recordtime_lst[i]) == 1751513654000:
                print('DEBUG')
            PumpStatChange.append({'PIPENAME':PipeConfig['PIPENAME'], 'PUMPCHANGETYPE':NewPumpState, 'PUMPCHANGETIME':int(Recordtime_lst[i]), 'PUMPCHANGETIMESTR':ChangeTime_str, 'PumpStateCode':int(PumpStateCode_lst[i]), 'PumpHz':float(PumpRunHz_lst[i])})
            lastPumpCode = PumpStateCode_lst[i]
            # 启停泵状态写入CSV文件
            PumpChangeInfo_df = pd.read_csv(PumpChangeInfo_Path, encoding='utf-8-sig')
            WritingPumpChange2CSV(PumpChangeInfo_df, PumpChangeInfo_Path, PumpStatChange)
    # 对高频数据的泵信息进行数值变换
    HighFrequencyPumpBZ_df = process_pump_data_new(HighFrequency_df)
    HighFrequencyPump_df = replace_values(HighFrequencyPumpBZ_df)
    if type(HighFrequencyPump_df) != type('Str'):
        # 最近没有状态变化，说明没有操控泵，如果当前时间段未发生启停泵变化则进行调泵判断，根据频率判断调泵判断失效，此处采用数据判断
        # 提取180秒的数据
        Check_Pump_Regulate(PipeConfig, HighFrequency_df, PumpChangeInfo_Path, HighFrequencyPump_df)

def Initialization_PumpChangeINFO_FileCsv(RunStart_df, PumpChangeInfo_Path, PipeConfig):
    """
    @description  : 启停泵信息文件不存在的情况下，初始化文件
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    # 泵的运行状态码
    PumpStateCode_lst = RunStart_df['PumpStateCode'].values
    StartStateCode = int(PumpStateCode_lst[0])
    PumpRunHz_lst = RunStart_df['PumpHz'].values
    StartPumpHz = float(PumpRunHz_lst[0])
    # 时间序列
    Recordtime_lst = RunStart_df['RECORDTIME'].values
    StartTime_int = int(Recordtime_lst[0])
    StartTime_str = datetime.fromtimestamp(int(StartTime_int/1000))
    if StartStateCode == 0:
        PumpStateChange = '停泵'
    else:
        PumpStateChange = '启泵'
    PumpChangeINFO_df = pd.DataFrame([{'PIPENAME':PipeConfig['PIPENAME'], 'PUMPCHANGETYPE':PumpStateChange, 'PUMPCHANGETIME':StartTime_int, 'PUMPCHANGETIMESTR':StartTime_str, 'PumpStateCode':StartStateCode, 'PumpHz': StartPumpHz}])
    PumpChangeINFO_df.to_csv(PumpChangeInfo_Path, encoding='utf-8-sig', index=False)
    

def CheckPumpRunState(HighFrequencyData_df, PipeConfig):
    """
    @description  :
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    PipeName = PipeConfig['PIPENAME']
    PumpChangeInfo_Path = f'temp/{PipeName}/PumpChangeInfo.csv'

    # 提取highFrequency.csv中的PumpInfo构建独立的DataFrame
    PumpInfo_df = process_pump_data_new(HighFrequencyData_df)
    # 提取最后60秒的数据进行逐一比较
    NowRundata_df = PumpInfo_df.iloc[-60:].reset_index(drop=True)
    # 将启停泵的True与False替换为数字便于比较启停泵变化
    PumpRun_df = replace_values(NowRundata_df)
    RunStart_df = replace_values(PumpInfo_df.iloc[0: 60].reset_index(drop=True))
    if type(PumpRun_df) != type('str'):
        # 检查启停泵信息文件是否存在， 不存在的话就根据当前管道数据写入
        if os.path.exists(f'temp/{PipeName}') is False:
            os.makedirs(f'temp/{PipeName}')
            Initialization_PumpChangeINFO_FileCsv(RunStart_df, PumpChangeInfo_Path, PipeConfig)
        elif os.path.lexists(f'temp/{PipeName}/PumpChangeInfo.csv') is False:
            # 如果启停泵文件不存在的情况
            Initialization_PumpChangeINFO_FileCsv(RunStart_df, PumpChangeInfo_Path)

        PumpStateChangeAnalysis(HighFrequencyData_df, PumpRun_df, PumpChangeInfo_Path, PipeConfig)
    else:
        print(f"第{NowRundata_df.iloc[-1]['RECORDTIME']}行泵状态数据采集错误")


def RunAnalysis():
    Rundata_Path = r'E:\工作管理\2、项目管理\6、第六采油厂\2025\07\故障状态\tempData\胡十转至胡四联集油管道\InfluxDB_Frequency.csv'
    Rundata_df = pd.read_csv(Rundata_Path, encoding='utf-8-sig')
    # AllConfig_df = pd.read_csv(r'.csv', encoding='utf-8-sig')
    AllConfig_df = pd.DataFrame([{'PIPENAME': '胡十转至胡四联集油管道', 'PUMPSPREAD':0.3}])
    PipeName = '胡十转至胡四联集油管道'
    PipeConfig_df = AllConfig_df[AllConfig_df['PIPENAME'] == PipeName].reset_index(drop=True)
    PipeConfig = PipeConfig_df.to_dict('records')[0]
    Rn = 210
    while Rn < Rundata_df.shape[0]:
        HighFrequencyData_df = Rundata_df.iloc[Rn-210: Rn].reset_index(drop=True)
        CheckPumpRunState(HighFrequencyData_df, PipeConfig)
        Rn += 60

RunAnalysis()