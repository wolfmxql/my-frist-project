'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2025-06-03 11:42:07
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2025-07-01 17:12:11
FilePath: \3、数据提取与输差运算\启停泵管道独立运行\根据泵数据分析启停.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import pandas as pd
import bisect
from datetime import datetime
import time
import os
from pathlib import Path
import numpy as np

def process_pump_data(df):
    """
    处理DataFrame中的PUMP列数据，将其拆分为多个列并删除异常行
    
    参数:
    df (pd.DataFrame): 包含Recordtime和PUMP列的DataFrame
    
    返回:
    pd.DataFrame: 处理后的DataFrame，包含Recordtime和拆分后的列
    """
    # 复制原始DataFrame以避免修改原数据
    processed_df = df.copy()
    
    # 定义有效数据的模式
    def is_valid_pump(value):
        # 检查是否为字符串类型
        if not isinstance(value, str):
            return False
        
        # 检查格式是否为"值1&值2、值3&值4、值5&值6、值7&值8"
        parts = value.split('、')
        if len(parts) != 4:
            return False
        
        # 检查每个部分是否包含&符号
        for part in parts:
            if '&' not in part:
                return False
        
        return True
    
    # 筛选出有效的PUMP数据
    valid_rows = processed_df['PUMP'].apply(is_valid_pump)
    processed_df = processed_df[valid_rows].reset_index(drop=True)
    
    # 拆分PUMP列
    split_data = processed_df['PUMP'].str.split('、', expand=True)
    
    # 定义列名映射
    column_mapping = {
        0: ['Pump1State', 'Pump2State'],
        1: ['Pump1EleFlow', 'Pump2EleFlow'],
        2: ['Pump1ElePres', 'Pump2ElePres'],
        3: ['Pump1Hz', 'Pump2Hz']
    }
    
    # 处理每个部分
    for i, col_names in column_mapping.items():
        # 进一步拆分每个部分为两个值
        part_data = split_data[i].str.split('&', expand=True)
        
        # 分配到对应的列
        for j, col_name in enumerate(col_names):
            processed_df[col_name] = part_data[j]
            
            # 尝试将数值列转换为浮点数
            if col_name != 'Pump1State' and col_name != 'Pump2State':
                processed_df[col_name] = pd.to_numeric(processed_df[col_name], errors='coerce')
    
    # 选择需要的列
    result_df = processed_df[['RECORDTIME'] + [col for cols in column_mapping.values() for col in cols]]
    
    return result_df


def process_pump_data_new(df):
    """
    处理DataFrame中的PUMP列数据，将其拆分为多个列并处理异常值
    
    参数:
    df (pd.DataFrame): 包含Recordtime和PUMP列的DataFrame
    
    返回:
    pd.DataFrame: 处理后的DataFrame，包含Recordtime和拆分后的列
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
    def process_pump_value(value):
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


def is_uniform_bool(lst):
    unique_values = set(lst)
    if unique_values == {'True'}:
        return '运行'  # 全为 True
    elif unique_values == {'False'}:
        return '关停'  # 全为 False
    else:
        return '发生变化'  # 包含 True 和 False

def PumpStateChange(PumpRun_df):
    """
    @description  : 根据Pump的状态参数检查是否有启停泵的情况
    ---------
    @param  :
    -------
    @Returns  : 
    -------
    """
    # 检查是否有异常数据  20250114需要根据MyPipeline的类型进行修改
    Pump1State = PumpRun_df['Pump1State'].values
    Pump2State = PumpRun_df['Pump2State'].values
    Pump1EleFlow = PumpRun_df['Pump1EleFlow'].values
    Pump2EleFlow = PumpRun_df['Pump2EleFlow'].values
    Pump1ElePres = PumpRun_df['Pump1ElePres'].values
    Pump2ElePres = PumpRun_df['Pump2ElePres'].values
    Pump1Hz = PumpRun_df['Pump1Hz'].values
    Pump2Hz = PumpRun_df['Pump2Hz'].values
    # 检查最近的泵状态
    # 提取数据前后30秒的泵状态
    Start_Pump1State_lst = Pump1State[0:60]
    End_Pump1State_lst = Pump1State[-60:]
    Start_Pump2State_lst = Pump2State[0:60]
    End_Pump2State_lst = Pump2State[-60:]
    StartState_Pump1 = is_uniform_bool(Start_Pump1State_lst)
    EndState_Pump1 = is_uniform_bool(End_Pump1State_lst)
    StartState_Pump2 = is_uniform_bool(Start_Pump2State_lst)
    EndState_Pump2 = is_uniform_bool(End_Pump2State_lst)
    if StartState_Pump1 == '发生变化' or StartState_Pump2 == '发生变化':
        # 'Pump1或Pump2状态发生变化，暂不处理'
        NewPumpState = '其他'
    else:
        if StartState_Pump1 == EndState_Pump1 and StartState_Pump2 == EndState_Pump2:
            # 'Pump1或Pump2状态无变化，不进行处理'
            NewPumpState = '其他'
        elif StartState_Pump1 != EndState_Pump1 and StartState_Pump2== EndState_Pump2:
            # Pump1状态发生变化，Pump2状态无变化，判断Pump1是否有启停
            if EndState_Pump1 == '关停' and EndState_Pump2 == '关停':
                NewPumpState = '停泵'
            elif EndState_Pump1 == '运行' and EndState_Pump2 == '关停':
                NewPumpState = '启泵'
            elif EndState_Pump1 == '关停' and EndState_Pump2 == '运行':
                NewPumpState = '下调'
            elif EndState_Pump1 == '运行' and EndState_Pump2 == '运行':
                NewPumpState = '上调'
            else:
                # 'Pump1或Pump2位置状态，暂不处理'
                NewPumpState = '其他'
        elif StartState_Pump1 == EndState_Pump1 and StartState_Pump2 != EndState_Pump2:
            # Pump1状态无变化，Pump2状态发生变化，判断Pump2是否有启停
            if EndState_Pump1 == '关停' and EndState_Pump2 == '关停':
                NewPumpState = '停泵'
            elif EndState_Pump1 == '运行' and EndState_Pump2 == '关停':
                NewPumpState = '下调'
            elif EndState_Pump1 == '关停' and EndState_Pump2 == '运行':
                NewPumpState = '启泵'
            elif EndState_Pump1 == '运行' and EndState_Pump2 == '运行':
                NewPumpState = '上调'
            else:
                # 'Pump1或Pump2位置状态，暂不处理'
                NewPumpState = '其他'
        else:
            # 'Pump1或Pump2位置状态，暂不处理'
            NewPumpState = '其他'
    return NewPumpState


'''
    方法二：或者根据变化时间点前后的数据进行判断
    0603 反馈出问题较多，需要进一步验证
'''

def find_change_indices(lst):
    """
    @description  : 检查该时间段内，泵状态是否发生变化
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    # 提取开始时间与结束时间的泵状态

    change_indices = []
    if len(lst) < 2:
        return change_indices  # 空列表或单一元素，无变化
    
    for i in range(1, len(lst)):
        prev = lst[i-1]
        curr = lst[i]
        if prev != curr:
            # 记录当前索引（变化发生在 i-1 到 i 之间，索引 i 为变化后的位置）
            change_indices.append(i)  # 或根据需求记录 i-1（变化前的位置）
    return change_indices

def AnalysisPumpChange(PumpStateRecord_df, PumpState_lst, Recordtime_lst):
    """
    @description  : 分别计算当前时刻 1#/2# 泵的变化情况
    ---------
    @param  :   PumpStateRecord_df  单泵的变化信息文件获取得DataFrame
                Pump1State_lst      当前时刻单泵的状态变化序列
    -------
    @Returns  :
    -------
    """
    PumpStateRecord_dict = []
    lastPumpState = PumpStateRecord_df.iloc[-1]['PumpState']
    if len(PumpState_lst) > 0:
        for i in range(1, len(PumpState_lst)):
            if lastPumpState != PumpState_lst[i]:
                if PumpState_lst[i] == 'True':
                    NewPumpState = '启泵'
                elif PumpState_lst[i] == 'False':
                    NewPumpState = '停泵'
                else:
                    NewPumpState = '泵状态数据错误'
                lastPumpState = PumpState_lst[i]
                PumpStateRecord_dict.append({'PumpStateChange':NewPumpState, 'PumpChangeTime':Recordtime_lst[i]})
    PumpStateRecord_df = pd.DataFrame(PumpStateRecord_dict)
    return PumpStateRecord_df
    
'''

def ExtractPumpChange(PipeName, NowRundata_df, PumpStateRecord_dict):
    """
    @description  : 根据记录的泵状态数据, 需要有两个记录文件，一个记录计算的过程主要是看计算到那个时间点了，二是存储泵的变化情况
    ---------
    @param  :   PumpStateRecord_dict 泵的变化信息数据文件获取，两个泵的拆解成相互独立DataFrame
                PumpRun_df 从Rundata_df中截取，用于判断泵变化的该时段数据
    -------
    @Returns  :
    -------
    """
    # 提取开始时间与结束时间的泵状态
    change_indices = []
    lastCalRecordTime = NowRundata_df['RECORDTIME'].values
    # 记录监控过程数据的文件路径
    CalRecordPath = f"/temp/{PipeName}/CalRecord.csv"
    CalRecord_df = pd.read_csv(CalRecordPath, encoding='utf-8-sig')
    # 上次运算完结的时间点数据
    lastCalRecordTime = CalRecord_df.iloc[-1]['Recordtime']
    # 提取用于本次识别得数据
    StartPosition = bisect.bisect_left(lastCalRecordTime, lastCalRecordTime)
    NowAnalysisRundata_df = NowRundata_df.iloc[StartPosition: ]
    
    # 1# 泵得变化判断
    Pump1State_lst = NowAnalysisRundata_df['Pump1State'].values
    Pump2State_lst = NowAnalysisRundata_df['Pump2State'].values
    Recordtime_lst = NowAnalysisRundata_df['RECORDTIME'].values

    # 提取变化文件中最后的状态信息
    Pump1ChangeInfo_Path = f"/temp/{PipeName}/Pump1ChangeInfo.csv"
    Pump2ChangeInfo_Path = f"/temp/{PipeName}/Pump2ChangeInfo.csv"
    Pump1StateRecord_df = pd.read_csv(Pump1ChangeInfo_Path, encoding='utf-8-sig')
    Pump2StateRecord_df = pd.read_csv(Pump2ChangeInfo_Path, encoding='utf-8-sig')

    # 提取1#/2# 泵数据 变化信息
    NewPump1StateRecord_df = AnalysisPumpChange(Pump1StateRecord_df, Pump1State_lst, Recordtime_lst)
    NewPump2StateRecord_df = AnalysisPumpChange(Pump2StateRecord_df, Pump2State_lst, Recordtime_lst)

    # 根据变化信息判断1#/2# 泵状态是否发生变化
    if NewPump1StateRecord_df.shape[0] > 0 and NewPump2StateRecord_df.shape[0] > 0:
        # 1#泵与2#泵 状态均发生变化
        if NewPump1StateRecord_df

'''
    


def PumpStateChange2(PumpRun_df):
    Pump1State_lst = PumpRun_df['Pump1State'].values
    Pump2State_lst = PumpRun_df['Pump2State'].values   
    Pump1ChangePosition = find_change_indices(Pump1State_lst)
    Pump2ChangePosition = find_change_indices(Pump2State_lst)
    if len(Pump1ChangePosition) == 0 and len(Pump2ChangePosition) == 0:
        # 'Pump1或Pump2状态无变化，不进行处理'
        NewPumpState = '其他'
        ChangeTime = 0
    elif len(Pump1ChangePosition) == 0 and len(Pump2ChangePosition) != 0:
        # Pump1状态无变化，Pump2状态发生变化，判断Pump2是否有启停
        if Pump1State_lst[-1] == 'False' and Pump2State_lst[-1] == 'True':
            NewPumpState = '启泵'
        elif Pump1State_lst[-1] == 'False' and Pump2State_lst[-1] == 'False':
            NewPumpState = '停泵'
        elif Pump1State_lst[-1] == 'True' and Pump2State_lst[-1] == 'True':
            NewPumpState = '上调'
        elif Pump1State_lst[-1] == 'True' and Pump2State_lst[-1] == 'False':
            NewPumpState = '下调'
        else:
            # 'Pump1或Pump2位置状态，暂不处理'
            NewPumpState = '其他'
        ChangeTime = Pump2ChangePosition[-1]
    elif len(Pump1ChangePosition) != 0 and len(Pump2ChangePosition) == 0:
        # Pump1状态发生变化，Pump2状态无变化，判断Pump1是否有启停
        if Pump1State_lst[-1] == 'False' and Pump2State_lst[-1] == 'True':
            NewPumpState = '下调'
        elif Pump1State_lst[-1] == 'True' and Pump2State_lst[-1] == 'True':
            NewPumpState = '上调'
        elif Pump1State_lst[-1] == 'False' and Pump2State_lst[-1] == 'False':
            NewPumpState = '停泵'
        elif Pump1State_lst[-1] == 'True' and Pump2State_lst[-1] == 'False':
            NewPumpState = '启泵'
        else:
            # 'Pump1或Pump2位置状态，暂不处理'
            NewPumpState = '其他'
        ChangeTime = Pump1ChangePosition[-1]
    elif len(Pump1ChangePosition) != 0 and len(Pump2ChangePosition) != 0:
        print('BUG')
    else:
        # 'Pump1或Pump2位置状态，暂不处理'
        NewPumpState = '其他'
        ChangeTime = Pump2ChangePosition[-1]
    return NewPumpState


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

def PumpStateChangeAnalysis(PumpRun_df, PumpChangeInfo_Path):
    """
    @description  : 通过对泵的状态进行数据变换后获取两泵相加的数值，从而判断泵的启停变化
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    PumpStatChange = []
    # 泵的运行状态码
    PumpStateCode_lst = PumpRun_df['PumpStateCode'].values
    # 泵的运行频率（1+2）
    PumpHz_lst = PumpRun_df['PumpHz'].values
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
            lastPumpHz = PumpHz_lst[0]
            PumpChangeTime = int(PumpRun_df.iloc[-1]['RECORDTIME'])
            PumpChangeTime_str = datetime.fromtimestamp(int(PumpChangeTime/1000))
            if lastPumpCode == 0: 
                PumpStateChange = '停泵'
            else:
                PumpStateChange = '启泵'
            PumpChangeInfo_df = pd.DataFrame([{'PumpStateCode':lastPumpCode,'PumpHz': lastPumpHz, 'PumpChangeTime':PumpChangeTime, 'PumpChangeTime_str':PumpChangeTime_str, 'PumpStateChange':PumpStateChange}])
    else:
        lastPumpCode = PumpStateCode_lst[0]
        lastPumpHz = PumpHz_lst[0]
        PumpChangeTime = int(PumpRun_df.iloc[-1]['RECORDTIME'])
        PumpChangeTime_str = datetime.fromtimestamp(int(PumpChangeTime/1000))
        if lastPumpCode == 0: 
            PumpStateChange = '停泵'
        else:
            PumpStateChange = '启泵'
        PumpChangeInfo_df = pd.DataFrame([{'PumpStateCode':lastPumpCode, 'PumpHz':lastPumpHz, 'PumpChangeTime':PumpChangeTime, 'PumpChangeTime_str':PumpChangeTime_str, 'PumpStateChange':PumpStateChange}])
    for i in range(1, len(PumpStateCode_lst)):
        if int(PumpStateCode_lst[i]) != lastPumpCode:
            # 泵的状态码发生了变化，说明有操控泵的情况
            if lastPumpCode == 0 and int(PumpStateCode_lst[i]) != 0:
                NewPumpState = '启泵'
            elif lastPumpCode != 0 and int(PumpStateCode_lst[i]) == 0:
                NewPumpState = '停泵'
            elif lastPumpCode != 0 and int(PumpStateCode_lst[i]) != 0:
                NewPumpState = '倒泵'
            else:
                # 20250626 泵状态没有产生变化的情况查看泵的频率是否发生变化
                if int(PumpHz_lst[i]) > lastPumpHz + 100:
                    NewPumpState = '上调'
                elif int(PumpHz_lst[i]) < lastPumpHz - 100:
                    NewPumpState = '下调'
                else:
                    NewPumpState = '其他'
        else:
            if int(PumpHz_lst[i]) > lastPumpHz + 100:
                NewPumpState = '上调'
            elif int(PumpHz_lst[i]) < lastPumpHz - 100:
                NewPumpState = '下调'
            else:
                NewPumpState = '无变化'
        if '无变化' not in NewPumpState:
            # 记录状态变化信息        
            ChangeTime_str = datetime.fromtimestamp(int(int(Recordtime_lst[i])/1000))
            PumpStatChange.append({'PumpStateChange':NewPumpState, 'PumpChangeTime':int(Recordtime_lst[i]), 'PumpChangeTime_str':ChangeTime_str, 'PumpStateCode':int(PumpStateCode_lst[i]), 'PumpHz':float(PumpHz_lst[i])})
            lastPumpCode = PumpStateCode_lst[i]
            lastPumpHz = PumpHz_lst[i]

        
    # 更新启停泵信息文件
    NewPumpChangeInfo_df = pd.DataFrame(PumpStatChange)
    AllPumpChangeInfo_df = pd.concat([PumpChangeInfo_df, NewPumpChangeInfo_df], axis=0)
    StartTime_int = int(time.time() * 1000) - 30 * 24 * 3600 * 1000
    WritePumpChangeInfo_df = AllPumpChangeInfo_df[AllPumpChangeInfo_df['PumpChangeTime'] > StartTime_int].reset_index(drop=True)
    WritePumpChangeInfo_df.to_csv(PumpChangeInfo_Path, encoding='utf-8-sig', index=False)

import cx_Oracle
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sqlalchemy

def query_oracle_old(query_str, oracleinfo):
    ora_host = oracleinfo['host']
    ora_port = oracleinfo['port']
    ora_service = oracleinfo['sid']
    ora_user = oracleinfo['username']
    ora_passwd = oracleinfo['usersc']
    dsn = cx_Oracle.makedsn(ora_host, ora_port, service_name=ora_service)
    connection = cx_Oracle.connect(ora_user, ora_passwd, dsn)
    # 执行查询
    cursor = connection.cursor()
    cursor.execute(query_str)
    # 获取列名
    column_names = [desc[0] for desc in cursor.description]
    # 将查询结果转为字典列表
    # result_dict = [dict(zip(column_names, row)) for row in cursor]
    # 将查询结果转换为DataFrame
    result_df = pd.DataFrame(cursor.fetchall(), columns=column_names)
    # 关闭连接
    cursor.close()
    connection.close()
    # 输出结果
    return result_df

def Get_PipeConfig_Oracle(oracleinfo):
    # oracle信息
    loaclmachine = oracleinfo['MachineName']
    # 查询语句
    query_str = "SELECT * FROM PIPECONFIG WHERE MACHINENAME = '" + loaclmachine + "'"
    Pipeconfig_df = query_oracle_old(query_str, oracleinfo)
    return Pipeconfig_df
    

def Delete_3Day_DataStatus(oracleinfo, pipeconfig_df):
    """
    @description  : 20250701 修复主从服务器产生的故障交互删除问题
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    Nowtime = int(time.time())
    # 20240708 修改删除数据为6小时，防止表内容过多导致故障
    Delete_time = Nowtime - 6 * 3600
    host = oracleinfo['host']
    port = oracleinfo['port']
    sid = oracleinfo['sid']
    username = oracleinfo['username']
    usersc = oracleinfo['usersc']
    connstr = username + '/' + usersc + '@' + host + ':' + str(port) + '/' + sid
    conn = cx_Oracle.connect(connstr)
    cur = conn.cursor()
    # 20250701 删除本机的管线故障信息
    PipeName_lst = pipeconfig_df['PIPENAME'].values.tolist()
    for PipeName in PipeName_lst:
        sql = f"delete from pipedatastatus where alarmtime < {Delete_time} and PIPENAME = '{PipeName}'"
        delete_NowDataStatus_sql = f"delete from nowpipedatastatus where alarmtime < {Nowtime} and PIPENAME = '{PipeName}'"
        print(sql)
        print(delete_NowDataStatus_sql)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        cur.execute(sql)
        cur.execute(delete_NowDataStatus_sql)
    conn.commit()  # 这里一定要commit才行，要不然数据是不会插入的
    conn.close()

# 示例使用
if __name__ == "__main__":
    oracleinfo_df = pd.read_csv('config/oracleinfo.csv', encoding='utf-8-sig').to_dict('records')
    oracleinfo = oracleinfo_df[-1]
    PipeConfig_df = Get_PipeConfig_Oracle(oracleinfo)
    PipeName_lst = PipeConfig_df['PIPENAME'].values.tolist()
    print(PipeName_lst)
    Delete_3Day_DataStatus(oracleinfo, PipeConfig_df)
    # 读取原始数据
    OragRundata_df = pd.read_csv(r'E:\工作管理\2、项目管理\6、第六采油厂\泵状态管道测试数\tempData\胡十转至胡四联集油管道\InfluxDB_Frequency.csv', encoding='utf-8-sig')
    PipeName = '胡十转至胡四联集油管道'
    OragRecordtime = OragRundata_df['RECORDTIME'].values
    NewPosition = bisect.bisect_left(OragRecordtime, 1747618524000) + 1
    OragRundata_df['PUMP'] = OragRundata_df['PUMP'].str.replace('[{}]', '', regex=True)
    df = OragRundata_df.iloc[NewPosition:].reset_index(drop=True)
    # 泵变化信息存储位置
    filePath = f'temp/{PipeName}'
    full_path = os.path.expanduser(filePath)
    if not os.path.exists(full_path):
        Path(full_path).mkdir(parents=True, exist_ok=True) 
        print(f"已创建文件夹：{full_path}")
    PumpChangeInfo_Path = f'temp/{PipeName}/PumpChangeInfo.csv'
    # 处理数据
    result = process_pump_data_new(df)
    Rn = 60
    while Rn < result.shape[0]:
        NowRundata_df = result.iloc[Rn-60:Rn].reset_index(drop=True)
        PumpRun_df = replace_values(NowRundata_df)
        if type(PumpRun_df) != type('str'):
            PumpStateChangeAnalysis(PumpRun_df, PumpChangeInfo_Path)
        else:
            print(f"第{NowRundata_df.iloc[-1]['RECORDTIME']}行泵状态数据采集错误")
        Rn += 60
    PumpChangeInfo_df = pd.read_csv(PumpChangeInfo_Path, encoding='utf-8-sig')
    print(PumpChangeInfo_df)