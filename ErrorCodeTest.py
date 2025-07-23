import pandas as pd
import numpy as np


def CheckAbnormalDataNum(Rundata_df):
    # 需要检查的列
    Column_list = ['SZPL', 'SZYL', 'MZPL', 'MZYL']
    # 定义需要检查的值
    values_to_check = [-1, -2, -3]
    # 检查每列中这些值的分别数量
    for column in Column_list:
        counts = Rundata_df[column].value_counts().reindex(values_to_check, fill_value=0)
        # print(f"列 {column} 中 -1、-2、-3 的数量分别为:")
        for value in values_to_check:
            # print(f"  {value}: {counts[value]}")
            if counts[value] / Rundata_df.shape[0] > 0.3:  # 检查是否超过30%
                # 如果超过30%，则输出警告信息，数据异常：-1、数据为空：-2、通信中断：-3
                if value == -1:
                    return "仪表故障&&采集的原始数据异常"
                elif value == -2:
                    return "采集故障&&采集空值数据较多"
                elif value == -3:
                    return "网络故障&&数据采集模块通信中断"
    return "采集异常数据较少"

def fill_first_n_nans(column):
    """
    @description  : 填充列中前N个连续NaN值为第N+1个位置的有效值
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    # 获取列中的NaN位置
    is_nan = column.isna()
    # 如果列中没有NaN，直接返回原列
    if not is_nan.any():
        return column
    # 20250610 如果列中只有NaN，直接返回原列，非主要数据的列
    if is_nan.all():
        return column
    # 找到第一个非NaN值的位置
    first_valid_idx = column.first_valid_index()
    # 20250616 处理first_valid_idx为None的情况
    if first_valid_idx is None:
        return column
    # 如果第一个非NaN值在位置0，无需处理
    if first_valid_idx == 0:
        return column
    # 获取第一个非NaN值
    first_valid_value = column[first_valid_idx]
    # 将第一个非NaN值之前的所有NaN替换为该值
    column.loc[:first_valid_idx - 1] = first_valid_value
    return column

def DataProcessing(Rundata_df):
    """
    @description  : 清洗采集端提供的异常数据信息
                    采集程序数据采集结果添加数据验证，数据异常：-1、数据为空：-2、通信中断：-3
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    try:
        # 20250616 复制DataFrame以避免修改原始数据
        cleaned_df = Rundata_df.copy()
        # 将所有列中的-1, -2, -3替换为NaN
        cleaned_df.replace([-1, -2, -3], np.nan, inplace=True)
        # 移除全NaN的列，避免插值时出错
        cleaned_df = cleaned_df.dropna(axis=1, how='all')
        # print(cleaned_df)
        # 对所有列进行线性填充，并设置limit_direction='forward'以确保正确插值， 20250618 inplace=True 与处理后的返回不能同时使用
        cleaned_df = cleaned_df.interpolate(method='linear', limit_direction='forward')
        # 如果需要，可以将NaN值替换为其他值（例如0）
        cleaned_df.fillna(method='ffill', inplace=True)
        cleaned_df = cleaned_df.apply(fill_first_n_nans, axis=0)
        return cleaned_df
    except Exception as e:
        print(f"数据清洗过程中发生异常: {e}")
        return None


def AbnormalDataProcessing(Rundata_df):
    """
    @description  : 检查采集数据的异常情况
    ---------
    @param  :   Rundata_df--读取的highFrequency.csv 数据
    -------
    @Returns  :     异常数据小于30%：数据清洗
                    异常数据大于30%：返回异常信息
    -------
    """
    CheckResult = CheckAbnormalDataNum(Rundata_df)
    if CheckResult == "采集异常数据较少":
        ProcessRundata_df = DataProcessing(Rundata_df)
        return False, ProcessRundata_df
    else:
        return True, CheckResult

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


DataPath = r'E:\工作管理\2、项目管理\9_10、第十采油厂\TestData\highFrequency.csv'

Original_highfdata_PD = pd.read_csv(DataPath, encoding = 'utf-8-sig')
DataState, highfdata_PD = AbnormalDataProcessing(Original_highfdata_PD)

print(DataState, highfdata_PD)

oracleinfo_df = pd.read_csv('config/oracleinfo.csv', encoding='utf-8-sig').to_dict('records')
oracleinfo = oracleinfo_df[-1]
PipeConfig_df = Get_PipeConfig_Oracle(oracleinfo)

PipeConfig_dict = PipeConfig_df.to_dict('records')

for PipeConfig in PipeConfig_dict:
    PipeName = PipeConfig['PIPENAME']
    if PipeName == '吴8增至吴九转集油管道':
        Original_highfdata_PD = pd.read_csv(DataPath, encoding = 'utf-8-sig')
        DataState, highfdata_PD = AbnormalDataProcessing(Original_highfdata_PD)
        if DataState:
            # 如果主线或复线数据异常，则提出本次循环，执行下一条管道监测
            print(f'&&&&&&&&{PipeName}&&&&&&&&')
            continue
        print(f'~~{PipeName}~~')


def Extent(FlowArr,PressureArr):
    if type(FlowArr) != type('Str') and type(PressureArr) != type('Str'):
        # 存在异常数据的情况，将异常数据进行替换
        print("ADASGFA")
        is_Flow_smaller_than_001 = all(x <= 0.01 for x in [0.1,0.5])
        is_Pressure_smaller_than_001 = all(x <= 0.01 for x in [1,2])
        if is_Flow_smaller_than_001 and is_Pressure_smaller_than_001:
            # 模块故障或供电问题，则会导致排量压力持续为0
            return True, "压力、排量持续为0"
        else:
            return False, "压力、排量未同步持续为0"
    elif type(FlowArr) == type('Str') and type(PressureArr) == type('Str'):
        return True, f"仪表故障&&{FlowArr}，{PressureArr}"
    elif type(FlowArr) == type('Str'):
        return True, f"仪表故障&&{FlowArr}"
    else:
        return True, f"仪表故障&&{PressureArr}"

import time

FaultTime_int = int(time.time()) - 8800
PipeName = '胡十转至胡四联集油管道'
FaultState_df = pd.DataFrame([{'PIPENAME': '胡十转至胡四联集油管道', 
                               'ALARMTIME': FaultTime_int,
                               'ALARMSTATUS':'网络故障&&采集器网络链接异常'},
                              {'PIPENAME': '胡十转至胡四联集油管道', 
                               'ALARMTIME': FaultTime_int,
                               'ALARMSTATUS':'网络故障&&采集器网络链接异常'},
                              {'PIPENAME': '胡十转至胡四联集油管道', 
                               'ALARMTIME': FaultTime_int,
                               'ALARMSTATUS':'网络故障&&采集器网络链接异常'},
                              {'PIPENAME': '胡十转至胡四联集油管道', 
                               'ALARMTIME': FaultTime_int,
                               'ALARMSTATUS':'软件故障&&缺少模块配置'}])

StartTime_int = int(time.time()) - 28800
PipeFaultState_df = FaultState_df[(FaultState_df['PIPENAME']==PipeName) & (FaultState_df['ALARMTIME'] > StartTime_int)].reset_index(drop=True)
# 故障总时长
PipeFaultLength = PipeFaultState_df.shape[0]
# 最主要的故障
# 分割并提取第一部分（不存在 && 时取整个值）
PipeFaultState_df['FaultType'] = PipeFaultState_df['ALARMSTATUS'].str.split('&&', expand=True)[0]
# 统计频率
counts = PipeFaultState_df['FaultType'].value_counts()
# 找出最频繁的值（可能有多个）
most_common = counts[counts == counts.max()].index.tolist()
print(f"故障总时长为{PipeFaultLength}min；出现频率最多的内容：{most_common}")





FlowArr1 = f"排量数据为{-7.13}m3/h超过常规有效范围"
PressureArr1 = f"压力数据为{100}Mpa超过常规有效范围"

FlowArr2 = [1,2,2,4]
PressureArr2 = [0.13,0.2,0.4,0.3]

SzFailureType, SzFailureINFO = Extent(FlowArr1,PressureArr1)
MzFailureType, MzFailureINFO = Extent(FlowArr1,PressureArr1)

print(SzFailureType, SzFailureINFO)

print(MzFailureType, MzFailureINFO)

if '仪表故障' in SzFailureINFO or '仪表故障' in MzFailureINFO:
    if '仪表故障' in SzFailureINFO and '仪表故障' in MzFailureINFO:
        SzFailureINFO = SzFailureINFO.replace('仪表故障&&', '始站')
        MzFailureINFO = MzFailureINFO.replace('仪表故障&&', '末站')
        FailureInfo = f"仪表故障&&{SzFailureINFO}，{MzFailureINFO}"
    elif '仪表故障' in SzFailureINFO:
        SzFailureINFO = SzFailureINFO.replace('仪表故障&&', '仪表故障&&始站')
        FailureInfo = SzFailureINFO
    else:
        MzFailureINFO = MzFailureINFO.replace('仪表故障&&', '仪表故障&&末站')
        FailureInfo = MzFailureINFO
    print('~~~~~~~~~~~~~~~~~~~~~')
    print(FailureInfo)