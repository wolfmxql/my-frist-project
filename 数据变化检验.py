'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2025-07-21 17:15:17
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2025-07-23 16:00:52
FilePath: \RunStateCheck\数据变化检验.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import pandas as pd
import numpy as np
import bisect
import sys
import os
import 累计法独立运算 as Cum


def find_linear_descending_regions(data, min_len=5, tolerance=5):
    """识别线性或平滑下降的区间，返回 [(start, end), ...]"""
    regions = []
    start = None
    for i in range(1, len(data)):
        if data[i-1] > data[i]:  # 正在下降
            if start is None:
                start = i - 1
        else:
            if start is not None:
                if i - start >= min_len:
                    region = (start, i)
                    regions.append(region)
                start = None
    # 收尾
    if start is not None and len(data) - start >= min_len:
        regions.append((start, len(data)))
    return regions

def in_any_region(i, regions):
    """判断索引 i 是否落在任意线性下降区域中"""
    return any(start <= i <= end for start, end in regions)

def is_sudden_drop(window):
    if window[0] == 0 or window[-1] != 0:
        return False
    diffs = [window[i] - window[i+1] for i in range(len(window)-1)]
    sharp_drop = any(d > window[0]*0.5 for d in diffs[:2])  # 前几步跳得大
    linear_like = all(abs(diffs[i] - diffs[i+1]) < 1e-2 for i in range(len(diffs)-1))
    return sharp_drop and not linear_like

def find_sudden_drops(data):
    linear_regions = find_linear_descending_regions(data, min_len=5)
    sudden_drop_indices = []
    for i in range(len(data) - 4):
        if in_any_region(i, linear_regions):
            continue
        window = data[i:i+5]
        if is_sudden_drop(window):
            sudden_drop_indices.append(i)
    return sudden_drop_indices


data = [100]*100 + [80, 70, 35, 20, 0, 0] + [0]*100 + [50]*10 + [30, 10, 0, 0, 0] + [0]*100
indices = find_sudden_drops(data)
print("Sudden drop positions:", indices)
for pos in indices:
    print(data[pos:pos+5])


''' ---数据恢复验证--- '''
# 需要注意小输差量情况下数据异常验证
# 1、数据本身存在波动，处于高点波动时的输差小于limit导致的报警消除
# 2、检查数据持续小于limit还是偶发性大输差导致的超过limit的报警
def recover_data(Rundata_df, Pytime, limitValue, PipeConfig):
    """
    @description  : 检查数据是否恢复
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    # 1、计算偏移后的flowloss
    # 2、针对存在突降掉0有恢复的情况，消除掉0阶段的Flowloss
    # 3、计算flowloss的恢复情况
    Sz_flow_lst = Rundata_df['SZPL'].values
    Mz_flow_lst = Rundata_df['MZPL'].values
    # 计算偏移后的flowloss
    Sz_flow = Sz_flow_lst[0: -Pytime]
    Mz_flow = Mz_flow_lst[Pytime: 0]
    flowloss_lst = [a - b for a, b in zip(Sz_flow, Mz_flow)]
    # 检查flowloss的恢复情况
    ComTime = int(PipeConfig['COMTIME'])
    CheckNum = int(PipeConfig['CHECKNUM'])
    # Warningflowloss = PipeConfig['WARNINGSHUCHA']
    # 提取最后阶段的数据
    ComdataLength = ComTime + (CheckNum - 1) * 60
    Comdata = flowloss_lst[-ComdataLength:]
    Flowloss_mean = np.mean(Comdata)
    if Flowloss_mean < limitValue:
        # 最后阶段的输差小于报警值，进一步考虑最大输差值影响
        # 此时说明数据已经恢复，不进行报警
        return False
    else:
        # 最后阶段的输差大于报警值，说明数据未恢复
        return True

''' =================数据清洗(待测试使用DataFrame线性填充方法)================= '''
def GetListMiss(list_data):
    """
    @description  :计算数据为空的位置及长度
    ---------
    @param  :list_data要计算的数据列,list类型
    -------
    @Returns  :返回缺失值的位置和长度
    -------
    """
    length = len(list_data)
    a = 0
    an = []
    while a < length:
        if list_data[a] == '' or list_data[a] == None or list_data[a] < -9999:
            an.append(a)
        a = a + 1
    cn = 0
    chalist = []
    if len(an) > 1:
        a0 = an[0]
        c0 = 0
        while cn < len(an):
            ancha = an[cn] - a0
            if ancha != cn - c0:
                chalist.append([an[c0], cn - c0])
                a0 = an[cn]
                c0 = cn
            cn = cn + 1
        chalist.append([an[c0], len(an) - c0])
    elif len(an) == 1:
        chalist.append([an[0], 1])
    return chalist

def UpdataMissValue(list_data, list_miss):
    """
    @description  : 填充更新数据
    ---------
    @param  : list_data：填充完空值后的数列；
                        list_miss：缺失值的长度和位置。
    -------
    @Returns  :
    -------
    """
    aaa = 0
    while aaa < len(list_miss):
        aa = list_miss[aaa]
        bb = 0
        while bb < aa[1]:
            if aa[0] == 0:
                # aa[0] 缺失值的位置，aa[1]缺失的长度
                list_data[aa[0]+bb] = list_data[aa[0]+aa[1]]
            elif aa[0]+aa[1] == len(list_data):
                list_data[aa[0]+bb] = list_data[aa[0]-1]
            else:
                if aa[1]+1 == 0:
                    # 20240717
                    print('DEBUG')
                list_data[aa[0]+bb] = list_data[aa[0]-1]+((list_data[aa[0]+aa[1]]-list_data[aa[0]-1])/(aa[1]+1))*(bb+1)
            bb = bb+1
        aaa = aaa+1
    return list_data

def DataProcessing(LowFData, LFLen, S_Flow_Coefficient, E_Flow_Coefficient):
    # 2024-01-22 修改NewHFDataList数据类型
    NewHFDataList = LowFData.iloc[-int(LFLen):].to_dict('records')
    LFSzpl = [float(x['SZPL'])*S_Flow_Coefficient for x in NewHFDataList]
    LFMzpl = [float(x['MZPL'])*E_Flow_Coefficient for x in NewHFDataList]
    LFSzplMissList = GetListMiss(LFSzpl)
    LFMzplMissList = GetListMiss(LFMzpl)
    if len(LFSzplMissList) != 0:
        LFSzplList = [
            float(x)
            for x in UpdataMissValue(LFSzpl, LFSzplMissList)
        ]
    else:
        LFSzplList = LFSzpl
    if len(LFMzplMissList) != 0:
        LFMzplList = [
            float(x)
            for x in UpdataMissValue(LFMzpl, LFMzplMissList)
        ]
    else:
        LFMzplList = LFMzpl
    '''##################异常值处理###################'''
    UnNormalMissSzplList_L = LFSzplList
    UnNormalMissMzplList_L = LFMzplList
    return UnNormalMissSzplList_L, UnNormalMissMzplList_L
''' ========================================= '''

def Get_PipeConfig(pipename):
    Pipeline_filePath = r''
    Pipeline_df = pd.read_csv(Pipeline_filePath, encoding='utf-8-sig')
    MyPipeline_df = Pipeline_df[Pipeline_df['PIPENAME']==pipename].reset_index(drop=True)
    MyPipeline_dict = MyPipeline_df.to_dict('records')
    PipeConfig_filePath = r''
    PipeConfig_df = pd.read_csv(PipeConfig_filePath, encoding='utf-8-sig')
    MyPipeConfig_df = PipeConfig_df[PipeConfig_df['PIPENAME']==pipename].reset_index(drop=True)
    MyPipeConfig_dict = MyPipeConfig_df.to_dict('records')
    if MyPipeline_df.shape[0] > 0 and MyPipeConfig_df.shape[0] > 0:
        return MyPipeline_dict[-1], MyPipeConfig_dict[-1]
    else:
        return None, None

def Get_OracleINFO():
    OracleINFO_filePath = r''
    OrcaleINFO_df = pd.read_csv(OracleINFO_filePath, encoding='utf-8-sig')
    OrcaleINFO_df = OrcaleINFO_df[OrcaleINFO_df['PIPENAME']==PipeName].reset_index(drop=True)
    if OrcaleINFO_df.shape[0] > 0:
        OrcaleINFO_dict = OrcaleINFO_df.to_dict('records')
        return OrcaleINFO_dict[-1]
    else:
        return None

''' ---数据恢复验证测试案例--- '''
Rundata_Path = r''
Rundata_df = pd.read_csv(Rundata_Path, encoding='utf-8-sig')
PipeName = ''
Pipeline, PipeConfig = Get_PipeConfig(PipeName)
OracleINFO = Get_OracleINFO()
if Pipeline != None and OracleINFO != None:
    Rn = 18000
    while Rn < Rundata_df.shape[0]:
        LowFData_PD = Rundata_df.iloc[Rn-18000: Rn].reset_index(drop=True)
        S_Flow_Coefficient = float(PipeConfig['S_FLOW_COEFFICIENT'])
        E_Flow_Coefficient = float(PipeConfig['E_FLOW_COEFFICIENT'])
        LFLen = int(PipeConfig['LOWFREQUENCYCOUNT'])
        ProcessingSzplList, ProcessingMzplList = DataProcessing(LowFData_PD, LFLen, S_Flow_Coefficient, E_Flow_Coefficient)
        Cum.CumulantsFlowlossCompare(LowFData_PD, Pipeline, PipeConfig, PipeName, OracleINFO, ProcessingSzplList, ProcessingMzplList, AlarmFilePath)
        Rn += 18000
else:
    print(f"{PipeName}参数文件错误")