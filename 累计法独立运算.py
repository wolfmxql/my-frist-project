import datetime
import pandas as pd
import numpy as np
import bisect
import os
import copy
import pywt
import time
import random
import string
import cx_Oracle
from sqlalchemy import create_engine
import shutil
import math as mh

def Get_CumulativeRundata(Recordtime_list, Rundata_LastTime, lengthtime, pytime):
    minCumulative_StartTime = Rundata_LastTime - (lengthtime + int(pytime)) * 1000
    minCumulative_StartPosition = bisect.bisect_left(Recordtime_list, minCumulative_StartTime)
    RundataLength = len(Recordtime_list[minCumulative_StartPosition:])
    if RundataLength > int((lengthtime + pytime) * 0.8):
        return True
    else:
        Numlength = lengthtime + pytime
        NeedNumlength = int(Numlength * 0.8)
        print(f"不能参与累计运算，需要的数据长度是{Numlength},需要的最小数据长度是{NeedNumlength},截取的数据长度是{RundataLength}")
        ##20250418 print(f"不能参与累计运算，需要的数据长度是{Numlength},需要的最小数据长度是{NeedNumlength},截取的数据长度是{RundataLength}")
        return False

def MaxMinNormalization(x,Max,Min):
    """
    @description  :数据（0，1）标准化处理
    ---------
    @param  :x需要处理的数据，Max数列中的最大值，Min数列中的最小值
    -------
    @Returns  : 返回标准处理后的数据
    -------
    """
    x1 = (x - Min) / (Max - Min)
    return x1

#Z-score标准化
def Z_ScoreNormalization(x,mu,sigma):
    """
    @description  :采用Z-score标准化
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    x = (x - mu) / sigma
    return x

#Sigmoid函数
def sigmoid(X, useStatus):
    """
    @description  :Sigmoid函数输出范围为（0，1），二分类概率计算
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    if useStatus:
        return 1.0 / (1 + np.exp(-float(X)))
    else:
        return float(X)

#获取平均数
def Get_Average(list):
    """
    @description  :数列平均数计算
    ---------
    @param  :list需要计算的数列
    -------
    @Returns  :返回float型数列平均值
    -------
    """
    sum = 0
    for item in list:
        sum += item
    return sum/len(list)

#数列的平均值
def avg_list(data_f,num_type):
    """
    @description  :返回DataFrame格式数据其中一列的平均值
    ---------
    @param  :data_f 输入的DF数据，num_type需要计算的列名称
    -------
    @Returns  : 返回平均值
    -------
    """   
    num_type_list = []
    for row in data_f:
        num_type_list.append(float(row[num_type]))
    num_avg = np.mean(num_type_list)
    return num_avg

#获取总和
def Get_Sum(list):
    """
    @description  :计算数列的总和
    ---------
    @param  :list输入的数列
    -------
    @Returns  :返回float型和
    -------
    """
    sum = 0
    for item in list:
        sum += item
    return sum

# 中位数
def get_median(data):
    """
    @description  :得到数列的中位数
    ---------
    @param  :data需要计算的数列
    -------
    @Returns  :返回中位数
    -------
    """    
    #对所有可迭代的对象进行排序操作
    #data = sorted(data)
    #对数列进行排序
    data.sort()
    size = len(data)
    if size % 2 == 0:
        # 判断列表长度为偶数
        median = (data[size // 2] + data[size // 2 - 1]) / 2
    if size % 2 == 1:
        # 判断列表长度为奇数
        median = data[(size - 1) // 2]
    return median

#快速排序
def quick_sort(data):
    """
    @description  :快速排序
    ---------
    @param  : 需要排序的数列
    -------
    @Returns  :
    -------
    """
    """quick_sort"""
    if len(data) >= 2:
        mid = data[len(data)//2]
        left,right = [], []
        data.remove(mid)
        for num in data:
            if num >= mid:
                right.append(num)
            else:
                left.append(num)
        return quick_sort(left) + [mid] + quick_sort(right)
    else:
        return data

#得到下四分位数
def Get_Q1(data_list):
    """
    @description  :获取数据列的下四分位
    ---------
    @param  :data_list：需要计算的数据列
    -------
    @Returns  :返回下四分位值
    -------
    """
    coutlist= len(data_list)+1
    Q1a= coutlist*0.2
    flootQ1=mh.fmod(Q1a,1)
    intQ1=mh.floor(Q1a)
    Q1 = float(data_list[intQ1])*flootQ1+float(data_list[intQ1+1])*(1-flootQ1)
    return Q1

#得到上四分位数
def Get_Q3(data_list):
    """
    @description  :获取数据列的上四分位
    ---------
    @param  :data_list：需要计算的数据列
    -------
    @Returns  :返回上四分位值
    -------
    """
    coutlist= len(data_list)+1
    Q3a= coutlist*0.8
    flootQ3=mh.fmod(Q3a,1)
    intQ3=mh.floor(Q3a)
    Q3 = float(data_list[intQ3])*flootQ3+float(data_list[intQ3+1])*(1-flootQ3)
    return Q3

#得到中位数
def Get_Q2(data_list):
    """
    @description  :获取数据列的中位
    ---------
    @param  :data_list：需要计算的数据列
    -------
    @Returns  :返回上中位数值
    -------
    """
    coutlist= len(data_list)+1
    Q2a= coutlist*0.5
    flootQ2=mh.fmod(Q2a,1)
    intQ2=mh.floor(Q2a)
    Q2 = float(data_list[intQ2])*flootQ2+float(data_list[intQ2+1])*(1-flootQ2)
    return Q2


# 众数(返回多个众数的平均值)
def Get_Most(list):
    """
    @description  : 计算众数并返回平均值
    ---------
    @param  :list 需要计算的数列
    -------
    @Returns  :返回多个众数的平均值
    -------
    """    
    most = []
    item_num = dict((item, list.count(item))
                    for item in list)
    for k, v in item_num.items():
        if v == max(item_num.values()):
            most.append(k)
    return sum(most) / len(most)


#归一化
def Get_NewList(re_list,P):
    n=0
    Max=0
    Min=0
    p = []
    x=[]
    for row in re_list:
        if P == 0:  # 始站压力
            p.append(row[0])
        elif P == 1:  # 始站排量
            p.append(row[1])
        n+=1
    Max = max(p)
    Min = min(p)
    for row in p:
        x.append(MaxMinNormalization(row,Max,Min))
    return x

def Get_Max_Limit(data_list):
    """
    @description  :根据上下四分位得到阈值上限
    ---------
    @param  :data_list：需要计算的数据列
    -------
    @Returns  :返回阈值上限
    -------
    """
    Q3=Get_Q3(data_list)
    Q1= Get_Q1(data_list)
    #print('上四分位置、下四分位')
    #print(Q3)
    #print(Q1)
    # return Q3+1.5*(Q3-Q1) #2022-06-08 排查异常数据过多情况
    return Q3+3*(Q3-Q1)

def Get_Min_Limit(data_list):
    """
    @description  :根据上下四分位得到阈值下限
    ---------
    @param  :data_list：需要计算的数据列
    -------
    @Returns  :返回阈值下限
    -------
    """
    Q3=Get_Q3(data_list)
    Q1= Get_Q1(data_list)
    # return Q1-1.5*(Q3-Q1) #2022-06-08 排查异常数据过多情况
    return Q1-3*(Q3-Q1)

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

def Get_ExceptionNum_num(datalist):
    """
    @description  :查找异常的数据，并返回异常数的缺失后的数列
    ---------
    @param  : datalist 需要进行处理数列
    -------
    @Returns  :
    -------
    """
    datalistS = copy.deepcopy(datalist)
    # data_list = DataCalculation.quick_sort(datalist)
    data_list = sorted(datalist)
    Q_Max_Limit = Get_Max_Limit(data_list)
    Q_Min_Limit = Get_Min_Limit(data_list)
    # print('最大限:',Q_Max_Limit,'；最小限:',Q_Min_Limit)
    ExceptionNum = []
    aa=0
    new_data_list = []
    while aa<len(datalistS):
        if float(datalistS[aa])>round(Q_Max_Limit,4) or float(datalistS[aa])<round(Q_Min_Limit,4):                                                                                                                            
            ExceptionNum.append([aa,float(datalistS[aa])])
            new_data_list.append(None)
        else:
            new_data_list.append(float(datalistS[aa]))
        aa = aa + 1
    if len(ExceptionNum)>len(datalistS)*0.85:
        return '异常数据过多，请检查仪表是否存在问题'
    else:
        misslist = GetListMiss(new_data_list)
        NewDataList = UpdataMissValue(new_data_list, misslist)
        return NewDataList

def FindDataLength(list_data, CumulativeTime, pytime):#2022-04-28
    # 20240822 判断lowfrequency的时间是否合适
    Rundata_df = pd.DataFrame(list_data)
    Recordtime_list = np.array(Rundata_df['RECORDTIME'].values)
    Rundata_LastTime = int(list_data[-1]['RECORDTIME'])
    # 20250106 测试，消除时间影响
    # NowTime = int(time.time())
    NowTime = int(Rundata_LastTime / 1000)
    if NowTime - (Rundata_LastTime / 1000) < 5 * 60:
        timelist = CumulativeTime.split('~')
        pytimelist = pytime.split('@')
        minpytime = int(pytimelist[0])
        midpytime = int(pytimelist[1])
        maxpytime = int(pytimelist[2])
        minlengthtime = int(timelist[0])
        midlengthtime = int(timelist[1])
        maxlengthtime = int(timelist[2])
        '''~~~~~~~计算各累计计算法需要的数据长度~~~~~~~~'''
        ISminCumulative = Get_CumulativeRundata(Recordtime_list, Rundata_LastTime, minlengthtime, minpytime)
        ISmidCumulative = Get_CumulativeRundata(Recordtime_list, Rundata_LastTime, midlengthtime, midpytime)
        ISmaxCumulative = Get_CumulativeRundata(Recordtime_list, Rundata_LastTime, maxlengthtime, maxpytime)
        '''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
        datalength = len(list_data)
        if datalength >= maxlengthtime + int(maxpytime) and ISmaxCumulative is True:
            return "可最大累计计算"
        elif datalength >= midlengthtime + int(midpytime) and ISmidCumulative is True:
            return "可正常累计计算"
        elif datalength >= minlengthtime + int(minpytime) and ISminCumulative is True:
            return "可最小累计计算"
        else:
            return "不满足累计计算需求"
    else:
        print(f"lowfrequency最后的时间为：{Rundata_LastTime},当前时间是{NowTime}")
        ##20250418 print(f"lowfrequency最后的时间为：{Rundata_LastTime},当前时间是{NowTime}")
        return "不满足累计计算需求，lowfrequency的最后时间与当前时间差距间隔较大"

def Wavelet(data):
    """
    @description  : 小波去噪得到新的数据
    ---------
    @param  : data：需要去噪的数据，一般为UpdataMissValue()函数填充处理完的数据
    -------
    @Returns  : 返回小波去噪后的数据
    -------
    """
    # 20240717 小波去噪运行错误则直接返回原数据
    try:
        if all(x == 0 for x in data):
            #20240717 增加数据全为0的判断，避免除数为0的情况
            return data
        else:
            # Create wavelet object and define parameters
            w = pywt.Wavelet('db8')  # 选用Daubechies8小波
            maxlev = pywt.dwt_max_level(len(data), w.dec_len)
            # print("maximum level is " + str(maxlev))
            threshold = 0.9  # Threshold for filtering
            # Decompose into wavelet components, to the level selected:
            coeffs = pywt.wavedec(data, 'db8', level=maxlev)  # 将信号进行小波分解
            # plt.figure()
            for i in range(1, len(coeffs)):
                coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]))  # 将噪声滤波
            datarec = pywt.waverec(coeffs, 'db8')  # 将信号进行小波重构
            return datarec
    except Exception as e:
        print(str(e))
        return data

def GetNewDataList(UnNormalDataList):
    if type(UnNormalDataList) == type('s'):
        #print(UnNormalDataList)
        return '退出'
    else:
        bbb = GetListMiss(UnNormalDataList)
        NewNumList = UpdataMissValue(UnNormalDataList,bbb)
        ListWavelet = Wavelet(NewNumList)
        return ListWavelet

def OutliersAndWaveLetOno(DataList):
    '''##################异常值处理###################'''
    UnNormalMissDataList_L = Get_ExceptionNum_num(DataList)
    '''##################小波去噪后数据###################'''
    if type(UnNormalMissDataList_L) != type('str'):
        HFNewSzplList = GetNewDataList(UnNormalMissDataList_L)
        # Alarminfo = UnNormalMissSzplList_L
        AlarmType = '正常'
    else:
        Alarminfo = UnNormalMissDataList_L
        AlarmType = '预警'
    return AlarmType

def remove_anomalies(lst_A, lst_B):
    # 定义异常值列表
    anomalies = [-1, -2, -3, '', None]
    
    # 收集A和B中异常值的索引
    abnormal_indices = set()
    
    # 检查列表A
    for i, value in enumerate(lst_A):
        if value in anomalies or (isinstance(value, (int, float)) and value < -9999):
            abnormal_indices.add(i)
    
    # 检查列表B
    for i, value in enumerate(lst_B):
        if value in anomalies or (isinstance(value, (int, float)) and value < -9999):
            abnormal_indices.add(i)
    
    # 转换为有序列表并降序排列，以便从后向前删除
    indices_to_remove = sorted(abnormal_indices, reverse=True)
    
    # 从后向前删除异常值，避免索引变化问题
    for i in indices_to_remove:
        if i < len(lst_A):
            del lst_A[i]
        if i < len(lst_B):
            del lst_B[i]
    
    return lst_A, lst_B

def CheckIncorrect(LowFData_PD, Cumulativetime, pytime):
    """
    @description  : 20250711 检查数据异常状态，
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    szpl = LowFData_PD['SZPL'].values
    SzplOrag_lst = szpl.tolist()
    mzpl = LowFData_PD['MZPL'].values
    MzplOrag_lst = mzpl.tolist()
    DataSumi = Cumulativetime + pytime
    Szpl_lst = SzplOrag_lst[-DataSumi:-pytime]
    Mzpl_lst = MzplOrag_lst[-Cumulativetime:]
    CalSzpl_lst, CalMzpl_lst = remove_anomalies(Szpl_lst, Mzpl_lst)
    if len(CalSzpl_lst) / len(Szpl_lst) < 0.8:
        return '异常'
    else:
        return '正常'


def GetListSum(Dlist):
    sum = 0
    for row in Dlist:
        sum = sum + float(row)
    return sum

def GetNowTimeStr():
    # 获取当前时间
    current_time = datetime.now()
    # 格式化时间字符串
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_time

def GetDataNowTimeStr(DataNowTime_int):
    """
    @description  : 获取当前计算点的时间字符串
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(DataNowTime_int / 1000)))
    return time_str
    
def GetShuChaComparaCumulative(LowFData_PD, szpl, mzpl, Cumulativetime, pytime,
                               normalshucha, warningshucha, PipeName):
    """
    @description  :累计法比较
    ---------
    @param  :
                    normalshucha    常规状态下的正常输差
                    warningshucha   报警输差阈值
    -------
    @Returns  :返回报警信息“'疑似发生泄漏，请核实!' + flowloss_info” or “'管道运行正常'”
    -------
    """
    if pytime == 0:
        pytime = 1
    DataSumi = Cumulativetime + pytime
    # 20250711 从原始数据进行计算与处理
    SzplOrag_array = LowFData_PD['SZPL'].values
    SzplOrag_lst = SzplOrag_array.tolist()
    MzplOrag_array = LowFData_PD['MZPL'].values
    MzplOrag_lst = MzplOrag_array.tolist()
    # 20250711 改变数据
    SzplAOrag = SzplOrag_lst[-DataSumi:-pytime]
    MzplAOrag = MzplOrag_lst[-Cumulativetime:]
    # 20250711 增加累计值计算中异常数据的筛查
    SzplA, MzplA = remove_anomalies(SzplAOrag, MzplAOrag)
    ShuChaS = (GetListSum(SzplA) - GetListSum(MzplA)) / Cumulativetime
    ##20250418 print(str(Cumulativetime)+'累计法计算输差：', ShuChaS)
    print(str(Cumulativetime)+'累计法计算输差：'+str(ShuChaS))
    cumulative_flowloss = (np.sum(SzplA) - np.sum(MzplA))/3600
    # 20250107 增加报警阈值的写入
    flowloss_info = f"{str(int(Cumulativetime/60))}分钟累计输差为：{str(round(cumulative_flowloss, 2))}方，平均输差为{round(ShuChaS, 2)}方/时，报警阈值为{warningshucha + normalshucha}方/时"
    if ShuChaS - normalshucha >= warningshucha:
        # 20241025 增加满足条件的信息记录
        alarmInfo = flowloss_info
        ##20250418 print(f"{PipeName}{flowloss_info}")
        Parameters = f"Cumulativetime{Cumulativetime}+pytime{pytime}"
        conditions = f"ShuChaS{ShuChaS} - normalshucha{normalshucha} >= warningshucha{warningshucha}"
        s0 = GetListSum(SzplA)
        s1 = GetListSum(MzplA)
        s2 = np.sum(SzplA)
        s3 = np.sum(MzplA)
        cumuData = f"GetListSum(SzplA){s0} - GetListSum(MzplA){s1}；np.sum(SzplA){s2} - np.sum(MzplA){s3}"
        # NowTimeStr = GetNowTimeStr()
        Nowtime_int = int(LowFData_PD.iloc[-1]['RECORDTIME'])
        NowTimeStr = GetDataNowTimeStr(Nowtime_int)
        strInfo = f"当前时间：{NowTimeStr};\r\n参数：{Parameters};\r\n运算数据：{cumuData};\r\n比较结果：{conditions};\r\n报警内容：{alarmInfo};\r\n"
        filepath = f"temp/{PipeName}"
        if os.path.exists(filepath) is False:
            os.makedirs(filepath)
        with open(filepath + "/CumulativeFlowloss.txt", 'a') as file:
            file.write(strInfo)
        # end
        return '疑似发生泄漏，请核实！' + flowloss_info
    else:
        return '管道运行正常'

def CreatFileFolder(PipeName, NOWTIME):
    """
    @description  :根据当前时间创建文件夹
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    year = time.strftime('%Y', time.localtime(int(NOWTIME/1000)))
    month = time.strftime('%m', time.localtime(int(NOWTIME/1000)))
    day = time.strftime('%d', time.localtime(int(NOWTIME/1000)))
    fileYear = os.getcwd() + '/temp/' + PipeName + '/' + str(year)
    fileMonth = fileYear + '/' + str(month)
    fileDay = fileMonth + '/' + str(day)
    AlarmInfoFilePath = 'temp/' + PipeName + '/' + str(year) + '/' + str(month) + '/' + str(day)
    if not os.path.exists(fileYear):
        os.mkdir(fileYear)
        os.mkdir(fileMonth)
        os.mkdir(fileDay)
    else:
        if not os.path.exists(fileMonth):
            os.mkdir(fileMonth)
            os.mkdir(fileDay)
        else:
            if not os.path.exists(fileDay):
                os.mkdir(fileDay)
    path2 = os.path.lexists(AlarmInfoFilePath + '/CumulantsFlowlossinfo.csv')
    if path2 is False:
        TilteName = ['PipeName', 'Recordtime', 'Recordtime_Str', 'CumuFlowloss', 'L_CumuFlowloss', 'M_CumuFlowloss', 'S_CumuFlowloss']
        RercordINFO_df = pd.DataFrame(columns=TilteName)
        RercordINFO_df.to_csv(AlarmInfoFilePath + '/CumulantsFlowlossinfo.csv', encoding='utf-8-sig', index=False)
    return AlarmInfoFilePath + '/CumulantsFlowlossinfo.csv'

def Warting_CumulantsFlowlossInfo_Csv(NowRecordtime, PipeConfig, Alarminfo):
    """
    @description  : 将累计计算的每分钟计算结果写入CSV文件
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    PipeName = str(PipeConfig['PIPENAME'])
    # 提取年月日
    AlarmInfoFilePath = CreatFileFolder(PipeName, NowRecordtime)
    # 拆分报警信息Alarminfo.split('&')
    flowloss_list = Alarminfo.split('&')
    if len(flowloss_list) == 3:
        L_CumuFlowloss = flowloss_list[2]
        M_CumuFlowloss = flowloss_list[1]
        S_CumuFlowloss = flowloss_list[0]
    elif len(flowloss_list) == 2:
        L_CumuFlowloss = ''
        M_CumuFlowloss = flowloss_list[1]
        S_CumuFlowloss = flowloss_list[0]
    elif len(flowloss_list) == 1:
        L_CumuFlowloss = ''
        M_CumuFlowloss = ''
        S_CumuFlowloss = flowloss_list[0]
    else:
        L_CumuFlowloss = ''
        M_CumuFlowloss = ''
        S_CumuFlowloss = ''
    if '疑似发生泄漏' in Alarminfo:
        if '疑似发生泄漏' in S_CumuFlowloss:
            CumuFlowloss = S_CumuFlowloss.replace(',','_')
        elif '疑似发生泄漏' in M_CumuFlowloss:
            CumuFlowloss = M_CumuFlowloss.replace(',','_')
        elif '疑似发生泄漏' in L_CumuFlowloss:
            CumuFlowloss = L_CumuFlowloss.replace(',','_')
        else:
            CumuFlowloss = '无累计记录'
    else:
        CumuFlowloss = '无累计记录'
    timeArray = time.localtime(int(NowRecordtime/1000))
    Time_Str = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    RecordINFO = {'PipeName':PipeName, 'Recordtime':int(NowRecordtime/1000), 'Recordtime_Str':Time_Str, 'CumuFlowloss':CumuFlowloss, 'L_CumuFlowloss':L_CumuFlowloss.replace(',','_'), 'M_CumuFlowloss':M_CumuFlowloss.replace(',','_'), 'S_CumuFlowloss':S_CumuFlowloss.replace(',','_')}
    RecordINFO_df = pd.DataFrame([RecordINFO])
    RecordINFO_df.to_csv(AlarmInfoFilePath, mode='a', index=False, header=False)
    return Time_Str+'累计计算信息写入完成！'

def GetCumulativeFlowBalance(LowFData_PD, szplist, mzpllist, CumulativeTime, CumPytime,
                             normalshucha, pipename, oracleinfo, datalength, warningShucha, PipeConfig, NowRecordtime): #2022-04-28
    """
    @description  : 长时累计法输差统计
    ---------
    @param  :
    -------
    @Returns  : “'疑似发生泄漏，请核实!' + flowloss_info” or “'管道运行正常'”
    -------
    """
    # 2023-12-31,增加常规输差的自动计算
    '''
    # 累计法不进行常规输差计算
    IsAutoNormalFlowloss = int(PipeConfig['ISAUTONORMALFLOWLOSS'])
    if IsAutoNormalFlowloss != 0:
        normalshucha = AutoCalculationFlowloss.Get_Normal_FlowLoss(PipeConfig)
    '''
    pytimelist = CumPytime.split('@')
    minpytime = int(pytimelist[0])
    midpytime = int(pytimelist[1])
    maxpytime = int(pytimelist[2])
    timelist = CumulativeTime.split('~')
    timelong0 = int(timelist[0])
    timelong1 = int(timelist[1])
    timelong2 = int(timelist[2])
    shuchalist = warningShucha.split('~')
    minshucha = float(shuchalist[0])
    midshucha = float(shuchalist[1])
    maxshucha = float(shuchalist[2])
    ''' ~~~~~累计报警信息跟踪~~~~~ '''
    filepath = f"temp/{pipename}"
    if os.path.exists(filepath) is False:
        os.makedirs(filepath)
    if os.path.lexists(filepath + "/CumulativeFlowloss.txt") is False:
        with open(filepath + "/CumulativeFlowloss.txt", 'a') as file:
            strInfo = f"累计偏移时间：{CumPytime}、累计计算时长：{CumulativeTime}、累计报警输差：{warningShucha}"
            file.write(strInfo)
    ''''''
    if datalength == '可最大累计计算':
        pytime = maxpytime
        Szpllist0Outliers = OutliersAndWaveLetOno(szplist[-(timelong0+pytime):])
        mzpllist0Outliers = OutliersAndWaveLetOno(mzpllist[-(timelong0+pytime):])
        IncorrectDataState = CheckIncorrect(LowFData_PD, timelong0, pytime)
        if Szpllist0Outliers == '正常' and mzpllist0Outliers == '正常' and IncorrectDataState == '正常':
            Alarminfo0 = GetShuChaComparaCumulative(LowFData_PD, szplist, mzpllist, timelong0,
                                                    minpytime, normalshucha,
                                                    minshucha, pipename)
        else:
            Alarminfo0 = '异常值过多，不进行累计计算'
        '''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
        Szpllist1Outliers = OutliersAndWaveLetOno(szplist[-(timelong1+pytime):])
        mzpllist1Outliers = OutliersAndWaveLetOno(mzpllist[-(timelong1+pytime):])
        IncorrectDataState = CheckIncorrect(LowFData_PD, timelong1, pytime)
        if Szpllist1Outliers == '正常' and mzpllist1Outliers == '正常' and IncorrectDataState == '正常':
            Alarminfo1 = GetShuChaComparaCumulative(LowFData_PD, szplist, mzpllist, timelong1,
                                                midpytime, normalshucha,
                                                midshucha, pipename)
            if '泄漏' in Alarminfo1:
                with open(filepath + "/CumulativeFlowloss.txt", 'a') as file:
                    strInfo = f"管道{pipename}偏移时间：{midpytime}、计算时长：{timelong1}、报警输差：{midshucha}"
                    file.write(strInfo)
        else:
            Alarminfo1 = '异常值过多，不进行累计计算'
        '''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
        Szpllist2Outliers = OutliersAndWaveLetOno(szplist[-(timelong2+pytime):])
        mzpllist2Outliers = OutliersAndWaveLetOno(mzpllist[-(timelong2+pytime):])
        IncorrectDataState = CheckIncorrect(LowFData_PD, timelong2, pytime)
        if Szpllist2Outliers == '正常' and mzpllist2Outliers == '正常' and IncorrectDataState == '正常':
            Alarminfo2 = GetShuChaComparaCumulative(LowFData_PD, szplist, mzpllist, timelong2,
                                                maxpytime, normalshucha,
                                                maxshucha, pipename)
        else:
            Alarminfo2 = '异常值过多，不进行累计计算'
        Alarminfo = Alarminfo0 + '&' + Alarminfo1 + '&' + Alarminfo2
    elif datalength == '可正常累计计算':
        pytime = midpytime
        Szpllist0Outliers = OutliersAndWaveLetOno(szplist[-(timelong0+pytime):])
        mzpllist0Outliers = OutliersAndWaveLetOno(mzpllist[-(timelong0+pytime):])
        IncorrectDataState = CheckIncorrect(LowFData_PD, timelong0, pytime)
        if Szpllist0Outliers == '正常' and mzpllist0Outliers == '正常' and IncorrectDataState == '正常':
            Alarminfo0 = GetShuChaComparaCumulative(LowFData_PD, szplist, mzpllist, timelong0,
                                                minpytime, normalshucha,
                                                minshucha, pipename)
        else:
            Alarminfo0 = '异常值过多，不进行累计计算'
        '''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
        Szpllist1Outliers = OutliersAndWaveLetOno(szplist[-(timelong1+pytime):])
        mzpllist1Outliers = OutliersAndWaveLetOno(mzpllist[-(timelong1+pytime):])
        IncorrectDataState = CheckIncorrect(LowFData_PD, timelong1, pytime)
        if Szpllist1Outliers == '正常' and mzpllist1Outliers == '正常' and IncorrectDataState == '正常':
            Alarminfo1 = GetShuChaComparaCumulative(LowFData_PD, szplist, mzpllist, timelong1,
                                                midpytime, normalshucha,
                                                midshucha, pipename)
            if '泄漏' in Alarminfo1:
                with open(filepath + "/CumulativeFlowloss.txt", 'a') as file:
                    strInfo = f"管道{pipename}偏移时间：{midpytime}、计算时长：{timelong1}、报警输差：{midshucha}"
                    file.write(strInfo)
        else:
            Alarminfo1 = '异常值过多，不进行累计计算'
        Alarminfo = Alarminfo0 + '&' + Alarminfo1
    elif datalength == '可最小累计计算':
        pytime = minpytime
        Szpllist0Outliers = OutliersAndWaveLetOno(szplist[-(timelong0+pytime):])
        mzpllist0Outliers = OutliersAndWaveLetOno(mzpllist[-(timelong0+pytime):])
        IncorrectDataState = CheckIncorrect(LowFData_PD, timelong0, pytime)
        if Szpllist0Outliers == '正常' and mzpllist0Outliers == '正常' and IncorrectDataState == '正常':
            Alarminfo0 = GetShuChaComparaCumulative(LowFData_PD, szplist, mzpllist, timelong0, minpytime, normalshucha, minshucha, pipename)
        else:
            Alarminfo0 = '异常值过多，不进行累计计算'
        Alarminfo = Alarminfo0
    else:
        Alarminfo = '不进行累积计算'
    Warting_CumulantsFlowlossInfo_Csv(NowRecordtime, PipeConfig, Alarminfo)
    # 20250107 写入累计计算信息记录
    ##20250418 print(f"▲▲▲▲▲{pipename}累计计算完成，{Alarminfo}")
    if Alarminfo.find('发生泄漏') != -1:
        # AlarmINFO = '疑似发生泄漏，请核实'
        # 20240515 累计输差分段报警信息，从最大的开始
        flowloss_list = Alarminfo.split('&')
        if len(flowloss_list) == 3:
            if '发生泄漏' in flowloss_list[2]:
                AlarmINFO = flowloss_list[2]
            elif '发生泄漏' in flowloss_list[1]:
                AlarmINFO = flowloss_list[1]
            elif '发生泄漏' in flowloss_list[0]:
                AlarmINFO = flowloss_list[0]
            else:
                AlarmINFO = '管道运行正常'
        elif len(flowloss_list) == 2:
            if '发生泄漏' in flowloss_list[1]:
                AlarmINFO = flowloss_list[1]
            elif '发生泄漏' in flowloss_list[0]:
                AlarmINFO = flowloss_list[0]
            else:
                AlarmINFO = '管道运行正常'
        elif len(flowloss_list) == 1:
            if '发生泄漏' in flowloss_list[0]:
                AlarmINFO = flowloss_list[0]
            else:
                AlarmINFO = '管道运行正常'
        else:
            AlarmINFO = '管道运行正常'
    else:
        AlarmINFO = '管道运行正常'
    return AlarmINFO

''' ======================================= '''
def all_values_same(lst):
    return len(set(lst)) == 1

def Calculation_Flowloss(Rundata_Part_0, Rundata_Part_1):
    SzplList = Rundata_Part_0['SZPL'].tolist()
    MzplList = Rundata_Part_1['MZPL'].tolist()
    Flowloss = np.mean(SzplList) - np.mean(MzplList)
    return Flowloss

def Get_Flow_Change_MaxMin(Rundata_PD_Com, AcceptableEFlow):
    # Rundata_PD_Com = Rundata_PD.iloc[-((Check_Num-1)*60 + Comtime):].reset_index(drop=True)
    Mz_Flow_list = Rundata_PD_Com['MZPL'].tolist()
    Fn = 20
    Max_list = []
    # 对比时间点前后20秒数据的平均值
    while Fn < len(Mz_Flow_list) - 20:
        Flow_0 = np.mean(Mz_Flow_list[Fn - 20:Fn])
        Flow_1 = np.mean(Mz_Flow_list[Fn:Fn + 20])
        Max_list.append(Flow_0 - Flow_1)
        Fn += 1
    # 找到变化最大/最小的数据
    Flow_Change_Max = np.max(Max_list)
    Flow_Change_Min = np.min(Max_list)
    # 找到数值变化最大/最小的位置
    Max_index = Max_list.index(Flow_Change_Max)
    Min_index = Max_list.index(Flow_Change_Min)
    # 获取数值变化最大前后的数据
    Flow_Before_Max = np.mean(Mz_Flow_list[Max_index + 10:Max_index + 20])
    Flow_After_Max = np.mean(Mz_Flow_list[Max_index + 20:Max_index + 30])
    # 获取数值变化最小前后的数据
    Flow_Before_Min = np.mean(Mz_Flow_list[Min_index + 10:Min_index + 20])
    Flow_After_Min = np.mean(Mz_Flow_list[Min_index + 20:Min_index + 30])
    # 数据掉线的情况下，数据下降较快，前后10秒的数据差值接近掉线前数据的平均值
    if Flow_Before_Max - Flow_After_Max > 0.9 * Flow_Before_Max and Flow_After_Min - Flow_Before_Min > 0.9 * Flow_After_Min:
        if Flow_After_Max < AcceptableEFlow and Flow_Before_Min < AcceptableEFlow:
            return '掉线'
        else:
            return '末站排量异常掉落'
    else:
        return '正常'

def Check_Datadead(Rundata_PD):
    """
    @description  : 检查最后15分钟的数据是否为死值
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    PartRundata_PD = Rundata_PD.iloc[-900:, ].reset_index(drop=True)
    Szpl = PartRundata_PD['SZPL'].tolist()
    Mzpl = PartRundata_PD['MZPL'].tolist()
    Szyl = PartRundata_PD['SZYL'].tolist()
    Mzyl = PartRundata_PD['MZYL'].tolist()
    SzplInfo = all_values_same(Szpl)
    MzplInfo = all_values_same(Mzpl)
    SzylInfo = all_values_same(Szyl)
    MzyllInfo = all_values_same(Mzyl)
    if SzplInfo is True and np.mean(Szpl) > 0.05:
        SzplCheckInfo = '始站排量数据死值；'
    else:
        SzplCheckInfo = ''
    if MzplInfo is True and np.mean(Mzpl) > 0.05:
        MzplCheckInfo = '末站排量数据死值；'
    else:
        MzplCheckInfo = ''
    if SzylInfo is True and np.mean(Szyl) > 0.05:
        SzylCheckInfo = '始站压力数据死值；'
    else:
        SzylCheckInfo = ''
    if MzyllInfo is True and np.mean(Mzyl) > 0.05:
        MzylCheckInfo = '末站压力数据死值；'
    else:
        MzylCheckInfo = ''
    # 20240801 增加停泵期间的死值判断、暂不处理
    CheckInfo = SzplCheckInfo + MzplCheckInfo + SzylCheckInfo + MzylCheckInfo
    return CheckInfo


def Get_Flowloss_Curve(PipeConfig, PipeLine, Rundata_PD):
    """
    @description  : 当产生报警时检查是否符合掉线的条件
    ---------
    @param  : Rundata_PD 从lowFre.csv中获取的数据，Comtime输差计算时间
                Check_Num 检查次数
    -------
    @Returns  :
    -------
    """
    PipeName = str(PipeConfig['PIPENAME'])
    # 20250107 修改lowFrequency.csv的数据，直接由参数导入
    Rundata_LastTime = int(Rundata_PD.iloc[-1]['RECORDTIME'])
    # 2024-01-22 由于PipeLine采用的Class类型不能用属性
    # 2024-02-05 增加PipeLine 的类型判断
    if type(PipeLine) == dict:
        Interval_Time = int(PipeLine['INTERVAL_TIME'])
        Shucha_Limit = float(PipeLine['SHUCHA_LIMIT'])
    else:
        ##20250418 print('▲检查PipeLine的数据类型：' + str(type(PipeLine)))
        print('▲检查PipeLine的数据类型：' + str(type(PipeLine)))
        Interval_Time = int(PipeLine.interval_time)
        Shucha_Limit = float(PipeLine.shucha_limit)
    Comtime = int(PipeConfig['COMTIME'])
    Check_Num = int(PipeConfig['CHECKNUM'])
    Rundata_Recordtime_List = Rundata_PD['RECORDTIME'].tolist()
    Cn = 0
    Flowloss_List = []
    DatadeadInfo = Check_Datadead(Rundata_PD)
    if '死值' in DatadeadInfo:
        print(str(DatadeadInfo))
        Check_info = DatadeadInfo
    else:
        while Cn < Check_Num:
            TPart0_Time_S = Rundata_LastTime - ((Check_Num - Cn - 1) * 60 + Comtime + Interval_Time) * 1000
            TPart0_Time_E = TPart0_Time_S + Comtime * 1000
            TPart1_Time_S = TPart0_Time_S + Interval_Time * 1000
            TPart1_Time_E = TPart0_Time_S + (Comtime + Interval_Time) * 1000
            # 根据时间点定位到数据位置
            TPart0_index_S = bisect.bisect_left(Rundata_Recordtime_List, TPart0_Time_S)
            TPart0_index_E = bisect.bisect_left(Rundata_Recordtime_List, TPart0_Time_E)
            TPart1_index_S = bisect.bisect_left(Rundata_Recordtime_List, TPart1_Time_S)
            TPart1_index_E = bisect.bisect_left(Rundata_Recordtime_List, TPart1_Time_E)
            # 计算不同时间段的输差
            Rundata_Part_0 = Rundata_PD.iloc[TPart0_index_S:TPart0_index_E].reset_index(drop=True)
            Rundata_Part_1 = Rundata_PD.iloc[TPart1_index_S:TPart1_index_E].reset_index(drop=True)
            Part_Flowloss = Calculation_Flowloss(Rundata_Part_0, Rundata_Part_1)
            Flowloss_List.append(Part_Flowloss)
            Cn += 1
        # 提取最后一次输差检测的数据1
        StartInspection_Rundata_Time = Rundata_LastTime - ((Check_Num - 1) * 60 + Comtime + Interval_Time) * 1000
        StartInspection_Rundata_Position = bisect.bisect_left(Rundata_Recordtime_List, StartInspection_Rundata_Time)
        Inspection_Rundata = Rundata_PD.iloc[StartInspection_Rundata_Position:,].reset_index(drop=True)
        Mz_Flow_List = Inspection_Rundata['MZPL'].tolist()
        # 找到数据中小于0.2的数据,常规输送下很难出现末站排量小于0.2的数据，改用参数设置，config中添加参数AcceptableEFlow（默认0.2）
        AcceptableEFlow = float(PipeConfig['ACCEPTABLEEFLOW'])
        if len(Flowloss_List) > 1:
            # 输差减小（可能是数据恢复正常的表现，所以要验证） 20250107 在输差变化判断中增加量级0.95避免波动的损失
            if Flowloss_List[-1] < Flowloss_List[-2] * 0.95:
                # 取最后45秒数据
                CheckData_StartTime_0 = Rundata_LastTime - 45000 - Interval_Time * 1000
                CheckData_EndTime_0 = Rundata_LastTime - 45000 
                CheckData_StartPosition_0 = bisect.bisect_left(Rundata_Recordtime_List, CheckData_StartTime_0)
                CheckData_EndPosition_0 = bisect.bisect_left(Rundata_Recordtime_List, CheckData_EndTime_0)
                CheckData_StartTime_1 = Rundata_LastTime - 45000
                CheckData_StartPosition_1 = bisect.bisect_left(Rundata_Recordtime_List, CheckData_StartTime_1)
                CheckData_PD_0 = Rundata_PD[CheckData_StartPosition_0:CheckData_EndPosition_0]
                CheckData_PD_1 = Rundata_PD[CheckData_StartPosition_1:]
                # 检查最后45秒的数据，判断是否已恢复到正常状态
                LastTime_Shucha = CheckData_PD_0['SZPL'].mean() - CheckData_PD_1['MZPL'].mean()
                # 最后45秒的数据输差小于输差要求，则需要判断是否为数据恢复
                if LastTime_Shucha < Shucha_Limit:
                    # 需要排除末站排量自动波动不稳定的情况（小量泄漏后仍有可能恢复到接近正常的数据）
                    # 检查数据是否存在连续的低值情况
                    # 如果数据出现陡降并且陡升的情况则认为是掉线
                    Check_info = Get_Flow_Change_MaxMin(Inspection_Rundata, AcceptableEFlow)
                else:
                    Check_info = '正常'
            else:
                Check_info = '正常'
        else:
            Fn = 20
            Max_list = []
            # 对比每个时间点前后20秒数据的平均值
            while Fn < len(Mz_Flow_List) - 20:
                Flow_0 = np.mean(Mz_Flow_List[Fn - 20:Fn])
                Flow_1 = np.mean(Mz_Flow_List[Fn:Fn + 20])
                Max_list.append(Flow_0 - Flow_1)
                Fn += 1
            # 找出数据下降最大的时间点位置
            Flow_Change_Max = np.max(Max_list)
            Flow_Change_Max_Position = Max_list.index(Flow_Change_Max)
            # 找到数值变化最大的位置
            # 2024-04-19 陈反馈问题
            # Max_index = Mz_Flow_List.index(Flow_Change_Max)
            Max_index = Flow_Change_Max_Position + 20
            # 获取数值变化最大前后的数据,因为数据的计算时间是从20起
            Flow_Before = np.mean(Mz_Flow_List[Max_index + 10:Max_index + 20])
            Flow_After = np.mean(Mz_Flow_List[Max_index + 20:Max_index + 30])
            # 数据掉线的情况下，数据下降较快，前后10秒的数据差值接近掉线前数据的平均值
            if Flow_Before - Flow_After > 0.9 * Flow_Before:
                if Flow_After < AcceptableEFlow:
                    Check_info = '掉线'
                else:
                    Check_info = '数据异常掉线'
            else:
                Check_info = '正常'
    return Check_info
''' ======================================= '''
def create_string_number(n):
    """ 生成一串指定位数的字符+数组混合的字符串 """
    m = random.randint(1, n)
    a = "".join([str(random.randint(0, 9)) for _ in range(m)])
    b = "".join([random.choice(string.ascii_letters) for _ in range(n - m)])
    return ''.join(random.sample(list(a + b), n))

def timeStamp(timeNum):
    """ 输入毫秒级的时间，转出正常格式的时间 """
    timeStamp = int(timeNum / 1000)
    timeArray = time.localtime(timeStamp)
    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    return otherStyleTime
''' ======================================= '''

def Query_For_Oracle(sql_query):
    Oracleinfo_df = pd.read_csv('Data/oracleinfo.csv', encoding='utf-8-sig')
    oracleinfo = Oracleinfo_df.to_dict('records')
    host = oracleinfo[-1]['host']
    port = oracleinfo[-1]['port']
    service_name = oracleinfo[-1]['sid']
    username = oracleinfo[-1]['username']
    password = oracleinfo[-1]['usersc']
    # 创建数据库连接引擎
    oracle_dsn = cx_Oracle.makedsn(host, port, service_name)
    engine = create_engine(f'oracle+cx_oracle://{username}:{password}@{oracle_dsn}')
    # 使用pandas读取数据
    data = pd.read_sql_query(sql_query, engine)
    # 查询结果字段转为大写
    data.columns = [col.upper() for col in data.columns]
    return data

def Verify_CumulantsFlowloss_Continuous_Csv(AlarmFilePath, PipeConfig, PipeLine, NewAlarmTime):
    """
    @description  : 验证累计法持续性输差过大的报警消除
    ---------
    @param  :
    -------
    @Returns  : 返回True 表示可以重新报警 返回False表示无需重新报警
    -------
    """
    # 20240820 增加过程记录文件
    CumulantsFlowloss_FilePath = AlarmFilePath + '/CumulantsFlowlossinfo.csv'
    if os.path.lexists(CumulantsFlowloss_FilePath) is False:
        # 20240820
        ##20250418 print("CumulantsFlowlossinfo.csv文件不存在，创建新文件")
        print("CumulantsFlowlossinfo.csv文件不存在，创建新文件")
        # 20240819 修改内容'AlarmType', 'AlarmInfo' → 'L_CumuFlowloss', 'M_CumuFlowloss', 'S_CumuFlowloss'
        TilteName = ['PipeName', 'Recordtime', 'Recordtime_Str', 'CumuFlowloss',  'L_CumuFlowloss', 'M_CumuFlowloss', 'S_CumuFlowloss']
        RercordINFO_df = pd.DataFrame(columns=TilteName)
        RercordINFO_df.to_csv(CumulantsFlowloss_FilePath, encoding='utf-8-sig', index=False)
        ISAlarm = False
        # 20240822
        cumulative_flowloss_IS = False
        IS_NowAlarm = False
        Flowloss_info = ''
    else:
        Alarm_df = pd.read_csv(CumulantsFlowloss_FilePath, encoding='utf-8-sig')
        PipeName = PipeConfig['PIPENAME']
        if Alarm_df.shape[0] == 0:
            # 20240820
            ##20250418 print("CumulantsFlowlossinfo.csv文件为空")
            print("CumulantsFlowlossinfo.csv文件为空")
            # pipealarminfo.csv 无数据记录的情况
            ISAlarm = False
            IS_NowAlarm = False
            Flowloss_info = ''
        else:
            # 20240619 提取累计输差记录文件的最后时间，作为Oracle中报警表的查询记录
            Last_Recordtime = int(Alarm_df.iloc[-1]['Recordtime']) * 1000
            Last_4h_Recordtime = Last_Recordtime - 4 * 3600 * 1000
            Alarm_Time_Str = timeStamp(Last_4h_Recordtime)
            # 找出BJNR列里包含'输差过大'的行 
            Cumulative_Time_Str = str(PipeLine.cumulative_time)
            Cumulative_Time_list = Cumulative_Time_Str.split('~')
            mask = ~Alarm_df['CumuFlowloss'].str.contains('疑似发生泄漏', regex=False)
            # 找mask中为True的行行号
            row_indices = np.where(mask)[0]
            # 20240820
            ##20250418 print(f"疑似发生泄漏的报警时间序列为：{row_indices}")
            # {位置：，长度：}
            Position_list = []
            if row_indices.shape[0] == 0:
                # pipealarminfo.csv全部都为‘疑似发生泄漏’或无数据记录的情况
                if Alarm_df.shape[0] == 0:
                    # pipealarminfo.csv 无数据记录的情况
                    ISAlarm = False
                    Flowloss_info = ''
                else:
                    # pipealarminfo.csv 全部行都包含‘疑似发生泄漏’的情况
                    # Start 20240619 调试版本用的是报警记录文件，改成Oracle读取 
                    SQL_Str = f"select * from MONITORALARM where BJTIME > TO_DATE('{Alarm_Time_Str}','yyyy-mm-dd hh24:mi:ss') and PIPENAME = '{PipeName}' and BJLX = '泄漏报警'"
                    ##20250418 print(SQL_Str)
                    PipeAlarmInfo_df = Query_For_Oracle(SQL_Str)
                    #  End 20240619 调试版本用的是报警记录文件，改成Oracle读取 
                    if PipeAlarmInfo_df.shape[0] == 0:
                        # 如果4小时内不存在该管道的报警，则将报警信息写入
                        ISAlarm = True
                        # 20240820
                        ##20250418 print(f"{Alarm_Time_Str}之后{PipeName}没有泄漏报警信息，可直接写入报警")
                        print(str(Alarm_Time_Str) + "之后" + str(PipeName) + "没有泄漏报警信息，可直接写入报警")
                        Flowloss_info = Alarm_df.iloc[-1]['CumuFlowloss']
                    else:
                        # PipeAlarminfo.csv记录的最后时间
                        LastRecordtime = int(Alarm_df.iloc[-1]['Recordtime'])
                        # 提取最后的报警时间，改用从数据库中记录的时间 20240619
                        # LastAlarmtime = int(PipeAlarmInfo_df.iloc[-1]['报警时间戳'])
                        LastAlarmTimeStr = str(PipeAlarmInfo_df.iloc[-1]['BJTIME'])
                        LastAlarm_TimeType = datetime.strptime(LastAlarmTimeStr, "%Y-%m-%d %H:%M:%S")
                        LastAlarmtime = int(LastAlarm_TimeType.timestamp())
                        if LastRecordtime - LastAlarmtime > 4 * 3600:
                            ISAlarm = True
                            # 20240820
                            ##20250418 print(f"数库中记录的{PipeName}最后的报警时间是：{LastAlarmTimeStr}，报警时间相差{LastRecordtime - LastAlarmtime}秒大于4小时，报警信息写入数据库")
                            Flowloss_info = Alarm_df.iloc[-1]['CumuFlowloss']
                        else:
                            ISAlarm = False
                            Flowloss_info = ''
                            # 20240820
                            ##20250418 print(f"数库中记录的{PipeName}最后的报警时间是：{LastAlarmTimeStr}，报警时间相差{LastRecordtime - LastAlarmtime}秒小于4小时，报警信息不写入数据库")
                        print("数库中记录的str(PipeName)最后的报警时间是：" + str(LastAlarmTimeStr) + "，报警时间相差" + str(LastRecordtime - LastAlarmtime) + "秒, 是否大于4小时")
            else:
                # 检验的记录数据中存在非报警信息的情况
                Position = row_indices[0]
                Rn = 1
                length = 1
                if Rn > len(row_indices) - 1:
                    # 当只有一条信息的时候
                    Position_list.append({'Position':Position, 'Length':length})
                while Rn < len(row_indices):
                    if Rn != len(row_indices) - 1:
                        if row_indices[Rn] == row_indices[Rn-1] + 1:
                            length += 1
                        else:
                            Position_list.append({'Position':Position, 'Length':length})
                            Position = row_indices[Rn]
                            length = 1
                    else:
                        if row_indices[Rn] == row_indices[Rn-1] + 1:
                            length += 1
                            Position_list.append({'Position':Position, 'Length':length})
                        else:
                            Position = row_indices[Rn]
                            Position_list.append({'Position':Position, 'Length':1})
                    Rn += 1
                # 20240820
                ##20250418 print(f"非输差过大的时间段为：{Position_list}")
                # 剔除未出现'输差过大'时间长度大于1/2计算时间长度的位置
                Pos_len_list = []
                for Pos_len in Position_list:
                    # 未报警时间30分钟内则认为是合理
                    if Pos_len['Length'] > int(int(Cumulative_Time_list[0]) / (2 * 60)):
                        Pos_len_list.append(Pos_len)
                # 20240820
                ##20250418 print(f"非报警时间段超过最小累计时间长度{int(int(Cumulative_Time_list[0]) / (2 * 60))}一半的时间段有{Pos_len_list}")
                # Pos_len_list表示所有非报警信息的开始时间及持续时间的记录
                if len(Pos_len_list) > 0:
                    Alarm_Position = Pos_len_list[-1]
                    # 找到最新‘疑似发生泄漏’的在CumulantsFlowlossinfo.csv中起点的位置
                    Alarm_Time_Str = Alarm_df.iloc[Alarm_Position['Position'] + Alarm_Position['Length'] - 1]['Recordtime_Str']
                    # 将时间字符串转换为datetime对象
                    dt = datetime.strptime(Alarm_Time_Str, "%Y-%m-%d %H:%M:%S")
                    # 将datetime对象转换为时间戳
                    timestamp = int(dt.timestamp())
                    # 20240820
                    ##20250418 print(f"最后出现输差过大的时间戳：{timestamp}，时间字符串为{Alarm_Time_Str}")
                    print("最后出现输差过大的时间戳：" + str(timestamp) + "，时间字符串为" + str(Alarm_Time_Str))
                    # 从数据库中读取管道的报警信息
                    SQL_Str = f"select * from MONITORALARM where BJTIME > TO_DATE('{Alarm_Time_Str}','yyyy-mm-dd hh24:mi:ss') and PIPENAME = '{PipeName}' and BJLX = '泄漏报警'"
                    ##20250418 print(SQL_Str)
                    Query_AlarmResult = Query_For_Oracle(SQL_Str)
                    # 提取最后的排量差记录
                    LastTITLEC = str(Alarm_df.iloc[-1]['Recordtime'])
                    Flowloss = LastTITLEC.replace("疑似发生泄漏，请核实！",'')
                    Flowloss_info = ""
                    if Query_AlarmResult.shape[0] == 0:
                        # 从连续输差过大开始报警时间后，数据库中没有报警记录则需要写入报警信息
                        ISAlarm = True
                        # 20240820
                        ##20250418 print(f"数据库中非输差过大的最后时间{Alarm_Time_Str}之后未出现{PipeName}的报警信息")
                        print("数据库中非输差过大的最后时间" + str(Alarm_Time_Str) + "之后未出现" + str(PipeName) + "的报警信息")
                        Flowloss_info = Alarm_df.iloc[-1]['CumuFlowloss']
                    else:
                        ISAlarm = False
                        # 如果连续8小时未产生报警则给出报警提示
                        LastRecordtime_Str = str(Alarm_df.iloc[-1]['Recordtime_Str'])
                        # 将时间字符串转换为datetime对象
                        LastRecordtime_dt = datetime.strptime(LastRecordtime_Str, "%Y-%m-%d %H:%M:%S")
                        # 将datetime对象转换为时间戳
                        LastRecordtime = int(LastRecordtime_dt.timestamp())
                        # 20240820 数据库记录的最后报警时间
                        LastAlarmTimeStr = str(Query_AlarmResult.iloc[-1]['BJTIME'])
                        LastAlarm_TimeType = datetime.strptime(LastAlarmTimeStr, "%Y-%m-%d %H:%M:%S")
                        LastAlarmtime = int(LastAlarm_TimeType.timestamp())
                        # 20240820
                        ##20250418 print(f"数据库中{Alarm_Time_Str}之后出现{PipeName}的报警信息是：")
                        print("数据库中" + str(Alarm_Time_Str) + "之后存在" + str(PipeName) + "的报警")
                        # print(Query_AlarmResult)
                        if LastRecordtime - LastAlarmtime > 4 * 3600:
                            ISAlarm = True
                            # 20240820
                            ##20250418 print(f"数库中记录的{PipeName}最后的报警时间是：{LastAlarmTimeStr}，报警时间相差{LastRecordtime - LastAlarmtime}秒大于4小时，报警信息写入数据库")
                            Flowloss_info = Alarm_df.iloc[-1]['CumuFlowloss']
                        else:
                            ISAlarm = False
                            Flowloss_info = ""
                            # 20240820
                            ##20250418 print(f"数库中记录的{PipeName}最后的报警时间是：{LastAlarmTimeStr}，报警时间相差{LastRecordtime - LastAlarmtime}秒小于4小时，报警信息不写入数据库")
                        print("数库中记录的" + str(PipeName) + "最后的报警时间是：" + str(LastAlarmTimeStr) + "，报警时间相差是否大于4小时")
                else:
                    ##20250418 print("非报警时间段数据中没有超过最小累计时间长度一半的记录，默认为当日累计输差全部超过报警阈值")
                    # 20240821 针对累计输差全部记录报警的情况
                    LastRecordtime_Str = str(Alarm_df.iloc[-1]['Recordtime_Str'])
                    # 将时间字符串转换为datetime对象
                    LastRecordtime_dt = datetime.strptime(LastRecordtime_Str, "%Y-%m-%d %H:%M:%S")
                    # 将datetime对象转换为时间戳
                    LastRecordtime = int(LastRecordtime_dt.timestamp())
                    '''~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
                    SQL_Str = f"select * from MONITORALARM where BJTIME > TO_DATE('{Alarm_Time_Str}','yyyy-mm-dd hh24:mi:ss') and PIPENAME = '{PipeName}' and BJLX = '泄漏报警'"
                    ##20250418 print(SQL_Str)
                    Query_AlarmResult = Query_For_Oracle(SQL_Str)
                    '''~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
                    if Query_AlarmResult.shape[0] > 0:
                        ISAlarm = False
                        Flowloss_info = ""
                        ##20250418 print("全部为输差过大的情况下，4小时内数据库中存在报警")
                    else:
                        ISAlarm = True
                        Flowloss_info = Alarm_df.iloc[-1]['CumuFlowloss']
                        ##20250418 print("全部为输差过大的情况下，4小时内数据库中无报警信息，需要新写入")
            # Start 20240619 检查数据库中最后的报警时间是否符合报警间隔要求
            Query_LastAlarm_Str = f"select * from (select * from MONITORALARM where PIPENAME = '{PipeName}' and BJLX = '泄漏报警' order by BJTIME desc) where rownum=1"
            Query_LastAlarm_Result = Query_For_Oracle(Query_LastAlarm_Str)
            if Query_LastAlarm_Result.shape[0] == 0:
                # 没有报警信息时
                IS_Interval_Time = True
            else:
                LastAlarmTimeStr_Orcale = str(Query_LastAlarm_Result.iloc[-1]['BJTIME'])
                LastAlarmTime_Oracle_Type = datetime.strptime(LastAlarmTimeStr_Orcale, "%Y-%m-%d %H:%M:%S")
                LastAlarmTime_Oracle_Int = int(LastAlarmTime_Oracle_Type.timestamp())
                # 报警时间间隔要求，取出最大累计时间一半和报警间隔时间一半的最大值
                AlarmIntervalTime = int(PipeConfig['ALARMINTERVALTIME'])
                MaxCumulativeTime = int(int(Cumulative_Time_list[-1])/2)
                IntervalTime = max(AlarmIntervalTime, MaxCumulativeTime)
                if int(NewAlarmTime)/1000 - LastAlarmTime_Oracle_Int > IntervalTime:
                    # 20240820
                    ##20250418 print(f"新的报警时间{NewAlarmTime}-数据库最后的报警时间{LastAlarmTime_Oracle_Int} > 时间间隔要求{IntervalTime}可进行报警")
                    IS_Interval_Time = True
                else:
                    # 20240820
                    ##20250418 print(f"新的报警时间{NewAlarmTime}-数据库最后的报警时间{LastAlarmTime_Oracle_Int} < 时间间隔要求{IntervalTime}不进行报警")
                    IS_Interval_Time = False
                print("新报警时间" + str(int(NewAlarmTime)/1000) + "数据库最后报警时间：" + str(LastAlarmTime_Oracle_Int) + "时间间隔要求" + str(IntervalTime))
            if IS_Interval_Time is True and ISAlarm is True:
                IS_NowAlarm = True
            else:
                IS_NowAlarm = False
                # 20240822
        # 20240822
        Last3Alarm_df = Alarm_df.iloc[-4:,].reset_index(drop=True)
        count_rows_with = Last3Alarm_df['CumuFlowloss'].str.contains('疑似发生泄漏').sum()
        if count_rows_with > 2:
            cumulative_flowloss_IS = True
        else:
            cumulative_flowloss_IS = False
    return IS_NowAlarm, Flowloss_info, cumulative_flowloss_IS

''' ====================================== '''
def GetLastAlarmInfo(pipename, oracleinfo, newbjnr):
    """
    @description  :查询管线的最后一条报警记录
    ---------
    @param  :pipename 管线名称, oracleinfo 数据库连接信息 ,newbjnr 查询内容
    -------
    @Returns  :
    -------
    """
    host = oracleinfo[0]['host']
    port = oracleinfo[0]['port']
    sid = oracleinfo[0]['sid']
    username = oracleinfo[0]['username']
    usersc = oracleinfo[0]['usersc']
    connstr = username + '/' + usersc + '@' + host + ':' + port + '/' + sid
    conn = cx_Oracle.connect(connstr)
    cursor = conn.cursor()
    ##20250418 print('数据库连接完成')
    print('数据库连接完成')
    # 20250718
    if '泄漏' in newbjnr and '站内产生' not in newbjnr:
        alarminfostr = newbjnr.split('。')
        cursor.execute("SELECT * FROM (SELECT * FROM MONITORALARM WHERE PIPENAME='" + pipename + "' and BJNR like '%" + alarminfostr[0] + "%' and BJNR not like '%站内产生%' ORDER BY BJTIME DESC) WHERE ROWNUM<=1")
    elif '请检查流量计是否故障' in newbjnr:
        # 20241018 增加流量计故障由于 输差数字不同产生的不同报警判断，导致连续报警
        cursor.execute("SELECT * FROM (SELECT * FROM MONITORALARM WHERE PIPENAME='" + pipename + "' and BJNR like '%请检查流量计是否故障%' ORDER BY BJTIME DESC) WHERE ROWNUM<=1")
    else:
        cursor.execute("SELECT * FROM (SELECT * FROM MONITORALARM WHERE PIPENAME='" + pipename + "' and BJNR = '" + newbjnr + "' ORDER BY BJTIME DESC) WHERE ROWNUM<=1")
    rows = cursor.fetchall()  # 得到所有数据集
    cursor.close()
    conn.close()
    return rows

def GetOraclePipeAlarm(pipename, oracleinfo, newbjnr, alarmintervaltime, bjlx, BJTIME):
    """
    @description  : 泄漏报警后10分钟内不进行预警信息写入数据库
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    if len(oracleinfo) == 1:
        bjnr = 0  # 填报的泄漏核实情况
        AlarmRecordtime = ''
        # 获取数据库中最后一条相同报警内容的报警信息
        rows = GetLastAlarmInfo(pipename, oracleinfo, newbjnr)
        ##20250418 print('☆☆☆☆☆新的报警内容是：'+str(newbjnr)+'☆☆☆☆☆')
        print('☆☆☆☆☆新的报警内容是：'+str(newbjnr)+'☆☆☆☆☆')
        # 获取数据库中最后一条泄漏报警的报警信息
        LastAlarmRows = GetLastAlarmInfo(pipename, oracleinfo, '泄漏')
        ##20250418 print('☆☆☆☆☆最后的泄漏报警信息：☆☆☆☆☆')
        ##20250418 print(LastAlarmRows)
        ##20250418 print('☆☆☆☆☆☆☆☆☆')
        LastAlarmRcordtime = ''
        # 获取最后一条泄漏报警信息的时间 LastAlarmRcordtime
        for LastAlarmInfo in LastAlarmRows:
            LastAlarmRcordtime = LastAlarmInfo[4]
        ##20250418 print('*****查询当前时间点最后的报警记录********')
        ##20250418 print(rows)
        print('查询当前时间点最后的报警记录：报警时间' + str(LastAlarmRcordtime))
        # 获取最后一条预警信息的时间 AlarmRecordtime
        for row in rows:
            bjnr = row[3]
            AlarmRecordtime = row[4]
        # 2024-01-22 修改报警时间比较，采用新的报警时间，而不是当前时间
        timeArray = time.strptime(str(BJTIME), "%Y-%m-%d %H:%M:%S")
        now = int(time.mktime(timeArray))
        # now = int(time.time())
        if AlarmRecordtime != '':
            timeArray = time.strptime(str(AlarmRecordtime), "%Y-%m-%d %H:%M:%S")
            timeStamp = int(time.mktime(timeArray))
        else:
            timeStamp = 1685087418 # 2023-06-26 15:50:18
        # 泄漏报警不为空，LastAlarmRcordtime 最后一条泄漏报警的时间 20231013
        if LastAlarmRcordtime != '':
            LasttimeArray = time.strptime(str(LastAlarmRcordtime), "%Y-%m-%d %H:%M:%S")
            LasttimeStamp = int(time.mktime(LasttimeArray))
            ##20250418 print('最后报警时间：'+str(LasttimeStamp)+'；当前时间：'+str(now)+'；间隔时间要求：'+str(int(alarmintervaltime)*60)+'；实际时间差：'+str(now - LasttimeStamp)+'；新报警内容：'+str(newbjnr))
            print('最后报警时间：'+str(LasttimeStamp)+'；当前时间：'+str(now)+'；间隔时间要求：'+str(int(alarmintervaltime)*60)+'；实际时间差：'+str(now - LasttimeStamp)+'；新报警内容：'+str(newbjnr))
            if bjlx == '泄漏报警':
                RecordAlarmTime = LasttimeStamp
            else:
                RecordAlarmTime = timeStamp
            if now - RecordAlarmTime < int(alarmintervaltime) * 60:
                ##20250418 print('当前时间：'+str(now)+'报警记录时间：'+str(RecordAlarmTime)+'报警间隔要求：'+str(int(alarmintervaltime) * 60))
                print('当前时间：'+str(now)+'报警记录时间：'+str(RecordAlarmTime)+'报警间隔要求：'+str(int(alarmintervaltime) * 60))
                return '重复报警'
            elif now - RecordAlarmTime < 600 and '泄漏' not in newbjnr:
                ##20250418 print('当前时间：'+str(now)+'报警记录时间：'+str(RecordAlarmTime)+'泄漏报警内容'+str(newbjnr))
                print('当前时间：'+str(now)+'报警记录时间：'+str(RecordAlarmTime)+'泄漏报警内容'+str(newbjnr))
                return '重复报警'
            else:
                return '重新开始计算'
        else:
            # 数据库中存在相同的预警
            if AlarmRecordtime != '':
                LasttimeArray = time.strptime(str(AlarmRecordtime), "%Y-%m-%d %H:%M:%S")
                LasttimeStamp = int(time.mktime(LasttimeArray))
                ##20250418 print('最后预警时间：'+str(timeStamp)+'；当前时间：'+str(now)+'；间隔时间要求：'+str(int(alarmintervaltime)*60)+'；实际时间差：'+str(now - timeStamp)+'；新报警内容：'+str(newbjnr))
                print('最后预警时间：'+str(timeStamp)+'；当前时间：'+str(now)+'；间隔时间要求：'+str(int(alarmintervaltime)*60)+'；实际时间差：'+str(now - timeStamp)+'；新报警内容：'+str(newbjnr))
                if bjlx == '泄漏报警':
                    RecordAlarmTime = LasttimeStamp
                else:
                    RecordAlarmTime = timeStamp
                if now - RecordAlarmTime < int(alarmintervaltime) * 60:
                    ##20250418 print('当前时间：'+str(now)+'报警记录时间：'+str(RecordAlarmTime)+'报警间隔要求'+str(int(alarmintervaltime) * 60))
                    print('当前时间：'+str(now)+'报警记录时间：'+str(RecordAlarmTime)+'报警间隔要求'+str(int(alarmintervaltime) * 60))
                    return '重复报警'
                elif now - RecordAlarmTime < 600 and '泄漏' not in newbjnr:
                    ##20250418 print('当前时间：'+str(now)+'报警记录时间：'+str(RecordAlarmTime)+'泄漏报警内容'+str(newbjnr))
                    print('当前时间：'+str(now)+'报警记录时间：'+str(RecordAlarmTime)+'泄漏报警内容'+str(newbjnr))
                    return '重复报警'
                else:
                    return '重新开始计算'
            else:
                return '重新开始计算'
    else:
        ##20250418 print('数据库信息异常，请查看oracleinfo.csv文件')
        print('数据库信息异常，请查看oracleinfo.csv文件')
        os._exit(1)


def insertdata(ID, PIPENAME, PIPESTATE, BJTIME, BJNR, BJLX, BJLEVEL,
               oracleinfo, ALARMINTERVALTIME, StrPumpInfo, AlarmFilePath):
    """
    @description  : 报警信息写入数据库
    ---------
    @param  :  ID：ID，PIPENAME：管线名称，
                        PIPESTATE：运行状态，
                        BJTIME：报警时间，
                        BJNR：报警内容，
                        BJLX：报警类型，
                        BJLEVEL：报警等级，
                        oracleinfo：数据库信息，
                        ALARMINTERVALTIME：报警间隔时间
    -------
    @Returns  :
    -------
    """
    #oracleinfo = CsvFileData.read_csv('Data/oracleinfo.csv')
    #ALARMINTERVALTIME分钟时间范围内 不重复写入数据库
    ISWOracle = GetOraclePipeAlarm(PIPENAME, oracleinfo, BJNR,
                                   ALARMINTERVALTIME,BJLX,BJTIME)
    ##20250418 print(ISWOracle)
    print('报警信息是否写入数据库：' + str(ISWOracle))
    if len(oracleinfo) == 1 and ISWOracle == '重新开始计算':
        if BJLX == '预警' or BJLX == '泄漏报警':
            ##20250418 print('★★★★★★★★★★'+AlarmFilePath+'/AlarmPumpInfo.csv★★★★★★★★★★★')
            host = oracleinfo[0]['host']
            port = oracleinfo[0]['port']
            sid = oracleinfo[0]['sid']
            username = oracleinfo[0]['username']
            usersc = oracleinfo[0]['usersc']
            connstr = username + '/' + usersc + '@' + host + ':' + str(port) + '/' + sid
            conn = cx_Oracle.connect(connstr)
            cur = conn.cursor()
            # 20250418 修改报警内容为“数据正常”时的报警类型
            if '数据正常' in BJNR:
                BJLX = '预警'
            if BJNR.find('启动中') > -1:
                # 如果程序启动中只做记录，不需要填写处置信息
                sql = "INSERT into MONITORALARM(ID,PIPENAME,PIPESTATE,BJTIME,BJNR,BJLX,BJLEVEL,VELEL,TRLEL,TRCON) VALUES('"+ID+"','"+PIPENAME + \
                    "','"+PIPESTATE + \
                    "',to_date('"+BJTIME+"','yyyy/mm/dd hh24:mi:ss'),'" + \
                    BJNR+"','"+BJLX + "','"+BJLEVEL + "','1','2','正常')"
            else:
                sql = "INSERT into MONITORALARM(ID,PIPENAME,PIPESTATE,BJTIME,BJNR,BJLX,BJLEVEL) VALUES('"+ID+"','"+PIPENAME + \
                    "','"+PIPESTATE + \
                    "',to_date('"+BJTIME+"','yyyy/mm/dd hh24:mi:ss'),'" + \
                    BJNR+"','"+BJLX + "','"+BJLEVEL + "')"
            ##20250418 print('~~~~~写入数据的语句是：'+sql)
            print('~~~~~写入数据的语句是：'+sql)
            cur.execute(sql)
            ##20250418 print(PIPENAME + '报警信息插入完成！')
            conn.commit()  # 这里一定要commit才行，要不然数据是不会插入的
            conn.close()
        else:
            ##20250418 print('管线运行正常，不写入数据库')
            print('管线运行正常，不写入数据库')
    elif ISWOracle == '重复报警':
        ##20250418 print('规定时间内有同样的报警信息',str(BJNR))
        print('规定时间内有同样的报警信息')
    else:
        ##20250418 print('数据库信息异常，请查看oracleinfo.csv文件')
        print('数据库信息异常，请查看oracleinfo.csv文件')
        os._exit(1)

def Check_PipeDataStatus_File():
    """
    @description  : 检测运行过程记录文件是否存在
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    year = time.strftime('%Y', time.localtime(time.time()))
    month = time.strftime('%m', time.localtime(time.time()))
    day = time.strftime('%d', time.localtime(time.time()))
    fileRunInfo = os.getcwd() + '/Data/RunInfo'
    filepath = fileRunInfo + '/' + str(year) + '/' + str(month) + '/' + str(day) + '日-PipeDataStatus.csv'
    return filepath

def copy_file(source_path, destination_path):
    try:
        shutil.copy2(source_path, destination_path)
        print(f"文件 {source_path} 已成功复制到 {destination_path}")
    except FileNotFoundError:
        print(f"错误：源文件 {source_path} 未找到。")
    except PermissionError:
        print(f"错误：没有权限复制文件到 {destination_path}。")
    except Exception as e:
        print(f"发生未知错误：{e}")

def Write_Pipeline_DataStatus(pipename, datastatutime, alarmstatus):
    """
    @description  : 预警信息都写入PipeDataStatus.csv
    ---------
    @param  :   pipename：管线名称, 
                datastatutime：预警信息时间,
                alarmstatus：预警信息内容
    -------
    @Returns  :
    -------
    """
    DataStatus_Filepath = 'Data/RunInfo/PipeDataStatus.csv'
    Diary_DataStatus_Filepath = Check_PipeDataStatus_File()
    # 20240715 测试程序将NowTime转为报警时间，正式程序不需要改该转换
    # NowTime = int(time.time())
    NowTime = int(time.time())
    # 20240715 防止因泄漏报警误判断为故障的情况
    # 20240805 部分故障信息不写入
    if '泄漏' not in alarmstatus and '正常' not in alarmstatus and '末站排量数据掉0较多' not in alarmstatus and '输差过大' not in alarmstatus and '数据采集不完整' not in alarmstatus:
        DataStatus_df = pd.DataFrame([{'PIPENAME': pipename, 'DATASTATUTIME':datastatutime, 'ALARMTIME':NowTime, 'ALARMSTATUS':alarmstatus}])
        if os.path.exists(Diary_DataStatus_Filepath):
            # 20250124 增加防止日-pipedatastatus.csv文件读取错误导致故障无法写入的问题
            try:
                Diary_DataStatus = pd.read_csv(Diary_DataStatus_Filepath, encoding='utf-8-sig')
                # 20240715
                Query_SameAlarm_Result = Diary_DataStatus[(Diary_DataStatus['PIPENAME'] == pipename) & (Diary_DataStatus['ALARMSTATUS'] == alarmstatus)].reset_index(drop=True)
                # 20240716 可能存在朝朝结果的情况
                if Query_SameAlarm_Result.shape[0] > 0:
                    Last_Result_Time = int(Query_SameAlarm_Result.iloc[-1]['ALARMTIME'])
                    # 测试20250114 
                    if NowTime - Last_Result_Time >= 50:
                        # if NowTime - Last_Result_Time >= 1:
                        # 如果两次故障时间大于50秒（不是同一次发现的的故障）则重新写入
                        Diary_DataStatus = pd.concat([Diary_DataStatus, DataStatus_df], ignore_index=True)
                else:
                    Diary_DataStatus = pd.concat([Diary_DataStatus, DataStatus_df], ignore_index=True)
            except Exception as e:
                print(e)
                ##20250418 print(f"{pipename}{Diary_DataStatus_Filepath}文件读取错误，重新生成故障文件")
                Diary_DataStatus = DataStatus_df
        else:
            Diary_DataStatus = DataStatus_df
        Diary_DataStatus.to_csv(Diary_DataStatus_Filepath, encoding='utf-8-sig', index=False)
        # 20240717 增加PipeDataStatus.csv文件的数据判断
        # 检查用于提交数据状态的文件是存在
        if os.path.exists(DataStatus_Filepath):
            try:
                All_DataStatus = pd.read_csv(DataStatus_Filepath, encoding='utf-8-sig')
                Query_Same_Result = All_DataStatus[(All_DataStatus['PIPENAME'] == pipename) & (All_DataStatus['ALARMSTATUS'] == alarmstatus)].reset_index(drop=True)
                if Query_Same_Result.shape[0] > 0:
                    try:
                        # 20240728 增加文件错误的强制执行时间为0
                        # 20240812 Query_SameAlarm_Result 修改为 Query_Same_Result
                        Last_Same_Result_Time = int(Query_Same_Result['ALARMTIME'].max())
                    except Exception as e:
                        print(str(e))
                        Last_Same_Result_Time = 0
                    if NowTime - Last_Same_Result_Time >= 50:
                        All_DataStatus = pd.concat([All_DataStatus, DataStatus_df], ignore_index=True)
                else:
                    All_DataStatus = pd.concat([All_DataStatus, DataStatus_df], ignore_index=True)
            except Exception as e:
                print(e)
                Nowtime_int = int(time.time())
                destination_path = DataStatus_Filepath.replace('.csv', f'{Nowtime_int}_error.csv')
                copy_file(DataStatus_Filepath, destination_path)
                print(str(e))
                All_DataStatus = DataStatus_df
        else:
            All_DataStatus = DataStatus_df
        # 只保留最近3天的管道状态信息
        AlarmTime_list = All_DataStatus['ALARMTIME'].values
        D3Time = NowTime - 3 * 24 * 3600
        D3Position = bisect.bisect_left(AlarmTime_list, D3Time)
        New_DataStatus_df = All_DataStatus.iloc[D3Position:]
        New_DataStatus_df.to_csv(DataStatus_Filepath, encoding='utf-8-sig', index=False)

''' ====================================== '''
def CumulantsFlowlossCompare(LowFData_PD, MyPipeline, PipeConfig, pipename, oracleinfo, ProcessingSzplList, ProcessingMzplList, AlarmFilePath):
    """
    @description  : 累计法流量平衡计算，2024-03-21增加累计法一半时间内是否存在报警的判断
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    # Start 20250107 测试，将日期的文件加创建更改未计算时间点的
    DatalastTime = int(int(LowFData_PD.iloc[-1]['RECORDTIME']) / 1000)
    dt_object = datetime.datetime.fromtimestamp(DatalastTime)
    year = dt_object.strftime("%Y")
    month = dt_object.strftime("%m")
    day = dt_object.strftime("%d")
    AlarmFilePath = f"temp/{pipename}/{year}/{month}/{day}"
    # end 20250107 测试，将日期的文件加创建更改未计算时间点的
    '''~~~~~~~~~~~~~~~累计法泄漏监测~~~~~~~~~~~~~~~~'''
    warningShucha = str(PipeConfig['WARNINGSHUCHA'])
    # 2024-01-24 由于参数引用是将LowFData修改为LowFData_PD导致后继的数据类型错误
    LowFData = LowFData_PD.reset_index(drop=True).to_dict('records')
    # 20240619 
    NowRecordtime = int(LowFData_PD.iloc[-1]['RECORDTIME'])
    # 根据数据长度判断采用不同的方法
    datalength = FindDataLength(LowFData, MyPipeline.cumulative_time, PipeConfig['CUMINTERVALTIME'])
    # 开始进行累计法监测计算 20240619 增加了NowRecordtime参数
    # 20250711 修改累计计算中异常值的处理
    TBCumulativeAlarm = GetCumulativeFlowBalance(LowFData_PD, ProcessingSzplList, ProcessingMzplList, MyPipeline.cumulative_time, PipeConfig['CUMINTERVALTIME'], MyPipeline.normalshucha, pipename, oracleinfo, datalength, warningShucha, PipeConfig, NowRecordtime)
    ##20250418 print('累计输差计算完成：', TBCumulativeAlarm)
    print('累计输差计算完成：'+str(TBCumulativeAlarm))
    # int(PipeConfig['CumulantsFLCompare']) == 1 累计法开启
    # if TBCumulativeAlarm == '疑似发生泄漏，请核实': 20240515 报警信息中写入了输差信息
    if '疑似发生泄漏，请核实' in TBCumulativeAlarm:
        DataInspection = Get_Flowloss_Curve(PipeConfig, MyPipeline, LowFData_PD)
        if DataInspection != '掉线' and '死值' not in DataInspection:
            # 泄漏点信息含定位
            Postion = random.randint(100, 2000) / 10
            # PipeAlarmInfo = '疑似发生泄漏（长时累计输差过大），请核实。泄漏点距离始站：' + str(Postion) + '米'
            # 20240515 增加累计输差具体数值
            PipeAlarmInfo = TBCumulativeAlarm + '泄漏点距离始站：' + str(Postion) + '米'
        else:
            if '死值' in DataInspection:
                PipeAlarmInfo = DataInspection
            else:
                PipeAlarmInfo = '末站排量数据短时掉线'
    else:
        PipeAlarmInfo = '运行正常'
    if PipeAlarmInfo != '运行正常':
        AlarmIntervalTime = int(PipeConfig['ALARMINTERVALTIME']) * 2
        ID = str(create_string_number(18))
        nowtimeInt = int(LowFData[-1]['RECORDTIME'])
        # 2024-01-24 修改报警时间点为数据的最后时刻
        alarm_time_1 = int(nowtimeInt)
        # alarm_time_1 = int(int(time.time()) * 1000)
        AlarmTime = timeStamp(int(alarm_time_1))
        ISAlarmTime = int(PipeConfig['ISALARMTIME'])
        nowtimeStr = timeStamp(nowtimeInt)
        StrPumpInfo = [str('当前数据时间：' + str(nowtimeInt / 1000) + '&' + str(nowtimeStr)), str('调整期限制：' + str(ISAlarmTime))]
        if '发生泄漏' in PipeAlarmInfo:
            AlarmType = '泄漏报警'
        else:
            AlarmType = '预警'
        # Strat 20240619 检查是否符合报警条件
        ISAlarm, Flowloss_info, IS_Cumulative_Csv = Verify_CumulantsFlowloss_Continuous_Csv(AlarmFilePath, PipeConfig, MyPipeline, alarm_time_1)
        # End 20240619 检查是否符合报警条件
        if ISAlarm is True and IS_Cumulative_Csv is True:
            # 20240822 将报警的数据写入 20250107 修改数据存储格式由csv 变为 feather 文件大小压缩60%
            WritingCumulativeRundataPath = f"temp/{pipename}/{NowRecordtime}_{int(time.time())}.feather"
            LowFData_PD.to_feather(WritingCumulativeRundataPath)
            insertdata(ID, pipename, '运行', AlarmTime, Flowloss_info, AlarmType, '2', oracleinfo, AlarmIntervalTime, StrPumpInfo, AlarmFilePath)
        if AlarmType == '预警':
            # 20240529 故障信息写入PipeDataStatus.csv文件
            Write_Pipeline_DataStatus(pipename, AlarmTime, PipeAlarmInfo)
    return '累计及算法完成，不进行综合流量平衡运算'