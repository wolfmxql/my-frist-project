'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2025-06-23 16:01:02
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2025-07-18 11:49:40
FilePath: \AnalysisTools\AnalysisTools\RunStateCheckMain.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import time
import pandas as pd
import bisect
from influxdb import InfluxDBClient
import cx_Oracle
from pathlib import Path
from datetime import datetime
import numpy as np

def query_oracle(query_str, oracleinfo):
    """
    @description  : 
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
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

def FindRunDataFilePath(Filename):
    RunDataFilePath = pd.read_csv('config/RunDataFilePath.csv').to_dict('records')
    FrequencyPath = ''
    for path in RunDataFilePath:
        if path['filename'] == Filename:
            FrequencyPath = path['filepath']
    return FrequencyPath

def CheckPathAndCreat(path):
    """
    @description  : 20250618 检查文件夹路径是否存在，不存在的情况下自动创建
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    full_path = os.path.expanduser(path)
    if not os.path.exists(full_path):
        Path(full_path).mkdir(parents=True, exist_ok=True)
        print(f"已创建文件夹：{full_path}")
    else:
        print(f"文件夹已存在：{full_path}")

def WriteFrequencydataFromInfluxdb(pipeconfig_dict, oracleinfo, timeLength):
    query_str = "select * from PLC_REGISTER"
    Pipe_Influxdb_Table_df = query_oracle(query_str, oracleinfo)
    # InfluxDBStr = pd.read_csv('config/InfluxDBInfo.csv')
    influxdb_pd = pd.read_csv('config/InfluxDBInfo.csv', encoding='utf-8-sig')
    InfluxDB_dict = influxdb_pd.to_dict('records')
    InfluxDBInfo = InfluxDB_dict[0]
    host1 = str(InfluxDBInfo['host'])
    port1 = int(InfluxDBInfo['port'])
    username1 = str(InfluxDBInfo['username'])
    password1 = str(InfluxDBInfo['password'])
    database1 = str(InfluxDBInfo['database'])
    client = InfluxDBClient(host=host1, port=port1, username=username1, password=password1, database=database1)
    FDataPath = 'temp/Rundata'
    Nowtime_int = int(time.time())
    # 将时间戳转换为 datetime 对象
    dt_object = datetime.fromtimestamp(Nowtime_int)
    # 格式化为年月日时分
    formatted_time = dt_object.strftime("%Y-%m-%d %H:%M")
    for pipeconfig in pipeconfig_dict:
        Pipe_Table_dict = Pipe_Influxdb_Table_df.loc[Pipe_Influxdb_Table_df['PIPENAME'] == pipeconfig['PIPENAME']].to_dict('records')
        # 20250616 增加管道信息的判断条件
        if len(Pipe_Table_dict) > 14400:
            Pipe_Table = Pipe_Table_dict[-1]['GROUPREG']
            if '_PART' in Pipe_Table:
                Pipe_Table_Name = Pipe_Table
            else:
                Pipe_Table_Name = f"{Pipe_Table}_PART"
            strSql = 'SELECT * FROM "' + Pipe_Table_Name + '" WHERE  time > now() - "' + timeLength + '" order by time asc tz(\'Asia/Shanghai\')'
            InfluxDbData = client.query(strSql, database=database1)
            InfluxListData = list(InfluxDbData.get_points())
            InfluxDB_Rundata_PD = pd.DataFrame(InfluxListData)
            if InfluxDB_Rundata_PD.shape[0] > 300:
                InfluxDB_Rundata_PD['RECORDTIME'] = pd.to_datetime(InfluxDB_Rundata_PD['RECORDTIME'], format='%Y/%m/%d %H:%M:%S.%f').astype('int64') // 10**6 - 28800000
                CheckPathAndCreat(FDataPath + '//' + pipeconfig['PIPENAME'])
                RunDataPath = FDataPath + '//' + pipeconfig['PIPENAME'] + '//RunData.csv'
                InfluxDB_Rundata_PD.to_csv(RunDataPath, index=False, encoding='utf-8-sig')
                HistoryDataPath = f"temp/Rundata/{pipeconfig['PIPENAME']}/Rundata{formatted_time}.csv"
                InfluxDB_Rundata_PD.to_csv(HistoryDataPath, index=False, encoding='utf-8-sig')
            else:
                print(f"{pipeconfig['PIPENAME']}当前时间点获取的数据量为{InfluxDB_Rundata_PD.shape[0]}，不支持运算要求，无法写入frequency.csv文件")
        else:
            print(f"{pipeconfig['PIPENAME']}对应的PLC_REGISTER配置信息不存在")

# 故障信息提取
def FindFaultInfo(FaultState_Path, PipeName):

    """
    @description  : 从运行数据中提取故障信息
    ---------
    @param  :       FaultState_Path:    故障状态文件路径
    -------
    @Returns  :
    -------
    """
    # 读取故障记录文件
    FaultINFO_df = pd.read_csv(FaultState_Path, encoding='utf-8-sig')
    # 提取当前管道的故障信息
    CurrentFaultINFO_df = FaultINFO_df[FaultINFO_df['PipeName'] == PipeName].reset_index(drop=True)
    # 本班次的故障记录
    NowTime_int = int(time.time())
    StartTime_int = NowTime_int - 8 * 3600 * 1000 
    CurrentFaultINFO_df = CurrentFaultINFO_df[CurrentFaultINFO_df['FaultTime'] >= StartTime_int].reset_index(drop=True)
    # 本班次故障总时长
    FaultState_TimeLength = int(CurrentFaultINFO_df.shape[0] / 60)
    if FaultState_TimeLength > 3:
        # 统计FaultType列中各值的出现次数
        value_counts = CurrentFaultINFO_df['FaultType'].value_counts()
        # 提取出现次数最多的值（可能有多个并列最多的情况）
        most_common_values = value_counts[value_counts == value_counts.max()].index.tolist()
        FaultTypeMax = most_common_values
        FaultType_MaxTime = int(value_counts.max()/60)
        return f"近8小时{FaultTypeMax}，故障时长{FaultType_MaxTime}分钟", f"故障总时长：{FaultState_TimeLength}分钟"
    else:
        return "/", "/"

def ExtractFaultState(FaultState_Path, PipeName):
    """
    @description  :
    ---------
    @param  :   FaultState_Path 故障状态文件路径
    -------
    @Returns  :
    -------
    """
    FaultState_df = pd.read_csv(FaultState_Path, encoding='utf-8-sig')
    # 根据时间和管道提取故障记录
    StartTime_int = int(time.time()) - 28800
    PipeFaultState_df = FaultState_df[(FaultState_df['PIPENAME']==PipeName) & (FaultState_df['ALARMTIME'] > StartTime_int)].reset_index(drop=True)
    # 故障总时长
    PipeFaultLength = PipeFaultState_df.shape[0]
    if PipeFaultLength > 10:
        # 总故障时间大于10分钟的情况下，提取最主要的故障信息
        # 分割并提取第一部分（不存在 && 时取整个值）
        PipeFaultState_df['FaultType'] = PipeFaultState_df['ALARMSTATUS'].str.split('&&', expand=True)[0]
        # 统计频率
        counts = PipeFaultState_df['FaultType'].value_counts()
        # 找出最频繁的值（可能有多个）
        most_common = counts[counts == counts.max()].index.tolist()
        print(f"故障总时长为{PipeFaultLength}分钟；出现频率最多的内容：{most_common}")
        PipeFaultLength = f"故障总时长为{PipeFaultLength}分钟"
        PipeFalutINFO = f"出现最多的故障为：{most_common}"
    else:
        PipeFaultLength = '/'
        PipeFalutINFO = '/'
    return PipeFaultLength, PipeFalutINFO


def check_and_create_table(username, password, host, port, sid, table_name):
    """
    @description  :
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    # 构建数据库连接字符串
    dsn = cx_Oracle.makedsn(host, port, sid=sid)
    # 建立数据库连接
    with cx_Oracle.connect(user=username, password=password, dsn=dsn) as connection:
        with connection.cursor() as cursor:
            # 检查表格是否存在
            cursor.execute(f"select count(*) from user_tables where table_name = upper('{table_name}')")
            result = cursor.fetchone()
            if result[0] > 0:
                # print(f"表格{table_name}已存在")
                return True
            else:
                print(f"表格{table_name}不存在，开始创建……")
                # 创建表的SQL语句
                create_table_sql = f"create table {table_name} (id number primary key, pipename varchar2(100), runtime varchar2(100), runtimelength varchar2(100), szflow_mean varchar2(100), mzflow_mean varchar2(100), flowloss_mean varchar2(100), weekflowloss_mean varchar2(255), cumuszflow varchar2(100), cumumzflow varchar2(100), cumuflowloss varchar2(100), faultstate varchar2(255), faulttimelength varchar2(255))"

                # 执行创建表SQL
                cursor.execute(create_table_sql)

                # 创建自增序列
                create_sequence_sql = f"create sequence {table_name}_seq START WITH 1 INCREMENT BY 1 NOCACHE NOCYCLE"
                cursor.execute(create_sequence_sql)

                # 创建触发器实现自增ID
                create_trigger_sql = f"create or replace trigger {table_name}_trg before insert on {table_name} for each row begin if :new.id is null then select {table_name}_seq.nextval into :new.id from dual; end if; end;"
                cursor.execute(create_trigger_sql)

                connection.commit()
                print(f"表 {table_name} 创建成功")
                return False

def WritingRunState2Oracle(RunStateINFO):
    # 读取oracle配置文件
    OracleConnectFilePath = r'config/oracleinfo.csv'
    OracleINFO_df = pd.read_csv(OracleConnectFilePath, encoding='utf-8-sig')
    OracleINFO_dict = OracleINFO_df.to_dict('records')
    OracleINFO = OracleINFO_dict[-1]
    # 数据库连接参数
    OracleINFO_dict = {
        'username': OracleINFO['username'],
        'password': OracleINFO['usersc'],
        'host': OracleINFO['host'],
        'port': int(OracleINFO['port']),
        'sid': OracleINFO['sid'],
        'table_name': 'runstateinfo'
    }
    
    # 检查并创建状态写入表格
    IsExist = check_and_create_table(**OracleINFO_dict)
    # 运行信息写入表格
    dsn = cx_Oracle.makedsn(OracleINFO['host'], OracleINFO['port'], sid=OracleINFO['sid'])
    # 创建数据库连接
    with cx_Oracle.connect(user=OracleINFO['username'], password=OracleINFO['usersc'], dsn=dsn) as connection:
        with connection.cursor() as cursor:
            # 构建insert语句
            columns = ','.join(RunStateINFO.keys())
            placeholders = ','.join([':'+str(i+1) for i in range(len(RunStateINFO))])
            insert_runstateInfo_str = f"insert into runstateinfo ({columns}) values ({placeholders})"
            print(f"要插入的数值为：{list(RunStateINFO.values())}")
            # 执行插入
            cursor.execute(insert_runstateInfo_str, list(RunStateINFO.values()))
            # 提交插入事务
            connection.commit()
            print(f"数据已成功写入数据库runstateinfo表")

def timestamp_to_hour_minute_simple(timestamp_ms):
    """将13位时间戳转换为时:分格式（不考虑时区）"""
    # 计算总秒数
    total_seconds = timestamp_ms // 1000
    
    # 计算小时和分钟
    hours = (total_seconds // 3600) % 24
    minutes = (total_seconds // 60) % 60
    
    # 格式化为"时:分"
    return f"{hours:02d}:{minutes:02d}"


def DataStateAnalysis(Rundata_df, AnalysisData_Path, WeekFlowloss_mean, FaultState_Path):
    """
    @description  : 计算运行数据状态
    ---------
    @param  :       Rundata_df 从fluxdb中提取的最近8小时的运行数据
                    AnalysisData_Path 分析数据存放路径
                    WeekFlowloss_mean 上周平均输差
    -------
    @Returns  :
    -------
    """
    # 提取始末站排量
    SzFlow_list = Rundata_df['SZPL'].tolist()
    MzFlow_list = Rundata_df['MZPL'].tolist()
    # 开始于结束时间
    StartTime_str = timestamp_to_hour_minute_simple(int(Rundata_df.iloc[0]['timestamp']))
    EndTime_str = timestamp_to_hour_minute_simple(int(Rundata_df.iloc[-1]['timestamp']))
    RunTime = f"{StartTime_str}-{EndTime_str}"
    # 管道名称
    PipeName = Rundata_df.iloc[-1]['PIPENAME']
    # 根据始站排量的数据计算运行时长
    RunLengthData_df = Rundata_df[Rundata_df['SZPL'] > 0.1].reset_index(drop=True)
    RunLength = RunLengthData_df.shape[0]
    # 累计外输
    CumuSzFlow = round(Rundata_df['SZPL'].sum() / 3600)
    # 累计接收
    CumuMzFlow = round(Rundata_df['MZPL'].sum() / 3600)
    # 累计输差
    CumuFlowloss = (Rundata_df['SZPL'].sum() - Rundata_df['MZPL'].sum()) / 3600
    if RunLength != 0:
        # 平均输差
        Flowloss_mean = round(CumuFlowloss / RunLength, 2)
        # 平均外输排量                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
        SzFlow_mean = round(CumuSzFlow / RunLength, 2)
        # 平均接受排量
        MzFlow_mean = round(CumuMzFlow / RunLength, 2)
    else:
        # 平均输差
        Flowloss_mean = "/"
        # 平均外输排量                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
        SzFlow_mean = "/"
        # 平均接受排量
        MzFlow_mean = "/"
    # 外输时长（分钟）
    RunTimeLength = round(RunLength / 60, 2)    
    # 故障状态/故障时长
    if os.path.exists(FaultState_Path):
        FaultState, FaultTimeLength = ExtractFaultState(FaultState_Path, PipeName)
    else:
        FaultState = "/"
        FaultTimeLength = "/"
    # 本班次的运行状态信息
    id = int(time.time() * 1000)
    RunStateINFO = {'id':id, 'PipeName':PipeName, 'RunTime':RunTime, 'RunTimeLength':RunTimeLength, 'SzFlow_mean':SzFlow_mean, 'MzFlow_mean':MzFlow_mean, 'Flowloss_mean':Flowloss_mean, 'WeekFlowloss_mean':WeekFlowloss_mean, 'CumuSzFlow':CumuSzFlow, 'CumuMzFlow':CumuMzFlow, 'CumuFlowloss':CumuFlowloss, 'FaultState':FaultState, 'FaultTimeLength':FaultTimeLength}
    print(RunStateINFO)
    if os.path.lexists(AnalysisData_Path):
        AnalysisData_df = pd.read_csv(AnalysisData_Path, encoding='utf-8-sig')
        NewAnalysisData_df = pd.concat([AnalysisData_df, pd.DataFrame([RunStateINFO])], ignore_index=True)
        # 写入文件
        NewAnalysisData_df.to_csv(AnalysisData_Path, index=False, encoding='utf-8-sig')
    else:
        NewAnalysisData_df = pd.DataFrame([RunStateINFO])
        # 写入文件
        NewAnalysisData_df.to_csv(AnalysisData_Path, index=False, encoding='utf-8-sig')
    # 写入数据库
    WritingRunState2Oracle(RunStateINFO)

def ErrorStateAnalysis(PipeName, ErrorINFO, AnalysisData_Path, FaultState_Path):
    """
    @description  : 错误状态写入数据库
    ---------
    @param  :       Rundata_df 从fluxdb中提取的最近8小时的运行数据
                    AnalysisData_Path 分析数据存放路径
                    WeekFlowloss_mean 上周平均输差
    -------
    @Returns  :
    -------
    """
    # 故障状态/故障时长
    if os.path.exists(FaultState_Path):
        FaultState, FaultTimeLength = ExtractFaultState(FaultState_Path, PipeName)
    else:
        FaultState = "/"
        FaultTimeLength = "/"
    # 数据检查时间
    NowTime_int = int(time.time())
    NowTime_Str = timestamp_to_hour_minute_simple(NowTime_int * 1000)
    StartTime_Str = timestamp_to_hour_minute_simple(NowTime_int*1000 - 28800*1000)
    RunTimeINFO = f"{NowTime_Str}-{StartTime_Str}"
    # 本班次的运行状态信息
    id = int(time.time()*1000)
    RunStateINFO = {'id':id, 'PipeName':PipeName, 'RunTime':RunTimeINFO, 'RunTimeLength':'/', 'SzFlow_mean':'/', 'MzFlow_mean':'/', 'Flowloss_mean':'/', 'WeekFlowloss_mean':'/', 'CumuSzFlow':'/', 'CumuMzFlow':'/', 'CumuFlowloss':'/', 'FaultState':ErrorINFO, 'FaultTimeLength':'/'}
    if os.path.lexists(AnalysisData_Path):
        AnalysisData_df = pd.read_csv(AnalysisData_Path, encoding='utf-8-sig')
        NewAnalysisData_df = pd.concat([AnalysisData_df, pd.DataFrame([RunStateINFO])], ignore_index=True)
        # 写入文件
        NewAnalysisData_df.to_csv(AnalysisData_Path, index=False, encoding='utf-8-sig')
    else:
        NewAnalysisData_df = pd.DataFrame([RunStateINFO])
        # 写入文件
        NewAnalysisData_df.to_csv(AnalysisData_Path, index=False, encoding='utf-8-sig')
    # 写入数据库
    WritingRunState2Oracle(RunStateINFO)

def WeekNormalFlowloss(Flowloss_lst):
    """
    @description  : 根据标准差出去异常输差
    ---------
    @param  :   
    -------
    @Returns  :
    -------
    """
    fn = 0
    while fn < 3:
        if len(Flowloss_lst) >= 15:
            Flowloss_std = np.std(Flowloss_lst)
            Flowloss_mean = np.mean(Flowloss_lst)
            minFlowloss = Flowloss_mean - 2 * Flowloss_std
            maxFlowloss = Flowloss_mean + 2 * Flowloss_std
            print(f"第{fn+1}次异常值排查，异常值范围：{minFlowloss}-{maxFlowloss}，标准差{Flowloss_std}")
            Flowloss_lst = [num for num in Flowloss_lst if minFlowloss < num < maxFlowloss]
            print(f"第{fn+1}次异常值排查后平均8小时输差列表{Flowloss_lst}")
        fn += 1
    print(f"标准差周平均8小时输差列表：{Flowloss_lst}")
    Flowloss_mean = np.mean(Flowloss_lst)
    print(f"上周标准差平均值{Flowloss_mean}")
    return Flowloss_mean

def WeekFlowloss(WeekAnalysisData_df):
    """
    @description  : 计算周输差的数据，不能用总值，需要排除故障产生的意外值，根据输差标准差作3次异常值排查
    ---------
    @param  :   
    -------
    @Returns  :
    -------
    """
    Flowloss_lst = []
    # 每次提取8小时的数据
    Recordtime_lst = WeekAnalysisData_df['RECORDTIME'].values
    EndTime = int(WeekAnalysisData_df.iloc[-1]['RECORDTIME'])
    StartTime = EndTime - 8 * 3600 * 1000
    Start0Time = int(WeekAnalysisData_df.iloc[0]['RECORDTIME'])
    while StartTime > Start0Time:
        StartPosition = bisect.bisect_left(Recordtime_lst, StartTime)
        EndPosition = bisect.bisect_left(Recordtime_lst, EndTime)
        PartRundata_df = WeekAnalysisData_df.iloc[StartPosition: EndPosition].reset_index(drop=True)
        # 提取运行时长
        ONPartRundata_df = PartRundata_df[PartRundata_df['SZPL'] > 0.2].reset_index(drop=True)
        RunTimeLength = ONPartRundata_df.shape[0]
        # 计算该时段的平均输差
        if RunTimeLength > 0:
            Flowloss_mean = (PartRundata_df['SZPL'].sum() - PartRundata_df['MZPL'].sum()) / RunTimeLength
            Flowloss_lst.append(Flowloss_mean)
        else:
            return "始站排量数据异常，不能进行周数据运算"
        EndTime = StartTime
        StartTime = EndTime - 8 * 3600 * 1000
    # 计算上周的平均输差
    print(f"周平均8小时输差列表：{Flowloss_lst}")
    if len(Flowloss_lst) > 20:
        WeekNormalFlowloss_mean = WeekNormalFlowloss(Flowloss_lst)
        return WeekNormalFlowloss_mean
    else:
        return "要求的数据长度不够，不能进行周数据运算"

def InfluxbDB_DataAnalysis(pipeconfig, oracleinfo):
    """
    @description  : 根据InfluxDB数据分析
    ---------
    @param  :   InfluxDB_INFO： influxDB数据
    -------
    @Returns  :
    -------
    """
    InfluxDB_pd = pd.read_csv('config/InfluxDBInfo.csv', encoding='utf-8-sig')
    InfluxDB_dict = InfluxDB_pd.to_dict('records')
    InfluxDBInfo = InfluxDB_dict[0]
    # influxdb连接信息
    host1 = str(InfluxDBInfo['host'])
    port1 = int(InfluxDBInfo['port'])
    username1 = str(InfluxDBInfo['username'])
    password1 = str(InfluxDBInfo['password'])
    database1 = str(InfluxDBInfo['database'])
    # 创建InfluxDB连接
    client = InfluxDBClient(host=host1, port=port1, username=username1, password=password1, database=database1)
    # 提取管道数据
    query_str = "select * from PLC_REGISTER"
    Pipe_Influxdb_Table_df = query_oracle(query_str, oracleinfo)
    Pipe_Table_dict = Pipe_Influxdb_Table_df[Pipe_Influxdb_Table_df['PIPENAME'] == pipeconfig['PIPENAME']].to_dict('records')
    if len(Pipe_Table_dict) > 0:
        Pipe_Table = Pipe_Table_dict[-1]['GROUPREG']
        if '_PART' in Pipe_Table:
            Pipe_Table_Name = Pipe_Table
        else:
            Pipe_Table_Name = f"{Pipe_Table}_PART"
        strSql = f'SELECT * FROM "{Pipe_Table_Name}" WHERE time > now() - 8d order by time asc tz(\'Asia/Shanghai\')'
        # 提出influxdb数据
        InfluxDbData = client.query(strSql, database=database1)
        InfluxListData = list(InfluxDbData.get_points())
        InfluxDB_Rundata_df = pd.DataFrame(InfluxListData)
        # 检查提取的InfluxDB数据是否符合时间长度要求
        if InfluxDB_Rundata_df.shape[0] > 5 * 24 * 3600:
            # InfluxDB 提取数据总长度超过5天
            InfluxDB_Rundata_df['RECORDTIME'] = pd.to_datetime(InfluxDB_Rundata_df['RECORDTIME'], format='%Y/%m/%d %H:%M:%S.%f').astype('int64') // 10**6 - 28800000
            print(InfluxDB_Rundata_df)
            WeekFlowloss_mean = WeekFlowloss(InfluxDB_Rundata_df)
        else:
            print('要求的数据长度不够，不能进行周数据运算')
            WeekFlowloss_mean = '要求的数据长度不够，不能进行周数据运算'
    else:
        WeekFlowloss_mean = '要求的数据长度不够，不能进行周数据运算'
    return WeekFlowloss_mean

def StatisticsFlowloss(WeekAnalysisData_df):
    """
    @description  : 根据上周记录的分析数据统计上周的输差
    ---------
    @param  :   WeekAnalysisData_df:    运行数据状态数据
    -------
    @Returns  :
    -------
    """
    # 提取上周记录的8小时班次输差数据
    Flowloss_lst = WeekAnalysisData_df['Flowloss_mean'].to_list()
    if len(Flowloss_lst) >= 15:
        Flowloss_mean = WeekNormalFlowloss(Flowloss_lst)
    else:
        Flowloss_mean = '记录数据小于5天，不能用作周输差分析'
    return Flowloss_mean

def CumulataWeekData(AnalysisData_Path, pipeconfig, oracleinfo):
    """
    @description  : 计算周累计数据
    ---------
    @param  :   AnalysisData_Path： 运行数据状态文件路径
    -------
    @Returns  :
    -------
    """
    if os.path.lexists(AnalysisData_Path):
        # 提取AnalysisData
        AnalysisData_df = pd.read_csv(AnalysisData_Path, encoding='utf-8-sig')
        # 提取周数据信息
        Recordtime_lst = AnalysisData_df['id'].values
        # 检查数据记录是否超过一周
        Nowtime_int = int(time.time() * 1000)
        StratTime_0 = Nowtime_int - 7 * 24 * 3600 * 1000
        StartPosition_0 = bisect.bisect_left(Recordtime_lst, StratTime_0)
        if StartPosition_0 > 0:
            WeekAnalysisData_df = AnalysisData_df.iloc[StartPosition_0: ].reset_index(drop=True)
            # 计算周数据
            WeekNormalFlowloss = StatisticsFlowloss(WeekAnalysisData_df)
        else:
            StratTime_1 = Nowtime_int - 5 * 24 * 3600 * 1000
            StartPosition_1 = bisect.bisect_left(Recordtime_lst, StratTime_1)
            if StartPosition_1 > 0:
                WeekAnalysisData_df = AnalysisData_df
                # 记录数据大于5天时，用全部数据
                WeekNormalFlowloss = StatisticsFlowloss(WeekAnalysisData_df)
            else:
                print("记录数据小于5天，不能用作周输差分析")
                # 从influxdb中读取数据分析上周的输差情况
                WeekNormalFlowloss = InfluxbDB_DataAnalysis(pipeconfig, oracleinfo)
    else:
        # 从influxdb中读取数据分析上周的输差情况
        WeekNormalFlowloss = InfluxbDB_DataAnalysis(pipeconfig, oracleinfo)
    return WeekNormalFlowloss

''' 提取本班次运行数据 '''
def ExtractRundata(oracleinfo, pipeconfig):
    """
    @description  : 从InfluxDB提取本班次8小时的数据
    ---------
    @param  :   InfluxDB_INFO： influxDB数据
    -------
    @Returns  :
    -------
    """
    InfluxDB_pd = pd.read_csv('config/InfluxDBInfo.csv', encoding='utf-8-sig')
    InfluxDB_dict = InfluxDB_pd.to_dict('records')
    InfluxDBInfo = InfluxDB_dict[0]
    # influxdb连接信息
    host1 = str(InfluxDBInfo['host'])
    port1 = int(InfluxDBInfo['port'])
    username1 = str(InfluxDBInfo['username'])
    password1 = str(InfluxDBInfo['password'])
    database1 = str(InfluxDBInfo['database'])
    # 创建InfluxDB连接
    client = InfluxDBClient(host=host1, port=port1, username=username1, password=password1, database=database1)
    # 提取管道数据
    query_str = "select * from PLC_REGISTER"
    Pipe_Influxdb_Table_df = query_oracle(query_str, oracleinfo)
    Pipe_Table_dict = Pipe_Influxdb_Table_df[Pipe_Influxdb_Table_df['PIPENAME'] == pipeconfig['PIPENAME']].to_dict('records')
    if len(Pipe_Table_dict) > 0:
        Pipe_Table = Pipe_Table_dict[-1]['GROUPREG']
        if '_PART' in Pipe_Table:
            Pipe_Table_Name = Pipe_Table
        else:
            Pipe_Table_Name = f"{Pipe_Table}_PART"
        strSql = f'SELECT * FROM "{Pipe_Table_Name}" WHERE time > now() - 8h order by time asc tz(\'Asia/Shanghai\')'
        # 提出influxdb数据
        InfluxDbData = client.query(strSql, database=database1)
        InfluxListData = list(InfluxDbData.get_points())
        InfluxDB_Rundata_df = pd.DataFrame(InfluxListData)
        if InfluxDB_Rundata_df.shape[0] > 0:
            # 将字符串转换为datetime类型
            InfluxDB_Rundata_df['RECORDTIME'] = pd.to_datetime(InfluxDB_Rundata_df['RECORDTIME'], format='%Y/%m/%d %H:%M:%S.%f')
            # 转换为13位时间戳（毫秒级）
            InfluxDB_Rundata_df['timestamp'] = InfluxDB_Rundata_df['RECORDTIME'].astype('int64') // 10**6
        else:
            InfluxDB_Rundata_df = f"{pipeconfig['PIPENAME']}在InfluxDB数据表中获取的数据为空"
    else:
        InfluxDB_Rundata_df = f"未在PLC_REGISTER表中找到{pipeconfig['PIPENAME']}的InfluxDB数据表信息"
    return InfluxDB_Rundata_df


def Get_PipeConfig_Oracle(OracleINFO):
    """
    @description  : 从数据库中提取pipeconfig
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    host = OracleINFO['host']
    port = int(OracleINFO['port'])
    username = OracleINFO['username']
    password = OracleINFO['usersc']
    sid = OracleINFO['sid']
    # 连接字符串
    dsn = cx_Oracle.makedsn(host, port, sid=sid)
    with cx_Oracle.connect(user=username, password=password, dsn=dsn) as connection:
        with connection.cursor() as cursor:
            # 查询语句
            loaclmachine = OracleINFO['MachineName']
            # 查询语句
            Query_PipeConfig = f"SELECT * FROM PIPECONFIG WHERE MACHINENAME = '{loaclmachine}' AND ISJC = 1"
            # 执行查询语句
            cursor.execute(Query_PipeConfig)
            # 提取查询结果
            Pipeconfig = cursor.fetchall()
            # 转换为DataFrame
            Pipeconfig_df = pd.DataFrame(Pipeconfig)
            # 提取列名
            columns = [col[0] for col in cursor.description]
            # 设置列名
            Pipeconfig_df.columns = columns
            connection.commit()
            return Pipeconfig_df

def CheckFilePath(filePath):
    """
    @description  : 检查文件路径是否存在，不存在则创建该路径下的所有文件夹
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    if os.path.exists(filePath) is False:
        os.makedirs(filePath)
    

def CheckRunState():
    """
    @description  : 计算管道8小时班次运行状态
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    OracleINFO_df = pd.read_csv('config/oracleinfo.csv', encoding='utf-8-sig')
    if OracleINFO_df.shape[0] > 0:
        OracleINFO_dict = OracleINFO_df.to_dict('records')
        OracleINFO = OracleINFO_dict[-1]
        Pipeconfig_df = Get_PipeConfig_Oracle(OracleINFO)
        FaultState_Path = r"" + OracleINFO['FaultStatePath']
        if type(Pipeconfig_df) is not bool and Pipeconfig_df.shape[0] > 0:
            Pipeconfig_dict = Pipeconfig_df.to_dict('records')
            # 提取周数据信息
            for Pipeconfig in Pipeconfig_dict:
                # 读取influxdb提取的数据文件
                PipeName = Pipeconfig['PIPENAME']
                # 检查文件路径是否存在，不存在则创建
                CheckFilePath(f"temp/Rundata/{PipeName}")
                # 运行状态记录文件路径
                AnalysisData_Path = f"temp/Rundata/{PipeName}/AnalysisData.csv"
                # 计算本班次运行状态
                NowDataPath = f"temp/Rundata/{PipeName}/Rundata{int(time.time())}.csv"
                # 提取本班次的运行数据并写入NowDataPath
                Rundata_df = ExtractRundata(OracleINFO, Pipeconfig)
                if type(Rundata_df) != type('string'):
                    # 写入csv文件
                    Rundata_df.to_csv(NowDataPath, encoding='utf-8-sig', index=False)
                    if Rundata_df.shape[0] > 600:
                        # 计算上周平均输差
                        WeekFlowloss_mean = CumulataWeekData(AnalysisData_Path, Pipeconfig, OracleINFO)
                        if type(WeekFlowloss_mean) == type('string'):
                            WeekFlowloss_mean = '/'
                        # 计算班次运行状态
                        DataStateAnalysis(Rundata_df, AnalysisData_Path, WeekFlowloss_mean, FaultState_Path)
                    else:
                        DataTimeLength = int(Rundata_df.shape[0] / 60)
                        print(f'本班次管道{PipeName}数据异常,只获取到{DataTimeLength}分钟的数据')
                        ErrorINFO = f'本班次管道{PipeName}数据异常,只获取到{DataTimeLength}分钟的数据'
                        ErrorStateAnalysis(PipeName, ErrorINFO, AnalysisData_Path, FaultState_Path)
                else:
                    print(f'本班次管道{PipeName}数据异常，{Rundata_df}')
                    ErrorINFO = Rundata_df
                    ErrorStateAnalysis(PipeName, ErrorINFO, AnalysisData_Path, FaultState_Path)
        else:
            print('管道Pipeconfig信息错误')
    else:
        print('缺少oracle配置信息')

def CheckAnalysis():
    while 1 < 2:
        IS_Check = input('是否开始运行状态监测程序？(Y/N/E)')
        if IS_Check == 'Y':
            CheckRunState()
        elif IS_Check == 'E':
            print('程序已退出')
            break


if __name__ == '__main__':
    CheckAnalysis()