'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2025-06-23 16:01:02
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2025-06-23 17:21:33
FilePath: \RunStateCheck\RunStateCheckMain.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import sys
import os
import time
import pandas as pd
import bisect
from influxdb import InfluxDBClient
import cx_Oracle
from pathlib import Path
from datetime import datetime


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
    try:
        RunDataFilePath = pd.read_csv('config/RunDataFilePath.csv').to_dict('records')
        FrequencyPath = ''
        for path in RunDataFilePath:
            if path['filename'] == Filename:
                FrequencyPath = path['filepath']
        return FrequencyPath
    except Exception as e:
        print(e)

def CheckPathAndCreat(path):
    """
    @description  : 20250618 检查文件夹路径是否存在，不存在的情况下自动创建
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    try:
        full_path = os.path.expanduser(path)
        if not os.path.exists(full_path):
            Path(full_path).mkdir(parents=True, exist_ok=True)
            print(f"已创建文件夹：{full_path}")
        else:
            print(f"文件夹已存在：{full_path}")
    except Exception as e:
        print(f"influxdb数据写入文件夹检查或创建失败{e}")

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
            strSql = 'SELECT * FROM "' + Pipe_Table + '" WHERE  time > now() - "' + timeLength + '" order by time asc tz(\'Asia/Shanghai\')'
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


def Get_PipeConfig_Oracle():
    """
    @description  : 从数据库中读取Pipeconfig
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    # oracle信息
    oracleinfo_df = pd.read_csv('Data/oracleinfo.csv', encoding='utf-8-sig').to_dict('records')
    oracleinfo = oracleinfo_df[-1]
    loaclmachine = oracleinfo['MachineName']
    # 查询语句
    query_str = "SELECT * FROM PIPECONFIG WHERE MACHINENAME = '" + loaclmachine + "'"
    Pipeconfig_df = query_oracle(query_str, oracleinfo)
    return Pipeconfig_df


def DataStateAnalysis(Rundata_df):
    """
    @description  : 计算运行数据状态
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    # 提取始末站排量
    SzFlow_list = Rundata_df['SZPL'].tolist()
    MzFlow_list = Rundata_df['MZPL'].tolist()
    # 根据始站排量的数据计算运行时长
    RunLengthData_df = Rundata_df[Rundata_df['SZPL'] > 0.1].reset_index(drop=True)
    RunLength = RunLengthData_df.shape[0]
    # 累计外输
    CumuSzFlow = round(Rundata_df['SZPL'].sum() / 3600)
    # 累计接收
    CumuMzFlow = round(Rundata_df['MZPL'].sum() / 3600)
    # 累计输差
    CumuFlowloss = (Rundata_df['SZPL'].sum() - Rundata_df['MZPL'].sum()) / 3600
    # 平均输差
    Flowloss_mean = round(CumuFlowloss / RunLength, 2)
    # 平均外输排量
    SzFlow_mean = round(CumuSzFlow / RunLength, 2)
    


def CheckRunState(Pipeconfig_df):
    """
    @description  : 计算管道8小时班次运行状态
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    Pipeconfig_dict = Pipeconfig_df.to_dict('recoder')
    for Pipeconfig in Pipeconfig_dict:
        # 读取influxdb提取的数据文件
        PipeName = Pipeconfig['PIPENAME']
        NowDataPath = f"temp/Rundata/{PipeName}/Rundata.csv"
        Rundata_df = pd.read_csv(NowDataPath, encoding='utf-8-sig')
        
