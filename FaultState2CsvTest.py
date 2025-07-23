'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2025-07-08 17:37:52
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2025-07-08 18:33:15
FilePath: \RunStateCheck\FaultState2CsvTest.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
'''
    读取Oracle故障状态写入文件测试
'''
import pandas as pd
import os
import time
from datetime import datetime
import bisect
import cx_Oracle
from sqlalchemy import create_engine

def Write_FaultState_Csv(FaultStateCsv_Path):
    """
    @description  : 20250708 读取故障写入全程故障记录CSV文件，文件保存24小时数据
    ---------
    @param  :   从Oracle中查询出的故障数据 PipedataStatus_df
                FaultStateCsv_Path 班次故障统计CSV文件路径
    -------
    @Returns  :
    -------
    """
    try:
        sql_query = 'SELECT * FROM NOWPIPEDATASTATUS'
        PipedataStatus_df = Query_For_Oracle(sql_query)
        if os.path.lexists(FaultStateCsv_Path):
            # 读取班次故障统计文件故障信息
            FaultState_df = pd.read_csv(FaultStateCsv_Path, encoding='utf-8-sig')
            if FaultState_df.shape[0] > 0:
                # 合并故障信息
                FaultState_df = pd.concat([FaultState_df, PipedataStatus_df], ignore_index=True)
            else:
                FaultState_df = PipedataStatus_df
            if FaultState_df.shape[0] > 0:
                # 20250708 故障信息去重
                FaultState_df.drop_duplicates(subset=['PIPENAME', 'ALARMSTATUS', 'DATASTATUTIME'], keep='first', inplace=True)
                # 删除24小时之前的数据
                StartTime_int = int(time.time()) - 86400
                FaultState_df['ALARMTIME'] = pd.to_numeric(FaultState_df['ALARMTIME'], errors='coerce').fillna(0).astype(int)
                FaultStateTime_lst = FaultState_df['ALARMTIME'].values
                StartPosition_int = bisect.bisect_left(FaultStateTime_lst, StartTime_int)
                # 20250708 截取24小时数据
                FaultState_df = FaultState_df.iloc[StartPosition_int:].reset_index(drop=True)
                FaultState_df.to_csv(FaultStateCsv_Path, encoding='utf-8-sig', index=False)
        else:
            # 班次故障信息记录文件不存在，直接写入
            PipedataStatus_df.to_csv(FaultStateCsv_Path, encoding='utf-8-sig', index=False)
    except Exception as e:
        print(f'写入班次故障信息文件错误：{str(e)}')


def Query_For_Oracle(sql_query):
    """
    20240728 执行Oracle数据库查询并返回结果。该函数首先从'Data/oracleinfo.csv'文件中读取Oracle数据库的连接信息，然后使用这些信息创建数据库连接引擎。接着，使用pandas的read_sql_query方法执行SQL查询并将结果存储在DataFrame中。最后，将查询结果的字段名转换为大写并返回。

    Parameters:
    sql_query (str): 需要执行的SQL查询语句。

    Returns:
    data (pd.DataFrame): 包含查询结果的DataFrame，其字段名为大写。
    """
    Oracleinfo_df = pd.read_csv('config/oracleinfo.csv', encoding='utf-8-sig')
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


FaultStateCsv_Path = 'temp/FaultStateAll.csv'
while 0 < 1:
    sql_query = 'SELECT * FROM NOWPIPEDATASTATUS'
    PipedataStatus_df = Query_For_Oracle(sql_query)
    Write_FaultState_Csv(PipedataStatus_df, FaultStateCsv_Path)