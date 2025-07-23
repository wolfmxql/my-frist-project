'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2025-07-14 14:55:06
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2025-07-14 18:45:10
FilePath: \RunStateCheck\check_create_table.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import cx_Oracle
import pandas as pd

def check_and_create_table(username, password, host, port, sid, table_name):
    """
    @description  :
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    try:
        # 构建数据库连接字符串
        dsn = cx_Oracle.makedsn(host, port, sid=sid)
        # 建立数据库连接
        with cx_Oracle.connect(user=username, password=password, dsn=dsn) as connection:
            with connection.cursor() as cursor:
                # 检查表格是否存在
                cursor.execute(f"select count(*) from user_tables where table_name = upper('{table_name}')")
                result = cursor.fetchone()
                if result[0] > 0:
                    print(f"表格{table_name}已存在")
                    return True
                else:
                    print(f"表格{table_name}不存在，开始创建……")
                    # 创建表的SQL语句
                    create_table_sql = f"create table {table_name} (id number primary key, pipename varchar2(50), runtime number, szflow_mean number, mzflow_mean number, flowloss_mean number, weekflowloss number, cumuszflow number, cumumzflow number, cumuflowloss number, faultstate varchar2(200), faulttimelength varchar2(255))"

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
    except cx_Oracle.Error as error:
        print(f"数据库操作错误“{error}")
        return None
    except Exception as e:
        print(f"发生未知错误：{e}")
        return None
    
if __name__ == "__main__":
    try:
        # 读取oracle配置文件
        OracleConnectFilePath = r'config/oracleinfo.csv'
        OracleINFO_df = pd.read_csv(OracleConnectFilePath, encoding='utf-8-sig')
        OracleINFO_dict = OracleINFO_df.to_dict('records')
        OracleINFO = OracleINFO_dict[-1]
        # 数据库连接参数
        OracleINFO = {
            'username': OracleINFO['username'],
            'password': OracleINFO['usersc'],
            'host': OracleINFO['host'],
            'port': int(OracleINFO['port']),
            'sid': OracleINFO['sid'],
            'table_name': 'runstateinfo'
        }
        check_and_create_table(**OracleINFO)
    except Exception as e:
        print(f"出现未知错误：{e}")