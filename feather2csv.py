'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2025-07-11 14:37:59
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2025-07-11 14:49:27
FilePath: \RunStateCheck\feather2csv.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import pandas as pd
import argparse

def convert_feather_to_csv(feather_path, csv_path):
    """
    将Feather格式文件转换为CSV格式
    
    参数:
    feather_path (str): Feather文件路径
    csv_path (str): 输出CSV文件路径
    """
    try:
        # 读取Feather文件
        df = pd.read_feather(feather_path)
        
        # 写入CSV文件，使用UTF-8编码，处理NaN值
        df.to_csv(csv_path, index=False, encoding='utf-8', na_rep='nan')
        
        print(f"成功将 {feather_path} 转换为 {csv_path}")
    except Exception as e:
        print(f"转换过程中出现错误: {e}")

def main():
    feather_path = r'E:\工作管理\2、项目管理\1、第一采油厂\4、日常报警统计分析\王东王96-40增\1751726042779_1751726066.feather'
    csv_path = r'E:\工作管理\2、项目管理\1、第一采油厂\4、日常报警统计分析\王东王96-40增\1751726042779_1751726066.csv'
    # 执行转换
    convert_feather_to_csv(feather_path, csv_path)

if __name__ == "__main__":
    main()    