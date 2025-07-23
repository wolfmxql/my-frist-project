'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2025-07-21 08:52:58
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2025-07-21 10:45:39
FilePath: \RunStateCheck\MonitorNetwork.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import time
import subprocess
import platform
import requests

def ping(host):
    param = '-n' if platform.system().lower() == 'windows' else '-c'
    command = ['ping', param, '4', host]
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT, timeout=5)
        print("网络通畅。")
        print(output.decode())
        return True
    except subprocess.CalledProcessError:
        print("网络可能不稳定或目标不可达")
        return False
    except subprocess.TimeoutExpired:
        print("连接超时")
        return False

def check_http_connection(url='')



def monitor_network(interval=10):
    while True:
        print("\n[检测网络状态]")
        ping_ok = ping("www.baidu.com")
