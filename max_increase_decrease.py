'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2025-07-04 17:41:02
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2025-07-04 17:41:38
FilePath: \RunStateCheck\max_increase_decrease.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np

def analyze_changes_np(data):
    """
    使用np.diff分析列表中的最大上升和下降位置及幅度
    
    参数:
    data (list): 输入数据列表
    
    返回:
    dict: 包含最大上升和下降信息的字典
    """
    if len(data) < 2:
        return {
            'max_increase': {'position': None, 'value': 0},
            'max_decrease': {'position': None, 'value': 0}
        }
    
    # 计算相邻元素的差值
    diffs = np.diff(data)
    
    # 最大上升
    max_increase = np.max(diffs)
    increase_indices = np.where(diffs == max_increase)[0]  # 可能有多个相同的最大值
    
    # 最大下降
    max_decrease = np.min(diffs)
    decrease_indices = np.where(diffs == max_decrease)[0]  # 可能有多个相同的最小值
    
    return {
        'max_increase': {
            'positions': increase_indices[-1],  # 所有最大上升的位置
            'value': max_increase
        },
        'max_decrease': {
            'positions': decrease_indices[-1],  # 所有最大下降的位置
            'value': max_decrease
        }
    }

# 示例用法
if __name__ == "__main__":
    # 示例数据
    data = [10, 15, 13, 8, 12, 20, 15, 7]
    
    # 分析变化
    result = analyze_changes_np(data)
    
    # 输出结果
    print("数据:", data)
    print(f"最大上升: 值 {result['max_increase']['value']}, 位置 {result['max_increase']['positions']}")
    print(f"最大下降: 值 {result['max_decrease']['value']}, 位置 {result['max_decrease']['positions']}")