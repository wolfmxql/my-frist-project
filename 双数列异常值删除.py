import pandas as pd

def remove_anomalies(A, B):
    # 定义异常值列表
    anomalies = [-1, -2, -3, '', None]
    
    # 收集A和B中异常值的索引
    abnormal_indices = set()
    
    # 检查列表A
    for i, value in enumerate(A):
        if value in anomalies or (isinstance(value, (int, float)) and value < -9999):
            abnormal_indices.add(i)
    
    # 检查列表B
    for i, value in enumerate(B):
        if value in anomalies or (isinstance(value, (int, float)) and value < -9999):
            abnormal_indices.add(i)
    
    # 转换为有序列表并降序排列，以便从后向前删除
    indices_to_remove = sorted(abnormal_indices, reverse=True)
    
    # 从后向前删除异常值，避免索引变化问题
    for i in indices_to_remove:
        if i < len(A):
            del A[i]
        if i < len(B):
            del B[i]
    
    return A, B

# 示例使用
if __name__ == "__main__":
    # 创建示例DataFrame
    data = {
        'A': [1, -1, 3, -10000, 5, ''],
        'B': [10, 20, -2, 40, None, 60]
    }
    df = pd.DataFrame(data)
    
    print("原始DataFrame:")
    print(df)
    
    # 从DataFrame中提取列并转换为列表
    arrayA = df['A'].values
    listA = arrayA.tolist()  # 或者直接使用 df['A'].tolist()
    
    arrayB = df['B'].values
    listB = arrayB.tolist()  # 或者直接使用 df['B'].tolist()
    
    print("\n转换后的列表:")
    print("列表A:", listA)
    print("列表B:", listB)
    
    # 处理异常值
    clean_A, clean_B = remove_anomalies(listA, listB)
    
    print("\n处理后的列表:")
    print("列表A:", clean_A)
    print("列表B:", clean_B)    