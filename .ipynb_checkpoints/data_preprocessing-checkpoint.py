import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import itertools

def reduce_mem_usage(df):
    """ 自动缩减 Pandas DataFrame 的内存占用
    如默认数据类型,int64, float64, object,将其转换为更节省空间的数据类型,如 int8, float16, category    
    """
    start_mem = df.memory_usage().sum() 
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() 
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


def process_outliers_and_issues(data):
    """
    处理数据中的异常值和数据质量问题
    
    参数:
        data: 原始DataFrame
    
    返回:
        处理后的DataFrame
    """
    #  处理日期异常
    def fix_zero_month(date_str):
        """处理日期字符串中的零月份问题"""
        month = int(date_str[4:6])
        if month == 0:
            month = 1
        return date_str[:4] + '-' + str(month) + '-' + date_str[6:]
    
    if 'regDate' in data.columns:
        data['regDate'] = data['regDate'].astype(str).apply(fix_zero_month)
    if 'creatDate' in data.columns:
        data['creatDate'] = data['creatDate'].astype(str).apply(fix_zero_month)
    
    #  处理notRepairedDamage列的'-'值
    if 'notRepairedDamage' in data.columns:
        data['notRepairedDamage'] = data['notRepairedDamage'].replace('-', 0).astype('float16')
    
    #  处理power列的异常值
    if 'power' in data.columns:
        data['power'] = data['power'].clip(upper=600)  # 将大于600的值设为600
    
    #  处理数值特征的异常值 (使用3σ原则)
    numeric_features = ['power',  'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 
                       'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12', 
                       'v_13', 'v_14'] #'kilometer'
    
    for col in numeric_features:
        if col in data.columns:
            # 跳过已经处理过的power列
            if col == 'power':
                continue
                
            # 计算均值和标准差
            mean_val = data[col].mean()
            std_val = data[col].std()
            
            # 定义异常值边界 (3σ原则)
            lower_bound = mean_val - 3 * std_val
            upper_bound = mean_val + 3 * std_val
            
            # 将超出范围的值裁剪到边界
            data[col] = data[col].clip(lower_bound, upper_bound)
    
    #  用众数填充缺失值
    data = data.fillna(data.mode().iloc[0,:])
    
    return data



