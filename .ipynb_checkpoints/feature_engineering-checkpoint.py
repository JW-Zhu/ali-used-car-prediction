import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import jn
from IPython.display import display, clear_output
import time
from tqdm import tqdm
import itertools
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.decomposition import PCA,FastICA,FactorAnalysis,SparsePCA
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import GridSearchCV,cross_val_score,StratifiedKFold,train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import scipy.signal as signal
from itertools import combinations
from scipy.stats import entropy

class FeatureEngineer:
    def __init__(self, data):
        self.data = data.copy()
        
    def expand_datetime_features(self, date_columns):
        """日期特征扩展（无需处理异常）"""
        for col in date_columns:
            self.data[col] = pd.to_datetime(self.data[col], format='%Y%m%d', errors='coerce')
            for attr in ['year', 'month']:
                self.data[f"{col}_{attr}"] = getattr(self.data[col].dt, attr)
        return self
    
    def add_time_features(self, reference_date='20200101'):
        """时间差特征（要求日期已预处理）"""
        ref_date = pd.to_datetime(reference_date, format='%Y%m%d')
        self.data['used_time1'] = (self.data['creatDate'] - self.data['regDate']).dt.days
        self.data['used_time2'] = (ref_date - self.data['regDate']).dt.days
        self.data['used_time3'] = (ref_date - self.data['creatDate']).dt.days
        return self
    
    def add_count_features(self, features):
        """添加计数特征"""
        for feat in tqdm(features, desc="Adding count features"):
            if self.data[feat].dtype == 'float16':
                self.data[feat] = self.data[feat].astype('float32')
            self.data[f"{feat}_count"] = self.data[feat].map(self.data[feat].value_counts())
        return self

    def add_binning_features(self, features, num_bins=50):
        """添加分箱特征"""
        for feat in tqdm(features, desc="Adding binning features"):
            value_range = (self.data[feat].max() - self.data[feat].min())
            bins = [i * value_range / num_bins for i in range(value_range)]
            self.data[f"{feat}_bin"] = pd.cut(self.data[feat], bins, labels=False)
        return self
    
    def add_cross_features(self, num_features, cat_features):
        """添加数值-类别交叉特征"""
        for cat in tqdm(cat_features, desc="Adding cross features"):
            group = self.data.groupby(cat, as_index=False)
            for num in num_features:
                stats = group[num].agg({
                    f'{cat}_{num}_max': 'max',
                    f'{cat}_{num}_min': 'min',
                    f'{cat}_{num}_median': 'median',
                    f'{cat}_{num}_mean': 'mean',
                    f'{cat}_{num}_mode': 'mode'
                })
                self.data = self.data.merge(stats, on=cat, how='left')
        return self
    
    def add_second_order_features(self, feature_pairs):
        """添加二阶交叉特征"""
        for f1, f2 in tqdm(feature_pairs, desc="Adding second-order features"):
            # 共现次数
            self.data[f'{f1}_{f2}_count'] = self.data.groupby([f1, f2])['SaleID'].transform('count')
            
            # 唯一值数量和熵
            for pair in [(f1, f2), (f2, f1)]:
                stats = self.data.groupby(pair[0], as_index=False)[pair[1]].agg({
                    f'{pair[0]}_{pair[1]}_nunique': 'nunique',
                    f'{pair[0]}_{pair[1]}_ent': lambda x: entropy(x.value_counts()/x.shape[0])
                })
                self.data = self.data.merge(stats, on=pair[0], how='left')
            
            # 比例偏好
            self.data[f'{f1}_in_{f2}_prop'] = self.data[f'{f1}_{f2}_count'] / (self.data[f'{f2}_count'] + 1e-5)
            self.data[f'{f2}_in_{f1}_prop'] = self.data[f'{f1}_{f2}_count'] / (self.data[f'{f1}_count'] + 1e-5)
        return self
    
    def add_interaction_features(self, v_features, cat_features):
        """添加交互特征"""
        # 数值特征间的加减乘
        for i, j in tqdm(combinations(v_features, 2), desc="Adding numeric interactions"):
            self.data[f'{i}+{j}'] = self.data[i] + self.data[j]
            self.data[f'{i}*{j}'] = self.data[i] * self.data[j]
            
        
        # 类别特征与数值特征的乘法
        for cat in tqdm(cat_features, desc="Adding cat-num interactions"):
            for num in v_features:
                self.data[f'{cat}*{num}'] = self.data[cat] * self.data[num]
        return self
    
    def get_processed_data(self):
        return self.data


# 使用示例
def process_features(data):
    # 初始化特征工程类
    fe = FeatureEngineer(data)
    
    # 定义各种特征组
    date_cols = ['regDate', 'creatDate']
    count_features = ['model', 'brand', 'regionCode', 'bodyType', 'fuelType',
                     'name', 'regDate_year', 'regDate_month','creatDate_year',
                      'creatDate_month','regDate', 'creatDate', 'kilometer']
    bin_features = ['power', 'used_time1', 'used_time2', 'used_time3', 'kilometer']
    cross_cat = ['model', 'brand', 'regDate_year']
    cross_num = ['v_0', 'v_3', 'v_8', 'v_11', 'v_12', 'power', 'kilometer']
    second_order_pairs = [['model', 'brand'], ['model', 'regionCode'], ['brand', 'regionCode']]
    v_features = [f'v_{i}' for i in range(15)]
    cat_features = ['model', 'brand', 'bodyType', 'fuelType', 'gearbox', 
                   'power', 'kilometer', 'notRepairedDamage', 'regionCode']
    
    # 执行特征工程
    processed_data = (fe.expand_datetime_features(date_cols)
                     .add_time_features()
                     .add_count_features(count_features)
                     .add_binning_features(bin_features[:4], num_bins=50)
                     .add_binning_features(bin_features[4:], num_bins=11)
                     .add_cross_features(cross_num, cross_cat)
                     .add_second_order_features(second_order_pairs)
                     .add_interaction_features(v_features, cat_features)
                     .get_processed_data())
    
    return processed_data


def expand_datetime_features(df, date_columns):
    """扩展日期时间特征
    
    功能:
        1. 将指定列转换为datetime类型
        2. 提取年、月、日、星期几等特征
        
    参数:
        df: 待处理的DataFrame
        date_columns: 需要处理的日期列名列表
        
    返回:
        处理后的DataFrame
    """
    for col in tqdm(date_columns):
        df[col] = pd.to_datetime(df[col].astype('str'))
        df[col + '_year'] = df[col].dt.year
        df[col + '_month'] = df[col].dt.month
        df[col + '_day'] = df[col].dt.day
        df[col + '_dayofweek'] = df[col].dt.dayofweek
    return df

def count_coding(df, fea_col):
    for f in fea_col:
        # 避免 float16 报错
        if df[f].dtype == 'float16':
            df[f] = df[f].astype('float32')
        df[f + '_count'] = df[f].map(df[f].value_counts())
    return df

#分桶操作
def cut_group(df,cols,num_bins=50):
    for col in cols:
        all_range = int(df[col].max()-df[col].min())
        bin = [i*all_range/num_bins for i in range(all_range)]
        df[col+'_bin'] = pd.cut(df[col], bin, labels=False)
    return df

### count编码
#def count_coding(df,fea_col):
#    for f in fea_col:
#        df[f + '_count'] = df[f].map(df[f].value_counts())
#    return(df)
def count_coding(df, fea_col):
    for f in fea_col:
        # 避免 float16 报错
        if df[f].dtype == 'float16':
            df[f] = df[f].astype('float32')
        df[f + '_count'] = df[f].map(df[f].value_counts())
    return df

#定义交叉特征统计
def cross_cat_num(df,num_col,cat_col):
    for f1 in tqdm(cat_col):
        g = df.groupby(f1, as_index=False)
        for f2 in tqdm(num_col):
            feat = g[f2].agg({
                '{}_{}_max'.format(f1, f2): 'max', '{}_{}_min'.format(f1, f2): 'min',
                '{}_{}_median'.format(f1, f2): 'median',
            })
            df = df.merge(feat, on=f1, how='left')
    return(df)
### 类别特征的二阶交叉
from scipy.stats import entropy
def cross_qua_cat_num(df):
    for f_pair in tqdm([
        ['model', 'brand'], ['model', 'regionCode'], ['brand', 'regionCode']
    ]):
        ### 共现次数
        df['_'.join(f_pair) + '_count'] = df.groupby(f_pair)['SaleID'].transform('count')
        ### n unique、熵
        df = df.merge(df.groupby(f_pair[0], as_index=False)[f_pair[1]].agg({
            '{}_{}_nunique'.format(f_pair[0], f_pair[1]): 'nunique',
            '{}_{}_ent'.format(f_pair[0], f_pair[1]): lambda x: entropy(x.value_counts() / x.shape[0])
        }), on=f_pair[0], how='left')
        df = df.merge(df.groupby(f_pair[1], as_index=False)[f_pair[0]].agg({
            '{}_{}_nunique'.format(f_pair[1], f_pair[0]): 'nunique',
            '{}_{}_ent'.format(f_pair[1], f_pair[0]): lambda x: entropy(x.value_counts() / x.shape[0])
        }), on=f_pair[1], how='left')
        ### 比例偏好
        df['{}_in_{}_prop'.format(f_pair[0], f_pair[1])] = df['_'.join(f_pair) + '_count'] / df[f_pair[1] + '_count']
        df['{}_in_{}_prop'.format(f_pair[1], f_pair[0])] = df['_'.join(f_pair) + '_count'] / df[f_pair[0] + '_count']
    return (df)



import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold,KFold
from itertools import product
class MeanTargetEncoder:
    def __init__(self, categorical_features, n_splits=10, target_type='classification', prior_weight_func=None):
        """
        categorical_features: 要编码的类别型列名列表

        n_splits: K 折交叉验证次数，用于防止数据泄露

        target_type: 任务类型，支持 'classification' 或 'regression'

        prior_weight_func: 平滑函数（用于防止某些类别样本太少导致编码不稳定）
        默认 prior_weight_func 是一个 sigmoid 函数:lambda x: 1 / (1 + np.exp((x - 2) / 1))
        样本数越多，平滑权重越小；样本数越少，平滑权重越大，编码更趋近于全局平均

        k: the number of observations needed for the posterior to be weighted equally as the prior
        f: larger f --> smaller slope
        """
        self.categorical_features = categorical_features
        self.n_splits = n_splits
        self.learned_stats = {}

        if target_type == 'classification':
            self.target_type = 'classification'
            self.target_values = []
        else:
            self.target_type = 'regression'
            self.target_values = None

        if isinstance(prior_weight_func, dict):
            self.prior_weight_func = eval(
                'lambda x: 1 / (1 + np.exp((x - k) / f))',
                dict(prior_weight_func, np=np)
            )
        elif callable(prior_weight_func):
            self.prior_weight_func = prior_weight_func
        else:
            self.prior_weight_func = lambda x: 1 / (1 + np.exp((x - 2) / 1))

    @staticmethod  #表示这个方法是 “静态方法”，不依赖self或cls就能运行。
    def _mean_encode_core(X_train, y_train, X_test, variable, target, prior_weight_func):
        X_train = X_train[[variable]].copy()
        X_test = X_test[[variable]].copy()

        if target is not None:
            encoded_col_name = f'{variable}_pred_{target}'
            X_train['temp_target'] = (y_train == target).astype(int)
        else:
            encoded_col_name = f'{variable}_pred'
            X_train['temp_target'] = y_train

        global_mean = X_train['temp_target'].mean()

        group_stats_df = X_train.groupby(by=variable)['temp_target'].agg(['mean', 'size']).rename(columns={'size': 'beta'})
        group_stats_df['beta'] = prior_weight_func(group_stats_df['beta'])

        group_stats_df[encoded_col_name] = (
            group_stats_df['beta'] * global_mean + 
            (1 - group_stats_df['beta']) * group_stats_df['mean']
        )
        group_stats_df.drop(['beta', 'mean'], axis=1, inplace=True)

        train_encoded = X_train.join(group_stats_df, on=variable)[encoded_col_name].values
        test_encoded = X_test.join(group_stats_df, on=variable).fillna(global_mean)[encoded_col_name].values

        return train_encoded, test_encoded, global_mean, group_stats_df

    def fit_transform(self, X, y):
        X_encoded = X.copy()
        if self.target_type == 'classification':
            kf = StratifiedKFold(self.n_splits)
            self.target_values = sorted(set(y))
            self.learned_stats = {
                f'{var}_pred_{cls}': [] for var, cls in product(self.categorical_features, self.target_values)
            }
            for var, cls in product(self.categorical_features, self.target_values):
                encoded_col_name = f'{var}_pred_{cls}'
                X_encoded[encoded_col_name] = np.nan
                for train_idx, val_idx in kf.split(y, y):
                    _, val_enc, global_mean, group_stats_df = self._mean_encode_core(
                        X_encoded.iloc[train_idx], y.iloc[train_idx],
                        X_encoded.iloc[val_idx], var, cls, self.prior_weight_func
                    )
                    X_encoded.iloc[val_idx, -1] = val_enc
                    self.learned_stats[encoded_col_name].append((global_mean, group_stats_df))
        else:
            kf = KFold(self.n_splits)
            self.learned_stats = {
                f'{var}_pred': [] for var in self.categorical_features
            }
            for var in self.categorical_features:
                encoded_col_name = f'{var}_pred'
                X_encoded[encoded_col_name] = np.nan
                for train_idx, val_idx in kf.split(y, y):
                    _, val_enc, global_mean, group_stats_df = self._mean_encode_core(
                        X_encoded.iloc[train_idx], y.iloc[train_idx],
                        X_encoded.iloc[val_idx], var, None, self.prior_weight_func
                    )
                    X_encoded.iloc[val_idx, -1] = val_enc
                    self.learned_stats[encoded_col_name].append((global_mean, group_stats_df))
        return X_encoded

    def transform(self, X):
        X_encoded = X.copy()
        if self.target_type == 'classification':
            for var, cls in product(self.categorical_features, self.target_values):
                encoded_col_name = f'{var}_pred_{cls}'
                X_encoded[encoded_col_name] = 0
                for global_mean, group_stats_df in self.learned_stats[encoded_col_name]:
                    X_encoded[encoded_col_name] += X_encoded[[var]].join(
                        group_stats_df, on=var
                    ).fillna(global_mean)[encoded_col_name]
                X_encoded[encoded_col_name] /= self.n_splits
        else:
            for var in self.categorical_features:
                encoded_col_name = f'{var}_pred'
                X_encoded[encoded_col_name] = 0
                for global_mean, group_stats_df in self.learned_stats[encoded_col_name]:
                    X_encoded[encoded_col_name] += X_encoded[[var]].join(
                        group_stats_df, on=var
                    ).fillna(global_mean)[encoded_col_name]
                X_encoded[encoded_col_name] /= self.n_splits
        return X_encoded
