from IPython.display import display, clear_output
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV,cross_val_score,StratifiedKFold,train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import entropy


# 计算类别特征的计数
def compute_category_counts(df, category_columns):
    for column in category_columns:
        # 避免 float16 报错
        if df[column].dtype == 'float16':
            df[column] = df[column].astype('float32')
        df[column + '_count'] = df[column].map(df[column].value_counts())
    return df

# 定义交叉特征统计：类别与数值特征的组合
def generate_cross_category_numeric_features(df, numeric_columns, category_columns):
    for category_column in tqdm(category_columns):
        grouped_data = df.groupby(category_column, as_index=False)
        for numeric_column in tqdm(numeric_columns):
            # 生成交叉特征的最大值、最小值和中位数
            feature = grouped_data[numeric_column].agg({
                f'{category_column}_{numeric_column}_max': 'max',
                f'{category_column}_{numeric_column}_min': 'min',
                f'{category_column}_{numeric_column}_median': 'median',
            })
            df = df.merge(feature, on=category_column, how='left')
    return df

# 生成类别与类别之间的交叉特征
def generate_cross_category_features(df):
    for pair in tqdm([
        ['model', 'brand'], ['model', 'regionCode'], ['brand', 'regionCode']
    ]):
        # 计算共现次数
        df[f'{pair[0]}_{pair[1]}_count'] = df.groupby(pair)['SaleID'].transform('count')
        
        # 计算类别间的唯一值数量和熵值
        df = df.merge(df.groupby(pair[0], as_index=False)[pair[1]].agg({
            f'{pair[0]}_{pair[1]}_nunique': 'nunique',
            f'{pair[0]}_{pair[1]}_entropy': lambda x: entropy(x.value_counts() / x.shape[0])
        }), on=pair[0], how='left')
        
        df = df.merge(df.groupby(pair[1], as_index=False)[pair[0]].agg({
            f'{pair[1]}_{pair[0]}_nunique': 'nunique',
            f'{pair[1]}_{pair[0]}_entropy': lambda x: entropy(x.value_counts() / x.shape[0])
        }), on=pair[1], how='left')
        
        # 计算类别间的比例偏好
        df[f'{pair[0]}_in_{pair[1]}_prop'] = df[f'{pair[0]}_{pair[1]}_count'] / df[f'{pair[1]}_count']
        df[f'{pair[1]}_in_{pair[0]}_prop'] = df[f'{pair[0]}_{pair[1]}_count'] / df[f'{pair[0]}_count']
    
    return df

from sklearn.model_selection import KFold

def apply_target_encoding_with_cv(X_data, Y_data, X_test, feature_cols, target_col='price', 
                                  target_encoding_stats=['mean', 'std', 'median', 'max', 'min', 'sum'], 
                                  n_splits=10, random_state=42):
    """
    对每个类别特征进行目标编码（target encoding），通过K折交叉验证防止目标泄漏。
    
    参数:
    - X_data: 训练集特征数据 (DataFrame)
    - Y_data: 训练集目标值 (Series)
    - X_test: 测试集特征数据 (DataFrame)
    - feature_cols: 特征列名称 (list)
    - target_col: 目标列，默认为 'price'
    - target_encoding_stats: 目标编码使用的统计量 (list)
    - n_splits: KFold 的折数，默认为 10
    - random_state: 随机种子
    
    返回:
    - X_data: 目标编码后的训练集特征数据
    - X_test: 目标编码后的测试集特征数据
    """
    
    # 所有目标编码字段将存入此列表
    encoded_feature_names = []

    # 默认统计值，用于填补测试集中缺失值
    price_stats_default = {
        'max': X_data[target_col].max(),
        'min': X_data[target_col].min(),
        'median': X_data[target_col].median(),
        'mean': X_data[target_col].mean(),
        'sum': X_data[target_col].sum(),
        'std': X_data[target_col].std(),
        'skew': X_data[target_col].skew(),
        'kurt': X_data[target_col].kurt(),
        'mad': (X_data[target_col] - X_data[target_col].mean()).abs().mean()
    }

    # 设置交叉验证方式
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # 对每一个类别特征进行多统计量目标编码
    for feature in tqdm(feature_cols):
        stat_map = {}

        # 初始化编码列
        for stat in target_encoding_stats:
            encoded_col_name = f'{feature}_target_{stat}'
            stat_map[encoded_col_name] = stat
            X_data[encoded_col_name] = 0
            X_test[encoded_col_name] = 0
            encoded_feature_names.append(encoded_col_name)

        # K折编码，防止目标泄漏
        for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(X_data, Y_data)):
            X_train_fold = X_data.iloc[train_idx].reset_index(drop=True)
            X_valid_fold = X_data.iloc[valid_idx].reset_index(drop=True)

            # 基于训练折生成统计量映射
            stat_encoding_df = X_train_fold.groupby(feature, as_index=False)[target_col].agg(stat_map)

            # 应用于验证集和测试集
            X_valid_encoded = X_valid_fold[[feature]].merge(stat_encoding_df, on=feature, how='left')
            X_test_encoded = X_test[[feature]].merge(stat_encoding_df, on=feature, how='left')

            # 填补缺失值，并赋值到原始 DataFrame
            for stat in target_encoding_stats:
                col = f'{feature}_target_{stat}'

                X_valid_encoded[col] = X_valid_encoded[col].fillna(price_stats_default[stat])
                X_test_encoded[col] = X_test_encoded[col].fillna(price_stats_default[stat])

                X_data.loc[valid_idx, col] = X_valid_encoded[col].values
                X_test[col] += X_test_encoded[col].values / kf.n_splits

    return X_data, X_test

from sklearn.model_selection import StratifiedKFold, KFold
from itertools import product
import numpy as np
import pandas as pd

class TargetMeanEncoder:
    """
    使用目标均值编码（Mean Encoding）的特征工程工具类。
    支持分类任务和回归任务，通过 KFold 或 StratifiedKFold 防止目标泄漏。
    """

    def __init__(self, categorical_features, n_splits=10, target_type='regression', prior_weight_func=None):
        """
        初始化 TargetMeanEncoder

        :param categorical_features: list[str], 需要进行均值编码的类别特征列名
        :param n_splits: int, KFold 划分的折数
        :param target_type: str, 'regression' 或 'classification'
        :param prior_weight_func: callable or dict, 控制先验权重的函数
                                  - 如果是 dict，默认使用指数衰减函数
                                  - 如果是 callable，自定义函数
        """
        self.categorical_features = categorical_features
        self.n_splits = n_splits
        self.learned_stats = {}

        if target_type == 'classification':
            self.target_type = target_type
            self.target_values = []
        else:
            self.target_type = 'regression'
            self.target_values = None

        if isinstance(prior_weight_func, dict):
            # 使用指数衰减函数
            self.prior_weight_func = eval(
                'lambda x: 1 / (1 + np.exp((x - k) / f))',
                dict(prior_weight_func, np=np)
            )
        elif callable(prior_weight_func):
            self.prior_weight_func = prior_weight_func
        else:
            # 默认函数
            self.prior_weight_func = lambda x: 1 / (1 + np.exp((x - 2) / 1))

    @staticmethod
    def _encode_fold(X_train, y_train, X_valid, variable, target, prior_weight_func):
        """
        针对单个折的均值编码子过程
        """
        X_train = X_train[[variable]].copy()
        X_valid = X_valid[[variable]].copy()

        if target is not None:  # 分类任务
            encoded_col = f'{variable}_mean_{target}'
            X_train['temp_target'] = (y_train == target).astype(int)
        else:  # 回归任务
            encoded_col = f'{variable}_mean'
            X_train['temp_target'] = y_train

        prior = X_train['temp_target'].mean()

        stats = X_train.groupby(variable)['temp_target'].agg(['mean', 'size']).rename(columns={'size': 'count'})
        stats['weight'] = prior_weight_func(stats['count'])
        stats[encoded_col] = stats['weight'] * prior + (1 - stats['weight']) * stats['mean']
        stats = stats[[encoded_col]]

        train_encoded = X_train.join(stats, on=variable)[encoded_col].values
        valid_encoded = X_valid.join(stats, on=variable).fillna(prior)[encoded_col].values

        return train_encoded, valid_encoded, prior, stats

    def fit_transform(self, X, y):
        """
        对训练集进行目标均值编码，并保存统计结果

        :param X: pandas.DataFrame, 输入特征
        :param y: pandas.Series or np.array, 目标值
        :return: pandas.DataFrame, 编码后的训练集
        """
        X_encoded = X.copy()

        if self.target_type == 'classification':
            skf = StratifiedKFold(self.n_splits)
            self.target_values = sorted(set(y))
            self.learned_stats = {f"{col}_mean_{t}": [] for col, t in product(self.categorical_features, self.target_values)}

            for col, t in product(self.categorical_features, self.target_values):
                new_col = f"{col}_mean_{t}"
                X_encoded[new_col] = np.nan

                for train_idx, valid_idx in skf.split(y, y):
                    _, valid_encoded, prior, stats = self._encode_fold(
                        X_encoded.iloc[train_idx], y.iloc[train_idx],
                        X_encoded.iloc[valid_idx], col, t, self.prior_weight_func
                    )
                    X_encoded.iloc[valid_idx, -1] = valid_encoded
                    self.learned_stats[new_col].append((prior, stats))

        else:  # 回归
            kf = KFold(self.n_splits)
            self.learned_stats = {f"{col}_mean": [] for col in self.categorical_features}

            for col in self.categorical_features:
                new_col = f"{col}_mean"
                X_encoded[new_col] = np.nan

                for train_idx, valid_idx in kf.split(y, y):
                    _, valid_encoded, prior, stats = self._encode_fold(
                        X_encoded.iloc[train_idx], y.iloc[train_idx],
                        X_encoded.iloc[valid_idx], col, None, self.prior_weight_func
                    )
                    X_encoded.iloc[valid_idx, -1] = valid_encoded
                    self.learned_stats[new_col].append((prior, stats))

        return X_encoded

    def transform(self, X):
        """
        对测试集或新数据进行编码

        :param X: pandas.DataFrame, 输入特征
        :return: pandas.DataFrame, 编码后的数据
        """
        X_encoded = X.copy()

        if self.target_type == 'classification':
            for col, t in product(self.categorical_features, self.target_values):
                new_col = f"{col}_mean_{t}"
                X_encoded[new_col] = 0
                for prior, stats in self.learned_stats[new_col]:
                    X_encoded[new_col] += X_encoded[[col]].join(stats, on=col).fillna(prior)[new_col]
                X_encoded[new_col] /= self.n_splits
        else:
            for col in self.categorical_features:
                new_col = f"{col}_mean"
                X_encoded[new_col] = 0
                for prior, stats in self.learned_stats[new_col]:
                    X_encoded[new_col] += X_encoded[[col]].join(stats, on=col).fillna(prior)[new_col]
                X_encoded[new_col] /= self.n_splits

        return X_encoded