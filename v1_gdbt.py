import pandas as pd
import datetime
import numpy as np
from sklearn.model_selection import KFold
import lightgbm as lgb
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
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
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

new_transactions = pd.read_csv('all/new_merchant_transactions.csv',
                               parse_dates=['purchase_date'])

historical_transactions = pd.read_csv('all/historical_transactions.csv',
                                      parse_dates=['purchase_date'])

def binarize(df):
    for col in ['authorized_flag', 'category_1']:
        df[col] = df[col].map({'Y':1, 'N':0})
    return df

# # use probability to fill -1 values in transactions
# def fill_nge_1(transactions, col):
#     null_vals = transactions[transactions[col] == -1]
#     id_list = null_vals['card_id'].unique()
#     for cid in id_list:
#         print(cid)
#         data = transactions[transactions['card_id'] == cid]
#         prob_table = data[col].value_counts().to_frame()
#         not_null = prob_table[prob_table.index != -1].sort_value()
#         if not_null.shape[0] == 0:
#             continue
#         null_count = prob_table.loc[-1]
#         fill_vals = np.zeros(null_count[0]) + prob_table[0]
#         transactions.loc[(transactions.card_id == cid) & (transactions[col] == -1), col] = fill_vals
#     return transactions

historical_transactions = binarize(historical_transactions)
new_transactions = binarize(new_transactions)

historical_transactions['month_diff'] = ((datetime.datetime.today() - historical_transactions['purchase_date']).dt.days)//30
historical_transactions['month_diff'] += historical_transactions['month_lag']
new_transactions['month_diff'] = ((datetime.datetime.today() - new_transactions['purchase_date']).dt.days)//30
new_transactions['month_diff'] += new_transactions['month_lag']

historical_transactions = pd.get_dummies(historical_transactions, columns=['category_2', 'category_3'])
new_transactions = pd.get_dummies(new_transactions, columns=['category_2', 'category_3'])

historical_transactions = reduce_mem_usage(historical_transactions)
new_transactions = reduce_mem_usage(new_transactions)

agg_fun = {'authorized_flag': ['mean']}
auth_mean = historical_transactions.groupby(['card_id']).agg(agg_fun)
auth_mean.columns = ['_'.join(col).strip() for col in auth_mean.columns.values]
auth_mean.reset_index(inplace=True)

authorized_transactions = historical_transactions[historical_transactions['authorized_flag'] == 1]
historical_transactions = historical_transactions[historical_transactions['authorized_flag'] == 0]
historical_transactions['purchase_month'] = historical_transactions['purchase_date'].dt.month.astype(int)
authorized_transactions['purchase_month'] = authorized_transactions['purchase_date'].dt.month.astype(int)
new_transactions['purchase_month'] = new_transactions['purchase_date'].dt.month.astype(int)
historical_transactions['purchase_day'] = historical_transactions['purchase_date'].dt.day.astype(int)
authorized_transactions['purchase_day'] = authorized_transactions['purchase_date'].dt.day.astype(int)
new_transactions['purchase_day'] = new_transactions['purchase_date'].dt.day.astype(int)
historical_transactions['purchase_year'] = historical_transactions['purchase_date'].dt.year.astype(int)
authorized_transactions['purchase_year'] = authorized_transactions['purchase_date'].dt.year.astype(int)
new_transactions['purchase_year'] = new_transactions['purchase_date'].dt.year.astype(int)
print('after preprocess')

def aggregate_transactions(history):
    history.loc[:, 'purchase_date'] = pd.DatetimeIndex(history['purchase_date']). \
                                          astype(np.int64) * 1e-9

    agg_func = {
        'category_1': ['sum', 'mean'],
        'category_2_1.0': ['mean'],
        'category_2_2.0': ['mean'],
        'category_2_3.0': ['mean'],
        'category_2_4.0': ['mean'],
        'category_2_5.0': ['mean'],
        'category_3_A': ['mean'],
        'category_3_B': ['mean'],
        'category_3_C': ['mean'],
        'merchant_id': ['nunique'],
        'merchant_category_id': ['nunique'],
        'state_id': ['nunique',lambda x: stats.mode(x)[0][0]],
        'city_id': ['nunique',lambda x: stats.mode(x)[0][0]],
        'subsector_id': ['nunique',lambda x: stats.mode(x)[0][0]],
        'purchase_amount': ['sum', 'mean', 'max', 'min', 'std'],
        'installments': ['sum', 'mean', 'max', 'min', 'std'],
        'purchase_day': ['mean', 'max', 'min', 'std'],
        'purchase_month': ['mean', 'max', 'min', 'std'],
        'purchase_year': ['mean', 'max', 'min', 'std'],
        'purchase_date': [np.ptp, 'min', 'max'],
        'month_lag': ['mean', 'max', 'min', 'std'],
        'month_diff': ['mean']
    }

    agg_history = history.groupby(['card_id']).agg(agg_func)
    agg_history.columns = ['_'.join(col).strip() for col in agg_history.columns.values]
    agg_history.reset_index(inplace=True)

    df = (history.groupby('card_id')
          .size()
          .reset_index(name='transactions_count'))

    agg_history = pd.merge(df, agg_history, on='card_id', how='left')

    return agg_history
history = aggregate_transactions(historical_transactions)
history.columns = ['hist_' + c if c != 'card_id' else c for c in history.columns]
del historical_transactions
authorized = aggregate_transactions(authorized_transactions)
authorized.columns = ['auth_' + c if c != 'card_id' else c for c in authorized.columns]
new = aggregate_transactions(new_transactions)
new.columns = ['new_' + c if c != 'card_id' else c for c in new.columns]


def aggregate_per_month(history):
    grouped = history.groupby(['card_id', 'month_lag'])

    agg_func = {
        'purchase_amount': ['count', 'sum', 'mean', 'min', 'max', 'std'],
        'installments': ['count', 'sum', 'mean', 'min', 'max', 'std'],
    }

    intermediate_group = grouped.agg(agg_func)
    intermediate_group.columns = ['_'.join(col).strip() for col in intermediate_group.columns.values]
    intermediate_group.reset_index(inplace=True)

    final_group = intermediate_group.groupby('card_id').agg(['mean', 'std'])
    final_group.columns = ['_'.join(col).strip() for col in final_group.columns.values]
    final_group.reset_index(inplace=True)

    return final_group

print('after aggregation')
# ___________________________________________________________
final_group = aggregate_per_month(authorized_transactions)

def successive_aggregates(df, field1, field2):
    t = df.groupby(['card_id', field1])[field2].mean()
    u = pd.DataFrame(t).reset_index().groupby('card_id')[field2].agg(['mean', 'min', 'max', 'std'])
    u.columns = [field1 + '_' + field2 + '_' + col for col in u.columns.values]
    u.reset_index(inplace=True)
    return u

additional_fields = successive_aggregates(new_transactions, 'category_1', 'purchase_amount')
additional_fields = additional_fields.merge(successive_aggregates(new_transactions, 'installments', 'purchase_amount'),
                                            on = 'card_id', how='left')
additional_fields = additional_fields.merge(successive_aggregates(new_transactions, 'city_id', 'purchase_amount'),
                                            on = 'card_id', how='left')
additional_fields = additional_fields.merge(successive_aggregates(new_transactions, 'category_1', 'installments'),
                                            on = 'card_id', how='left')


def read_data(input_file):
    df = pd.read_csv(input_file)
    df['first_active_month'] = pd.to_datetime(df['first_active_month'])
    df['first_active_month'] = df['first_active_month'].fillna(datetime.date(2018,2,1))
    print(df['first_active_month'].isna().value_counts())
    df['elapsed_time'] = (datetime.date(2018, 2, 1) - df['first_active_month'].dt.date).dt.days.astype(int)
    df = df.drop('first_active_month', axis=1)
    return df
#_________________________________________

train = read_data('all/train.csv')
# test = read_data('all/test.csv')

target = train['target']

del train['target']

train = pd.merge(train, history, on='card_id', how='left')
test = pd.merge(test, history, on='card_id', how='left')
del history

train = pd.merge(train, authorized, on='card_id', how='left')
test = pd.merge(test, authorized, on='card_id', how='left')
del authorized

train = pd.merge(train, new, on='card_id', how='left')
test = pd.merge(test, new, on='card_id', how='left')
del new

train = pd.merge(train, final_group, on='card_id', how='left')
test = pd.merge(test, final_group, on='card_id', how='left')
del final_group

train = pd.merge(train, auth_mean, on='card_id', how='left')
test = pd.merge(test, auth_mean, on='card_id', how='left')
del auth_mean

train = pd.merge(train, additional_fields, on='card_id', how='left')
test = pd.merge(test, additional_fields, on='card_id', how='left')
del additional_fields

print('after combining')

# add a PCA here
train = pd.read_csv('all/newtrain.csv')
categorical_feats = ['feature_1', 'feature_2', 'feature_3']
test = pd.read_csv('all/newtest.csv')
train_id = train['card_id']
train_feats = train[categorical_feats]
train = train.drop(['card_id'], axis=1)
train = train.drop(categorical_feats, axis=1)

for columns in train.columns:
    tryimputed = train[columns]
    tryimputed = tryimputed.values.reshape(-1, 1)
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(tryimputed)
    train[columns] = imp.transform(tryimputed).astype(int)

test_id = test['card_id']
test = test.drop(['card_id'], axis=1)
test_feats = test[categorical_feats]
test = test.drop(categorical_feats, axis=1)

for columns in test.columns:
    tryimputed = test[columns]
    tryimputed = tryimputed.values.reshape(-1, 1)
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(tryimputed)
    test[columns] = imp.transform(tryimputed).astype(int)

pca = PCA(n_components=30)
train = pd.DataFrame(pca.fit_transform(train))
train = pd.concat([train, train_feats], axis=1)
test = pd.DataFrame(pca.transform(test))
test = pd.concat([test, test_feats], axis=1)
print('after pca')

param = {'num_leaves': 111,
         'min_data_in_leaf': 149,
         'objective':'regression',
         'max_depth': 9,
         'learning_rate': 0.001,
         "boosting": "gbdt",
         "feature_fraction": 0.7522,
         "bagging_freq": 1,
         "bagging_fraction": 0.7083 ,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.2634,
         "random_state": 133,
         "verbosity": -1}

folds = KFold(n_splits=5, shuffle=True, random_state=17)
oof = np.zeros(len(train))
predictions = np.zeros(len(test))
# feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
    print("fold n°{}".format(fold_))
    trn_data = lgb.Dataset(train.iloc[trn_idx],
                           label=target.iloc[trn_idx],
                           # categorical_feature=categorical_feats
                           )
    val_data = lgb.Dataset(train.iloc[val_idx],
                           label=target.iloc[val_idx],
                           # categorical_feature=categorical_feats
                           )

    num_round = 10000
    clf = lgb.train(param,
                    trn_data,
                    num_round,
                    valid_sets=[trn_data, val_data],
                    verbose_eval=100,
                    early_stopping_rounds=300)

    oof[val_idx] = clf.predict(train.iloc[val_idx], num_iteration=clf.best_iteration)

    predictions += clf.predict(test, num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(mean_squared_error(oof, target) ** 0.5))

sub_df = pd.DataFrame({"card_id":test_id})
sub_df["target"] = predictions
sub_df.to_csv("submit.csv", index=False)
