import pandas as pd
# 决策树分类器，用于特征提取
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
# 随机森林分类器
from sklearn.ensemble import RandomForestClassifier
# train_test_split自动分训练集和测试集
from sklearn.model_selection import train_test_split
# 验证方式cross_val_score交叉验证
from sklearn.model_selection import cross_val_score, GridSearchCV
# accuracy_score获取正确率
from sklearn.metrics import accuracy_score
# 模型的保存和加载
import pickle
# 计算准确率
from sklearn.metrics import classification_report

import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("20200705上海数据3141.csv", encoding="UTF-8")
print("data pre-processing start", "#" * 50)
# 查询原数据统计列基本信息
print(df.info())
# 获取行，列数
row_counts = df.shape[0]
column_counts = df.shape[1]
print("row:", row_counts)
print("column:", column_counts)
# 填充一些可以填充的行
P_is_null = df['高血压史'].isnull()
Q_is_null = df['高血压年数'].isnull()
Q_modify_num = 0
R_is_null = df['糖尿病史'].isnull()
S_is_null = df['糖尿病年数'].isnull()
S_modify_num = 0
for i in range(0, row_counts, 1):
    # 填充高血压
    if (P_is_null[i] == False) & (Q_is_null[i] == True):
        df.loc[i, '高血压年数'] = 0
        Q_modify_num = Q_modify_num + 1
        # print("第", i, "行填充高血压年数")
    # 填充糖尿病
    if (R_is_null[i] == False) & (S_is_null[i] == True):
        df.loc[i, '糖尿病年数'] = 0
        S_modify_num = S_modify_num + 1
        # print("第", i, "行填充糖尿病年数")
print("高血压年数列填充", Q_modify_num, "行")
print("糖尿病年数列填充", S_modify_num, "行")
print(df.info())
# 删除空值> 1/4的行
# 获取每一行的空值数
null_counts = df.isnull().sum(axis=1)
drop_max_counts = column_counts // 4
print("删除前总行数:", row_counts)
for i in range(0, row_counts, 1):
    if null_counts[i] >= drop_max_counts:
        df.drop(i, inplace=True)
print(df.info())
print("删除空值过多后总行数:", df.shape[0])

# 填充剩下的缺省行
D_mean = df['收缩压'].mean()
df['收缩压'].fillna(D_mean, inplace=True)
E_mean = df['舒张压'].mean()
df['舒张压'].fillna(E_mean, inplace=True)
F_mean = df['身高'].mean()
df['身高'].fillna(F_mean, inplace=True)
G_mean = df['体重'].mean()
df['体重'].fillna(G_mean, inplace=True)
H_mean = df['BMI'].mean()
df['BMI'].fillna(H_mean, inplace=True)
I_mean = df['腰臀比'].mean()
df['腰臀比'].fillna(I_mean, inplace=True)
K_mode = df['婚姻1=未2=离3=已婚4=丧偶'].mode()
df['婚姻1=未2=离3=已婚4=丧偶'].fillna(K_mode[0], inplace=True)
L_mean = df['MMSE'].mean()
df['MMSE'].fillna(L_mean, inplace=True)
df['吸烟'].fillna(0, inplace=True)
df['饮酒'].fillna(0, inplace=True)

P_mode = df['高血压史'].mode()
df['高血压史'].fillna(P_mode[0], inplace=True)
Q_mode = df['高血压年数'].mode()
df['高血压年数'].fillna(Q_mode[0], inplace=True)
R_mode = df['糖尿病史'].mode()
df['糖尿病史'].fillna(R_mode[0], inplace=True)
S_mode = df['糖尿病年数'].mode()
df['糖尿病年数'].fillna(S_mode[0], inplace=True)

T_mode = df['0=physical inactivity'].mode()
df['0=physical inactivity'].fillna(T_mode[0], inplace=True)
V_mode = df['ADL '].mode()
df['ADL '].fillna(V_mode[0], inplace=True)
W_mean = df['CESD>=16为抑郁'].mean()
df['CESD>=16为抑郁'].fillna(W_mean, inplace=True)
X_mean = df['空腹血糖值'].mean()
df['空腹血糖值'].fillna(X_mean, inplace=True)
print(df.info())
print("data pre-processing ending", "#" * 50)
# 数据预处理，ending --------------------------------------------

# 特征提取，starting ------------------------------------------

x = df.drop(['ID', '认知诊断1=正常2=MCI3=痴呆'], axis=1)
y = df['认知诊断1=正常2=MCI3=痴呆']

# 随机森林作为基模型的特征选择
selector = SelectFromModel(RandomForestClassifier(), max_features=12).fit(x, y)
support_list = selector.get_support()
print("特征是否保留", support_list)
indices = []
for i in range(0, x.shape[1], 1):
    if support_list[i]:
        indices.append(i)
print("特征提取序号:", indices)
print("未特征提取前属性:", x.columns)
pre_columns = x.columns[indices]
print("特征提取属性结果:", pre_columns)

seed = 5
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=seed)


# 根据重要性重新划分训练集和测试集
# preIndex = indices[:20]
# print("-" * 30)
# print(x.columns[indices])
# x2 = x[x.columns[indices]]
# x_train, x_test, y_train, y_test = train_test_split(x2, y, test_size=0.3, random_state=seed)


# 划分测试集和训练集
x2 = x[pre_columns]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=seed)


def self_predict(x):
    x = x[['文化年', 'MMSE', 'ADL ']]
    rows = x.shape[0]
    result_list = []
    for i in range(0, rows, 1):
        education = x.iloc[i, 0]
        MMSE = x.iloc[i, 1]
        ADL = x.iloc[i, 2]
        if ADL > 50:
            result_list.append(3)
        else:
            if education < 1:
                if MMSE > 18:
                    result_list.append(1)
                elif int(MMSE <= 18) & int(MMSE > 16):
                    result_list.append(2)
                else:
                    result_list.append(3)
            elif int(education >= 1) & int(education <= 6):
                if MMSE > 22:
                    result_list.append(1)
                elif int(MMSE <= 22) & int(MMSE > 19):
                    result_list.append(2)
                else:
                    result_list.append(3)
            else:
                if MMSE > 27:
                    result_list.append(1)
                elif int(MMSE <= 27) & int(MMSE > 23):
                    result_list.append(2)
                else:
                    result_list.append(3)
    print(result_list)
    result_array = np.array(result_list)
    print(result_array)
    return result_array


self_predict = self_predict(x_test)
# acc = accuracy_score(y_test, self_predict, normalize=True)
# print(acc)
print(classification_report(y_test, self_predict, digits=4))
