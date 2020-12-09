import pandas as pd
# 决策树分类器，用于特征提取
from sklearn.feature_selection import SelectFromModel
# 随机森林分类器
from sklearn.ensemble import RandomForestClassifier
# train_test_split自动分训练集和测试集
from sklearn.model_selection import train_test_split
# accuracy_score获取正确率
from sklearn.metrics import accuracy_score
# 模型的保存和加载
import pickle

import time
import math
import random
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


# 遗传次数
gene_time = 100
# 种群中个体数量
pop_size = 10
# 个体长度/染色体长度
chrom_length = 14
param_chrom_length = [4, 4, 3, 3]
# 交叉概率
pc = 0.6
# 变异概率
pm = 0.01
# 存储每一代最优解 auc, n_estimators, max_depth
results =[[]]
# 个体适应度
# 创建一维数组，每个元素的内容为[0, 1, 0, 1,       0, 1, 0, 1,     0, 1, 0,    1, 0, 1]
def ind_value_and_params(individual_dparam):
    n_estimators = 1 + (individual_dparam[0]) * 8   # 0-15  1 - 129
    max_depth = individual_dparam[1] + 1
    max_features = 1 + individual_dparam[2] * 3
    min_samples_leaf = individual_dparam[3] + 1
    auc = randomforest_auc(n_estimators, max_depth, max_features, min_samples_leaf)
    return [auc, n_estimators, max_depth, max_features, min_samples_leaf]
# 存储每次繁衍过程中种群中的那个的最终解
results = []

def create_pop(chrom_length, pop_size):
    pop = []
    for i in range(pop_size):
        individual = []
        for j in range(chrom_length):
            temp = random.randint(0, 1)
            individual.append(temp)
        pop.append(individual)
    return pop

def ga(gene_time= None, pop=None):
    start = time.time()
    for i in range(gene_time):
        # 一次繁衍对应一个种群
        print("第" + str(i+1) + "代开始繁衍###############")
        # 计算第i代中，每个染色体/个体的目标函数值，即在此参数上的准确率
        [pop_values, pop_values_and_params] = cal_obj_value(pop)
        # 冒泡排序，选出适应度最好的个体
        [best_value, best_params] = best(pop_values_and_params)
        # 记录本次种群最优解
        # results.append([best_value, best_params])
        results.append(best_value)
        print("第" + str(i + 1) + "种群表现最好的个体准确率和参数值为" + str(best_value))
        # 根据个体函数值，计算个体适应度,此时设置的是舍弃准确率<0（可修改，否则没有意义）
        fit_values = cal_fit_values(pop_values)
        # 自然选择，轮转赌算法淘汰部分基因
        print("处理前" + str(pop))
        pop = selection(pop, fit_values)
        print("淘汰后" + str(pop))
        crossover(pop, pc)  # 两两交叉繁殖
        print("交叉后" + str(pop))
        mutation(pop, pm)  # 基因突变
        print("突变后" + str(pop))
    end = time.time()
    print("运行时间：%.2f秒" %(end-start))
    plt_x = np.arange(1, len(results) + 1, 1)  #1-gene_time

    plt.xlim(0, 101)
    plt.xlabel("gene_time")
    plt.ylabel("accuracy_score")
    plt.ylim(0.70, 0.82)
    plt.plot(plt_x, results)
    plt.show()






def cal_obj_value(pop):
    # 计算单个种群中各个个体对应的函数值，此时是准确率
    pop_values = []
    pop_values_and_params = []
    # 将种群中每个个体的参数都逐个解密
    for i in range(len(pop)):
        # 将个体化为10进制数
        individual_dparam = b2d(pop[i], param_chrom_length)
        # 计算个体的实际准确率和参数实际值
        ind = ind_value_and_params(individual_dparam)
        pop_values.append(ind[0])
        pop_values_and_params.append(ind)
    print(pop_values_and_params)
    return [pop_values, pop_values_and_params]

def b2d(individual, param_chrom_length):
    d_params = []
    param_size = len(param_chrom_length)
    first_index = 0
    last_index = 0
    for i in range(param_size):
        b_param_length = param_chrom_length[i]
        last_index = last_index + param_chrom_length[i]
        b_param = individual[first_index:last_index]
        d_param = 0
        for k in range(b_param_length):
            d_param += b_param[k] * (math.pow(2, b_param_length-k-1))
        d_params.append(int(d_param))
        first_index = last_index
    return d_params





def randomforest_auc(n_estimators, max_depth, max_features, min_samples_leaf):
    estimator = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                       max_features=max_features, min_samples_leaf=min_samples_leaf)
    estimator = estimator.fit(x_train, y_train)
   # print(estimator.predict_proba(x_train.head(5)))
    auc = accuracy_score(y_test, estimator.predict(x_test), normalize=True)
    return auc

def cal_fit_values(pop_values):
    fit_values = []
    min_fit_value = min(pop_values)
    max_fit_value = max(pop_values)
    # temp = 0.0
    # 舍弃负值
    for i in range(len(pop_values)):
        # if pop_values[i] > 0:
        #     temp = pop_values[i]
        # else:
        #     temp = 0.0
        # fit_values.append(max_fit_value-pop_values[i])
        fit_values.append(pop_values[i]-min_fit_value)
    return fit_values


def best(pop_values_and_params):
    # 找出种群中适应度最好的个体
    best_value_and_params = pop_values_and_params[0]
    best_value = best_value_and_params[0]
    best_params = best_value_and_params[1:]
    for i in range(1, len(pop_values_and_params)):
        ind_value_and_params = pop_values_and_params[i]
        if ind_value_and_params[0] > best_value:
            best_value = ind_value_and_params[0]
            best_params = ind_value_and_params[1:]
    return [best_value, best_params]

def selection(pop, fit_values):
    # 自然选择，淘汰一部分适应度低的个体
    new_fit_values = []
    total_fit = sum(fit_values)
    for i in range(len(fit_values)):
        new_fit_values.append(fit_values[i]/total_fit)
    # 计算每个个体的适应度累积值1-i
    cumsum_fit_values = cumsum(new_fit_values)
    print("累计适应度" + str(cumsum_fit_values))
    # 为种群中各个个体生成对应个数随机浮点数
    ms = []
    min_fit_values = new_fit_values[1]
    for i in range(len(fit_values)):
        ms.append(random.uniform(min_fit_values, 1))
    ms.sort()
    print("ms = " + str(ms))
    # 轮盘赌算法(选中的个体成为下一轮，没有选中的直接被淘汰，被选中的个体替代)
    # 以随机数为标准，取上一个随机数取值开始对应的下一个比随机数大的值
    new_pop = pop
    m = 0
    j = 0
    while m < len(pop):
        if ms[m] < cumsum_fit_values[j]:
            new_pop[m] = pop[j]
            m = m + 1
        else:
            j = j + 1
    return new_pop

def cumsum(fit_values):
    cumsum_fit_value = []
    cumsum_fit_value.append(fit_values[0])
    for i in range(1, len(fit_values)):
        cumsum_fit_value.append(cumsum_fit_value[i-1] + fit_values[i])
    return cumsum_fit_value

def crossover(pop, pc):
    len_pop = len(pop)
    for i in range(len_pop):
        if random.random() < pc:
            father = pop[i]
            mother = pop[random.randint(0, len_pop-1)]
            cpoint = random.randint(0, len(pop[0])-1)
            temp1 = []
            temp1.extend(father[0:cpoint])
            temp1.extend(mother[cpoint:])
            pop[i] = temp1



def mutation(pop, pm):
    len_pop = len(pop)
    len_individual = len(pop[0])
    for i in range(len_pop):
        if random.random() < pm:
            mpoint = random.randint(0, len_individual-1)
            if pop[i][mpoint] == 1:
                pop[i][mpoint] = 0
            else:
                pop[i][mpoint] = 1


pop = [[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
       [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
       [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
       [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
       [1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
       [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0],
       [0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
       [1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0],
       [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
       [0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0]]
# pop = create_pop(chrom_length, pop_size)
print(pop)
ga(gene_time=gene_time, pop=pop)

