# -*- coding: utf-8 -*-
"""
此代码用于统计分析
"""
import numpy as np
import pandas as pd
from scipy import stats
import scikit_posthocs as sp
from statsmodels.stats.inter_rater import fleiss_kappa
import numpy as np
import matplotlib.pyplot as plt
import krippendorff
from scipy.stats import norm
from tableone import TableOne

class Statistical():
    def __init__(self):
        self.colors =[
            [246, 111, 105],  # 红色
            [254, 179, 174],  # 粉色
            [21, 151, 165],   # 青色
            [14, 96, 107],    # 深青色
            [255, 194, 75]    # 黄色
        ]
        self.colors = np.array(self.colors) / 255

    def _get_table_one(self, table_one_file):
        data=pd.read_csv(table_one_file, encoding='gbk')
        columns = ['检查号', '性别', '年龄', '检查类型', '病人来源', '检查项目', 'word_count']
        data = data[columns]
        data.set_index('检查号', inplace=True)
        data = data[~data.index.duplicated(keep='first')]  # 去掉重复的index
        
        # Preprocess Exam item: upper abdomen, Middle abdomen, Lower abdomen
        item = {}
        enhancement = {}
        for idx in data.index:
            item[idx] = ''
            if ('肝' in data.loc[idx, '检查项目']) or ('上腹' in data.loc[idx, '检查项目']):
                item[idx] += 'Upper abdomen plus '
            if ('肾' in data.loc[idx, '检查项目']) or ('中腹' in data.loc[idx, '检查项目']):
                item[idx] += 'Middle abdomen plus '
            if ('盆腔' in data.loc[idx, '检查项目']) or ('下腹' in data.loc[idx, '检查项目']):
                item[idx] += 'Lower abdomen plus '
            
            # Enhancement
            if '增强' in data.loc[idx, '检查项目']:
                enhancement[idx] = 'Contrast-enhanced'
            else:
                enhancement[idx] = 'Non-contrast-enhanced'
        
        # delete the last 'plus '
        for idx in item.keys():
            item[idx] = item[idx][:-6]
        # replace data['检查项目'] with item
        for idx in item.keys():
            data.loc[idx, '检查项目'] = item[idx]
        
        # add enhancement
        data['Enhancement'] = enhancement.values()

        # Preprocess Gender: 男->Male, 女->Female
        data.loc[:, '性别'] = data.loc[:, '性别'].apply(lambda x: 'Male' if x == '男' else 'Female')

        # Preprocess Age: 保留数字部分
        data.loc[:, '年龄'] = data.loc[:, '年龄'].astype(str).apply(lambda x: x.split('岁')[0])

        # Preprocess Source: 门诊->Outpatient, 住院->Inpatient, 急诊->Emergency
        data.loc[:, '病人来源'] = data.loc[:, '病人来源'].apply(lambda x: 'Outpatient' if x == '门诊' else 'Inpatient' if x == '住院' else 'Emergency')

        # Rename columns
        rename={
            '性别': 'Gender',
            '年龄': 'Age',
            '检查类型': 'Modality',
            '病人来源': 'Patient source',
            '检查项目': 'Exam site',
            'word_count': 'Length of the the findings section'
        }
        data.rename(columns=rename, inplace=True)

        # Reorder columns
        data = data[['Gender', 'Age', 'Patient source', 'Modality', 'Enhancement', 'Exam site', 'Length of the the findings section']]
        vc = data.value_counts

        # TableOne
        columns = ['Gender', 'Age', 'Patient source', 'Modality', 'Enhancement', 'Exam site', 'Length of the the findings section']
        categorical = ['Gender', 'Patient source', 'Modality', 'Enhancement','Exam site']
        groupby = None
        nonnormal = ['Age','Length of the the findings section']
        numeric = ['Age','Length of the the findings section']
        data[numeric] = data[numeric].apply(pd.to_numeric, errors='coerce')

        mytable = TableOne(
                    data, 
                    columns=columns,
                    categorical=categorical,
                    groupby=groupby,
                    nonnormal=nonnormal,
                    pval=False
        )

        print(mytable.tabulate(tablefmt = "grid"))

        mytable.to_excel('mytable.xlsx')

    def load_data(self, file_paths, match_id_path, icc_id_path):
        """
        读取数据

        Parameters
        ----------
        file_paths: path str or list
            文件路径，可以是csv、excel等格式
        match_id_path: path str
            匹配ID，用于匹配模型输出和模型名称
        icc_id_path: path str
            用于做icc的ID
        
        Returns
        -------
        data: dict
            数据集，key为文件路径，value为数据集
        """
        data = {}
        for file_path in file_paths:
            if file_path.endswith('.csv'):
                data_ = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                data_ = pd.read_excel(file_path)
            elif file_path.endswith('.xls'):
                data_ = pd.read_excel(file_path)
            data[file_path] = data_

        match_id = pd.read_json(match_id_path, orient='records')
        icc_id = pd.read_json(icc_id_path, orient='records')

        return data, match_id, icc_id

    def preprocess_data(self, data, match_id, icc_id):
        """
        数据预处理

        Parameters
        ----------
        data: np.array or list
            数据集
        match_id: pd.DataFrame
            匹配ID，用于匹配模型输出和模型名称
        icc_id: pd.DataFrame
            用于做icc的ID
        
        Returns
        -------
        data: np.array
            预处理后的数据集
        """
        #%% 将data中的每个data的第一列的列名改为idx,并将其设置为index
        for key in data.keys():
            data[key].columns = ['idx'] + data[key].columns[1:].tolist()
            data[key].set_index('idx', inplace=True)

        # 每个data取前20列
        for key in data.keys():
            data[key] = data[key].iloc[:, :19]

        #%% 将data按照model name重新对列名进行排序
        data_preprocessed = {}
        for key in data.keys():
            data_preprocessed[key] = data[key].copy(deep=True)
        # 修改colname:将colname中的model1改为wxyy，model2改为claude，model3改为human
        [data_preprocessed_.rename(columns={'model1': 'wxyy', 'model2': 'claude', 'model3': 'human'}, inplace=True)\
            for data_preprocessed_ in data_preprocessed.values()]
        
        # 将data_preprocessed的值全部置为0
        for key in data_preprocessed.keys():
            data_preprocessed[key].iloc[:, :] = None

        # 将相应的data的值填充到data_preprocessed中
        match_id = match_id.T
        for key in data_preprocessed.keys():
            data_preprocessed[key]["影像所见"] = data[key]["影像所见"]
            for idx_ in data_preprocessed[key].index:
                # wxyy
                wxyy_uid = match_id.loc[idx_]['wxyy']
                in_which_col = [wxyy_uid in data[key].loc[idx_][mn] for mn in ['model1', 'model2', 'model3']]
                which_model = ['model1', 'model2', 'model3'][in_which_col.index(True)]
                model_num = which_model[-1]
                random_col = [which_model, f'完整性{model_num}', f'幻觉性{model_num}', f'准确性{model_num}', f'表达力{model_num}', f'修改度{model_num}']
                data_preprocessed[key].loc[idx_, ['wxyy', '完整性1', '幻觉性1', '准确性1', '表达力1', '修改度1']] = data[key].loc[idx_, random_col].values

                # claude
                claude_uid = match_id.loc[idx_]['claude']
                in_which_col = [claude_uid in data[key].loc[idx_][mn] for mn in ['model1', 'model2', 'model3']]
                which_model = ['model1', 'model2', 'model3'][in_which_col.index(True)]
                model_num = which_model[-1]
                random_col = [which_model, f'完整性{model_num}', f'幻觉性{model_num}', f'准确性{model_num}', f'表达力{model_num}', f'修改度{model_num}']
                data_preprocessed[key].loc[idx_, ['claude', '完整性2', '幻觉性2', '准确性2', '表达力2', '修改度2']] = data[key].loc[idx_, random_col].values

                # human
                human_uid = match_id.loc[idx_]['human']
                in_which_col = [human_uid in data[key].loc[idx_][mn] for mn in ['model1', 'model2', 'model3']]
                which_model = ['model1', 'model2', 'model3'][in_which_col.index(True)]
                model_num = which_model[-1]
                random_col = [which_model, f'完整性{model_num}', f'幻觉性{model_num}', f'准确性{model_num}', f'表达力{model_num}', f'修改度{model_num}']
                data_preprocessed[key].loc[idx_, ['human', '完整性3', '幻觉性3', '准确性3', '表达力3', '修改度3']] = data[key].loc[idx_, random_col].values

        #%% 检查data_preprocessed是否有空值,以及那个key那个idx有空值
        null_idx = []
        for key in data_preprocessed.keys():
            for idx_ in data_preprocessed[key].index:
                if data_preprocessed[key].loc[idx_].isnull().sum() > 0:
                    print(key, idx_)
                    null_idx.append(idx_)

        #%% 将data_preprocessed中的4~5类似的变成4，同理3~4类似的变成3，2~3类似的变成2，1~2类似的变成1
        # 这是因为X.C.评估时出现了这种情况，我们取第一个值，即最低值
        num_col = ['完整性1', '幻觉性1', '准确性1', '表达力1', '修改度1',
                    '完整性2', '幻觉性2', '准确性2', '表达力2', '修改度2',
                    '完整性3', '幻觉性3', '准确性3', '表达力3', '修改度3'
        ]
        for key in data_preprocessed.keys():
            # 将每个元素用~分割，取第一个元素，然后将其转换为float
            data_preprocessed[key][num_col] = data_preprocessed[key][num_col].astype(str).applymap(lambda x:x.split('~')[0])
            data_preprocessed[key][num_col] = data_preprocessed[key][num_col].astype(str).applymap(lambda x:x.split('-')[0])
            data_preprocessed[key][num_col] = data_preprocessed[key][num_col].astype(float)

        #%% 将data中的每个data的icc部分提取出来，存放到icc_data中
        icc_id = icc_id.values.reshape(-1,)
        icc_data = {}
        for key in data_preprocessed.keys():
            icc_data[key] = data_preprocessed[key].loc[data_preprocessed[key].index.isin(icc_id)]
            # 从data_preprocessed中删除icc_data
            data_preprocessed[key].drop(index=icc_data[key].index, inplace=True)
        data_preprocessed_all = pd.concat([data_preprocessed[key] for key in data_preprocessed.keys()], axis=0)
        
        return data_preprocessed_all, icc_data
    
    def my_fleiss_kappa(self, data):
        """
        计算Cohen's Kappa系数

        Parameters
        ----------
        data: list of pd.DataFrame
            两个或多个文件的数据集

        Returns
        -------
        kappa: float
            Cohen's Kappa系数
        """
        # 求多个data重复的index并排序
        idx = data[0].index
        for i in range(1, len(data)):
            idx = idx.intersection(data[i].index)
        data = [d.loc[idx] for d in data]

        # assumes subjects in rows, and categories in columns
        kappa_results = {}
        rate_percent = {}
        for key in data[0].columns:
            dd = [d[key].values for d in data]
            df = pd.DataFrame(dd)
            dd_transformed = self.transform(*dd)
            rate_percent[key] = dd_transformed

            k = fleiss_kappa(dd_transformed)
            K = self.fleiss_kappa_(dd_transformed)
            kappa_results[key] = K

            print(f'{key}的加权Kappa系数为：{K}')
        
        # plot percent
        value_counts_all = {}
        for i, key in enumerate(rate_percent.keys()):
            rate_percent[key] = pd.DataFrame(rate_percent[key], columns=['1', '2', '3', '4', '5'])
            dd = rate_percent[key].values
            # 去掉0,1,2,即只看3个人以上一致的比例
            uni_d = [5, 4, 3]
            value_counts = {f"{int(ud)} Raters": f"{np.sum(dd == ud)/len(dd):.2f}" for ud in uni_d}
            value_counts['<3 Raters'] = 1 - np.sum(np.float16(list(value_counts.values())))
            value_counts_all[key] = value_counts

        # Plot
        new_keys = ["Comprehensiveness" , "Hallucinations", "Accuracy", "Expressiveness", "Accept without revision",
                    "Comprehensiveness of claude" , "Hallucinations of claude", "Accuracy of claude", "Expressiveness of claude", "Accept without revision of claude",
                    "Comprehensiveness of human" , "Hallucinations of human", "Accuracy of human", "Expressiveness of human", "Accept without revision of human"
        ]
        # rename key
        value_counts_all = dict(zip(new_keys, list(value_counts_all.values())))
        models = ['ERNIE', 'Claude', 'Human']
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Times New Roman']
        plt.rcParams['axes.unicode_minus'] = False  # 确保负号正常显示
        fig, ax = plt.subplots(3, 5, figsize=(12, 8))
        for i, key in enumerate(value_counts_all.keys()):
            # plot percent:value_counts,圈圈图
            i_p = i // 5
            j_p = i % 5
            wedges, texts, autotexts  = \
                ax[i_p, j_p].pie(value_counts_all[key].values(), 
                            startangle=90, 
                            wedgeprops={'width':0.3}, 
                            colors=self.colors,
                            autopct='%1.f%%',
                            pctdistance=1.15,
                            # labeldistance=1,
                            rotatelabels=True
            )

            # 手动旋转百分比标签
            for autotext, wedge in zip(autotexts, wedges):
                angle = (wedge.theta2 - wedge.theta1) / 2 + wedge.theta1  # 计算角度
                x = np.cos(np.radians(angle))  # 计算标签的x坐标
                y = np.sin(np.radians(angle))  # 计算标签的y坐标
                ha = 'left' if x > 0 else 'right'  # 水平对齐方式
                rotation = angle if x > 0 else angle + 180  # 旋转角度
                autotext.set_rotation(rotation)  # 设置旋转角度
                autotext.set_ha(ha)  # 设置水平对齐方式
                autotext.set_va('center')  # 设置垂直对齐方式

            # title
            if i_p == 0:
                ax[i_p, j_p].set_title(key, fontsize=15)  # 标题字体大小
                # title离图像的距离
                ax[i_p, j_p].title.set_position([0.5, 2])
            # y轴显示为各个模型
            if j_p == 0:
                ax[i_p, j_p].set_ylabel(models[i_p], fontsize=15)  # y轴字体大小
                # y轴label离图像的距离
                ax[i_p, j_p].yaxis.set_label_coords(-0.2, 0.5)
            # y轴字体大小
            ax[i_p, j_p].tick_params(axis='y', labelsize=15)
        
        # legend: 设置到框外,水平铺开，放在框外上
        legend = ax[i_p, j_p].legend(value_counts_all[key].keys(), 
                            loc='upper center', 
                            bbox_to_anchor=(-0.5, -0.1),
                            ncol=4,
        )

        # 字图之间的间隔
        plt.subplots_adjust(hspace=0, wspace=0.2)
        plt.tight_layout()
        print("save figure...")
        plt.savefig('../manuscript/fig/percent.jpg', dpi=1000)

    def fleiss_kappa_(self, dd_transformed: np.array):
        """
        Calculates Fleiss' kappa coefficient for inter-rater agreement.
        Args:
            dd_transformed: numpy array of shape (subjects, categories), where each element represents
                the number of raters who assigned a particular category to a subject.
        Returns:
            kappa: Fleiss' kappa coefficient.
        """
        subjects, _ = dd_transformed.shape
        n_rater = np.sum(dd_transformed[0])
    
        # p_j = np.sum(dd_transformed, axis=0) / (n_rater * subjects)
        # P_e_bar = np.sum(p_j ** 2)
    
        # P_i = (np.sum(dd_transformed ** 2, axis=1) - n_rater) / (n_rater * (n_rater - 1))
        # P_bar = np.sum(P_i)/subjects
    
        # K = (P_bar - P_e_bar) / (1 - P_e_bar + 1e-10)
        P_hat = np.sum((np.sum(np.power(dd_transformed, 2), axis=1) - n_rater) / (n_rater * (n_rater - 1)))/subjects

        p_e_hat = np.sum(np.power(np.sum(dd_transformed, axis=0) / (n_rater * subjects), 2))

        K = (P_hat - p_e_hat)/(1-p_e_hat)
    
        # tmp = (1 - P_e_bar) ** 2
        # var = 2 * (tmp - np.sum(p_j * (1 - p_j) * (1 - 2 * p_j)) + 1e-10) / (tmp * subjects * n_rater * (n_rater - 1) + 1e-10)
        
        # # standard error
        # SE = np.sqrt(var) 
        # Z = K / SE
        # p_value = 2 * (1 - norm.cdf(np.abs(Z)))
        # ci_bound = 1.96 * SE / subjects
        # lower_ci_bound = K - ci_bound
        # upper_ci_bound = K + ci_bound
        # return K, SE, Z, p_value, lower_ci_bound, upper_ci_bound
        return K

    def transform(self, *raters, categories=5):
        """
        Transforms the ratings of multiple raters into the required data format for Fleiss' Kappa calculation.
        Args:
            *raters: Multiple raters' ratings. Each rater's ratings should be a list or array of annotations.
        Returns:
            data: numpy array of shape (subjects, categories), where each element represents the number of raters
                who assigned a particular category to a subject.
        """
        assert all(len(rater) == len(raters[0]) for rater in raters), "Lengths of raters are not consistent."
        subjects = len(raters[0])
        data = np.zeros((subjects, categories))
        for i in range(subjects):
            for rater in raters:
                data[i, int(rater[i])-1] += 1
        
        return data
 
    def tranform2(self, weighted):
        """
        Transforms weighted data into the required data format for Fleiss' Kappa calculation.
        Args:
            weighted: List of weighted ratings. Each row represents [rater_0_category, rater_1_category, ..., rater_n_category, weight].
        Returns:
            data: numpy array of shape (subjects, categories), where each element represents the number of raters
                who assigned a particular category to a subject.
        """
        n_rater = len(weighted[0]) - 1
        raters = [[] for _ in range(n_rater)]
        for i in range(len(weighted)):
            for j in range(len(raters)):
                raters[j] = raters[j] + [weighted[i][j] for _ in range(weighted[i][n_rater])]
        
        data = self.transform(*raters)
        
        return data
 
    def test(self):
        # Example data provided by wikipedia https://en.wikipedia.org/wiki/Fleiss_kappa
        dd_transformed = np.array([
            [0, 0, 0, 0, 14],
            [0, 2, 6, 4, 2],
            [0, 0, 3, 5, 6],
            [0, 3, 9, 2, 0],
            [2, 2, 8, 1, 1],
            [7, 7, 0, 0, 0],
            [3, 2, 6, 3, 0],
            [2, 5, 3, 2, 2],
            [6, 5, 2, 1, 0],
            [0, 2, 2, 3, 7]
        ])

        dd_transformed = np.array([
            [5, 0, 0],
            [4, 1, 0],
            [4, 0, 1],
        ])

    
        self.fleiss_kappa_(dd_transformed)
    
        # need transform
        rater1 = [1, 2, 3,4,4]
        rater2 = [1, 2, 4,4,5]
        rater3 = [1, 2, 3,4,5]
        raters = np.array([rater1, rater2, rater3])
    
        data1 = self.transform(rater1, rater2, rater3)
        self.fleiss_kappa_(data1)
    
        # The first row indicates that both rater 1 and 2 rated as category 0, this case occurs 8 times.
        # need transform2
        weighted_data = [
            [0, 0, 8],
            [0, 1, 2],
            [0, 2, 0],
            [1, 0, 0],
            [1, 1, 17],
            [1, 2, 3],
            [2, 0, 0],
            [2, 1, 5],
            [2, 2, 15]
        ]
        data = self.tranform2(weighted_data)
        fleiss_kappa(data)
    
    def friedman_test(self, data):
        """
        Friedman检验

        Parameters
        ----------
        data: np.array
            数据集，每一行为一个样本，每一列为一个case
        
        Returns
        -------
        s: float
            Friedman检验的统计量
        p: float
            Friedman检验的p值
        """
        s, p = stats.friedmanchisquare(*data)
        return s, p

    def _friedman_nemenyi_test(self, data):
        # friedman_test
        friedman_test_result = {'完整性': {}, '幻觉性': {}, '准确性': {}, '表达力': {}, '修改度': {}}
        nemenyi_test_result = {'完整性': {}, '幻觉性': {}, '准确性': {}, '表达力': {}, '修改度': {}}
        for key in friedman_test_result:
            group1 = data[f'{key}1'].values
            group2 = data[f'{key}2'].values
            group3 = data[f'{key}3'].values
            x = np.array([group1, group2, group3])
            s, p = ss.friedman_test(x)
            mean = np.mean(x, axis=1)
            friedman_test_result[key]['s'] = s
            friedman_test_result[key]['p'] = p
            friedman_test_result[key]['mean'] = mean

            # 使用scikit-posthocs库进行后续的Nemenyi多重比较检验，使用np.array格式
            pmat = sp.posthoc_nemenyi_friedman(x.T)
            pmat.index = ['wxyy', 'claude', 'human']
            pmat.columns = ['wxyy', 'claude', 'human']
            nemenyi_test_result[key] = pmat

        return friedman_test_result, nemenyi_test_result

    def _visulization_performances(self, data, friedman_test_result, nemenyi_test_result):
        """
        可视化

        Parameters
        ----------
        data: pd.DataFrame
            数据集
        """

        # 堆叠百分比柱状图
        # 设置字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Times New Roman']
        plt.rcParams['axes.unicode_minus'] = False  # 确保负号正常显示
        titles = ["Comprehensiveness" , "Hallucinations", "Accuracy", "Expressiveness", "Accept without revision"]
        fig, ax = plt.subplots(1, 5, figsize=(10, 3))
        for i, key in enumerate(['完整性', '幻觉性', '准确性', '表达力', '修改度']):
            data_ = data[[f'{key}1', f'{key}2', f'{key}3']]
            # 每一列数据中，1,2,3,4,5的比例
            data_perc = {1: data_.apply(lambda x: np.sum(x == 1)/len(x), axis=0),
                        2: data_.apply(lambda x: np.sum(x == 2)/len(x), axis=0),
                        3: data_.apply(lambda x: np.sum(x == 3)/len(x), axis=0),
                        4: data_.apply(lambda x: np.sum(x == 4)/len(x), axis=0),
                        5: data_.apply(lambda x: np.sum(x == 5)/len(x), axis=0)
            }
            data_perc = pd.DataFrame(data_perc)
            x = range(data_perc.index.shape[0])
            for i_ in data_perc.columns:
                if i_ == data_perc.columns[0]:
                    bottom = [0]*data_perc.shape[0]
                else:
                    bottom = np.sum([data_perc[j].values for j in np.arange(1, i_)], axis=0)
                ax[i].bar(x, data_perc[i_].values, bottom=bottom, color=self.colors[i_-1])
            ax[i].set_title(titles[i], fontsize=10)
            # x ticks and tick labels
            ax[i].set_xticks(range(data_perc.shape[0]))
            ax[i].set_xticklabels(['ERNIE', 'Claude', 'Human'], rotation=0, fontsize=8)
            # y range
            ax[i].set_ylim(0, 1.5)
            # i >0时，y轴不显示
            if i > 0:
                ax[i].set_yticks([])
            
            # 框外，左侧
            if i ==0:
                ax[i].set_ylabel('Percentage')
            if i == len(titles)-1:
                legend = ax[i].legend(['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'], 
                            loc='upper center',
                            bbox_to_anchor=(1.6, 1),
                            ncol=1,
                            prop={'size': 8}
                )
            
            # add signature line
            p_wc = nemenyi_test_result[key].loc["wxyy", "claude"]
            p_wc = f"p={p_wc:.3f}" if p_wc >=0.001 else "p<0.001"
            p_wh = nemenyi_test_result[key].loc["wxyy", "human"]
            p_wh = f"p={p_wh:.3f}" if p_wh >=0.001 else "p<0.001"
            p_ch = nemenyi_test_result[key].loc["claude", "human"]
            p_ch = f"p={p_ch:.3f}" if p_ch >=0.001 else "p<0.001"
            # line wc
            ax[i].plot([0, 1], [1.05, 1.05], color='black', linewidth=1)
            ax[i].plot([0, 0], [1.02, 1.05], color='black', linewidth=1)
            ax[i].plot([1, 1], [1.02, 1.05], color='black', linewidth=1)
            # line wh
            ax[i].plot([0, 2], [1.2, 1.2], color='black', linewidth=1)
            ax[i].plot([0, 0], [1.17, 1.2], color='black', linewidth=1)
            ax[i].plot([2, 2], [1.17, 1.2], color='black', linewidth=1)
            # line ch
            ax[i].plot([1, 2], [1.35, 1.35], color='black', linewidth=1)
            ax[i].plot([1, 1], [1.32, 1.35], color='black', linewidth=1)
            ax[i].plot([2, 2], [1.32, 1.35], color='black', linewidth=1)

            ax[i].text(0.5, 1.09, p_wc, ha='center', va='center', fontsize=8)
            ax[i].text(1, 1.24,p_wh, ha='center', va='center', fontsize=8)
            ax[i].text(1.5, 1.39, p_ch, ha='center', va='center', fontsize=8)
        plt.tight_layout()
        plt.savefig('../manuscript/fig/stacked_bar.jpg', dpi=1000)

    def error_analysis(self, data):
        data['完整性1'].value_counts()

if __name__ == '__main__':

    ss = Statistical()

    #%% tableone
    table_one_file = r'F:\work\research\GPT\data\reports_one_year\csv\sample_reports.csv'
    ss._get_table_one(table_one_file)

    #%% model comparison
    file_paths = ['../data/evaluated/task_陈秀珍_20240702200659_evaluated.xlsx', 
                '../data/evaluated/task_陈耀萍_20240702200659_evaluated.xlsx',
                '../data/evaluated/task_董梦实_20240702200659_evaluated.xlsx',
                '../data/evaluated/task_段亚妮_20240702200659_evaluated.xlsx',
                '../data/evaluated/task_黎超_20240702200659_evaluated.xlsx',               
    ]
    match_id_path = '../data/match_id_300.json'
    icc_id_path = '../data/icc_id.json'
    data, match_id, icc_id = ss.load_data(file_paths, match_id_path, icc_id_path)
    data_preprocessed_all, icc_data = ss.preprocess_data(data, match_id, icc_id)

    # 对比模型以及人类之间生成结论的性能时，由于有50个用于检验观测者之间一致性的病例是5个评估者都评估过，我们取被相对高年资的那位医生评估的结果，整合到250个病例中，用于对比性能。
    data_preprocessed_all_plus_chenxiuzhen = pd.concat([data_preprocessed_all, icc_data['../data/evaluated/task_陈秀珍_20240702200659_evaluated.xlsx']], axis=0)

    # icc
    icc_list = list(icc_data.values())
    num_col = ['完整性1', '幻觉性1', '准确性1', '表达力1', '修改度1',
                '完整性2', '幻觉性2', '准确性2', '表达力2', '修改度2',
                '完整性3', '幻觉性3', '准确性3', '表达力3', '修改度3'
    ]
    icc_list = [icc[num_col] for icc in icc_list]
    cohen_kappa_results = ss.my_fleiss_kappa(icc_list)
    print("加权Kappa系数:", cohen_kappa_results)

    # friedman_test
    friedman_test_result, nemenyi_test_result = ss._friedman_nemenyi_test(data_preprocessed_all_plus_chenxiuzhen)

    for key in nemenyi_test_result:
        print(f'{key}friedman_test_result:')
        print(friedman_test_result[key])
        print(f'{key}neimenyi_test_result:')
        print(nemenyi_test_result[key])

    # vis
    ss._visulization_performances(data_preprocessed_all_plus_chenxiuzhen, friedman_test_result, nemenyi_test_result)

    # Error analysis
    ss.error_analysis(data_preprocessed_all_plus_chenxiuzhen)

    



