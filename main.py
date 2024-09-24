import pandas as pd
import numpy as np
import jieba
import time
from tqdm import tqdm
from datetime import timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import os,difflib,math


def load_region_dict(dict_path):
    """加载自定义区域字典并生成区域集合"""
    jieba.load_userdict(dict_path)  # 加载自定义区域字典
    region_set = set()
    with open(dict_path, 'r', encoding='utf-8-sig') as file:
        for line in file:
            line = (line + ' ').split(' ', 1)[0].strip()
            if line:
                region_set.add(line)  # 将区域添加到集合中
    return region_set

def preprocess_text(text):
    """对文本进行分词"""
    tokens = jieba.lcut(text)  
    return tokens

def normalize_text(text):
    """规范化文本格式"""
    return text.replace('（', '(').replace('）', ')').replace(' ', '')  

def weighted_jaccard(tokens,steepness: float = 0.0):
    """计算tokens权数"""
    n = len(tokens)
    
    #steepness 参数非0时可以控制衰减的速率。值越大，前期衰减越快，后期减缓越明显；值越小，则衰减较为平缓。
    #结合了反比例和指数衰减，使得权重在初期快速减少，随后趋于平稳。
    #steepness = 0.0 时，不会有指数衰减的效果。所有的权重变化完全由反比例决定，因此前面的词对相似度的贡献显著高于后面的词
    steepness =0.0
    weights = {token: 1.0 / (i + 1) * math.exp(-steepness * i / n) for i, token in enumerate(tokens)}  # 计算文本中每个词的权重

    return weights # 返回tokens的权重字典


def weighted_jaccard_similarity(tokens1, tokens2,weights1=None,weights2=None,steepness: float = 0.0):
    """计算加权 Jaccard 相似度"""
    if weights1 is None:
        weights1 = weighted_jaccard(tokens1,steepness)
    if weights2 is None:
        weights2 = weighted_jaccard(tokens2,steepness)

    intersection = set(tokens1) & set(tokens2)  # 计算两个文本的交集
    weighted_intersection = sum(min(weights1.get(token, 0), weights2.get(token, 0)) for token in intersection)  # 计算加权交集

    union = set(tokens1) | set(tokens2)  # 计算两个文本的并集
    weighted_union = sum(weights1.get(token, 0) + weights2.get(token, 0) for token in union)  # 计算加权并集

    return weighted_intersection / weighted_union if weighted_union != 0 else 0  # 返回加权 Jaccard 相似度


def weighted_SequenceMatcher_similarity(tokens1, tokens2, weights=None):
    """计算SequenceMatcher相似度"""
    # 将 tokens 转换为字符串
    s1 = ''.join(tokens1)
    s2 = ''.join(tokens2)

    if weights is None:
        #weights = {'insert': 1, 'delete': 1, 'substitute': 1}
        #三个权重都为1，等同于直接使用:
        similarity=difflib.SequenceMatcher(None, s1, s2).ratio()
    else:
        matcher = difflib.SequenceMatcher(None, s1, s2)
        total_cost = 0
        matches = 0

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                total_cost += weights['substitute'] * (i2 - i1)
            elif tag == 'delete':
                total_cost += weights['delete'] * (i2 - i1)
            elif tag == 'insert':
                total_cost += weights['insert'] * (j2 - j1)
            else:  # 'equal'
                matches += (i2 - i1)

        # 计算最大长度
        max_len = max(len(s1), len(s2))
        similarity = 1 - (total_cost / max_len) if max_len > 0 else 1.0

    return similarity

def weighted_levenshtein_similarity(tokens1, tokens2):
    """计算莱温斯坦相似度"""
    # 此算法不使用分词后的字符串 
    # 传入company1和company2两个字符串，即不考虑地域信息
    s1=tokens1
    s2=tokens2  

    # 计算莱温斯坦距离
    if len(s1) < len(s2):
        s1, s2 = s2, s1

    if len(s2) == 0:
        return 1.0 if len(s1) == 0 else 0.0  # 如果 s2 为空，且 s1 也为空则相似度为 1

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1  # 插入操作的成本
            deletions = current_row[j] + 1  # 删除操作的成本
            substitutions = previous_row[j] + (c1 != c2)  # 替换操作的成本
            current_row.append(min(insertions, deletions, substitutions))  # 选择成本最低的操作
        previous_row = current_row

    distance = previous_row[-1]  # 莱温斯坦距离
    max_len = max(len(s1), len(s2))  # 两个字符串的最大长度
    similarity = 1 - (distance / max_len) if max_len > 0 else 1.0  # 确保 max_len 不为 0

    return similarity


def standardize_area(area, region_set):
    """标准化区域字段"""
    return area.strip()  

def split_leftmost_regions(company_name, region_list):
    """拆解公司名称中的地域信息"""
    words = jieba.lcut(company_name)  # 对公司名称进行分词
    split_regions = []

    for word in words:
        if word in region_list:  # 如果当前词是地域词，则添加到 split_regions 列表中
            split_regions = [word] + split_regions
        else:
            break
    return ''.join(words[len(split_regions):]), ''.join(split_regions) ,words[len(split_regions):] + split_regions # 返回公司名称和地域信息,以及公司名称token列表


def preprocess_company_area(company_name, region_set, similarity_type):
    """公司名称和地区的处理"""
    company, area, tokens = split_leftmost_regions(company_name, list(region_set))  # 拆解公司名称中的地域信息
    company = normalize_text(company)  # 规范化公司名称
    area = standardize_area(area, region_set)  # 标准化区域字段
    #重新计算tokens，结果有许些差异
    tokens = preprocess_text(f'{company} {area or ""}')  # 对公司名称和区域字段进行分词
    weigths = weighted_jaccard(tokens, steepness=0.0) if similarity_type[1] == '1' else None # 对公司名称分词后的 tokens 进行加权    

    return company, area, tokens, weigths  # 返回公司名称、地区、公司名称分词后的 tokens、weights

def load_and_preprocess_data(input_file_path,dict_path,similarity_type):
    """加载并预处理数据"""
    region_set = load_region_dict(dict_path)  # 加载自定义区域字典
    df1 = pd.read_excel(input_file_path, sheet_name='Sheet1')#.head(100)  # 读取第一个工作表的数据
    df2 = pd.read_excel(input_file_path, sheet_name='Sheet2')#.head(100)  # 读取第二个工作表的数据

    # 检查是否存在 'id' 列
    if 'id' not in df1.columns or 'id' not in df2.columns:
        raise ValueError("输入数据中缺少 'id' 列")

    df1[['company', 'area', 'tokens', 'weights']] = df1['company_name'].apply(lambda x: pd.Series(preprocess_company_area(x, region_set,similarity_type)))  # 处理第一个工作表中的公司名称和地区,并对公司名称进行分词
    df2[['company', 'area', 'tokens', 'weights']] = df2['company_name'].apply(lambda x: pd.Series(preprocess_company_area(x, region_set,similarity_type)))  # 处理第二个工作表中的公司名称和地区,并对公司名称进行分词
 
    return df1, df2

def select_better_similarity_1(sim1, threshold1, sim2, threshold2):
    """选择更好的相似度"""
    # 计算偏差
    deviation1 = abs(sim1 - threshold1)  # 计算第一个相似度与阈值的偏差
    deviation2 = abs(sim2 - threshold2)  # 计算第二个相似度与阈值的偏差

    # 计算均值和标准差
    similarities = [sim1, sim2]
    deviations = [deviation1, deviation2]

    mean_similarity = np.mean(similarities)  # 计算相似度的均值
    std_dev_similarity = np.std(similarities)  # 计算相似度的标准差
    mean_deviation = np.mean(deviations)  # 计算偏差的均值
    std_dev_deviation = np.std(deviations)  # 计算偏差的标准差

    # 计算变异系数
    cv_similarity = std_dev_similarity / mean_similarity if mean_similarity > 0 else float('inf')  # 计算相似度的变异系数
    cv_deviation = std_dev_deviation / mean_deviation if mean_deviation > 0 else float('inf')  # 计算偏差的变异系数

    # 打印调试信息
    #print("相似度1:", sim1, "偏差:", deviation1)
    #print("相似度2:", sim2, "偏差:", deviation2)

    # 处理小于阈值的情况
    valid_sim1 = sim1 >= threshold1  # 判断第一个相似度是否大于等于阈值
    valid_sim2 = sim2 >= threshold2  # 判断第二个相似度是否大于等于阈值

    if valid_sim1 and not valid_sim2:
        return sim1, "相似度2小于阈值，选择相似度1", 1  # 如果第一个相似度大于等于阈值，第二个相似度小于阈值，则选择第一个相似度
    elif valid_sim2 and not valid_sim1:
        return sim2, "相似度1小于阈值，选择相似度2", 2  # 如果第二个相似度大于等于阈值，第一个相似度小于阈值，则选择第二个相似度
    elif not valid_sim1 and not valid_sim2:
        return None, "两者都小于阈值", None  # 如果两个相似度都小于阈值，则返回 None

    # 判断哪个相似度更好
    if sim1 > sim2:
        better_similarity = sim1
        better_reason = "更高的相似度"  # 如果第一个相似度大于第二个相似度，则选择第一个相似度
        selected_group = 1
    elif sim2 > sim1:
        better_similarity = sim2
        better_reason = "更高的相似度"  # 如果第二个相似度大于第一个相似度，则选择第二个相似度
        selected_group = 2
    else:
        # 当相似度相同时，比较偏差
        if deviation1 < deviation2:
            better_similarity = sim1
            better_reason = "较小的偏差"  # 如果第一个相似度的偏差小于第二个相似度的偏差，则选择第一个相似度
            selected_group = 1
        elif deviation2 < deviation1:
            better_similarity = sim2
            better_reason = "较小的偏差"  # 如果第二个相似度的偏差小于第一个相似度的偏差，则选择第二个相似度
        else:
            # 偏差也相等，选择标准差
            if std_dev_similarity < std_dev_deviation:
                better_similarity = sim1
                better_reason = "较小的标准差"  # 如果第一个相似度的标准差小于第二个相似度的标准差，则选择第一个相似度
                selected_group = 1
            else:
                better_similarity = sim2
                better_reason = "较小的标准差"  # 如果第二个相似度的标准差小于第一个相似度的标准差，则选择第二个相似度
                selected_group = 2

    # 在最后阶段比较变异系数
    if cv_similarity < cv_deviation:
        better_similarity = sim1 if better_similarity == sim1 else sim2
        better_reason = "较小的变异系数"  # 如果第一个相似度的变异系数小于第二个相似度的变异系数，则选择第一个相似度
        selected_group = 1 if better_similarity == sim1 else 2

    return better_similarity, better_reason, selected_group

def select_better_similarity(sim1, threshold1, sim2, threshold2):
    """选择更好的相似度

    Args:
        sim1 (float): 第一个相似度值
        threshold1 (float): 第一个相似度的阈值
        sim2 (float): 第二个相似度值
        threshold2 (float): 第二个相似度的阈值

    Returns:
        tuple: 包含选择的相似度值、选择原因和组别标识
    """
    valid_sim1 = sim1 >= threshold1
    valid_sim2 = sim2 >= threshold2

    # 处理有效相似度
    if valid_sim1 and not valid_sim2:
        return sim1, "相似度2小于阈值，选择相似度1", 1
    elif valid_sim2 and not valid_sim1:
        return sim2, "相似度1小于阈值，选择相似度2", 2
    elif not valid_sim1 and not valid_sim2:
        return None, "两者都小于阈值", None

    # 计算相似度与阈值的比例
    ratio1 = sim1 / threshold1 if threshold1 > 0 else float('inf')
    ratio2 = sim2 / threshold2 if threshold2 > 0 else float('inf')

    # 比较比例
    if ratio1 > ratio2:
        return sim1, "相似度1相对阈值表现更好", 1
    elif ratio2 > ratio1:
        return sim2, "相似度2相对阈值表现更好", 2
    else:
        # 当比例相同时，选择相似度更高的
        if sim1 > sim2:
            return sim1, "相似度1更高", 1
        else:
            return sim2, "相似度2更高", 2


def find_best_match(row1, remaining_rows, jaccard_threshold, SequenceMatcher_threshold, similarity_type):
    """为给定的公司名称寻找最佳匹配"""
    # id1, company_name1, company1, area1 = row1[['id', 'company_name', 'company', 'area']]  
    # 获取第一个公司信息
    id1 = row1.id           
    company_name1 = row1.company_name
    company1 = row1.company
    area1 = row1.area
    tokens1 = row1.tokens
    weights1 = row1.weights

    jaccard_best_score = 0  # 初始化加权 Jaccard 相似度的最佳分数
    jaccard_best_match = None  # 初始化加权 Jaccard 相似度的最佳匹配
    SequenceMatcher_best_score = 0  # 初始化加权 SequenceMatcher 相似度的最佳分数
    SequenceMatcher_best_match = None  # 初始化加权 SequenceMatcher 相似度的最佳匹配
    none_match = {  # 初始化未匹配的结果
        'ID': None,
        'similarity_type': None,
        'Company_Name': None,
        'Company': None,
        'Area': None
    }

    for row2 in remaining_rows:  # 遍历剩余的公司
        id2 = row2['id']
        company_name2 = row2['company_name']
        company2 = row2['company']
        area2 = row2['area']
        tokens2 = row2['tokens']
        weights2 = row2['weights']
        
        if similarity_type[1] == '1':  # 判断是否需要计算加权 Jaccard 相似度
            group = 1 
            jaccard_similarity = weighted_jaccard_similarity(tokens1,tokens2,weights1,weights2)  # 计算加权 Jaccard 相似度
            if jaccard_similarity > jaccard_best_score:  # 更新加权 Jaccard 相似度的最佳分数和最佳匹配
                jaccard_best_score = jaccard_similarity
                jaccard_best_match = {
                    'ID': id2,
                    'similarity_type': '01',
                    'Company_Name': company_name2,
                    'Company': company2,
                    'Area': area2
                }
            
        if similarity_type[0] == '1':  # 判断是否需要计算加权 SequenceMatcher 相似度
            group = 2
            SequenceMatcher_similarity = weighted_SequenceMatcher_similarity(tokens1,tokens2)  # 计算加权 SequenceMatcher 相似度
            if SequenceMatcher_similarity > SequenceMatcher_best_score:  # 更新加权 SequenceMatcher 相似度的最佳分数和最佳匹配
                SequenceMatcher_best_score = SequenceMatcher_similarity
                SequenceMatcher_best_match = {
                    'ID': id2,
                    'similarity_type': '10',
                    'Company_Name': company_name2,
                    'Company': company2,
                    'Area': area2
                }
            
        if similarity_type == '11':  # 如果both需要选择更好的相似度
            result_similarity, reason, group = select_better_similarity(
                jaccard_best_score, jaccard_threshold, SequenceMatcher_best_score, SequenceMatcher_threshold  # 选择更好的相似度
            )
        
    if group == 1 and jaccard_best_score >= jaccard_threshold:  # 如果选择了加权 Jaccard 相似度，并且分数大于等于阈值
        best_score = jaccard_best_score
        best_match = jaccard_best_match
    elif group == 2 and SequenceMatcher_best_score >= SequenceMatcher_threshold:  # 如果选择了加权 SequenceMatcher 相似度，并且分数大于等于阈值
        best_score = SequenceMatcher_best_score
        best_match = SequenceMatcher_best_match
    else:  # 否则，没有找到匹配
        best_score = None
        best_match = none_match

    return [  # 返回匹配结果
        id1, company_name1, best_score, best_match['similarity_type'], 
        company1, area1, best_match['ID'], best_match['Company_Name'], 
        best_match['Company'], best_match['Area']
    ]



def find_best_match_batch(batch, remaining_rows, jaccard_threshold, SequenceMatcher_threshold, similarity_type):
    # 处理整个批次
    matches = []
    for row1 in batch.itertuples(index=False):
        match = find_best_match(row1, remaining_rows, jaccard_threshold, SequenceMatcher_threshold, similarity_type)
        matches.append(match)
    return matches

def find_matches(df1, df2, jaccard_threshold, SequenceMatcher_threshold, similarity_type, exact_match_score, num_workers=1, batch_size=10):
    """查找匹配的公司"""
    matches = []  # 初始化匹配结果列表
    
    merged_df = pd.merge(df1, df2, on=['company', 'area'], suffixes=('_1', '_2'), how='outer', indicator=True)  # 合并两个工作表的数据，并标记匹配类型

    # 完全匹配的记录
    for _, row in tqdm(merged_df.iterrows(), total=merged_df.shape[0], desc="Matching companies"):
        if row['_merge'] == 'both':  # 如果是完全匹配
            matches.append([  # 将匹配结果添加到列表中
                row['id_1'], row['company_name_1'], exact_match_score, 0,
                row['company'], row['area'], 
                row['id_2'], row['company_name_2'], row['company'], row['area']
            ])

    remaining_df2 = df2[~df2['id'].isin(merged_df.loc[merged_df['_merge'] == 'both', 'id_2'])]  # 获取第二个工作表中未完全匹配的记录
    matched_ids = merged_df[merged_df['_merge'] == 'both']['id_1'].unique()  # 获取第一个工作表中完全匹配的记录的 id
    df1_remaining = df1[~df1['id'].isin(matched_ids)]  # 获取第一个工作表中未完全匹配的记录 
    
    # 多进程处理
    # 准备剩余行的列表,转化为列表
    remaining_rows = remaining_df2.to_dict(orient='records')
    # 计算剩余记录数
    total_records = len(df1_remaining)  
    # 接收任务返回结果
    def receive_future(futures, matches, pbar):
        # 处理已完成的任务
        for future in as_completed(futures):
            batch_matches = future.result()  # 获取结果
            matches.extend(batch_matches)  # 合并所有批次的匹配结果
            pbar.update(len(batch_matches))  # 更新进度条
            futures.remove(future)  # 从 futures 列表中移除已完成的任务

    # 创建进程池并处理批次
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []  # 存储分配的任务
        max_tasks = num_workers * 2  # 设置最大任务数为进程数的两倍
        # 进度条
        with tqdm(total=total_records, desc="Finding best matches") as pbar:
            for i in range(0, len(df1_remaining), batch_size):
                batch = df1_remaining.iloc[i:i + batch_size]  # 提取当前批次
                future = executor.submit(find_best_match_batch, batch, remaining_rows, jaccard_threshold, SequenceMatcher_threshold, similarity_type)
                futures.append(future)  # 将每个分配任务添加到 futures 列表

                # 处理已完成的任务
                while len(futures) >= max_tasks:  # 如果 futures 已达到可处理数量
                    receive_future(futures, matches, pbar)  # 接收并处理已完成的任务

            # 确保所有任务都已完成
            while futures:  # 如果仍然有未处理的任务
                receive_future(futures, matches, pbar)  # 处理剩余的任务

            

    matches_df = pd.DataFrame(matches, columns=[  # 将匹配结果列表转换为 DataFrame
        'id1', 'Company_Name1', 'similarity', 'similarity_type',
        'Company1', 'Area1', 'BestMatchID', 'BestMatchCompany_Name', 
        'BestMatchCompany', 'BestMatchArea'
    ])
    
    return preprocess_matches(matches_df)  # 处理匹配结果，并返回最终结果

def preprocess_matches(matches_df):
    """处理匹配结果，保留最大相似度的记录"""
    max_similarity_indices = matches_df.groupby('BestMatchCompany_Name')['similarity'].idxmax()  # 找到每个公司名称对应的最大相似度记录的索引
    final_matches = matches_df.copy()  # 复制匹配结果
    
    # 清空所有 BestMatch 字段
    final_matches[['similarity', 'BestMatchID', 'BestMatchCompany_Name', 'BestMatchCompany', 'BestMatchArea']] = None  # 清空所有 BestMatch 字段
    final_matches.loc[max_similarity_indices, ['similarity', 'BestMatchID', 'BestMatchCompany_Name', 'BestMatchCompany', 'BestMatchArea']] = \
        matches_df.loc[max_similarity_indices, ['similarity', 'BestMatchID', 'BestMatchCompany_Name', 'BestMatchCompany', 'BestMatchArea']]  # 将最大相似度记录的值复制到最终结果中
    
    final_matches.sort_values(by='id1', inplace=True)  # 按照 id1 排序
    return final_matches  # 返回最终结果

def save_results(matches_df, output_file):
    """保存匹配结果"""
    with pd.ExcelWriter(output_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        matches_df.to_excel(writer, sheet_name='SimilarityResults', index=False)  # 将匹配结果保存到 Excel 文件的 SimilarityResults 工作表中
    print(f"加权相似度结果已保存到 {output_file} 的 'SimilarityResults' 工作表")

if __name__ == '__main__':
    file_path = r''  # 设置文件路径
    input_file = file_path + 'company.xlsx' # 数据文件
    output_file = input_file 
    user_dict_path = file_path + 'region_dict.txt' # 自定义区域字典
    jaccard_threshold = 0.391  #jaccard相似度阈值
    SequenceMatcher_threshold = 0.9 #SequenceMatcher相似度阈值
    #levenshtein_threshold = 0.71  #levenshtein相似度阈值
    similarity_type = '01'  #01表示加权 Jaccard 相似度，10表示加权 SequenceMatcher 相似度,11表示两者均衡
    exact_match_score = 100 # 完全匹配的分数
    num_cores = os.cpu_count()  # 获取 CPU 核心数
    max_workers = num_cores * 1 #2  # 根据需要调整进程数
    batch_size = 10  # 每次处理的记录数 # 将 df1_remaining 分批处理

    start_time = time.time()  # 记录开始时间
    df1, df2 = load_and_preprocess_data(input_file, user_dict_path,similarity_type)  # 加载并预处理数据
    matches_df = find_matches(df1, df2, jaccard_threshold, SequenceMatcher_threshold, similarity_type, exact_match_score, num_workers=max_workers, batch_size=batch_size)  # 查找匹配的公司
    save_results(matches_df, output_file)  # 保存匹配结果

    # 计算并显示总耗时
    total_time = time.time() - start_time
    formatted_time = str(timedelta(seconds=int(total_time)))
    print(f"总耗时: {formatted_time}")
