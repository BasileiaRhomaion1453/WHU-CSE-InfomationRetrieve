import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from searchfinal import ChnSearchEngine 
from itertools import chain

def evaluate_metrics(true_labels, bool_query_results, topn_query_results):
    # 将结果转化为0和1的标签
    # true_labels是正确的标签，用1表示匹配，0表示不匹配
    x = [0] * 84
    for i in range(len(true_labels)):
        x[true_labels[i] - 1] = 1
    y_true = np.array(x)

    # 布尔查询结果
    y = [0] * 84
    for i in range(len(bool_query_results)):
        y[bool_query_results[i] - 1] = 1
    y_bool_pred = np.array(y)

    # 计算布尔查询的评价指标
    bool_metrics = {
        'accuracy': accuracy_score(y_true, y_bool_pred),
        'precision': precision_score(y_true, y_bool_pred),
        'recall': recall_score(y_true, y_bool_pred),
        'f1_score': f1_score(y_true, y_bool_pred)
    }

    # Top-N查询结果
    y = [0] * 84
    if len(topn_query_results) >= 1:
        for i in range(len(topn_query_results)):
            y[topn_query_results[i] - 1] = 1
        y_topn_pred = np.array(y)
    else:
        y_topn_pred = np.array([0] * 84)

    # 计算Top-N查询的评价指标
    topn_metrics = {
        'accuracy': accuracy_score(y_true, y_topn_pred),
        'precision': precision_score(y_true, y_topn_pred),
        'recall': recall_score(y_true, y_topn_pred),
        'f1_score': f1_score(y_true, y_topn_pred)
    }

    return {
        'bool_query': bool_metrics,
        'topn_query': topn_metrics
    }

def run_evaluation(engine, queries, true_labels):
    all_bool_results = []
    all_topn_results = []

    for query in queries:
        # 执行布尔查询
        bool_results = engine.run_query(query, query_type='boolean')
        # 执行Top-N查询
        topn_results = engine.run_query(query, query_type='topn', top_n=5)

        if bool_results is not None:
            all_bool_results.append(bool_results)
        if topn_results is not None:
            all_topn_results.append(topn_results)

    # 合并所有查询结果
    bool_query_results = list(set(chain.from_iterable(all_bool_results)))
    topn_query_results = list(set(chain.from_iterable(all_topn_results)))

    # 计算评价指标
    metrics = evaluate_metrics(true_labels, bool_query_results, topn_query_results)

    # 输出布尔查询的评价指标
    print("Boolean Query Metrics:")
    print("Accuracy:", metrics['bool_query']['accuracy'])
    print("Precision:", metrics['bool_query']['precision'])
    print("Recall:", metrics['bool_query']['recall'])
    print("F1 Score:", metrics['bool_query']['f1_score'])

    # 输出Top-N查询的评价指标
    print("\nTop-N Query Metrics:")
    print("Accuracy:", metrics['topn_query']['accuracy'])
    print("Precision:", metrics['topn_query']['precision'])
    print("Recall:", metrics['topn_query']['recall'])
    print("F1 Score:", metrics['topn_query']['f1_score'])

if __name__ == "__main__":
    # 定义正确的标签
    true_labels = [56]
    # 定义查询
    queries = ["中二"] 
    # 创建搜索引擎实例
    engine = ChnSearchEngine(directory='./dataset')
    # 运行评价
    run_evaluation(engine, queries, true_labels)

'''
魔女,文档ID:[
1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 
20, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 
37, 39, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 
55, 56, 57, 59, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 73, 
75, 76, 77, 78, 79, 80, 81, 82, 83, 84]
#旅行, 文档ID:[
1, 3, 4, 6, 10, 12, 13, 14, 15, 16, 18, 20, 21, 23, 25, 27, 
28, 30, 32, 33, 35, 36, 37, 41, 44, 45, 46, 48, 51, 53, 54, 
55, 56, 57, 59, 62, 64, 65, 66, 67, 68, 69, 70, 71, 73, 75, 
76, 78, 79, 80, 82, 83, 84]
#伊蕾娜, 文档ID: [
1, 2, 3, 8, 9, 11, 14, 15, 16, 17, 18, 20, 23, 25, 28, 30, 
31, 33, 34, 36, 38, 39, 41, 45, 46, 47, 48, 49, 51, 53, 54, 
55, 56, 57, 59, 61, 62, 63, 64, 66, 67, 68, 69, 70, 71, 73, 
75, 76, 77, 78, 80, 82, 83, 84]
'''
