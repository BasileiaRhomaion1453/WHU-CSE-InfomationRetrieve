import re
import time
import os
import pickle
from collections import Counter, defaultdict
from itertools import chain
import numpy as np
import jieba
from ebooklib import epub
from bs4 import BeautifulSoup
import ebooklib

# 设置 Jieba 词典
jieba.set_dictionary('./dict.txt.big')

# 定义中文搜索引擎类
class ChnSearchEngine:
    def __init__(self, directory, index_file='index.pkl', allow_single_char=False, length_norm=False):
        self.docs = []  # 存储文档内容
        self.directory = directory  # 文档目录
        self.index_file = index_file  # 索引文件
        self.vocab_to_num = None  # 词汇表映射到编号
        self.length_norm = length_norm  # 是否进行长度归一化
        self.allow_single_char = allow_single_char  # 是否允许单字符词

        # 如果索引文件存在，则加载索引，否则构建索引
        if os.path.exists(self.index_file):
            self.load_index()
        else:
            self.build_index()

    # 加载数据
    def load_data(self):
        epub_files = self.search_epub_files(self.directory)
        lines = []
        for epub_file in epub_files:
            content = self.read_epub(epub_file)
            for i, chapter in content:
                lines.append((epub_file, i, chapter))
        self.docs = lines

    # 读取 EPUB 文件内容
    def read_epub(self, epub_file):
        book = epub.read_epub(epub_file)
        content = []
        for i, item in enumerate(book.get_items()):
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_body_content(), 'html.parser')
                text = soup.get_text()
                content.append((i + 1, text))
        return content

    # 搜索 EPUB 文件
    def search_epub_files(self, directory):
        epub_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".epub"):
                    epub_files.append(os.path.join(root, file))
        return epub_files

    # 解析文本
    def parse(self, text: str):
        words = list(jieba.cut(text))
        word_list = []
        for word in words:
            if not self.allow_single_char and len(word) == 1:
                continue
            word_list.append(word)
        return word_list

    # 解析并构建倒排索引
    def parse_and_build(self):
        words_list = []
        inverted_idx = defaultdict(set)
        for i, (file_name, chapter_num, doc) in enumerate(self.docs):
            words = self.parse(doc)
            words_list.append(words)
            for word in words:
                inverted_idx[word].add(i)
        return words_list, inverted_idx

    # 计算 TF（词频）
    def __cal_tf(self) -> np.ndarray:
        tf = np.zeros((len(self.vocab), len(self.docs)))
        for i, words in enumerate(self.docs_words):
            counter = Counter(words)
            for word in counter.keys():
                tf[self.vocab_to_num[word], i] = counter[word] / len(words)
        return tf

    # 计算 IDF（逆文档频率）
    def __cal_idf(self) -> np.ndarray:
        freq = np.zeros((len(self.num_to_vocab), 1))
        for i in range(len(self.num_to_vocab)):
            count = 0
            for doc_words in self.docs_words:
                if self.num_to_vocab[i] in doc_words:
                    count += 1
            freq[i, 0] = count
        idf = np.log(len(self.docs) / (freq + 1))
        return idf

    # 计算 TF-IDF
    def cal_tf_idf(self) -> np.ndarray:
        return self.__cal_tf() * self.__cal_idf()

    # 计算余弦相似度
    def cos_sim(self, docs_tf_idf, text_tf_idf):
        a = text_tf_idf / np.sqrt(np.sum(np.square(text_tf_idf), axis=0, keepdims=True))
        b = docs_tf_idf / np.sqrt(np.sum(np.square(docs_tf_idf), axis=0, keepdims=True))
        return b.T.dot(a).ravel()

    # 获取文本的得分
    def get_score(self, text):
        text_words = self.parse(text)
        unk_word_num = 0
        target_doc_idxes = set()
        target_word_idxes = set()
        for word in set(text_words):
            target_doc_idxes = target_doc_idxes | self.inverted_idx[word]
            if word not in self.vocab_to_num:
                self.vocab_to_num[word] = len(self.vocab_to_num)
                self.num_to_vocab[len(self.vocab_to_num) - 1] = word
                unk_word_num += 1
            target_word_idxes.add(self.vocab_to_num[word])
        if len(target_doc_idxes) == 0:
            return None
        xs, ys = list(target_word_idxes), list(target_doc_idxes)
        idf, tf_idf = self.idf, self.tf_idf
        if unk_word_num > 0:
            idf = np.concatenate((idf, np.zeros((unk_word_num, 1), dtype=np.float64)), axis=0)
            tf_idf = np.concatenate((tf_idf, np.zeros((unk_word_num, self.tf_idf.shape[1]), dtype=np.float64)), axis=0)
        counter = Counter(text_words)
        tf = np.zeros((len(idf), 1))
        for word in counter.keys():
            tf[self.vocab_to_num[word], 0] = counter[word]
        text_tf = tf[xs, :]
        text_idf = idf[xs, :]
        doc_tf_idf = tf_idf[:, ys][xs, :]
        text_tf_idf = text_tf * text_idf
        scores = self.cos_sim(doc_tf_idf, text_tf_idf)
        if self.length_norm:
            doc_lens = np.array([len(doc) for doc in self.docs_words])[ys]
            scores = scores / doc_lens
        return list(zip(ys, scores))

    # 获取得分最高的前 N 个文档
    def get_top_n(self, text, n=5):
        scores = self.get_score(text)
        if scores is None:
            print('Oops! No matches.')
            return time.perf_counter(), [], []
        filenum = [x[0] for x in scores]
        n = np.minimum(n, len(scores))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[:n]
        query_time = time.perf_counter()
        print('Top {} docs for "{}": \n'.format(n, text))
        top_docs = []
        for i, _ in scores:
            file_name, chapter_num, chapter = self.docs[i]
            chapter_name = self.extract_chapter_name(chapter)
            context = self.findtext(chapter, text)
            print(f"文件名: {file_name[10:]}")
            print(f"章节名: {chapter_name}")
            print(f"上下文: \n{context}")
            print("-----------------------------")
            top_docs.append((f"{file_name[10:]}", f"{chapter_name}", f"{context}"))
        print(f"章节号: {filenum}")
        print(f"查询章节数量: {len(self.docs)}")
        return query_time, top_docs, filenum

    # 查找文本中的上下文
    def findtext(self, chapter, text):
        context_index = chapter.lower().find(text.lower())
        context_start = max(0, context_index - 50)
        context_end = min(len(chapter), context_index + 50 + len(text))
        return chapter[context_start:context_end]

    # 提取章节名称
    def extract_chapter_name(self, chapter_text):
        pattern = r'(第[一二三四五六七八九十]+章)\s+(.*?)\n'
        match = re.search(pattern, chapter_text)
        if match:
            chapter_name = match.group(1) + ' ' + match.group(2)
        else:
            chapter_name = "未知章节名"
        return chapter_name

    # 布尔查询
    def boolean_query(self, query):
        terms = query.split()
        result_set = None
        i = 0
        while i < len(terms):
            term = terms[i].upper()
            if term == "AND":
                i += 1
                if i < len(terms):
                    if result_set is None:
                        result_set = self.inverted_idx.get(terms[i], set())
                    else:
                        result_set = result_set.intersection(self.inverted_idx.get(terms[i], set()))
            elif term == "OR":
                i += 1
                if i < len(terms):
                    if result_set is None:
                        result_set = set()
                    result_set = result_set.union(self.inverted_idx.get(terms[i], set()))
            elif term == "NOT":
                i += 1
                if i < len(terms):
                    if result_set is None:
                        result_set = set(range(len(self.docs)))
                    result_set = result_set.difference(self.inverted_idx.get(terms[i], set()))
            else:
                if result_set is None:
                    result_set = self.inverted_idx.get(terms[i], set())
                else:
                    result_set = result_set.intersection(self.inverted_idx.get(terms[i], set()))
            i += 1
        if result_set is None:
            return [], []
        return [(self.docs[i][0], self.docs[i][1], self.docs[i][2]) for i in result_set], result_set

    # 运行查询
    def run_query(self, query, query_type='topn', top_n=5):
        start_time = time.perf_counter()
        if query_type == 'boolean':
            result, x = self.boolean_query(query)
            query_time = time.perf_counter() - start_time
            if len(result) != 0:
                print(f'Boolean query results for "{query}": ')
                for file_name, chapter_num, chapter in result:
                    print(f"文件名: {file_name[10:]}")
                    print(f"章节名: {self.extract_chapter_name(chapter)}")
                    print(f"上下文: \n{self.findtext(chapter, query)}")
                    print("-----------------------------")
                print(f"章节号: {x}")
                print('Query time: {} sec'.format(query_time) + '\n')
                return x
            else:
                print('Oops! No matches.')
                return None
        else:
            query_time, _, num = self.get_top_n(query, top_n)
            query_time = query_time - start_time
            print('Query time: {} sec'.format(query_time) + '\n')
            return num

    # 构建索引
    def build_index(self):
        self.load_data()
        start_time = time.perf_counter()
        self.docs_words, self.inverted_idx = self.parse_and_build()
        self.vocab = set(chain(*self.docs_words))
        self.vocab_to_num = {v: i for i, v in enumerate(self.vocab)}
        self.num_to_vocab = {i: v for v, i in self.vocab_to_num.items()}
        self.tf = self.__cal_tf()
        self.idf = self.__cal_idf()
        self.tf_idf = self.tf * self.idf
        self.index_time = time.perf_counter() - start_time
        self.save_index()
        print("索引创建完成。")
        print("创建的索引：")
        for term, doc_ids in self.inverted_idx.items():
            print(f"词: {term}, 文档/章节ID: {list(doc_ids)}")
        print(f"创建的章节/文档数量: {len(self.docs)}")
        print(f"索引创建时间: {self.index_time:.2f} 秒")

    # 保存索引
    def save_index(self):
        with open(self.index_file, 'wb') as f:
            pickle.dump((self.docs, self.docs_words, self.inverted_idx, self.vocab, self.vocab_to_num, self.num_to_vocab, self.tf, self.idf, self.tf_idf), f)
        print("索引保存完成。")

    # 加载索引
    def load_index(self):
        with open(self.index_file, 'rb') as f:
            self.docs, self.docs_words, self.inverted_idx, self.vocab, self.vocab_to_num, self.num_to_vocab, self.tf, self.idf, self.tf_idf = pickle.load(f)
        print("索引加载完成。")
        print(f"章节数量: {len(self.docs)}")


# 主函数，提供用户交互界面
if __name__ == '__main__':
    searcher = ChnSearchEngine('./dataset')
    while True:
        x = input('请选择查询方式：\n1. Top-k查询\n2. 布尔查询\n3.输入exit退出\n注意：请输入繁体字进行查询\n')
        if x == '1':
            query_type = 'top_n'
            target = input('Enter your query: ')
        elif x == '2':
            query_type = 'boolean'
            target = input('Enter your query: ')
        elif x == 'exit':
            break
        else:
            print('非法输入，请重新输入！\n')
            continue
        searcher.run_query(target, query_type, top_n=5)
