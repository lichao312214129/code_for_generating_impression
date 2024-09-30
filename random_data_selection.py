"""
目的：纳入研究的样本数据
    1、提取检查部位列中所有包含"上腹"、"下腹"、"盆腔"的报告。
    2、检查印象的字数。
"""
import os
import glob
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt

root = r"F:\work\research\GPT\data\reports_one_year\csv"
# 包含xls但不包含~$的文件
files = glob.glob(os.path.join(root, "*.xls"))
files = [file for file in files if "~$" not in file]
def read_file_(file):
    print(f"正在处理文件：{file}")
    data = pd.read_excel(file)
    data["检查部位"] = data["检查部位"].astype(str)
    data = data[data["检查部位"].str.contains("上腹|下腹|盆腔")]
    return data
    
def read_files():
    # 多进程，保留结果
    with ProcessPoolExecutor(max_workers=5) as executor:  # 创建 ThreadPoolExecutor
        # 使用as_completed方法异步获取任务结果
            future_to_data = {executor.submit(read_file_, x): x for x in files}
            results = []
            for future in as_completed(future_to_data):
                # 收集每个任务的结果
                results.append(future.result())

    # 将结果转换为DataFrame
    df = pd.concat(results, ignore_index=True)
    return df

def del_report_less_than_n_words(file, n=100):
    df = pd.read_csv(file)
    findings = df["影像所见"].values
    # 删除空格
    findings = [x.replace(" ", "") for x in findings]
    # 替换
    df["影像所见"] = findings
    word_count = [len(x) for x in findings]
    df["word_count"] = word_count
    # distribution of word count
    plt.hist(word_count, bins=30)
    plt.xlabel("word count")
    df = df[df["word_count"] > n]
    df.to_csv("./data/reports_one_year/csv/all_reports.csv", index=False, encoding="utf-8-sig")

def random_choose_sample(file, n=None):
    df = pd.read_csv(file)
    df = df.sample(n)
    df.to_csv("./data/reports_one_year/csv/sample_reports.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    df = read_files()
    df.to_csv("./data/reports_one_year/csv/all_reports.csv", index=False, encoding="utf-8-sig")
    del_report_less_than_n_words("./data/reports_one_year/csv/all_reports.csv")
    random_choose_sample("./data/reports_one_year/csv/all_reports.csv", n=300)