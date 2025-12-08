import pandas as pd
import re
import jsonlines
import argparse
import os

parser = argparse.ArgumentParser(description="Run multi-turn response generation with customizable file paths.")
parser.add_argument('--data_path', type=str, 
                    default='', 
                    help='Path to the data file (e.g., /path/to/train.parquet or .jsonl)')
parser.add_argument('--gen_file', type=str, 
                    default='', 
                    help='Path to the gen_file')
args = parser.parse_args()


def normalize(text):
    # 标准化文本：转大写，去除两端空格，仅保留字母数字
    text = text.upper().strip()
    text = re.sub(r'[^a-zA-Z0-9]', '', text)
    return text


gt_answer = dict()

# 根据文件后缀判断读取方式
if args.data_path.endswith('.jsonl'):
    # 处理新的 JSONL benchmark 格式
    print(f"正在加载 JSONL 格式的基准数据: {args.data_path}")
    with jsonlines.open(args.data_path) as reader:
        for row in reader:
            # 直接从 jsonl 中获取 question 和 answer
            question = row.get("question")
            gt = row.get("answer")
            
            # 确保字段存在
            if question and gt:
                gt_answer[question] = normalize(gt)
else:
    # 处理原有的 Parquet 格式
    print(f"正在加载 Parquet 格式的基准数据: {args.data_path}")
    data_df = pd.read_parquet(args.data_path)
    for i, row in data_df.iterrows():
        # 原有逻辑：从 extra_info 中提取
        if "extra_info" in row and row["extra_info"]:
            question = row["extra_info"].get("question")
            gt = row["extra_info"].get("selected_answer")
            if question and gt:
                gt_answer[question] = normalize(gt)


# 读取生成文件进行评估
with jsonlines.open(args.gen_file) as reader:   
    gen_data = list(reader)

steps = 0
suc = 0
emp = 0

for data in gen_data:
    content = data['trajectory']
    input_seq = content[-1]['input_seq']
    output_seq = content[-1]['output_seq']

    # 提取问题，这里假设 input_seq 格式包含 "Objective: ... \nObservation"
    # 如果新 benchmark 的 prompt 格式不同，这里可能需要相应调整 regex
    try:
        question = re.findall(r'Objective: (.*?)\nObservation', input_seq)[0]
    except IndexError:
        # 如果匹配不到问题，跳过该条目或记录错误
        continue

    # 提取生成的答案
    if not re.findall(r"```(.*?)```", output_seq):
        answer = " "
    else:
        answer = re.findall(r"```(.*?)```", output_seq)[0]
    
    # 检查是否包含 stop 以及答案匹配逻辑
    if 'stop' in answer:
        try:
            # 尝试提取 [] 中的内容
            ans = normalize(re.findall(r"\[(.*?)\]", answer)[0])
        except:
            ans = ""
        
        # 获取该问题的标准答案
        if question in gt_answer:
            ground_truth = gt_answer[question]
            
            # 判断正确性：包含关系 或 排序后的字符匹配
            if ground_truth in ans or ''.join(sorted(ground_truth)) == ''.join(sorted(ans)):
                suc += 1
                steps += data['trajectory_length']
        else:
            # 如果问题在标准答案中找不到（可能是数据对齐问题）
            pass 
    else:
        emp += 1


print(f"问题数目：{len(gen_data)}")
print(f"回答正确数目：{suc}")
# 避免除以零错误
if len(gen_data) > 0:
    print(f"正确率：{suc/len(gen_data)}")
else:
    print("正确率：0")
print(f"未回答数目：{emp}")
if suc > 0:
    print(f"平均步数：{steps/suc}")
else:
    print("平均步数：0")