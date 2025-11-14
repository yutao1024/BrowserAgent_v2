import pandas as pd
import re
import jsonlines
import json
import argparse

parser = argparse.ArgumentParser(description="Run multi-turn response generation with customizable file paths.")
parser.add_argument('--data_path', type=str, 
                    default='', 
                    help='Path to the data file (e.g., /path/to/train.parquet)')
parser.add_argument('--gen_file', type=str, 
                    default='', 
                    help='Path to the gen_file')
parser.add_argument('--output_file', type=str, 
                    default='./test.jsonl', 
                    help='Output file path for writing the correct data (e.g., /path/to/output.jsonl)')
parser.add_argument('--output_file_wrong', type=str, 
                    default='./test_wrong.jsonl', 
                    help='Output file path for writing wrong answers with stop (e.g., /path/to/output_wrong.jsonl)')
parser.add_argument('--output_file_no_stop', type=str, 
                    default='./test_no_stop.jsonl', 
                    help='Output file path for writing data without stop (e.g., /path/to/output_no_stop.jsonl)')

args = parser.parse_args()



# 注意：eval_prompt 变量未使用，但保留读取操作以防后续需要
try:
    with open("sys_eval_prompt.txt","r",encoding="utf-8") as f:
        eval_prompt = f.read()
except FileNotFoundError:
    eval_prompt = ""

def normalize(text):
    text = text.upper().strip()
    text = re.sub(r'[^a-zA-Z0-9]', '', text)
    return text


data_df = pd.read_parquet(args.data_path)
gt_answer = dict()
for i, row in data_df.iterrows():
    question = row["extra_info"]["question"]
    gt = row["extra_info"]["selected_answer"]
    gt_answer[question] = normalize(gt)


with jsonlines.open(args.gen_file) as reader:
    gen_data = list(reader)

# 初始化三个结果列表
res_right = list()  # 包含 stop 且答案正确
res_wrong = list()  # 包含 stop 但答案错误
res_no_stop = list()  # 不包含 stop

# 用于跟踪不同问题（分组）的集合
questions_right = set()  # 正确答案的问题集合
questions_wrong = set()  # 错误答案的问题集合
questions_no_stop = set()  # 无stop的问题集合

# 用于追踪当前问题的步骤序列
pre_q = None
tmp = list()

for data in gen_data:
    input = data['input_seq']
    output = data['output_seq']
    
    # 提取问题
    question_match = re.findall(r'Objective: (.*?)\nObservation', input)
    if not question_match:
        # 如果无法提取问题，将该数据归类为 no_stop
        res_no_stop.append(data)
        continue
    question = question_match[0]
    
    # 检查问题是否切换
    if pre_q is not None and pre_q != question:
        # 问题切换时，如果 tmp 中还有未处理的步骤（没有 stop），添加到 no_stop
        if tmp:
            res_no_stop.extend(tmp)
            questions_no_stop.add(pre_q)  # 记录未完成的问题
        tmp = list()
    
    pre_q = question
    tmp.append(data)
    
    # 尝试提取答案
    answer_match = re.findall(r"```(.*?)```", output)
    if not answer_match:
        # 没有找到 ```...``` 格式，说明不包含 stop，继续累积步骤
        # 这些步骤会在问题切换或文件结束时处理
        continue
    
    answer = answer_match[0]
    
    # 检查是否包含 stop
    if "stop" in answer:
        # 提取答案内容
        answer_content_match = re.findall(r"\[(.*?)\]", answer)
        if answer_content_match:
            answer_content = normalize(answer_content_match[0])
        else:
            answer_content = normalize(answer)
        
        # 获取标准答案并判断是否正确
        if question in gt_answer:
            ground_truth = gt_answer[question]
            # 判断答案是否正确
            if ground_truth in answer_content or ''.join(sorted(ground_truth)) == ''.join(sorted(answer_content)):
                res_right.extend(tmp)
                questions_right.add(question)  # 记录正确答案的问题
            else:
                res_wrong.extend(tmp)
                questions_wrong.add(question)  # 记录错误答案的问题
        else:
            # 如果没有标准答案，归类为错误
            res_wrong.extend(tmp)
            questions_wrong.add(question)  # 记录错误答案的问题
        
        # 清空当前步骤序列
        tmp = list()
    # 如果找到 ```...``` 但不包含 stop，继续累积（会在问题切换或文件结束时处理）

# 处理最后一个问题的未完成步骤（如果有）
if tmp:
    res_no_stop.extend(tmp)
    if pre_q is not None:
        questions_no_stop.add(pre_q)  # 记录最后一个未完成的问题

# 统计信息
total_input = len(gen_data)
total_output = len(res_right) + len(res_wrong) + len(res_no_stop)


print(f"\n不同问题（分组）总数：")
print(f"  正确问题数: {len(questions_right)} 个")
print(f"  错误问题数: {len(questions_wrong)} 个")
print(f"  无stop问题数: {len(questions_no_stop)} 个")


# 写入三个输出文件
with open(args.output_file, 'w', encoding='utf-8') as f:
    for item in res_right:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

with open(args.output_file_wrong, 'w', encoding='utf-8') as f:
    for item in res_wrong:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

with open(args.output_file_no_stop, 'w', encoding='utf-8') as f:
    for item in res_no_stop:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")