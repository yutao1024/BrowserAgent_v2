import json
import re
import os

def split_test_dataset(dataset_path, test_path):
    """
    根据 dataset.jsonl 中的 difficulty_level 对 test.jsonl 进行分类拆分。
    """
    
    # 1. 建立问题到难度的映射字典
    # key: question (stripped), value: difficulty_level
    question_difficulty_map = {}
    
    print(f"正在读取 {dataset_path} 构建索引...")
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # 提取问题文本，并去除首尾空格以确保匹配准确
                    question = data.get("question", "").strip()
                    # 提取难度等级信息
                    difficulty = data.get("info", {}).get("difficulty_level", "unknown")
                    
                    if question:
                        question_difficulty_map[question] = difficulty
                except json.JSONDecodeError:
                    print(f"跳过无法解析的行: {line[:50]}...")
    except FileNotFoundError:
        print(f"错误: 找不到文件 {dataset_path}")
        return

    print(f"已索引 {len(question_difficulty_map)} 个问题。")

    # 2. 准备输出文件句柄
    # 根据常见的难度等级分类，外加一个 unknown 用于处理未匹配到的情况
    output_handles = {
        'easy': open('r_test_easy.jsonl', 'w', encoding='utf-8'),
        'medium': open('r_test_medium.jsonl', 'w', encoding='utf-8'),
        'hard': open('r_test_hard.jsonl', 'w', encoding='utf-8'),
        'unknown': open('test_unknown.jsonl', 'w', encoding='utf-8')
    }

    # 3. 处理 test.jsonl
    # 正则表达式用于从 input_seq 中提取 Objective 之后、Observation 之前的内容
    # 使用 re.DOTALL 让 . 可以匹配换行符
    objective_pattern = re.compile(r'Objective:\s*(.*?)\s*Observation:', re.DOTALL)
    
    print(f"正在处理 {test_path} 并分类写入...")
    processed_count = 0
    
    try:
        with open(test_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    
                    # 获取 input_seq，通常在 trajectory 的第一个元素中
                    trajectory = data.get("trajectory", [])
                    if not trajectory:
                        continue
                        
                    first_seq = trajectory[0].get("input_seq", "")
                    
                    # 使用正则提取问题部分
                    match = objective_pattern.search(first_seq)
                    if match:
                        extracted_question = match.group(1).strip()
                        
                        # 在映射中查找难度
                        difficulty = question_difficulty_map.get(extracted_question)
                        
                        # 确定目标文件，如果难度未知或未找到，放入 unknown
                        target_file = output_handles.get(difficulty, output_handles['unknown'])
                        
                        # 写入数据
                        # 根据你的要求：处理完一条后，在 JSONL 中增加一个空行 (即写入两个换行符)
                        target_file.write(json.dumps(data, ensure_ascii=False) + '\n')
                        processed_count += 1
                    else:
                        # 无法提取到问题的情况
                        output_handles['unknown'].write(json.dumps(data, ensure_ascii=False) + '\n')
                        
                except json.JSONDecodeError:
                    print(f"跳过无法解析的测试数据行: {line[:50]}...")
                    
    except FileNotFoundError:
        print(f"错误: 找不到文件 {test_path}")

    # 4. 关闭所有文件
    for handle in output_handles.values():
        handle.close()
        
    print(f"处理完成。共处理了 {processed_count} 条数据。")

# 运行函数 (请确保当前目录下存在这两个文件)
if __name__ == "__main__":
    split_test_dataset('/data/yutao/browseragent2_dev/BrowserAgent/test_gen_data/webwalker/main-00000-of-00001.jsonl', '/data/yutao/browseragent2_dev/BrowserAgent/test_gen_data/webwalker/result/20251204_141524_webarena_results_rft.jsonl')