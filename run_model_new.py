import json
import pandas as pd
from openai import OpenAI
import threading
import json
from typing import List, Dict, Any
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import random
import uuid
import re

# 线程锁，用于文件写入
lock = threading.Lock()

# 配置 API Key 和 Client
api_key = "sk-proj-1234567890"
client = OpenAI(api_key=api_key, base_url="http://localhost:5001/v1")

# 读取 System Prompt
with open("system_prompt_with_history_info.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

# 全局变量，用于存储生成的输出文件名
global_filename = None

def generate_filename():
    """生成带时间戳的文件名"""
    now = datetime.now()
    return f"{now.strftime('%Y%m%d_%H%M%S')}_webarena_results.jsonl"

def write_a_data(action_list, filename=None):
    """线程安全地将结果写入文件"""
    global global_filename
    if filename is None:
        if global_filename is None:
            global_filename = generate_filename()
        filename = global_filename

    trajectory_data = {
        "trajectory": action_list,
        "trajectory_length": len(action_list)
    }
    
    lock.acquire()
    try:
        with open(filename, "a", encoding="utf-8") as fw:
            fw.write(json.dumps(trajectory_data, ensure_ascii=False) + "\n")
    finally:
        lock.release()

def call_tool_server(trajectory_ids: List[str], actions: List[str], finish: List[bool], start_url: str = None, **kwargs: Dict[str, List[Any]]) -> Dict[str, Any]:
    """
    通过 HTTP 请求查询 tool server 获取观察结果 (observation) 和结束状态
    增加了 start_url 参数支持
    """
    env_url = "http://localhost:30810/get_observation"

    # 如果没有指定 start_url，使用默认的维基百科 URL
    if start_url is None:
        start_url = "https://tigerai.ca/wiki/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
    
    extra_fields = [{
        "url": start_url
    }]
    data = {
        "trajectory_ids": trajectory_ids,
        "actions": actions,
        "finish": finish,
        "extra_fields": extra_fields
    }
    
    try:
        resp = requests.post(env_url, json=data, timeout=1200)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e), "observations": [""], "dones": [False], "valids": [False]}

user_prompt = """
Objective: {}
Observation: {}
HISTORY_ACTION: {}
HISTORY_info: {}
"""

def get_response(prompt, model="qwen2.5-7b", temperature=0):
    """调用大模型获取回复"""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=1024
    )

    model_answer = response.choices[0].message.content
    return model_answer

def extract_command(text):
    """提取模型输出中的命令部分"""
    blocks = re.findall(r'```\s*([^\s].*?[^\s])\s*```', text, re.DOTALL)
    
    if not blocks:
        return " "

    last_command = blocks[-1].strip()
    last_command = last_command.replace("```", "")
    return last_command.strip()

def extract_conclusion(text):
    """提取模型输出中的结论部分"""
    blocks = re.findall(r'<conclusion>\s*(.*?)\s*</conclusion>', text, re.DOTALL)

    if not blocks:
        return " "

    last_conclusion = blocks[-1].strip()
    return last_conclusion

def Get_multi_turn_response(question, answer, start_url=None):
    """
    核心循环：执行多轮对话和浏览器操作
    """
    tar_id = str(uuid.uuid4())
    history = "\n"
    obj = question
    history_info = "\n"
    action_list = []
    is_error = False
    error_msg = ""
    
    try:
        # 初始化环境，传入 start_url
        print(f"start_url: {start_url}")
        jsoned_data = call_tool_server([tar_id], [''], [False], start_url=start_url)
        # 增加健壮性检查
        if 'observations' in jsoned_data and len(jsoned_data['observations']) > 0:
            obs = jsoned_data['observations'][0]
        else:
            obs = ""
        
        for i in range(30): # 最多执行30步
            try:
                # 尝试清洗 observation 字符串
                obs_str = str(obs)
                if 'Observation:\n' in obs_str:
                    obs_str = obs_str.split('Observation:\n')[1]
                if '\nParsed Previous Action:' in obs_str:
                    obs_str = obs_str.split('\nParsed Previous Action:')[0]
                obs = obs_str
            except Exception:
                pass
            
            real_prompt = user_prompt.format(obj, obs, history, history_info)
            prompt = system_prompt + "\n\n" + real_prompt
            
            try:
                response = get_response(prompt, temperature=0)
                last_command = extract_command(response)
                last_info = extract_conclusion(response)
                
                history = history + last_command + "\n"
                history_info = history_info + last_info + "\n"
                
                action_list.append({"input_seq": prompt, "output_seq": response})
                
                # 执行动作
                if "stop" in last_command:
                    call_tool_server([tar_id], [response], [True], start_url=start_url)
                    break
                else:
                    jsoned_data = call_tool_server([tar_id], [response], [False], start_url=start_url)
                    if 'observations' in jsoned_data and len(jsoned_data['observations']) > 0:
                        obs = jsoned_data['observations'][0]
                    else:
                        obs = ""
                    
            except Exception as e:
                is_error = True
                error_msg = str(e)
                print(f"Step {i} error: {e}")
                break
                
    except Exception as e:
        is_error = True
        error_msg = str(e)
        print(f"Workflow error: {e}")

    # 记录结果
    if action_list:
        action_list[-1]["is_error"] = is_error
        action_list[-1]["error_msg"] = error_msg
    else:
        action_list.append({
            "input_seq": f"question: {question}",
            "output_seq": "error",
            "is_error": is_error,
            "error_msg": error_msg
        })

    write_a_data(action_list)

max_threads = 16 

def process_standardized_item(item):
    """
    处理标准化的数据项
    item 结构: {"question": str, "answer": str, "url": str or None}
    """
    return Get_multi_turn_response(item["question"], item["answer"], start_url=item["url"])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run multi-turn response generation with customizable file paths.")
    
    # 原始参数
    parser.add_argument('--data_path', type=str, default='', 
                        help='Path to the parquet data file (e.g., /path/to/train.parquet)')
    
    # 新增参数
    parser.add_argument('--jsonl_path', type=str, default=None, 
                        help='Path to the JSONL data file (e.g., /path/to/data.jsonl)')
    parser.add_argument('--start_url', type=str, default=None, 
                        help='Default starting URL for browser navigation')
    parser.add_argument('--force_url', action='store_true',
                        help='Force use of --start_url, ignoring URLs in dataset')
    parser.add_argument('--num_samples', type=int, default=None, 
                        help='Number of samples to process (default: all)')
    
    args = parser.parse_args()

    # 统一的数据列表，包含 {"question": ..., "answer": ..., "url": ...}
    tasks_to_process = []

    # 模式 1: 处理 JSONL 文件
    if args.jsonl_path:
        print(f"正在读取 JSONL 文件: {args.jsonl_path}")
        try:
            with open(args.jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        # 移植过滤逻辑：只保留 single_source 类型
                        if data.get("info", {}).get("type") == "single_source":
                            # URL 逻辑判断
                            if args.force_url and args.start_url:
                                task_url = args.start_url
                            else:
                                # 优先使用数据中的 root_url，否则使用 start_url
                                task_url = data.get("root_url") or args.start_url

                            tasks_to_process.append({
                                "question": data["question"],
                                "answer": data.get("answer", ""),
                                "url": task_url
                            })
                    except json.JSONDecodeError:
                        continue
            print(f"从 JSONL 中筛选出 {len(tasks_to_process)} 个 'single_source' 任务")
        except Exception as e:
            print(f"读取 JSONL 失败: {e}")
            exit(1)

    # 模式 2: 处理 Parquet 文件
    elif args.data_path:
        print(f"正在读取 Parquet 文件: {args.data_path}")
        try:
            data_df = pd.read_parquet(args.data_path)
            # 随机打乱
            data_df = data_df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            for index, row in data_df.iterrows():
                extra_info = row.get("extra_info", {})
                
                # URL 逻辑判断
                if args.force_url and args.start_url:
                    task_url = args.start_url
                else:
                    # 尝试从 extra_info 中获取 url，字段名假设为 'url'
                    task_url = extra_info.get("url") or args.start_url

                tasks_to_process.append({
                    "question": extra_info.get("question", ""),
                    "answer": extra_info.get("selected_answer", ""),
                    "url": task_url
                })
            print(f"从 Parquet 中读取了 {len(tasks_to_process)} 个任务")
        except Exception as e:
            print(f"读取 Parquet 失败: {e}")
            exit(1)
    
    else:
        print("错误: 必须提供 --data_path (Parquet) 或 --jsonl_path (JSONL)")
        exit(1)

    # 截取指定数量的样本
    if args.num_samples is not None and args.num_samples > 0:
        tasks_to_process = tasks_to_process[:args.num_samples]
        print(f"根据配置，截取前 {len(tasks_to_process)} 个任务进行处理")

    # 开始多线程处理
    print(f"开始处理 {len(tasks_to_process)} 个数据项，使用 {max_threads} 个线程")
    
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        # 提交任务到线程池
        future_to_item = {
            executor.submit(process_standardized_item, item): idx 
            for idx, item in enumerate(tasks_to_process)
        }
        
        completed_count = 0
        for future in as_completed(future_to_item):
            idx = future_to_item[future]
            try:
                future.result() 
                completed_count += 1
                if completed_count % 10 == 0:
                    print(f"已完成 {completed_count}/{len(tasks_to_process)} 个任务")
            except Exception as e:
                print(f"任务 {idx} 执行出错: {e}")
                completed_count += 1
    
    print(f"所有任务完成！总计处理了 {completed_count} 个数据项")