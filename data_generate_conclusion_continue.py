import json
import pandas as pd
from typing import List, Dict, Any
import requests
from openai import OpenAI
import argparse
import random
import uuid
import re

# 配置 API
api_key = "sk-5TOLjHJSn7uyRj2gXZLxYsRe9vxmr8N9XWK2lQHalvgXiBoc"
client = OpenAI(api_key=api_key, base_url="https://open.xiaojingai.com/v1/")

# 读取 System Prompt
with open("/data/yutao/browseragent2_dev/BrowserAgent/system_prompt_with_history_info_conclusion_v2.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

def call_tool_server(trajectory_ids: List[str], actions: List[str], finish: List[bool], start_url: str = None, **kwargs: Dict[str, List[Any]]) -> Dict[str, Any]:
    """
    查询 Tool Server 获取观察结果 (Observation) 和结束标志
    """
    env_url = "http://localhost:30810/get_observation"

    # 使用提供的 URL 或默认为 Wikipedia
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
        result = resp.json()
        return result
    except Exception as e:
        print(f"[ERROR] Tool server call failed: {e}")
        return {"error": str(e), "observations": [""], "dones": [False], "valids": [False]}


user_prompt = """
Objective: {}
Observation: {}
HISTORY_ACTION: {}
HISTORY_info: {}
"""

def get_response(prompt, model="gpt-4.1", temperature=0.5):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=1024
    )
    model_answer = response.choices[0].message.content
    return model_answer

def extract_command(text):
    blocks = re.findall(r'```\s*([^\s].*?[^\s])\s*```', text, re.DOTALL)
    if not blocks:
        return " "
    last_command = blocks[-1].strip()
    last_command = last_command.replace("```", "")
    return last_command.strip()

def extract_conclusion(text):
    blocks = re.findall(r'<conclusion>\s*(.*?)\s*</conclusion>', text, re.DOTALL)
    if not blocks:
        return " "
    last_conclusion = blocks[-1].strip()
    return last_conclusion

def extract_obs(text):
    """
    从 <obs></obs> 标签中提取最相关的 HTML 元素
    """
    blocks = re.findall(r'<obs>\s*(.*?)\s*</obs>', text, re.DOTALL)
    if not blocks:
        return " "
    last_obs = blocks[-1].strip()
    return last_obs

def write_a_data(input_seq, output_seq, output_file, extracted_obs=None):
    """
    写入训练数据
    """
    written_data = {
        "input_seq": input_seq,
        "output_seq": output_seq
    }
    
    if extracted_obs and extracted_obs.strip():
        written_data["extracted_obs"] = extracted_obs
    
    with open(output_file, "a", encoding="utf-8") as fw:
        fw.write(json.dumps(written_data, ensure_ascii=False) + "\n")

def Get_multi_turn_response(question, answer, output_file, start_url=None, golden_path=None):
    """
    生成多轮浏览器交互响应
    """
    tar_id = str(uuid.uuid4())
    history = "\n"
    history_info = "\n"
    obj = question
    
    # 初始化环境
    try:
        jsoned_data = call_tool_server([tar_id], [''], [False], start_url=start_url)
        obs = jsoned_data['observations'][0]    
    except Exception as e:
        print(f"[ERROR] Failed to get observation: {e}")
        obs = ""

    for i in range(30):
        # 改进 observation 解析逻辑
        try:
            if isinstance(obs, str) and 'Observation:\n' in obs:
                obs_parts = obs.split('Observation:\n')
                if len(obs_parts) > 1:
                    obs_content = obs_parts[1]
                    if '\nParsed Previous Action:' in obs_content:
                        obs = obs_content.split('\nParsed Previous Action:')[0]
                    else:
                        obs = obs_content
                else:
                    obs = obs_parts[0] if obs_parts else ""
            elif isinstance(obs, str):
                obs = obs
            else:
                obs = str(obs) if obs is not None else ""
        except Exception as e:
            print(f"[WARNING] Error parsing observation: {e}")
            obs = str(obs) if obs is not None else ""
        
        print(f"[DEBUG] Processed observation length: {len(obs)} chars") 
        
        real_prompt = user_prompt.format(obj, obs, history, history_info)
        # print(real_prompt) # 可以取消注释以调试
        prompt = system_prompt + "\n\n" + real_prompt
        response = get_response(prompt, temperature=0.3)

        last_command = extract_command(response)
        last_info = extract_conclusion(response)
        last_obs = extract_obs(response)

        history = history + last_command + "\n"
        history_info = history_info + last_info + "\n"

        try:
            jsoned_data = call_tool_server([tar_id], [response], [False], start_url=start_url)
            obs = jsoned_data['observations'][0]
        except Exception as e:
            print(f"[ERROR] Failed to update observation: {e}")
        
        # 写入数据
        write_a_data(prompt, response, output_file, extracted_obs=last_obs)

        if "stop" in last_command:
            call_tool_server([tar_id], [response], [True], start_url=start_url)
            return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-turn response generation with customizable file paths and URLs.")
    parser.add_argument('--output_file', type=str, 
                        default='./test.jsonl', 
                        help='Output file path for writing the data')
    parser.add_argument('--data_path', type=str, 
                        default='', 
                        help='Path to the data file (Parquet)')
    parser.add_argument('--jsonl_path', type=str, 
                        default=None, 
                        help='Path to the JSONL data file')
    parser.add_argument('--start_url', type=str, 
                        default=None, 
                        help='Default starting URL for browser navigation')
    parser.add_argument('--url_field', type=str, 
                        default='url', 
                        help='Field name in the dataset that contains the URL')
    parser.add_argument('--force_url', action='store_true',
                        help='Force use of --start_url, ignoring URLs in dataset')
    parser.add_argument('--num_samples', type=int, 
                        default=5000, 
                        help='Number of samples to process')
    
    # === 新增参数：start_index ===
    parser.add_argument('--start_index', type=int, default=0, 
                        help='Index to start processing from (skips previous items after shuffle)')
    
    parser.add_argument('--question', type=str, 
                        default=None, 
                        help='Single question to process')
    parser.add_argument('--gt', type=str, 
                        default=None, 
                        help='Ground truth answer')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Configuration:")
    print(f"  Output file: {args.output_file}")
    
    if args.question is not None:
        print("  Mode: Single Question Mode")
    elif args.jsonl_path is not None:
        print("  Mode: JSONL File Mode")
        print(f"  Data file: {args.jsonl_path}")
    elif args.data_path is not None:
        print("  Mode: Batch File Mode (Parquet)")
        print(f"  Data file: {args.data_path}")
    
    print(f"  Start URL: {args.start_url or 'Wikipedia (default)'}")
    print(f"  Force URL: {args.force_url}")
    print(f"  Start Index: {args.start_index}")
    print(f"  Num Samples: {args.num_samples}")
    print("=" * 80)

    # 模式1：直接指定问题和答案
    if args.question is not None:
        print("\n[Single Question Mode] Processing single question...")
        Get_multi_turn_response(args.question, args.gt or "", args.output_file, start_url=args.start_url, golden_path=None)
        print(f"Completed! Processed 1 question.")

    # 模式3：从JSONL文件批量处理
    elif args.jsonl_path is not None:
        print(f"\n[JSONL File Mode] Processing file: {args.jsonl_path}...")
        
        processed_lines = []
        try:
            with open(args.jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if data.get("info", {}).get("type") == "single_source":
                            processed_lines.append(data)
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            print(f"[ERROR] Failed to read JSONL file: {e}")
            exit(1)

        print(f"[INFO] Found {len(processed_lines)} 'single_source' items in the file.")
        
        random.seed(43)
        random.shuffle(processed_lines)

        cnt = 0 # 实际处理计数
        
        for idx, data in enumerate(processed_lines):
            # 跳过指定数量的样本
            if idx < args.start_index:
                continue

            question = data["question"]
            gt = data.get("answer", "")
            
            golden_path_list = data.get("info", {}).get("golden_path", [])
            golden_path_str = golden_path_list[0] if golden_path_list else None 

            if args.force_url and args.start_url:
                final_url = args.start_url
            else:
                task_url = data.get("root_url")
                final_url = task_url or args.start_url
            
            if cnt == 0 or cnt % 10 == 0:
                print(f"\n[Progress] Global Index {idx} | Batch Progress {cnt + 1}/{args.num_samples}")
                print(f"  Question: {question[:80]}...")
                print(f"  URL: {final_url or 'Default Wikipedia'}")

            Get_multi_turn_response(question, gt, args.output_file, start_url=final_url, golden_path=golden_path_str)
            
            cnt += 1
            if cnt >= args.num_samples:
                break
        
        print(f"\nCompleted! Processed {cnt} samples (Started from index {args.start_index}).")

    # 模式2：从文件批量处理（Parquet）
    elif args.data_path:
        print(f"\n[Batch File Mode] Processing file: {args.data_path}...")
        
        data_df = pd.read_parquet(args.data_path)
        # 保持 random_state=42，确保顺序和之前一致
        data_df = data_df.sample(frac=1, random_state=42).reset_index(drop=True)

        cnt = 0 # 实际处理计数
        
        for i, row in data_df.iterrows():
            # 跳过指定数量的样本
            if i < args.start_index:
                continue
                
            question = row["extra_info"]["question"]
            gt = row["extra_info"]["selected_answer"]
            
            if args.force_url and args.start_url:
                final_url = args.start_url
            else:
                task_url = None
                try:
                    if args.url_field in row.get("extra_info", {}):
                        task_url = row["extra_info"][args.url_field]
                    elif args.url_field in row:
                        task_url = row[args.url_field]
                except Exception:
                    pass
                final_url = task_url or args.start_url
            
            if cnt == 0 or cnt % 10 == 0:
                print(f"\n[Progress] Global Index {i} | Batch Progress {cnt + 1}/{args.num_samples}")
                print(f"  Question: {question[:80]}...")
                print(f"  URL: {final_url or 'Default Wikipedia'}")

            Get_multi_turn_response(question, gt, args.output_file, start_url=final_url, golden_path=None)
            
            cnt += 1
            if cnt >= args.num_samples:
                break
        
        print(f"\n{'='*80}")
        print(f"Completed! Processed {cnt} samples (Started from index {args.start_index}).")
        print(f"Results saved to: {args.output_file}")
        print(f"{'='*80}")

    else:
        print("[ERROR] Must provide either --question, --jsonl_path, or --data_path")
        parser.print_help()
        exit(1)