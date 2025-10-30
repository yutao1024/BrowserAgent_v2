import json
import pandas as pd
from typing import List, Dict, Any
import requests
from openai import OpenAI
import json
import argparse

api_key = "sk-5TOLjHJSn7uyRj2gXZLxYsRe9vxmr8N9XWK2lQHalvgXiBoc"
client = OpenAI(api_key = api_key, base_url= "https://open.xiaojingai.com/v1/")
with open("system_prompt_with_history_info.txt","r",encoding = "utf-8") as f:
    system_prompt = f.read()

def call_tool_server(trajectory_ids: List[str], actions: List[str], finish: List[bool], start_url: str = None, **kwargs: Dict[str, List[Any]]) -> Dict[str, Any]:
    """
    Query the tool server for the observation and done flag
    
    Args:
        trajectory_ids: List of trajectory IDs
        actions: List of actions to execute
        finish: List of finish flags
        start_url: Starting URL for the browser (if None, uses default Wikipedia)
        **kwargs: Additional keyword arguments
    """
    env_url = "http://localhost:30815/get_observation"

    # Use provided URL or default to Wikipedia
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

# Prompt for filtering observation to extract key elements
filter_observation_prompt = """You are an intelligent assistant helping to filter webpage observations for browser automation tasks.

Given the user's objective, previous actions, and a long observation containing many webpage elements, your task is to identify and extract the 5-10 MOST RELEVANT elements that are most likely to help achieve the objective.

**Objective:** {}

**Previous Actions (HISTORY_ACTION):**
{}

**Previous Information Found (HISTORY_info):**
{}

**Full Observation:**
{}

**Your Task:**
Analyze the objective and select ONLY the 5-10 most relevant elements from the observation that will help complete the task.

**Selection Criteria (in priority order):**
1. **Answer-containing elements**: Elements that directly contain information related to the objective (e.g., if looking for "when founded", prioritize elements with dates, "Established", "Founded" keywords)
2. **Semantic relevance**: Elements whose content keywords match the objective's topic
3. **Interactive elements**: Clickable links, buttons, or input fields that could lead to the answer
4. **Navigation elements**: Links to relevant sections (History, About, Details, etc.)
5. **Avoid redundancy**: Don't select elements already interacted with (check HISTORY_ACTION)
6. **Complementary information**: Select elements that extend or verify HISTORY_info

**Important Rules:**
- Extract EXACTLY 5-10 elements (not more, not less)
- Preserve the EXACT format: `[id] element_type 'content'`
- Include the full line as it appears in the observation
- Prioritize quality over quantity - better to have 5 highly relevant elements than 10 mediocre ones
- If an element contains the answer, ALWAYS include it
- Remove line numbers (e.g., "    10|") if present in the original

**Output Format:**
Return ONLY the filtered elements, one per line, in the exact format shown:

[5177] rowheader 'Established' required: False
[5178] gridcell 'September 15, 1764' required: False
[5200] button 'Search'
[2933] link 'History'

Do NOT include:
- Explanations or reasoning
- Headers or labels
- Line numbers
- Any additional text"""

user_prompt = """
Objective: {}
Observation: {}
HISTORY_ACTION: {}
HISTORY_info: {}
"""

def get_response(prompt , model = "gpt-4.1", temperature = 0.5):

    response = client.chat.completions.create(
        model = model,
        messages = [{"role": "user", "content": prompt}],
        temperature = temperature,
        max_tokens = 1024
    )
    model_answer = response.choices[0].message.content
    return model_answer

def filter_observation(obs, objective, history, history_info, model="gpt-4o-mini", max_obs_length=3000):
    """
    Filter observation to extract the most relevant 5-10 elements using LLM.
    
    Args:
        obs: The full observation string
        objective: The user's objective/question
        history: Previous actions taken
        history_info: Information already found
        model: The model to use for filtering (default: gpt-4o-mini for cost efficiency)
        max_obs_length: Maximum length of observation before filtering is applied
    
    Returns:
        Filtered observation string with key elements
    """
    # If observation is already short enough, return as is
    if len(obs) <= max_obs_length:
        print(f"[INFO] Observation length ({len(obs)}) is within limit, skipping filtering")
        return obs
    
    print(f"[INFO] Observation too long ({len(obs)} chars), applying filtering...")
    
    # Prepare the filtering prompt
    filter_prompt = filter_observation_prompt.format(
        objective,
        history if history.strip() else "None",
        history_info if history_info.strip() else "None",
        obs
    )
    
    try:
        # Use a smaller, faster model for filtering to save costs
        filtered_obs = get_response(filter_prompt, model=model, temperature=0.3)
        print(f"[INFO] Filtered observation from {len(obs)} to {len(filtered_obs)} chars")
        return filtered_obs
    except Exception as e:
        print(f"[ERROR] Failed to filter observation: {e}")
        # Fallback: if filtering fails, truncate observation
        print(f"[WARNING] Using truncation fallback")
        return obs[:max_obs_length] + "\n... [truncated]"

import re
def extract_command(text):
    blocks = re.findall(r'```\s*([^\s].*?[^\s])\s*```', text, re.DOTALL)
    
    if not blocks:
        return " "

    last_command = blocks[-1].strip()
    last_command = last_command.replace("```","")
    return last_command.strip()

def extract_conclusion(text):
    blocks = re.findall(r'<conclusion>\s*(.*?)\s*</conclusion>', text, re.DOTALL)

    if not blocks:
        return " "

    last_conclusion = blocks[-1].strip()
    return last_conclusion

def write_a_data(input,output,output_file):
    written_data = {"input_seq":input,"output_seq":output}
    with open(output_file,"a",encoding = "utf-8") as fw:
        fw.write(json.dumps(written_data,ensure_ascii=False) + "\n")

import uuid

# Configuration for observation filtering
ENABLE_OBSERVATION_FILTERING = False  
OBSERVATION_MAX_LENGTH = 1000  
FILTER_MODEL = "gpt-4o-mini"  

def Get_multi_turn_response(question, answer, output_file, start_url=None):
    """
    Generate multi-turn browser interaction responses
    
    Args:
        question: The question/objective to accomplish
        answer: The ground truth answer (for evaluation)
        output_file: Path to save the interaction data
        start_url: Starting URL for browser navigation (optional)
    """
    tar_id = str(uuid.uuid4())
    history = "\n"
    history_info = "\n"
    obj = question
    
    # Initialize environment with optional custom URL
    try:
        jsoned_data = call_tool_server([tar_id], [''], [False], start_url=start_url)
        obs = jsoned_data['observations'][0]    
    except Exception as e:
        print(f"[ERROR] Failed to get observation: {e}")
        obs = ""

    for i in range(30):
        # 改进observation解析逻辑
        try:
            if isinstance(obs, str) and 'Observation:\n' in obs:
                # 尝试提取observation部分
                obs_parts = obs.split('Observation:\n')
                if len(obs_parts) > 1:
                    obs_content = obs_parts[1]
                    # 如果包含"Parsed Previous Action"，则截取到该位置
                    if '\nParsed Previous Action:' in obs_content:
                        obs = obs_content.split('\nParsed Previous Action:')[0]
                    else:
                        obs = obs_content
                else:
                    obs = obs_parts[0] if obs_parts else ""
            elif isinstance(obs, str):
                # 如果observation数据不包含标准格式，直接使用原始数据
                obs = obs
            else:
                # 如果obs不是字符串，转换为字符串
                obs = str(obs) if obs is not None else ""
        except Exception as e:
            print(f"[WARNING] Error parsing observation: {e}")
            # 如果解析失败，保持原始obs或设为空字符串
            obs = str(obs) if obs is not None else ""
        
        print(f"[DEBUG] Processed observation length: {len(obs)} chars") 
        print(f"[DEBUG] Processed observation: {obs}") 
        
        # Apply observation filtering to extract key elements if enabled
        if ENABLE_OBSERVATION_FILTERING:
            filtered_obs = filter_observation(
                obs, obj, history, history_info, 
                model=FILTER_MODEL, 
                max_obs_length=OBSERVATION_MAX_LENGTH
            )
        else:
            filtered_obs = obs
        
        # Use filtered observation in the prompt
        real_prompt = user_prompt.format(obj, filtered_obs, history, history_info)
        prompt = system_prompt + "\n\n" + real_prompt
        response = get_response(prompt, temperature=0.3)

        last_command = extract_command(response)
        last_info = extract_conclusion(response)

        history = history + last_command + "\n"
        history_info = history_info + last_info + "\n"

        try:
            jsoned_data = call_tool_server([tar_id], [response], [False], start_url=start_url)
            obs = jsoned_data['observations'][0]
        except Exception as e:
            print(f"[ERROR] Failed to update observation: {e}")
            # 如果更新失败，保持当前observation
        write_a_data(prompt,response,output_file)

        if "stop" in last_command:
            call_tool_server([tar_id], [response], [True], start_url=start_url)
            return


number_to_process = 5000
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-turn response generation with customizable file paths and URLs.")
    parser.add_argument('--output_file', type=str, 
                        default='./test.jsonl', 
                        help='Output file path for writing the data (e.g., /path/to/output.jsonl)')
    parser.add_argument('--data_path', type=str, 
                        default='', 
                        help='Path to the data file (e.g., /path/to/train.parquet)')
    parser.add_argument('--start_url', type=str, 
                        default=None, 
                        help='Default starting URL for browser navigation (e.g., https://www.google.com)')
    parser.add_argument('--url_field', type=str, 
                        default='url', 
                        help='Field name in the dataset that contains the URL (default: "url")')
    parser.add_argument('--force_url', action='store_true',
                        help='Force use of --start_url, ignoring URLs in dataset')
    parser.add_argument('--num_samples', type=int, 
                        default=5000, 
                        help='Number of samples to process (default: 5000)')
    
    # 新增：支持直接指定单个问题和答案
    parser.add_argument('--question', type=str, 
                        default=None, 
                        help='Single question to process (bypasses data file)')
    parser.add_argument('--gt', type=str, 
                        default=None, 
                        help='Ground truth answer for the question')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Configuration:")
    print(f"  Output file: {args.output_file}")
    
    # 判断使用哪种输入模式
    if args.question is not None:
        print("  Mode: Single Question Mode")
        print(f"  Question: {args.question}")
        print(f"  Ground Truth: {args.gt or 'Not provided'}")
    else:
        print("  Mode: Batch File Mode")
        print(f"  Data file: {args.data_path}")
        print(f"  Samples to process: {args.num_samples}")
    
    print(f"  Start URL: {args.start_url or 'Wikipedia (default)'}")
    print(f"  URL field: {args.url_field}")
    print(f"  Force URL: {args.force_url}")
    if args.force_url:
        print(f"  ⚠️  Will IGNORE URLs in dataset, use --start_url for all tasks")
    else:
        print(f"  ℹ️  Will use dataset URLs if available, fallback to --start_url")
    print("=" * 80)

    # 模式1：直接指定问题和答案
    if args.question is not None:
        print("\n[Single Question Mode] Processing single question...")
        print(f"  Question: {args.question}")
        print(f"  URL: {args.start_url or 'Default Wikipedia'}")
        
        Get_multi_turn_response(args.question, args.gt or "", args.output_file, start_url=args.start_url)
        
        print(f"\n{'='*80}")
        print(f"Completed! Processed 1 question.")
        print(f"Results saved to: {args.output_file}")
        print(f"{'='*80}")
    
    # 模式2：从文件批量处理（原有方式）
    else:
        if not args.data_path:
            print("[ERROR] Must provide either --question or --data_path")
            parser.print_help()
            exit(1)
        
        data_df = pd.read_parquet(args.data_path)
        data_df = data_df.sample(frac=1, random_state=42).reset_index(drop=True)

        cnt = 0
        for i, row in data_df.iterrows():
            question = row["extra_info"]["question"]
            gt = row["extra_info"]["selected_answer"]
            
            # Determine which URL to use based on --force_url flag
            if args.force_url and args.start_url:
                # Force use command-line URL, ignore dataset
                final_url = args.start_url
            else:
                # Try to get URL from dataset, fall back to command line arg or default
                task_url = None
                try:
                    # First try to get from extra_info
                    if args.url_field in row.get("extra_info", {}):
                        task_url = row["extra_info"][args.url_field]
                    # Then try direct field
                    elif args.url_field in row:
                        task_url = row[args.url_field]
                except Exception as e:
                    print(f"[INFO] No URL found in dataset for row {i}, using default")
                
                # Use task-specific URL, or command-line URL, or system default
                final_url = task_url or args.start_url
            
            if cnt == 0 or cnt % 10 == 0:
                print(f"\n[Progress] Processing sample {cnt + 1}/{args.num_samples}")
                print(f"  Question: {question[:80]}...")
                print(f"  URL: {final_url or 'Default Wikipedia'}")

            cnt += 1
            Get_multi_turn_response(question, gt, args.output_file, start_url=final_url)
            
            if cnt >= args.num_samples:
                break
        
        print(f"\n{'='*80}")
        print(f"Completed! Processed {cnt} samples.")
        print(f"Results saved to: {args.output_file}")
        print(f"{'='*80}")