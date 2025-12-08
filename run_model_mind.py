import json
import argparse
import threading
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# 引入必要的库
import lxml.html
from datasets import load_dataset
from openai import OpenAI

# -----------------------------------------------------------------------------
# 1. HTML 解析与格式化模块 (保留优化版)
# -----------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """清理文本中的多余空白"""
    if not text:
        return ""
    return " ".join(text.split())

def get_node_role(element) -> str:
    """根据 HTML 标签估算 Accessibility Role"""
    tag = element.tag
    role_map = {
        'a': 'link',
        'button': 'button',
        'input': 'textbox',
        'select': 'combobox',
        'img': 'img',
        'h1': 'heading', 'h2': 'heading', 'h3': 'heading',
        'table': 'table', 'tr': 'row', 'td': 'gridcell', 'th': 'columnheader',
        'ul': 'list', 'li': 'listitem',
        'div': 'generic', 'span': 'generic'
    }
    return element.get('role', role_map.get(tag, 'generic'))

# def parse_mind2web_html(html_str: str) -> str:
#     """
#     【优化版】激进过滤非交互式元素和冗余文本，以减少 Token 消耗。
#     """
#     if not html_str:
#         return ""

#     try:
#         tree = lxml.html.fromstring(html_str)
#     except Exception as e:
#         return f"HTML Parse Error: {e}"

#     obs_lines = []
    
#     # 定义关键交互标签和 Role
#     INTERACTIVE_TAGS = {
#         'input', 'button', 'select', 'a', 'textarea', 'option', 
#         'form', 'img', 'label'
#     }
#     INTERACTIVE_ROLES = {
#         'button', 'link', 'checkbox', 'menuitem', 'tab', 'combobox', 
#         'textbox', 'searchbox', 'radio', 'switch', 'scrollbar'
#     }

#     for element in tree.iter():
#         backend_id = element.get('backend_node_id') or element.get('backend-id')
#         if not backend_id:
#             continue

#         tag = element.tag
#         role = get_node_role(element)
#         raw_text = element.text_content()
#         text_content = clean_text(raw_text)
        
#         # 激进过滤逻辑
#         is_interactive_tag = tag in INTERACTIVE_TAGS
#         has_aria_label = element.get('aria-label') is not None
#         is_interactive_role = role in INTERACTIVE_ROLES
#         has_meaningful_text = text_content and len(text_content) > 3 and tag not in ['html', 'body']

#         should_keep = False
#         if is_interactive_tag or has_aria_label or is_interactive_role:
#             should_keep = True
#         elif has_meaningful_text:
#             if tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'span', 'strong', 'em']:
#                  should_keep = True
#             elif tag == 'div' and len(text_content) < 50:
#                 should_keep = True

#         if not should_keep:
#             continue

#         # 截断与格式化
#         display_text = text_content[:50] if text_content else ""
#         line = f"[{backend_id}] {role} '{display_text}'"
        
#         if tag == 'input' or tag == 'textarea':
#             val = element.get('value') or element.get('placeholder')
#             if val:
#                 line += f" value: {clean_text(val)[:30]}"
        
#         attrs = []
#         if element.get('required'): attrs.append("req")
#         if element.get('aria-label'): attrs.append(f"label: {clean_text(element.get('aria-label'))[:30]}")
#         if element.get('type'): attrs.append(f"type: {element.get('type')}")
#         if element.get('checked') or element.get('selected'): attrs.append("checked")

#         if attrs:
#             line += " " + " ".join(attrs)
            
#         obs_lines.append(line)

#     return "\n".join(obs_lines)

def parse_mind2web_html(html_str: str) -> str:
    """
    【优化版】激进过滤非交互式元素和冗余文本，以减少 Token 消耗。
    """
    if not html_str:
        return ""

    try:
        tree = lxml.html.fromstring(html_str)
    except Exception as e:
        return f"HTML Parse Error: {e}"

    obs_lines = []
    
    # 定义关键交互标签和 Role
    INTERACTIVE_TAGS = {
        'input', 'button', 'select', 'a', 'textarea', 'option', 
        'form', 'img', 'label'
    }
    INTERACTIVE_ROLES = {
        'button', 'link', 'checkbox', 'menuitem', 'tab', 'combobox', 
        'textbox', 'searchbox', 'radio', 'switch', 'scrollbar'
    }

    for element in tree.iter():
        backend_id = element.get('backend_node_id') or element.get('backend-id')
        if not backend_id:
            continue

        tag = element.tag
        role = get_node_role(element)
        raw_text = element.text_content()
        text_content = clean_text(raw_text)
        
        # 激进过滤逻辑
        is_interactive_tag = tag in INTERACTIVE_TAGS
        has_aria_label = element.get('aria-label') is not None
        is_interactive_role = role in INTERACTIVE_ROLES
        has_meaningful_text = text_content and len(text_content) > 3 and tag not in ['html', 'body']

        should_keep = False
        if is_interactive_tag or has_aria_label or is_interactive_role:
            should_keep = True
        elif has_meaningful_text:
            if tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'span', 'strong', 'em']:
                 should_keep = True
            elif tag == 'div' and len(text_content) < 50:
                should_keep = True

        if not should_keep:
            continue

        # 截断与格式化
        display_text = text_content[:50] if text_content else ""
        line = f"[{backend_id}] {role} '{display_text}'"
        
        if tag == 'input' or tag == 'textarea':
            val = element.get('value') or element.get('placeholder')
            if val:
                line += f" value: {clean_text(val)[:30]}"
        
        attrs = []
        if element.get('required'): attrs.append("req")
        if element.get('aria-label'): attrs.append(f"label: {clean_text(element.get('aria-label'))[:30]}")
        if element.get('type'): attrs.append(f"type: {element.get('type')}")
        if element.get('checked') or element.get('selected'): attrs.append("checked")

        if attrs:
            line += " " + " ".join(attrs)
            
        obs_lines.append(line)

    return "\n".join(obs_lines)


# -----------------------------------------------------------------------------
# 2. 模型交互与主逻辑
# -----------------------------------------------------------------------------

lock = threading.Lock()

# 配置 API
api_key = "sk-proj-1234567890"  # 请替换为实际 Key
client = OpenAI(api_key=api_key, base_url="http://localhost:5001/v1") 



with open("/data/yutao/browseragent2_dev/BrowserAgent/system_prompt_with_history_info.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()


# User Prompt 模板
user_prompt_template = """
Objective: {}
Observation: {}
HISTORY_ACTION: {}
HISTORY_info: {}
"""

global_filename = None

def generate_filename(split_name):
    now = datetime.now()
    return f"{now.strftime('%Y%m%d_%H%M%S')}_mind2web_{split_name}_results.jsonl"

def write_a_data(action_list, annotation_id, filename=None):
    global global_filename
    if filename is None:
        if global_filename is None:
            global_filename = f"mind2web_results.jsonl"
        filename = global_filename
    
    trajectory_data = {
        "annotation_id": annotation_id,
        "trajectory": action_list,
        "trajectory_length": len(action_list)
    }
    
    lock.acquire()
    try:
        with open(filename, "a", encoding="utf-8") as fw:
            fw.write(json.dumps(trajectory_data, ensure_ascii=False) + "\n")
    finally:
        lock.release()

# 【修改点 1】: 增加异常抛出逻辑
def get_response(prompt, model="qwen2.5-7b", temperature=0):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        error_msg = str(e)
        # 检测是否为 Context Length 错误 (OpenAI API 通常返回 400 和特定消息)
        if "maximum context length" in error_msg or "context_length_exceeded" in error_msg:
            # 抛出特定 ValueError，以便上层捕获并跳过
            raise ValueError("TokenLimitExceeded")
        
        print(f"LLM Error: {e}")
        return "Error"

def process_mind2web_instance(instance):
    objective = instance['confirmed_task']
    annotation_id = instance['annotation_id']
    actions = instance['actions']
    action_reprs = instance['action_reprs']
    
    action_list = []
    
    for i, action_step in enumerate(actions):
        
        # 1. 准备 Observation
        raw_cleaned_html = action_step['cleaned_html']
        observation = parse_mind2web_html(raw_cleaned_html)
        
        # 2. 准备 History
        history_actions_list = action_reprs[0:i]
        history_action_str = "\n".join(history_actions_list) if history_actions_list else "None"
        history_info_str = "" 
        
        # 3. 拼接 Prompt
        real_prompt = user_prompt_template.format(
            objective,
            observation,
            history_action_str,
            history_info_str
        )
        prompt = system_prompt + "\n\n" + real_prompt
        
        # 【修改点 2】: 捕获 TokenLimitExceeded 异常并直接退出
        try:
            model_output = get_response(prompt, temperature=0)
        except ValueError as ve:
            if str(ve) == "TokenLimitExceeded":
                print(f"⚠️ [Skip] 任务 {annotation_id} 第 {i} 步 Token 超限，直接跳过不保存。")
                return f"Skipped Task {annotation_id} (Token Limit)"
            else:
                # 其他 ValueError 继续抛出或处理
                model_output = "Error"

        action_record = {
            "input_seq": prompt,
            "output_seq": model_output,
            "ground_truth": {
                "operation": action_step['operation'],
                "pos_candidates": action_step['pos_candidates']
            },
            "step_index": i
        }
        
        action_list.append(action_record)

    # 只有当循环完整执行且没有触发 return 时，才会执行到这里
    write_a_data(action_list, annotation_id)
    return f"Finished Task {annotation_id} ({len(action_list)} steps)"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Mind2Web Evaluation on Test Sets")
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to process')
    parser.add_argument('--max_threads', type=int, default=8, help='Thread pool size')
    parser.add_argument('--data_dir', type=str, default="./Mind2Web/data", 
                        help='Mind2Web 数据集 data 目录的路径')
    parser.add_argument('--test_split', type=str, default="test_task", 
                        choices=["test_task", "test_website", "test_domain"],
                        help='选择要运行的测试集拆分')
    
    args = parser.parse_args()

    global_filename = generate_filename(args.test_split)

    split_path = os.path.join(args.data_dir, args.test_split, f"{args.test_split}_*.json")
    
    print(f"正在从本地加载测试集: {args.test_split}")
    print(f"搜索路径: {split_path}")

    try:
        dataset = load_dataset("json", data_files={args.test_split: split_path}, split=args.test_split)
        print(f"✅ 成功加载 {args.test_split}。总样本数: {len(dataset)}")
    except Exception as e:
        print(f"❌ 加载数据集失败: {e}")
        exit(1)

    total_samples = len(dataset)
    num_process = min(args.num_samples, total_samples) if args.num_samples else total_samples
    tasks_to_process = [dataset[i] for i in range(num_process)]

    print(f"开始处理 {len(tasks_to_process)} 个任务，使用 {args.max_threads} 个线程")

    with ThreadPoolExecutor(max_workers=args.max_threads) as executor:
        future_to_item = {
            executor.submit(process_mind2web_instance, item): idx 
            for idx, item in enumerate(tasks_to_process)
        }
        
        completed_count = 0
        for future in as_completed(future_to_item):
            idx = future_to_item[future]
            try:
                res = future.result()
                completed_count += 1
                if completed_count % 1 == 0:
                    print(f"进度: {completed_count}/{len(tasks_to_process)} - {res}")
            except Exception as e:
                print(f"任务 {idx} 执行出错: {e}")

    print(f"处理完成。结果已保存至 {global_filename}")