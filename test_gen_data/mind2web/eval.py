import json
import argparse
import re
import numpy as np
import os
from datetime import datetime

# -----------------------------------------------------------------------------
# 核心解析与评估函数
# -----------------------------------------------------------------------------

def parse_model_output(output_str):
    """
    解析模型输出，提取操作类型(op)、目标元素ID(id)和值(value)。
    策略：
    1. 优先查找 markdown 代码块 ```...``` 中的内容。
    2. 如果有多个代码块，优先使用最后一个（通常是最终决定的动作）。
    3. 如果没有代码块，则在全文中尝试搜索。
    """
    if not output_str or output_str.strip().lower() == "error":
        return None, None, None

    # 1. 尝试提取所有 markdown 代码块的内容
    # re.DOTALL 允许 . 匹配换行符
    code_blocks = re.findall(r'```(.*?)```', output_str, re.DOTALL)
    
    # 准备待检查的文本列表
    # 如果找到了代码块，就只检查代码块（倒序，优先看最后生成的）
    # 如果没找到，就检查整个原始字符串
    texts_to_check = code_blocks[::-1] if code_blocks else [output_str]
    
    # 正则表达式：匹配 CLICK [id] 或 TYPE [id] [val]
    # 格式说明：
    #   ([A-Z]+)       -> Group 1: 动作类型 (CLICK, TYPE, SELECT...)
    #   \s+            -> 空格
    #   \[([^\]]+)\]   -> Group 2: ID (中括号内的任意非中括号字符)
    #   (?:\s+\[(.*?)\])? -> Group 3 (可选): Value (中括号内的任意字符)
    action_pattern = r"([A-Z]+)\s+\[([^\]]+)\](?:\s+\[(.*?)\])?"
    
    # 特殊匹配 STOP [answer] 或 CHECK [answer]
    stop_pattern = r"(STOP|CHECK)(?:\s+\[(.*?)\])?"

    for text in texts_to_check:
        text = text.strip()
        if not text: continue
        
        # A. 优先尝试匹配 STOP/CHECK (通常没有 ID)
        match_stop = re.search(stop_pattern, text, re.IGNORECASE)
        # 如果匹配到 STOP 且不是误判（例如文本中只是提到了 stop）
        # 这里加一个简单判断：如果 STOP 是由大写字母组成，或者在由 ``` 包裹的代码块中
        if match_stop:
             op = match_stop.group(1).upper()
             val = match_stop.group(2).strip() if match_stop.group(2) else ""
             return op, None, val

        # B. 匹配常规动作 (CLICK, TYPE, etc.)
        match = re.search(action_pattern, text, re.IGNORECASE)
        if match:
            op = match.group(1).upper()
            ele_id = match.group(2).strip()
            val = match.group(3).strip() if match.group(3) else ""
            
            # 过滤掉可能的占位符 (例如模型有时会输出 `goto [url]`)
            if ele_id.lower() in ['url', 'id']:
                continue 
                
            return op, ele_id, val

    return None, None, None

def calculate_metrics(results_file, log_filename):
    total_steps = 0
    total_tasks = 0
    
    element_acc_count = 0
    action_match_count = 0 
    step_success_count = 0
    task_success_count = 0
    
    print(f"正在评估文件: {results_file} ...")
    
    # 初始化日志文件
    with open(log_filename, 'w', encoding='utf-8') as log_file:
        log_file.write(f"Mind2Web 详细对比日志 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"输入文件: {results_file}\n\n")

    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                task_data = json.loads(line)
            except json.JSONDecodeError:
                continue
                
            trajectory = task_data.get("trajectory", [])
            annotation_id = task_data.get("annotation_id", "N/A")
            
            if not trajectory:
                continue

            total_tasks += 1
            is_task_success = True
            
            # 尝试提取任务描述
            first_input = trajectory[0].get('input_seq', '')
            task_desc = "N/A"
            if 'Objective:' in first_input:
                try:
                    task_desc = first_input.split('Objective:')[1].split('Observation:')[0].strip()
                except:
                    pass

            # --- 任务级日志 ---
            task_log_lines = [
                f"--- 任务ID: {annotation_id} ---",
                f"任务描述: {task_desc}"
            ]
            
            for step_idx, step in enumerate(trajectory):
                total_steps += 1
                
                # 1. 获取模型预测
                pred_raw = step.get("output_seq", "")
                pred_op, pred_id, pred_val = parse_model_output(pred_raw)
                
                # 2. 获取真值 (Ground Truth)
                gt = step.get("ground_truth", {})
                gt_op_dict = gt.get("operation", {})
                gt_candidates = gt.get("pos_candidates", [])
                
                gt_op = gt_op_dict.get("op", "").upper()
                gt_val = gt_op_dict.get("value", "")
                
                # 获取所有合法的目标 ID 列表
                valid_ids = [str(c.get("backend_node_id")) for c in gt_candidates]
                
                # --- 指标计算 ---
                
                # A. Element Accuracy (元素准确率)
                is_element_correct = False
                if pred_id and str(pred_id) in valid_ids:
                    is_element_correct = True
                elif not valid_ids and not pred_id:
                     # 特殊情况：如果真值无需元素（如 STOP），且预测也无元素，算对
                     is_element_correct = True

                # B. Action Match (操作匹配)
                is_op_correct = False
                if pred_op == gt_op:
                    if gt_op in ["CLICK", "HOVER", "ENTER", "STOP", "CHECK"]:
                        is_op_correct = True
                    else:
                        # TYPE 和 SELECT 的值匹配
                        # 宽松匹配: 预测包含真值，或真值包含预测，或完全相等
                        p_val_norm = str(pred_val).strip().lower()
                        g_val_norm = str(gt_val).strip().lower()
                        
                        if p_val_norm == g_val_norm:
                             is_op_correct = True
                        elif g_val_norm and g_val_norm in p_val_norm:
                             is_op_correct = True
                        elif p_val_norm and p_val_norm in g_val_norm:
                             is_op_correct = True
                
                # C. Step Success (步骤成功)
                is_step_success = is_element_correct and is_op_correct
                
                # 更新计数
                if is_element_correct:
                    element_acc_count += 1
                if is_op_correct:
                    action_match_count += 1
                if is_step_success:
                    step_success_count += 1
                else:
                    is_task_success = False

                # --- 记录当前步骤的详细对比到日志 ---
                # 截取原始输出的前200字符，去掉换行符以便日志整洁
                raw_preview = pred_raw.replace('\n', ' ')[:200]
                
                step_log_lines = [
                    f"  [步骤 {step_idx+1}] ------------------------------------------",
                    f"  原始输出(前200): {raw_preview}...", 
                    f"  预测解析: Op={pred_op} | ID={pred_id} | Val='{pred_val}'",
                    f"  真值数据: Op={gt_op} | ID={valid_ids} | Val='{gt_val}'",
                    f"  --- 结果 ---",
                    f"  Element: {'✅' if is_element_correct else '❌'} | Action: {'✅' if is_op_correct else '❌'} | Step: {'✅' if is_step_success else '❌'}",
                ]
                task_log_lines.extend(step_log_lines)
            
            if is_task_success:
                task_success_count += 1
            
            task_log_lines.append(f"  >>> 任务结果: {'✅ 成功' if is_task_success else '❌ 失败'}\n")

            # 实时写入日志
            with open(log_filename, 'a', encoding='utf-8') as log_file:
                 log_file.write("\n".join(task_log_lines) + "\n")

    # --- 打印最终结果 ---
    if total_steps == 0:
        print("未找到任何步骤数据。")
        return

    print("\n" + "="*40)
    print(f"Mind2Web 评估结果 (修正版)")
    print("="*40)
    print(f"总任务数 (Tasks): {total_tasks}")
    print(f"总步骤数 (Steps): {total_steps}")
    print(f"详细对比日志已输出至: {log_filename}")
    print("-" * 40)
    print(f"Element Acc (元素选择准确率): {element_acc_count / total_steps:.2%}")
    print(f"Action Score (动作+值准确率): {action_match_count / total_steps:.2%}")
    print(f"Step Success Rate (步骤成功率): {step_success_count / total_steps:.2%}")
    print(f"Task Success Rate (任务成功率): {task_success_count / total_tasks:.2%}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str, help='Path to the output .jsonl file')
    args = parser.parse_args()
    
    base_name = os.path.basename(args.file_path).replace('.jsonl', '')
    log_filename = f"{base_name}_compare_fixed.log"
    
    calculate_metrics(args.file_path, log_filename)