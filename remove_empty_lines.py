#!/usr/bin/env python3
"""
删除 JSONL 文件中的空行
用法: python remove_empty_lines.py <input_file> [output_file]
如果未指定 output_file，将覆盖原文件
"""
import sys
import json
import argparse
from pathlib import Path


def remove_empty_lines(input_file, output_file=None):
    """
    删除 JSONL 文件中的空行
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径，如果为 None 则覆盖原文件
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"错误: 文件 '{input_file}' 不存在")
        return False
    
    # 如果未指定输出文件，使用临时文件然后覆盖
    if output_file is None:
        output_file = str(input_path) + '.tmp'
        overwrite = True
    else:
        overwrite = False
    
    valid_lines = []
    empty_count = 0
    total_lines = 0
    
    # 读取文件，跳过空行
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                total_lines += 1
                stripped_line = line.strip()
                
                # 跳过空行
                if not stripped_line:
                    empty_count += 1
                    continue
                
                # 验证是否为有效的 JSON（可选，用于检查格式）
                try:
                    json.loads(stripped_line)
                    valid_lines.append(line)  # 保留原始行（包括换行符）
                except json.JSONDecodeError as e:
                    print(f"警告: 第 {line_num} 行不是有效的 JSON: {e}")
                    print(f"      内容: {stripped_line[:50]}...")
                    # 可以选择跳过或保留
                    # valid_lines.append(line)  # 如果想保留无效行，取消注释
    
    except Exception as e:
        print(f"错误: 读取文件时出错: {e}")
        return False
    
    # 写入输出文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(valid_lines)
        
        # 如果覆盖原文件，移动临时文件
        if overwrite:
            import shutil
            shutil.move(output_file, str(input_path))
            output_file = str(input_path)
        
        print(f"成功处理文件: {input_file}")
        print(f"  总行数: {total_lines}")
        print(f"  空行数: {empty_count}")
        print(f"  有效行数: {len(valid_lines)}")
        print(f"  输出文件: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"错误: 写入文件时出错: {e}")
        if overwrite and Path(output_file).exists():
            Path(output_file).unlink()  # 删除临时文件
        return False


def main():
    parser = argparse.ArgumentParser(
        description='删除 JSONL 文件中的空行',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python remove_empty_lines.py data.jsonl
  python remove_empty_lines.py data.jsonl output.jsonl
  python remove_empty_lines.py data.jsonl --in-place
        """
    )
    
    parser.add_argument(
        'input_file',
        type=str,
        help='输入的 JSONL 文件路径'
    )
    
    parser.add_argument(
        'output_file',
        type=str,
        nargs='?',
        default=None,
        help='输出文件路径（可选，如果不指定则覆盖原文件）'
    )
    
    parser.add_argument(
        '--in-place',
        action='store_true',
        help='直接修改原文件（与不指定 output_file 效果相同）'
    )
    
    args = parser.parse_args()
    
    # 如果指定了 --in-place，忽略 output_file
    if args.in_place:
        output_file = None
    else:
        output_file = args.output_file
    
    success = remove_empty_lines(args.input_file, output_file)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()





