from datasets import load_dataset
import json

# --- 设置 ---
DATASET_NAME = "osunlp/Mind2Web"
SPLIT_NAME = "train"
# --- 结束设置 ---

# 尝试重新加载以确保 train_dataset 存在
print(f"--- 正在确保 {DATASET_NAME} 训练集已加载... ---")
try:
    train_dataset = load_dataset(DATASET_NAME, split=SPLIT_NAME)
    print(f"✅ 训练集加载成功。总实例数: {len(train_dataset)}")
except Exception as e:
    print(f"❌ 加载数据集失败，无法打印实例字段信息。错误: {e}")
    exit()

# 获取第一个实例 (索引 0)
first_instance = train_dataset[0]

print("\n--- Mind2Web 第一个实例的字段和子字段信息 ---")

def print_field_info(data, prefix=""):
    """
    递归打印字段名称和部分信息。
    """
    if isinstance(data, dict):
        for key, value in data.items():
            current_prefix = f"{prefix}{key}"
            
            # 对于字典本身，我们先打印它的键，然后递归处理它的值
            if not prefix: # 只有顶级字段才打印粗体
                 print(f"字段名: **{current_prefix}**")
            else:
                 print(f"  子字段: {current_prefix}")
            
            if isinstance(value, (dict, list)):
                # 嵌套结构不直接显示值，递归或显示长度
                if isinstance(value, dict):
                    print_field_info(value, prefix=f"{current_prefix}.")
                elif isinstance(value, list):
                    length = len(value)
                    first_element_type = type(value[0]).__name__ if length > 0 else 'N/A'
                    print(f"    信息: (列表, 长度: {length}, 包含 {first_element_type} 元素)")
                    # 仅查看第一个元素的结构，避免打印整个列表
                    if length > 0 and isinstance(value[0], dict):
                        # 递归查看列表中的第一个字典元素的子字段
                        print(f"    - 第一个元素的结构预览:")
                        print_field_info(value[0], prefix=f"{current_prefix}[0].")
            else:
                # 简单值，打印其预览信息
                preview = str(value)
                if isinstance(value, str) and len(preview) > 100:
                    preview = f"'{preview[:100]}...'"
                elif isinstance(value, str):
                    preview = f"'{preview}'"
                
                # 只有非嵌套字段才显示值预览
                if prefix:
                    print(f"    信息: {preview}")
                else:
                    print(f"  信息: {preview}")


# 从根目录开始打印
print_field_info(first_instance)

print("\n--- 打印结束 ---")