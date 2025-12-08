from datasets import load_dataset
import pandas as pd # 导入 pandas 方便查看数据结构

# --- 用户设置 ---
# 数据集在 Hugging Face 上的名称
DATASET_NAME = "osunlp/Mind2Web"

# 要获取的拆分：仅训练集
SPLIT_NAME = "train"

# 您想查看的实例数量
NUM_INSTANCES_TO_SHOW = 5
# --- 结束设置 ---


print(f"--- 正在从 Hugging Face 加载 {DATASET_NAME} 数据集的 {SPLIT_NAME} 拆分... ---")

try:
    # 加载数据集。这会自动下载数据到本地缓存
    train_dataset = load_dataset(DATASET_NAME, split=SPLIT_NAME)

    print(f"\n✅ 数据集加载成功!")
    print(f"总实例数（训练集）: {len(train_dataset)}")
    print(f"特征 (Schema): {train_dataset.column_names}")
    
    # 转换为 Pandas DataFrame，方便查看前几条数据
    # 取前 NUM_INSTANCES_TO_SHOW 个实例
    df = pd.DataFrame(train_dataset[:NUM_INSTANCES_TO_SHOW])

    print(f"\n--- 前 {NUM_INSTANCES_TO_SHOW} 个实例的关键信息 ---")
    
    # 打印关键字段，例如任务ID、网站和任务描述
    for i in range(len(df)):
        instance = df.iloc[i]
        
        # 任务描述可能位于 'confirmed_task' 字段
        task_description = instance.get('confirmed_task', 'N/A')
        website = instance.get('website', 'N/A')
        num_actions = len(instance.get('actions', [])) # 计算轨迹中的操作步数
        
        print(f"\n[{i+1}] 任务ID: {instance.get('annotation_id', 'N/A')}")
        print(f"  任务描述: {task_description[:80]}...") # 限制长度防止输出过多
        print(f"  网站: {website}")
        print(f"  操作步骤数 (Actions): {num_actions}")

except Exception as e:
    print(f"\n❌ 加载数据集时出错: {e}")
    print("请检查您的网络连接或确保 datasets 库已正确安装。")