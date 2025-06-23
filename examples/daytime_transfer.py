# author:"flt"
# data:8/22/2024 10:02 PM
import pandas as pd

# 读取CSV文件
df = pd.read_csv('transfer_output.csv')

# 定义一个函数来将 daytime 映射到 1, 2, 3
def map_daytime_to_period(daytime):
    # 提取小时部分
    hour = int(daytime.split(":")[0])
    if 6 <= hour < 12:
        return 1
    elif 12 <= hour < 19:
        return 2
    else:
        return 3

# 应用映射函数
df['daytime'] = df['daytime'].apply(map_daytime_to_period)

# 可以选择将结果保存到新文件
df.to_csv('filtered_output.csv', index=False)