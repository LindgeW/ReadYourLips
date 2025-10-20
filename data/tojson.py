import json

# 输入txt文件路径（请替换为实际路径）
txt_file_path = "unseen_cmd.txt"
# 输出json文件路径
json_file_path = "unseen_cmd.json"

# 初始化存储数据的字典
command_dict = {}

# 读取txt文件并处理
with open(txt_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        # 去除每行首尾的空白字符（如换行符、空格）
        line = line.strip()
        if not line:  # 跳过空行
            continue
        # 按分号分割为键和值
        key_value = line.split(';', 1)  # 限制分割1次，避免值中含分号的情况
        if len(key_value) == 2:
            key, value = key_value
            command_dict[key.strip()] = value.strip()  # 去除键值两边的空格
        else:
            print(f"警告：跳过格式不正确的行 - {line}")

# 将字典转换为JSON并保存到文件
with open(json_file_path, 'w', encoding='utf-8') as f:
    json.dump(command_dict, f, ensure_ascii=False)

print(f"转换完成，JSON文件已保存至：{json_file_path}")

