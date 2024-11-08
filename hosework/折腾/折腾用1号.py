import os


def get_data_loader(data_folder, is_train, batch_size=100, shuffle=True):
    # 使用条件表达式直接在字典中生成文件名
    data_dict = {
        'data_name': 'train-images-idx3-ubyte.gz' if is_train else 't10k-images-idx3-ubyte.gz',
        'label_name': 'train-labels-idx1-ubyte.gz' if is_train else 't10k-labels-idx1-ubyte.gz'
    }

    # 现在可以使用 data_dict['data_name'] 和 data_dict['label_name'] 来访问文件名
    data_path = os.path.join(data_folder, data_dict['data_name'])
    label_path = os.path.join(data_folder, data_dict['label_name'])

    # 接下来，你可以使用 data_path 和 label_path 来加载数据
    # ... 加载数据的代码 ...

    return data_path, label_path  # 或者返回一个包含这些路径的数据加载器对象

# 示例调用
data_folder = '/path/to/data'
is_train = True
data_path, label_path = get_data_loader(data_folder, is_train)
print(data_path)  # 输出: /path/to/data/train-images-idx3-ubyte.gz
print(label_path)  # 输出: /path/to/data/train-labels-idx1-ubyte.gz