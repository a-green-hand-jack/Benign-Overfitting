# 这个模块就是为了处理那些pre process 得到数据，然后绘制

# 首先需要的是得到pkl路径下的文件
# 然后根据传入的参数得到我感兴趣的那部分
#

# %% 加载目标pkl文件
import pickle

def load_and_print_pkl(file_path):
    try:
        # 使用pickle加载Pickle文件
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            print("Contents of the Pickle file:")
            print(data)
    except FileNotFoundError:
        print("File not found. Please provide a valid file path.")
    except Exception as e:
        print("An error occurred:", e)

# 用法示例：替换为你的Pickle文件路径
pkl_file_path = '.\pre_process_outputs\angle\MLP\0\BOF.pkl'

# 调用函数加载并打印Pickle文件内容
load_and_print_pkl(pkl_file_path)

# %% 
























