import os

def get_list(save_dir, category):
    """
    将b文件路径及标签写入txt
    :param save_dir: 保存文件路径
    :param usage: 数据集类别
    :return:
    """
    tname = os.path.join(save_dir, category + "_list.txt")
    data_dir = os.path.join(save_dir, category)
    # 获取子文件夹下的文件列表
    sub_dir = os.listdir(data_dir)

    with open(tname, "w") as f:
        for i, subname in enumerate(sub_dir):
            subpath = os.path.join(data_dir, subname)
            for filename in os.listdir(subpath):
                # 文件路径 标签
                # line = os.path.join(root_dir, subpath, filename) + " " + str(sub_dir[i]) + "\n"
                line = os.path.join(root_dir, subpath, filename) + "\n"
                f.write(line)

root_dir = os.path.abspath(os.getcwd())
# root_dir = os.path.abspath(os.path.join(os.getcwd(), "../.."))



if __name__ == "__main__":
    category ='train'
    data_dir = 'wav_2classify'
    get_list(data_dir, category)


