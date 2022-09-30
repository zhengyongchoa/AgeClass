# encoding: utf-8

import numpy as np
import _pickle as cPickle
from torch.utils.data import Dataset ,DataLoader
import  options as opt
import csv

def data_from_train():
    dataset = AudioData(file_list = opt.train_file )
    print('train data num_data:{}'.format(len(dataset.filenames)))
    loader = DataLoader(dataset,
                        num_workers=opt.num_workers, drop_last=True,
                        shuffle=True, batch_size= opt.batch_size)
    return (dataset, loader)

def data_from_valid():
    dataset = AudioData(file_list = opt.valid_file )
    print('valid data num_data:{}'.format(len(dataset.filenames)))
    loader = DataLoader(dataset,
                        num_workers=opt.num_workers, drop_last=True,
                        shuffle=True, batch_size= opt.batch_size)
    return (dataset, loader)


def data_from_test(batchsize= opt.test_batch_size , filelist = opt.test_file):
    dataset = AudioData( file_list = filelist)
    print('test data num_data:{}'.format(len(dataset.filenames)))

    loader = DataLoader(dataset, num_workers= opt.num_workers, drop_last= True,
                        shuffle=True, batch_size= opt.test_batch_size)
    return (dataset, loader)


class AudioData(Dataset):
    def __init__(self, file_list ):
        # image_filenames - 音频路径集合
        self.filenames = self.load_files( file_list)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        file_name = self.filenames[idx]
        dir = file_name[0]
        x = cPickle.load(open(dir, 'rb'))
        x = x.astype(np.float32)
        y = int(dir.split('/')[-1].split('.')[0].split('_')[0][0])
        y= y - 1  # 若num_classes=3，则y应为[0,1,2]
        return x ,y

    def load_files(self, csv_path):
        filpath = []
        if csv_path != "":
            f = open(csv_path)
            data = csv.reader(f)  #
            for line in data:
                tmp = line
                filpath.append(tmp)
        else:
            print('没有特征文件！')
        return filpath

