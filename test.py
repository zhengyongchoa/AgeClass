# 说明：
# test：基于GPU/CPU 快速评估model在测试集的准确率
# testflask1：对文件所有的音频进行测试
# testflask2：对单独的wav文件进行测试，目前只完成到：input为特征数据. 将来检查异常音频



import os
os.environ['OPENBLAS_NUM_THREADS'] = '0,1'
import torch
from torch   import load
import torch.nn.functional as F
from data import  data_from_test
import  options as opt
from model import Simple_Classify

if opt.test_device == 'GPU':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device ='cpu'


if __name__ == '__main__':
    model = Simple_Classify( ).to(device)
    model.load_state_dict(load('./weights/2class_cuda_epoch_39_acc_97.57_valAcc_86.46.pt'))
    (test_dataset, test_loader) = data_from_test()

    # model.eval()
    with torch.no_grad():
        correct = 0
        sumcorrect = 0
        for (idx, batch) in enumerate(test_loader):
            (inputs, label) = batch[0].to(device), batch[1].to(device)
            logit = model(inputs)
            predtxt = torch.argmax(logit.data, 1)
            correct = (predtxt == label).sum().cpu() .numpy()
            # print(idx)
            # print(correct)
            acc = float(correct * 100 / opt.batch_size)
            print(acc)
            sumcorrect = sumcorrect  +  correct


    acc  = float(  sumcorrect *100 / (idx+1)/ opt.batch_size ) # 存在问题
    print( 'the test acc is: ', acc)



                
                
