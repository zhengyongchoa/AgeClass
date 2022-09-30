import os
os.environ['OPENBLAS_NUM_THREADS'] = '0,1'
import torch
from torch import load
import options as opt
from model import Simple_Classify
import _pickle as cPickle
import numpy as np

if opt.test_device == 'GPU':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = 'cpu'

if __name__ == '__main__':
    # load Model:
    model = Simple_Classify().to(device)
    model.load_state_dict(load('./models/2class_cpu_epoch_14_acc_94.11_val_90.23_debug3.pth'))

    # load Data:
    datadir = '/home/zyc/zycode/PYcode/AgeRecognition/AgeClass/feature_2classify_3/test'
    names = [na for na in os.listdir( datadir ) if na.endswith(".p")]

    model.eval()
    with torch.no_grad():
        correct = 0
        sumcorrect =0
        for (idx, name) in enumerate(names):
            label = int(name.split('_')[0]) -1
            inputs = cPickle.load(open(os.path.join(datadir, name ), 'rb'))
            inputs = inputs.astype(np.float32)
            inputs = torch.tensor(inputs)
            inputs = inputs.unsqueeze(0)
            inputs = inputs.to(device)

            logit = model(inputs)
            # t1 =logit.data.cpu().numpy()
            predtxt = torch.argmax(logit.data, 1)
            predtxt =predtxt.data.cpu().numpy()
            if  predtxt == label:
                correct = 1
                print('序列：%2d, %s, %d  '%( idx+1, name,  predtxt+1))
                sumcorrect += correct

    acc = float(sumcorrect * 100 / (idx + 1) )
    print('the test accuracy is: ', acc)





