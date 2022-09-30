import os
os.environ['OPENBLAS_NUM_THREADS'] = '0,1'
import time
import torch
import torch.nn as nn
from torch import optim ,save ,load
from data import data_from_train  ,data_from_valid

from model import  Simple_Classify
from tensorboardX import SummaryWriter
import  options as opt

# 调试时需要选择CPU模式：
if opt.device_mode =='GPU':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = 'cpu'
print('Using device:', device)





if __name__ == '__main__':
    model = Simple_Classify( ).to(device)
    if False:
        model.load_state_dict(load('./weights/epoch_49_acc_98.28.pt'))
    else:
        model.apply(model.weights_init)

    writer = SummaryWriter()

    (train_dataset, train_loader) = data_from_train()
    (valid_dataset, valid_loader) =  data_from_valid()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr= 1e-3 )
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)

    iteration = 0
    m = 0

    for epoch in range(opt.max_epoch):
        start_time = time.time()
        exp_lr_scheduler.step()
        batchcorrect =0
        val_correct = 0
        for (i, batch) in enumerate(train_loader):
            time.sleep(0.01)
            (inputs, label) = batch[0].to(device), batch[1].to(device)
            # print(inputs.size())
            logit = model(inputs)
            loss = criterion(logit, label)  # 计算两者的误差
            optimizer.zero_grad() # 清空上一步的残余更新参数值,loss关于weight的导数变成0
            loss.backward() # 沿着计算图反向传播
            optimizer.step() #更新参数

            iteration += 1

            tot_iter = epoch*len(train_loader)+i
            writer.add_scalar('data/CE_loss', loss.detach().cpu().numpy(), iteration)
            train_loss = loss.item()

            # 每个epoch内设置 [batch * N] 打印一次loss
            if i% 10==0:
                print('epoch:%d, bath :%d, train_loss:%.6f'%(epoch ,i, train_loss))

            #学习率的变化
            if False:
                print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))
            # 每个batch的准确率
            predicted = torch.argmax(logit.data,1)
            batchcorrect += (predicted == label).sum()

            n = int(len(train_dataset.filenames) / opt.batch_size)
            if (i+1)==n:
                tra_acc = float(batchcorrect*100)/float(opt.batch_size*n)
                # print('epoch:%d,  train_loss:%.6f, Accuracy:%.2f' % (epoch, train_loss , tra_acc))  # 如果没有验证集可放开print
                break

        end_time = time.time()
        T_time = end_time - start_time
        # 计算每个epoch中验证集的准确率
        with torch.no_grad(): # 测试条件下
            corrects = 0
            for (idx,batch) in enumerate(valid_loader):
                (inputs, label) = batch[0].to(device), batch[1].to(device)
                logit = model(inputs)
                predtxt = torch.argmax(logit.data, 1)
                val_correct += (predtxt == label).sum().cpu()

            valid_Acc = 100 * val_correct / ( (idx+1)*opt.batch_size)
            # writer.add_scalar('acc/accuracy', acc, m)
            print('epoch:%d, loss:%.6f, trian_Acc:%.2f, valid_Acc:%.2f'  % (epoch, train_loss, tra_acc , valid_Acc))

    savename = os.path.join(opt.savemodel_dir, '2class_{}_epoch_{}_acc_{:.2f}_val_{:.2f}_debug3.pth'.format(device, epoch, tra_acc, valid_Acc) )
    savepath = opt.savemodel_dir
    if(not os.path.exists(savepath)): os.makedirs(savepath)
    torch.save(model.state_dict(), savename)
