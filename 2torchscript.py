import  options as opt
import torch
from model import  Simple_Classify
from torch import optim ,save ,load


if opt.device_mode =='GPU':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = 'cpu'

if __name__ == '__main__':
    # model = Simple_Classify( ).to(device)
    model = Simple_Classify( )

    model.load_state_dict(load('./models/2class_cpu_epoch_14_acc_94.11_val_90.23_debug3.pth'))
    model.eval()
    dummy_input = torch.rand( 1 , 21 , 63)

    # print(model)

    with torch.no_grad():
        jit_model = torch.jit.trace(model, dummy_input)
        jit_model.save('models/2class_cpu_cpp_debug04.pt')



    # load_jit_model = torch.jit.load('models/2class_cpu_cpp.pt')
    # print(load_jit_model(torch.rand(1, 21, 63)))
    # print(load_jit_model.code)
