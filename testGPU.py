from tqdm import tqdm
import torch , time



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print(torch.__version__)
print("GPU型号： ",torch.cuda.get_device_name(0))

a = torch.randn(10000, 1000)
b = torch.randn(1000, 2000)

t0 = time.time()
c = torch.matmul(a, b)
t1 = time.time()
print(a.device,",time=", t1 - t0, c.norm(2))

device = torch.device('cuda')
a = a.to(device)
b = b.to(device)

# print('t0:%d', time.time())
t0 = time.time()
c = torch.matmul(a, b)
t2 = time.time()
print(a.device,",time=", t2 - t0, c.norm(2))

t0 = time.time()
c = torch.matmul(a, b)
t2 = time.time()
print(a.device,",time=", t2 - t0, c.norm(2))

for i in tqdm(range(10000)):
     time.sleep(0.01)