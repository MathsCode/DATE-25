import torch

N = 1
d = 4096
D = 14336 # 1024*14

a = torch.randn(N, d).cuda()
b = torch.randn(d, D).cuda()

torch.cuda.cudart().cudaProfilerStart() 
c = torch.matmul(a, b)
torch.cuda.cudart().cudaProfilerStop()