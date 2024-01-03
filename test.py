import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn as nn


NUM = 50


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(NUM, 100),
            nn.Tanh()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(100, 150),
            nn.Tanh()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(150, 100),
            nn.Tanh()
        )
        self.fc4 = nn.Sequential(
            nn.Linear(100, NUM),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

        return x


PATH = './net.pth'
checkpoint = torch.load(PATH)
net = Net()
net.load_state_dict(checkpoint)


# 时间长度
N = 4000
t = [0]
for i in range(1, N):
    t.append(t[-1] + 1)


def v_est(t):
    if 1000 < t < 3000:
        return 10 * abs(np.sin(t / (50 * np.pi)))
    else:
        return 0

v = np.array([v_est(ti) for ti in t], dtype=np.float64)

alpha0 = 1
alpha1 = 0.5
alpha2 = 0.1

m_est = [20000]  #肥料初始重量
for k in range(1, N):
    m_k = m_est[k - 1] - (alpha0 + alpha1 * v[k - 1] + alpha2 * v[k - 1] ** 2) * (t[k] - t[k-1])
    if m_k < 0:
        m_k = 0
    m_est.append(m_k)

#设置高斯白噪声
np.random.seed(1)
m_noise = np.random.normal(0, 500, N)
m_mea = m_est + m_noise    #测量重量值


data = np.zeros([1, NUM])
mn_max = max(m_mea)
mn_min = min(m_mea)
me_max = max(m_est)
me_min = min(m_est)
result = []


for i in range(N):
    data[0][i%NUM] = 2 * (m_mea[i] - mn_min)/(mn_max - mn_min) - 1
    if i%NUM == NUM - 1 or i == N - 1:
        data1 = torch.tensor(data).float()
        b_x = Variable(data1)
        output = net(b_x)
        output1 = output.detach().numpy()
        for j in range(NUM):
            if len(result) < N:
                result.append((output1[0][j] + 1) * (me_max - me_min)/2 + me_min)
        data = np.zeros([1, NUM])


fig, ax1= plt.subplots(1, 1, figsize=(8, 8))

ax1.plot(t, m_mea, 'lightblue')
ax1.plot(t, m_est, 'k')
ax1.plot(t, result, 'r')


plt.show()






