import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as thdat
import numpy as np
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(DEVICE)
def np_to_th(x):
    n_samples = len(x)
    return torch.from_numpy(x).to(torch.float).to(DEVICE).reshape(n_samples, -1)

def nderv(f, wrt, n):
    for i in range(n):
        grads = torch.autograd.grad(f, wrt, create_graph=True)[0]
        f = grads.sum()

    return grads
    

class Net(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        n_units=250,
        epochs=5000,
        loss=nn.MSELoss(),
        lr=1e-3,
        loss2=None,
        loss2_weight=0.8,
    ) -> None:
        super().__init__()

        self.epochs = epochs
        self.loss = loss
        self.loss2 = loss2
        self.loss2_weight = loss2_weight
        self.lr = lr
        self.n_units = n_units

        self.layers = nn.Sequential(
            nn.Linear(input_dim, self.n_units),
            nn.Tanh(),
            nn.Linear(self.n_units, self.n_units),
            nn.Tanh(),
            nn.Linear(self.n_units, self.n_units),
            nn.Tanh(),
            nn.Linear(self.n_units, self.n_units),
            nn.Tanh(),
        )
        self.out = nn.Linear(self.n_units, output_dim)

    def forward(self, x):
        h = self.layers(x)
        out = self.out(h)

        return out

    def fit(self, X, y):
        Xt = np_to_th(X)
        yt = np_to_th(y)
        
        Xt.requires_grad=True

        optimiser = optim.Adam(self.parameters(), lr=self.lr)
        self.train()
        losses = []
        for ep in range(self.epochs):
            optimiser.zero_grad()
            outputs = self.forward(Xt)
            loss = self.loss(yt, outputs)
            if self.loss2:
                for i in range(1,len(Xt)):
                	loss += self.loss2_weight * (((nderv(outputs.sum(),Xt,1)[i].item())-torch.cos(2*np.pi*Xt[i]).item())**2)/len(Xt)    
            loss.backward()
            optimiser.step()
            losses.append(loss.item())
            if ep % int(self.epochs / 10) == 0:
                print(f"Epoch {ep}/{self.epochs}, loss: {losses[-1]:.2f}")
        return losses

    def predict(self, X):
        self.eval()
        out = self.forward(np_to_th(X))
        return out.detach().cpu().numpy()

        
'''
 =========================================================================================

'''
# Masses
m1 = 1.0
m2 = 1.0
# Spring constants
k1 = 5.0
k2 = 2.0
# Natural lengths
L1 = 0.5
L2 = 0.5
# Initial conditions
# d1_0 and d2_0 are the initial displacements; v1_0 and v2_0 are th
d1_0 = 0.5
d2_0 = 3.25
v1_0 = 0.0
v2_0 = 0.0
b1 = 0
b2 = 0





train_t = np.linspace(0, 2.5, 20)
test_t = np.linspace(0, 2.5,200)
train_u = np.sin(2*np.pi*train_t)/(2*np.pi) + 1
NN=Net(1,1,loss2=True)
NN.fit(train_t,train_u,)



true_u = np.sin(2*np.pi*test_t)/(2*np.pi) + 1
pred_u = NN.predict(test_t)


plt.figure(figsize = (10,8))
plt.plot(train_t, train_u, 'ok', label = 'Train')
plt.plot(test_t, true_u, '-k',label = 'True')
plt.plot(test_t, pred_u, '--r', label = 'Prediction')
plt.legend(fontsize = 15)
plt.xlabel('t', fontsize = 15)
plt.ylabel('u', fontsize = 15)
plt.show()
 
 
