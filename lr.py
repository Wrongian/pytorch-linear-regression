import pandas as pd
import torch
import matplotlib.pyplot as plt

#import dataset 
df_path = "datasets/salary.csv"
df = pd.read_csv(df_path)
df = df.drop(columns=["i"])
df = df.astype(float)
dataset = torch.tensor(df.values).reshape((-1,2))
x = dataset[:, : 1]
y = dataset[:, 1: 2]

#model
epochs = 100
learning_rate = 0.01
class LR(torch.nn.Module):
    def __init__(self,inputSize,outputSize):
        super(LR,self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize).double()
        
    def forward(self, x):
        y = self.linear(x)
        return y
    

#linear regression
model = LR(1,1)
criterion = torch.nn.MSELoss()
optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)

#gpu speedup
# if torch.cuda.is_available():
#     model.cuda()
#     x = x.cuda()
#     y = y.cuda()

prediction = model(x).detach()
#before
plt.plot(x.numpy(),prediction.numpy(),"--",label="Before",alpha = 0.5)
for epoch in range(epochs):
    #zero grads from last epoch
    optimiser.zero_grad()
    
    #forward
    outputs = model(x)

    #get mse loss
    loss = criterion(outputs, y)

    #backpropogation based on loss
    loss.backward()

    #update based on grads
    optimiser.step()
    
    print(f"epoch:{epoch} loss {loss.item()}")

prediction = model(x).detach()
# plt.clf()
plt.plot(x.numpy(),y.numpy(),"go",label="Data",alpha = 0.5)
plt.plot(x.numpy(),prediction.numpy(),"--",label="After",alpha = 0.5)
plt.legend(loc= "best")
plt.show()