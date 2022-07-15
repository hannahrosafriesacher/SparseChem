import torch
import numpy as np

class SmallModel(torch.nn.Module):

    def __init__(self, hidden_size, dropout):
        super(SmallModel, self).__init__()

        self.linear1 = torch.nn.Linear(1, hidden_size)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!???????
        x = self.dropout(x)
        x = self.linear2(x)
        return x


net=SmallModel()
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!????
optimizer=torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
loss=torch.nn.BCELoss()
#!!!!!
epoch=20

for epoch in range(0, epoch):
    print('Epoch: ', epoch)
    running_loss=0
    for datapoint in range(0, x.shape[0]):
        inputs=x[datapoint]
        print(input.shape)
        labels=y[datapoint]
        print(labels, shape)

        optimizer.zero_grad()

        outputs=net(inputs)
        BCE_loss=loss(outputs, labels)
        BCE_loss.backward()
        optimizer.step()

        running_loss+=loss

        # when to stop?