import numpy as np
from scipy.io import loadmat
import csv
import torch as pt
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv1d(3, 6, 5)
#         self.pool = nn.MaxPool1d(2, 2)
#         self.conv2 = nn.Conv1d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# loaddata takes in the file names of training and testing data and returns
# tensors xTr, yTr, and xTe.
def loaddata(training,testing):
	training_data = loadmat(training)
	testing_data = loadmat(testing)
	xTr = training_data["x"]
	yTr = np.round(training_data["y"])
	xTe = testing_data["x"]
	xTr_tensor = pt.from_numpy(xTr)
	yTr_tensor = pt.from_numpy(yTr)
	xTe_tensor = pt.from_numpy(xTe)

	return xTr_tensor,yTr_tensor,xTe_tensor


def train(xTr,yTr,net):
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
		
	running_loss = 0.0
	d,n = xTr.size()
	for i in range(n):
		inputs = xTr[i]
        labels = yTr[i]
        # print(xTr[i])
        # print(yTr[i])
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.data[0]
        if i % 700 == 699:
        	print (running_loss)
        	running_loss = 0.0		


def test(xTe,w):
	return 0

if __name__ == "__main__":
	xTr,yTr,xTe = loaddata("train.mat","test.mat")
	print(xTr.size())
	net = Net()
	train(xTr, yTr, net)
	
	#need to write to the csv file.
