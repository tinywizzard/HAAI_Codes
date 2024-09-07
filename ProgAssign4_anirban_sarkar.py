"""
This program is a partial MNIST classifier using AlexNet. It accepts three parameters provided as a command line input. The first two inputs are two digits between 0-9 which are used to train and test the classifier and the third parameter controls the number of training epochs.
Syntax: python assignment.py <number> <number> <number>

For example, to train and test AlexNet with 1 and 2 MNIST samples with 4 training epochs, the command line input should be:
python assignment.py 1 2 4
"""

"""
ALERT: * * * No changes are allowed to import statements  * * *
"""
import sys
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim


"""
* * * Changes allowed from here  * * *
"""

class AlexNet(nn.Module):   
    def __init__(self, num=10):
        super(AlexNet, self).__init__()
        self.feature = nn.Sequential(
            # Define feature extractor here...
        )
        self.classifier = nn.Sequential(
            # Define classifier here...
        )
    
    def forward(self, x):
        # define forward network 'x' that combines feature extractor and classifier
        return x

"""
ALERT: * * * No changes are allowed after this comment  * * *
"""

def load_subset(full_train_set, full_test_set, label_one, label_two):
    # Sample the correct train labels
    train_set = []
    data_lim = 20000
    for data in full_train_set:
        if data_lim>0:
            data_lim-=1
            if data[1]==label_one or data[1]==label_two:
                train_set.append(data)
        else:
            break

    test_set = []
    data_lim = 1000
    for data in full_test_set:
        if data_lim>0:
            data_lim-=1
            if data[1]==label_one or data[1]==label_two:
                test_set.append(data)
        else:
            break

    return train_set, test_set

def train(model,optimizer,train_loader,epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

def test(model,test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target, reduction='sum').item()#size_average=False
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
    
    test_loss /= len(test_loader.dataset)
    acc=100. * float(correct.to(torch.device('cpu')).numpy())
    test_accuracy = (acc / len(test_loader.dataset))
    return test_accuracy

if __name__ == '__main__':
 
    if len(sys.argv) == 3:
        print("Usage: python assignment.py <number> <number>")
        sys.exit(1)

    input_data_one = sys.argv[1].strip()
    input_data_two = sys.argv[2].strip()
    epochs = sys.argv[3].strip()
    
    """  Call to function that will perform the computation. """
    if input_data_one.isdigit() and input_data_two.isdigit() and epochs.isdigit():

        label_one = int(input_data_one)
        label_two = int(input_data_two)
        epochs = int(epochs)
        
        if label_one!=label_two and 0<=label_one<=9 and 0<=label_two<=9:
            torch.manual_seed(42)
            # Load MNIST dataset
            trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
            full_train_set = dset.MNIST(root='./data', train=True, transform=trans, download=True)
            full_test_set = dset.MNIST(root='./data', train=False, transform=trans)
            batch_size = 16
            # Get final train and test sets
            train_set, test_set = load_subset(full_train_set,full_test_set,label_one,label_two)
            
            train_loader = torch.utils.data.DataLoader(dataset=train_set,batch_size=batch_size,shuffle=False)
            test_loader = torch.utils.data.DataLoader(dataset=test_set,batch_size=batch_size,shuffle=False)

            model = AlexNet()
            if torch.cuda.is_available():
                model.cuda()
                    
            optimizer = optim.SGD(model.parameters(), lr=0.01)
            
            for epoch in range(1, epochs+1):
                train(model,optimizer,train_loader,epoch)
                accuracy = test(model,test_loader)

            print(round(accuracy,2))
            
            
        else:
           print("Invalid input")
    else:
        print("Invalid input")
 
    
    """ End to call """