import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
from tqdm import tqdm

class Net(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.fc1 = nn.Linear(4096, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x), 2)
        x = F.relu(F.max_pool2d(self.conv2(x), 2)) 
        x = F.relu(self.conv3(x), 2)
        x = F.relu(F.max_pool2d(self.conv4(x), 2)) 
        x = x.view(-1, 4096)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class Model():
    def __init__(self, train_loader, test_loader, lr=0.01, momentum=0.9):
        self.init_cuda()
        self.model = Net().to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr, momentum)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=15, gamma=0.1, verbose=True)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.results = {
            "train_losses": [],
            "test_losses": [],
            "train_acc": [],
            "test_acc": []
        }
    
    @staticmethod
    def check_cuda_availability():
        use_cuda = torch.cuda.is_available()
        print("CUDA Available?", use_cuda)
  
    def init_cuda(self):
        use_cuda = torch.cuda.is_available()
        print("CUDA Available?", use_cuda)

        self.device = torch.device("cuda" if use_cuda else "cpu")
    
    def train(self):
        self.model.train()
        pbar = tqdm(self.train_loader)

        train_loss = 0
        correct = 0
        processed = 0

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            # Predict
            pred = self.model(data)

            # Calculate loss
            loss = F.cross_entropy(pred, target)
            train_loss+=loss.item()

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            
            correct += self.get_correct_predicted_count(pred, target)
            processed += len(data)

            pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

        self.results["train_acc"].append(100*correct/processed)
        self.results["train_losses"].append(train_loss/len(self.train_loader))
    
    def test(self):
        self.model.eval()
        
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                test_loss += F.cross_entropy(output, target).item()  # sum up batch loss

                correct += self.get_correct_predicted_count(output, target)
        
        test_loss /= len(self.test_loader.dataset)
        self.results["test_acc"].append(100. * correct / len(self.test_loader.dataset))
        self.results["test_losses"].append(test_loss)

        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(self.test_loader.dataset),
        100. * correct / len(self.test_loader.dataset)))
    

    def get_correct_predicted_count(self, pPrediction, pLabels):
        return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

    def run(self, num_epochs = 10):
        for epoch in range(1, num_epochs+1):
            print(f'Epoch {epoch}')
            self.train()
            self.test()
            self.scheduler.step()
    
    def show_summary(self):
        summary(self.model, input_size=(1 ,28 ,28))

