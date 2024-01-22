import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchsummary

# Define model
class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# torch.manual_seed(0)

# Initialize model
model = TheModelClass()
for param in model.parameters():
    param.data = nn.parameter.Parameter(torch.ones_like(param) * 0.1)
    print(param.shape)

# Initialize optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

# torch.save(model, 'foo.pkl')

x = torch.ones(3, 32, 32)
# print('x =', x)
print('model(x) =', model(x))

traced_script_module = torch.jit.trace(model, x)
print('traced_script_module(x) =', traced_script_module(x))
# print('traced_script_module =', traced_script_module)

traced_script_module.save('model.pt')

torchsummary.summary(model, input_size=(3,32,32))
