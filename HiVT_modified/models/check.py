import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

class check(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features=3, out_features=1),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        return self.network(x)

class Node:
    def __init__(self, value):
        self.value = value
        self.children = []
    
    def add_children(self, children, weight):
        self.children.append((children, weight))

class PR:
    def __init__(self):
        self.queue= []
    
    def push(self, item, priority):
        if len(self.queue) == 0:
            self.queue.append((item, priority))
        else:
            for i in range(self.queue):
                if priority < self.queue[i][0]

# class SubModule1(torch.nn.Module):
#     def __init__(self):
#         super(SubModule1, self).__init__()
#         self.conv1 = torch.nn.Conv2d(1, 20, 5)

#     def forward(self, x):
#         with record_function("submodule1_forward"):
#             x = self.conv1(x)
#         return x

# class SubModule2(torch.nn.Module):
#     def __init__(self):
#         super(SubModule2, self).__init__()
#         self.conv2 = torch.nn.Conv2d(20, 20, 5)

#     def forward(self, x):
#         with record_function("submodule2_forward"):
#             x = self.conv2(x)
#         return x

# class MyModel(torch.nn.Module):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         self.module1 = SubModule1()
#         self.module2 = SubModule2()

#     def forward(self, x):
#         x = self.module1(x)
#         x = self.module2(x)
#         return x

# # Create the model and inputs
# model = MyModel()
# inputs = torch.randn(5, 1, 28, 28)

# # Enable the profiler
# with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
#     model(inputs)

# # Print the profiling results
# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
