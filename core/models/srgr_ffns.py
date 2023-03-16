import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    
def make_model(depth, width):
    return nn.Sequential(
        *(
            [Flatten(), nn.Linear(784, width), nn.ReLU()] + 
            sum([[nn.Linear(width, width), nn.ReLU()] for _ in range(depth - 1)], []) + 
            [nn.Linear(width, 10)]
        )
    )

def ffn2():
    return make_model(6, 20)

def ffn3():
    return make_model(3, 50)

def ffn4():
    return make_model(3, 100)

def ffn5():
    return make_model(6, 100)

def ffn6():
    return make_model(6, 200)

def ffn7():
    return make_model(9, 200)

def ffn8():
    return make_model(5, 600)