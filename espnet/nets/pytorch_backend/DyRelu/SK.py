import torch
import torch.nn as nn


class SKLayer(nn.Module):
    def __init__(self, dim):
        super(SKLayer, self).__init__()

        assert dim % 8 == 0, 'Dimension should be divisible by 8.'
        self.dim = dim

        self.reduce_linear = nn.Linear(dim, dim // 8)
        self.act = nn.ReLU()
        self.span_linear1 = nn.Linear(dim // 8, dim)
        self.span_linear2 = nn.Linear(dim // 8, dim)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs1, inputs2):

        inputs1 = inputs1.unsqueeze(1) # Bx1xTxD
        inputs2 = inputs2.unsqueeze(1) # Bx1xTxD

        inputs = torch.cat((inputs1, inputs2), 1) # Bx2xTxD
        x = inputs.sum(dim=1, keepdim=True).mean(dim=2, keepdim=True) # Bx1x1xD
        x = self.act(self.reduce_linear(x)) # Bx1x1xD_reduce
        
        x1 = self.span_linear1(x)  # Bx1x1xD
        x2 = self.span_linear2(x)  # Bx1x1xD

        x = torch.cat((x1,x2), 1)  # Bx2x1xD
        weight = self.softmax(x)  # Bx2x1xD
        
        return torch.mul(weight, inputs).sum(dim=1)  # BxTxD

