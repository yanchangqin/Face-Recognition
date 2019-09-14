import torch
import torch.nn as nn

class CenterLoss(nn.Module):
    def __init__(self,cls_num,feature_num):
        super().__init__()
        self.center = nn.Parameter(torch.randn(cls_num,feature_num))

    def forward(self, xs,label):
        batch_size = xs.size(0)
        self.batch_size_tensor = xs.new_empty(1).fill_(batch_size)
        # print(batch_size)
        # print(batch_size_tensor)
        # print(self.center)
        # print(label)
        self.centers_batch = self.center.index_select(0, label.long())
        # print(centers_batch)
        return (xs - self.centers_batch).pow(2).sum()/0.5 /self.batch_size_tensor
    def backward(self,xs,label):
        diff = self.centers_batch - xs
        counts = self.center.new_ones(self.center.size(0))
        ones = self.center.new_ones(label.size(0))
        grad_centers = self.center.new_zeros(self.center.size())
        counts = counts.scatter_add_(0, label.long(), ones)
        grad_centers.scatter_add_(0, label.unsqueeze(1).expand(xs.size()).long(), diff)
        # print(counts)
        grad_centers = grad_centers / counts.view(-1, 1)
        # print((xs - self.centers_batch).pow(2))
        # print((grad_centers ))
        # print((- 1 * diff / batch_size_tensor).sum(), None, (grad_centers / batch_size_tensor).sum(), None)
        return (- 1 * diff / self.batch_size_tensor).sum(), None, (grad_centers / self.batch_size_tensor).sum(), None

# xs = torch.Tensor([[1,2],[3,4],[5,6]])
# label=torch.LongTensor([0,0,1])
# center = CenterLoss(2,2)
# center.forward(xs,label)
