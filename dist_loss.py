import torch
import torch.nn as nn
import torch.nn.functional as F


def nuclear_norm_dist(x1, x2):
    nx1 = F.normalize(x1, dim=-1)
    nx2 = F.normalize(x2, dim=-1)
    loss = torch.abs(torch.norm(nx1, 'nuc') - torch.norm(nx2, 'nuc')) / x1.shape[0]
    return loss


class CMD(nn.Module):
    """
    Adapted from https://github.com/wzell/cmd/blob/master/models/domain_regularizer.py
    """
    def __init__(self):
        super(CMD, self).__init__()

    def forward(self, x1, x2, n_moments=5):
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        sx1 = x1-mx1
        sx2 = x2-mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(n_moments - 1):
            scms = scms + self.scm(sx1, sx2, i + 2)
        return scms / x1.shape[0]

    def matchnorm(self, x1, x2):
        power = torch.pow(x1-x2,2)
        summed = torch.sum(power)
        sqrt = summed ** 0.5
        return sqrt
        # return ((x1-x2)**2).sum().sqrt()

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)


def cmd(X, X_test, K=5):
    """
    central moment discrepancy (cmd)
    objective function for keras models (theano or tensorflow backend)

    - Zellinger, Werner, et al. "Robust unsupervised domain adaptation for
    neural networks via moment alignment.", TODO
    - Zellinger, Werner, et al. "Central moment discrepancy (CMD) for
    domain-invariant representation learning.", ICLR, 2017.
    """
    x1 = X
    x2 = X_test
    mx1 = x1.mean(0)
    mx2 = x2.mean(0)
    sx1 = x1 - mx1
    sx2 = x2 - mx2
    dm = l2diff(mx1,mx2)
    scms = [dm]
    for i in range(K-1):
        # moment diff of centralized samples
        scms.append(moment_diff(sx1,sx2,i+2))
        # scms+=moment_diff(sx1,sx2,1)
    return sum(scms) / X.shape[0]

def l2diff(x1, x2):
    """
    standard euclidean norm
    """
    return (x1-x2).norm(p=2)

def moment_diff(sx1, sx2, k):
    """
    difference between moments
    """
    ss1 = sx1.pow(k).mean(0)
    ss2 = sx2.pow(k).mean(0)
    #ss1 = sx1.mean(0)
    #ss2 = sx2.mean(0)
    return l2diff(ss1, ss2)


class CORAL(nn.Module):
    def __init__(self):
        super(CORAL, self).__init__()

    def forward(self, source, target):
        d = source.size(1)
        # source covariance
        xm = torch.mean(source, 0, keepdim=True) - source
        xc = xm.t() @ xm
        # target covariance
        xmt = torch.mean(target, 0, keepdim=True) - target
        xct = xmt.t() @ xmt
        # frobenius norm between source and target
        loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
        loss = loss / (4 * d * d)
        return loss


def CORAL2(source, target, **kwargs):
    d = source.data.shape[1]
    ns, nt = source.data.shape[0], target.data.shape[0]
    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm / (ns - 1)
    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt / (nt - 1)
    # frobenius norm between source and target
    loss = torch.mul((xc - xct), (xc - xct))
    loss = torch.sum(loss) / (4 * d * d)
    return loss


# Deep CORAL
def CORAL3(source, target):
    DEVICE = source.device
    d = source.size(1)
    ns, nt = source.size(0), target.size(0)
    # source covariance
    tmp_s = torch.ones((1, ns)).to(DEVICE) @ source
    cs = (source.t() @ source - (tmp_s.t() @ tmp_s) / ns) / (ns - 1)
    # target covariance
    tmp_t = torch.ones((1, nt)).to(DEVICE) @ target
    ct = (target.t() @ target - (tmp_t.t() @ tmp_t) / nt) / (nt - 1)
    # frobenius norm
    loss = (cs - ct).pow(2).sum().sqrt()
    loss = loss / (4 * d * d)
    return loss


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)

    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)


# 多核MMD
def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss


def ranking_loss(p1, n1, p2, n2, m=1.):
    #p1, n1 = F.normalize(p1, dim=-1), F.normalize(n1, dim=-1)
    #p2, n2 = F.normalize(p2, dim=-1), F.normalize(n2, dim=-1)
    #l1 = F.relu(m + torch.norm(F.relu(p2 - p1), p=2, dim=-1)**2 - torch.norm(F.relu(n2 - p1), p=2, dim=-1)**2)
    #l2 = F.relu(m + torch.norm(F.relu(p2 - p1), p=2, dim=-1)**2 - torch.norm(F.relu(p2 - n1), p=2, dim=-1)**2)
    #l1 = torch.clamp(m + torch.norm(p2 - p1, p=2, dim=-1) - torch.norm(n1 - p1, p=2, dim=-1), min=0.0)
    #l2 = torch.clamp(m + torch.norm(p1 - p2, p=2, dim=-1) - torch.norm(n2 - p2, p=2, dim=-1), min=0.0)
    l1 = torch.clamp(m - F.cosine_similarity(p1, p2, dim=-1) + F.cosine_similarity(p1, n1, dim=-1), min=0.0)
    l2 = torch.clamp(m - F.cosine_similarity(p2, p1, dim=-1) + F.cosine_similarity(p2, n2, dim=-1), min=0.0)
    return l1.mean() + l2.mean()


def diff_loss(x1, x2, mode='cos'):  # (B, D1)  (B, D2)
    #if x1.ndim == 3 and x2.ndim == 3:
    #    x1 = torch.flatten(x1, 0, 1)
    #    x2 = torch.flatten(x2, 0, 1)
    if mode == 'l2':
        nx1 = F.normalize(x1, dim=-1)   # 先L2归一化约束在[-1, 1]之间
        nx2 = F.normalize(x2, dim=-1)
        return torch.mean(torch.clamp(1. - torch.norm(nx1 - nx2, p=2, dim=-1), min=0.0).pow(2))
        #return torch.mean(torch.clamp(1. - torch.norm(x1 - x2, p=2, dim=-1), min=0.0).pow(2))
    elif mode == 'fnorm':
        #nx1 = F.normalize(x1 - torch.mean(x1, 0), dim=-1)
        #nx2 = F.normalize(x2 - torch.mean(x2, 0), dim=-1)
        #return torch.mean(torch.matmul(nx1.transpose(-1, -2), nx2).pow(2))  # D1 x D2   
        return DecorrLoss(x1, x2)   # 相关性损失 (去相关)
    elif mode == 'cos':
        return torch.mean(torch.clamp(F.cosine_similarity(x1, x2, dim=-1), min=0.0))  
        #return torch.mean(torch.clamp(F.cosine_similarity(x1-torch.mean(x1, 0), x2-torch.mean(x2, 0), dim=-1), min=0.0))  # 皮尔逊相关性
        #return F.cosine_embedding_loss(x1, x2, torch.tensor([[-1]]).to(x1.device))  # (B, D)
    else:
        raise ValueError("Unknown distance type")


# (B, D)
def decorr_loss(h1, h2):
    # 计算协方差矩阵
    h1_centered = h1 - h1.mean(dim=0, keepdims=True)
    h2_centered = h2 - h2.mean(dim=0, keepdims=True)
    # 交叉协方差矩阵
    cov_matrix = torch.matmul(h1_centered.T, h2_centered) / (h1.size(0) - 1)
    # 去相关损失为交叉协方差矩阵的Frobenius范数
    loss = torch.norm(cov_matrix, p='fro')
    return loss

# (B, T, D)
def DecorrLoss(h1, h2):
    if h1.ndim == 3:
        # 将三维的数据展平为二维
        B, T, C = h1.size()
        # 计算每个时间步的均值
        h1_mean = h1.mean(dim=1, keepdim=True)
        h2_mean = h2.mean(dim=1, keepdim=True)
        # 去中心化
        h1_centered = h1 - h1_mean
        h2_centered = h2 - h2_mean
        # 将每个时间步的数据拼接在一起
        h1_centered = h1_centered.reshape(-1, C)  # (B*T, D)
        h2_centered = h2_centered.reshape(-1, C)  # (B*T, D)
    else:
        # 计算协方差矩阵
        h1_centered = h1 - h1.mean(dim=0, keepdims=True)
        h2_centered = h2 - h2.mean(dim=0, keepdims=True)
    # 计算交叉协方差矩阵
    cov_matrix = torch.matmul(h1_centered.T, h2_centered) / (h1_centered.size(0) - 1)
    # 去相关损失为交叉协方差矩阵的 Frobenius 范数
    loss = torch.norm(cov_matrix, p='fro')  # 平方和开方
    return loss


def orth_loss(input1, input2):
    batch_size = input1.size(0)
    input1 = input1.reshape(batch_size, -1)
    input2 = input2.reshape(batch_size, -1)
    input1_mean = torch.mean(input1, dim=0, keepdims=True)
    input2_mean = torch.mean(input2, dim=0, keepdims=True)
    input1 = input1 - input1_mean
    input2 = input2 - input2_mean
    input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
    input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)
    input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
    input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)
    diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))
    return diff_loss


class JSDLoss(nn.Module):
    def __init__(self, hid_size1, hid_size2, norm=False):
        super(JSDLoss, self).__init__()
        self.norm = norm
        hid_size = hid_size1 + hid_size2
        self.net = nn.Sequential(nn.Linear(hid_size, hid_size//2),   # 可以换成Conv(k=1, s=1)
                                 nn.ReLU(),
                                 #nn.Linear(hid_size//2, hid_size//2))
                                 #nn.ReLU(),
                                 nn.Linear(hid_size//2, 1))
        #self.fc = nn.Linear(hid_size, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.1, 0.1)
                nn.init.zeros_(m.bias)

    def forward(self, x, y):
        bs = x.size(0)
        tiled_x = torch.cat((x, x), dim=0)
        shuffled_y = torch.cat((y[1:], y[0].unsqueeze(0)), dim=0)
        concat_y = torch.cat((y, shuffled_y), dim=0)
        inputs = torch.cat((tiled_x, concat_y), dim=-1)
        logits = self.net(inputs)
        #logits = self.fc(F.relu(inputs))
        if self.norm:
            logits = F.normalize(logits, p=2, dim=-1)
        pred_xy = logits[:bs]
        pred_x_y = logits[bs:]
        mi_loss = (-F.softplus(-pred_xy)).mean() - F.softplus(pred_x_y).mean()  # max jsd
        return -mi_loss


def jsd_loss(T: torch.Tensor, T_prime: torch.Tensor):
    """Estimator of the Jensen Shannon Divergence see paper equation (2)

      Args:
        T (torch.Tensor): Statistique network estimation from the marginal distribution P(x)P(z)
        T_prime (torch.Tensor): Statistique network estimation from the joint distribution P(xz)

      Returns:
        float: DJS estimation value
    """
    joint_expectation = (-F.softplus(-T)).mean()   # pxy
    marginal_expectation = F.softplus(T_prime).mean()  # pxpy
    mutual_info = joint_expectation - marginal_expectation
    return -mutual_info

