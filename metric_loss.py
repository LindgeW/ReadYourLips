# AM-softmax / AAM-softmax / ArcFace / CosFace
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# https://www.kaggle.com/code/nanguyen/arcface-loss
# https://kevinmusgrave.github.io/pytorch-metric-learning/losses


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


class ArcFaceLoss(nn.Module):
    def __init__(self, embedding_size, class_num, s=30.0, m=0.50):   # m in [0.3, 0.7]
        """ArcFace formula:
            cos(m + theta) = cos(m)cos(theta) - sin(m)sin(theta)
        Note that:
            0 <= m + theta <= Pi
        So if (m + theta) >= Pi, then theta >= Pi - m. In [0, Pi]
        we have:
            cos(theta) < cos(Pi - m)
        So we can use cos(Pi - m) as threshold to check whether
        (m + theta) go out of [0, Pi]

        Args:
            embedding_size: usually 128, 256, 512 ...
            class_num: num of people when training
            s: scale, see normface https://arxiv.org/abs/1704.06369
            m: margin, see SphereFace, CosFace, and ArcFace paper
        """
        super().__init__()
        self.in_features = embedding_size
        self.out_features = class_num
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(class_num, embedding_size))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = ((1.0 - cosine.pow(2)).clamp(0, 1)).sqrt()
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)  # drop to CosFace
        # update y_i by phi in cosine
        output = cosine * 1.0  # make backward works
        batch_size = len(output)
        output[range(batch_size), label] = phi[range(batch_size), label]
        logits = output * self.s
        return F.cross_entropy(logits, label)


class ArcFaceLoss2(nn.Module):
    def __init__(self, embedding_size, class_num, scale=64, margin=0.7):
        super().__init__()
        self.embedding_size = embedding_size
        self.class_num = class_num
        self.scale = scale
        self.margin = margin
        self.weights = nn.Parameter(torch.FloatTensor(class_num, embedding_size))
        nn.init.xavier_normal_(self.weights)

    def forward(self, features, targets):
        cos_theta = F.linear(features, F.normalize(self.weights), bias=None)
        cos_theta = cos_theta.clip(-1 + 1e-7, 1 - 1e-7)
        arc_cos = torch.acos(cos_theta)
        M = F.one_hot(targets, num_classes=self.out_features) * self.margin
        arc_cos = arc_cos + M
        cos_theta_2 = torch.cos(arc_cos)
        logits = cos_theta_2 * self.scale
        return F.cross_entropy(logits, targets)


class ArcFaceLoss3(nn.Module):
    def __init__(self, num_classes, embedding_size, margin=0.3, scale=30.0):
        """
        ArcFace: Additive Angular Margin Loss for Deep Face Recognition
        (https://arxiv.org/pdf/1801.07698.pdf)
        Args:
            num_classes: The number of classes in your training dataset
            embedding_size: The size of the embeddings that you pass into
            margin: m in the paper, the angular margin penalty in radians
            scale: s in the paper, feature scale
        """
        super().__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.margin = margin
        self.scale = scale
        self.W = torch.nn.Parameter(torch.Tensor(num_classes, embedding_size))
        nn.init.xavier_normal_(self.W)

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (None, embedding_size)
            labels: (None,)
        Returns:
            loss: scalar
        """
        cosine = self.get_cosine(embeddings)  # (None, n_classes)
        mask = self.get_target_mask(labels)  # (None, n_classes)
        cosine_of_target_classes = cosine[mask == 1]  # (None, )
        modified_cosine_of_target_classes = self.modify_cosine_of_target_classes(
            cosine_of_target_classes
        )  # (None, )
        diff = (modified_cosine_of_target_classes - cosine_of_target_classes).unsqueeze(1)  # (None,1)
        logits = cosine + (mask * diff)  # (None, n_classes)
        logits = self.scale_logits(logits)  # (None, n_classes)
        return F.cross_entropy(logits, labels)

    def get_cosine(self, embeddings):
        """
        Args:
            embeddings: (None, embedding_size)
        Returns:
            cosine: (None, n_classes)
        """
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.W))
        return cosine

    def get_target_mask(self, labels):
        """
        Args:
            labels: (None,)
        Returns:
            mask: (None, n_classes)
        """
        batch_size = labels.size(0)
        onehot = torch.zeros(batch_size, self.num_classes, device=labels.device)
        onehot.scatter_(1, labels.unsqueeze(-1), 1)
        return onehot

    def modify_cosine_of_target_classes(self, cosine_of_target_classes):
        """
        Args:
            cosine_of_target_classes: (None,)
        Returns:
            modified_cosine_of_target_classes: (None,)
        """
        eps = 1e-6
        # theta in the paper
        angles = torch.acos(torch.clamp(cosine_of_target_classes, -1 + eps, 1 - eps))
        return torch.cos(angles + self.margin)

    def scale_logits(self, logits):
        """
        Args:
            logits: (None, n_classes)
        Returns:
            scaled_logits: (None, n_classes)
        """
        return logits * self.scale


class CosFaceLoss(nn.Module):
    def __init__(self, embedding_size, class_num, s=30.0, m=0.40):
        """
        Args:
            embedding_size: usually 128, 256, 512 ...
            class_num: num of people when training
            s: scale, see normface https://arxiv.org/abs/1704.06369
            m: margin, see SphereFace, CosFace, and ArcFace paper
        """
        super().__init__()
        self.embedding_size = embedding_size
        self.class_num = class_num
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(class_num, embedding_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        output = cosine * 1.0  # make backward works
        batch_size = len(output)
        output[range(batch_size), label] = phi[range(batch_size), label]
        logits = output * self.s
        return F.cross_entropy(logits, label)

