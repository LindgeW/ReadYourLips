
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ctc_decode import *
from multiprocessing import Pool
from conformer import Conformer
import random


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, se=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.se = se

        if self.se:
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.conv3 = conv1x1(planes, planes // 16)
            self.conv4 = conv1x1(planes // 16, planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.se:
            w = self.gap(out)
            w = self.conv3(w)
            w = self.relu(w)
            w = self.conv4(w).sigmoid()
            out = out * w

        out = out + residual
        out = self.relu(out)
        return out


# ResNet18
class ResNet(nn.Module):
    def __init__(self, block, layers, se=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.se = se
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.bn = nn.BatchNorm1d(512)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, se=self.se))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, se=self.se))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        #x = self.bn(x)
        return x



class DRLModel(nn.Module):
    def __init__(self, vocab_size, num_spk, se=False):
        super(DRLModel, self).__init__()
        # shallow 3DCNN
        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        # resnet18
        self.resnet18 = ResNet(BasicBlock, [2, 2, 2, 2], se=se)
        self.cfm = Conformer(512, 4, 512*4, 3, 31, 0.1)
        self.dropout = nn.Dropout(p=0.1)
        self.fc = nn.Linear(512, vocab_size-1)  # including blank label, excluding bos

        self.spk_fc1 = nn.Linear(1, 512//2)
        self.spk_fc2 = nn.Linear(512//2, 512)
        self.spk_cls = nn.Linear(512, num_spk)

        # initialize
        #self._initialize_weights()

    def visual_frontend_forward(self, x):
        x = x.transpose(1, 2).contiguous()   # (b, c, t, h, w)
        x = self.frontend3D(x)
        x = x.transpose(1, 2).contiguous()
        x = x.reshape(-1, 64, x.size(3), x.size(4))
        x = self.resnet18(x)
        return x

    # 特征正交分解
    def factorize(self, vids):
        b, t = vids.size()[:2]
        x = self.visual_frontend_forward(vids)
        x = x.reshape(b, -1, 512)
        vid_feat = x / torch.norm(x, p=2, dim=-1, keepdim=True)  # BTD
        spk_feat = self.spk_fc1(torch.norm(x, p=2, dim=-1, keepdim=True))  # BT1 -> BTD
        spk_feat = self.spk_fc2(spk_feat.mean(dim=1))  # BD
        return vid_feat, spk_feat

    def calc_orth_loss(self, vids, tgts, spk_ids, xlens, ylens):  # (b, t, c, h, w)
        # vids: (B, T, C, H, W)
        vid_feat, spk_feat = self.factorize(vids)

        spk_loss = F.cross_entropy(self.spk_cls(self.dropout(spk_feat)), spk_ids)

        vsr_feat = self.cfm(vid_feat, xlens)
        logits = self.fc(self.dropout(vsr_feat))
        log_probs = logits.transpose(0, 1).log_softmax(dim=-1)  # (T, B, V)
        vsr_loss = F.ctc_loss(log_probs, tgts[:, 1:], xlens.reshape(-1), ylens.reshape(-1), zero_infinity=True)

        diff_loss = self.diff_loss(vsr_feat, spk_feat.unsqueeze(1))  
        loss = vsr_loss + spk_loss + diff_loss
        return loss

    def calc_drl_loss(self, vids, tgts, xlens, ylens):  # (b, t, c, h, w)
        vids = torch.flatten(vids, 0, 1)
        tgts = torch.flatten(tgts, 0, 1)
        xlens = torch.flatten(xlens, 0, 1)
        ylens = torch.flatten(ylens, 0, 1)
        # vids: (16x2, T, C, H, W)
        vid_feat, spk_feat = self.factorize(vids)

        s1, s2 = spk_feat.chunk(2, dim=0)   # 不相同
        l = random.choice(range(1, s1.shape[0]))  # [1: T-1]
        s1_sfl = torch.cat((s1[l:, ...], s1[:l, ...]), dim=0).contiguous()
        s2_sfl = torch.cat((s2[l:, ...], s2[:l, ...]), dim=0).contiguous()
        y = torch.ones(s1.shape[0], device=vids.device)
        spk_loss1 = F.cosine_embedding_loss(s1, s1_sfl, target=y) + \
                    F.cosine_embedding_loss(s2, s1_sfl, target=-1. * y, margin=0.2)
        spk_loss2 = F.cosine_embedding_loss(s2, s2_sfl, target=y) + \
                    F.cosine_embedding_loss(s1, s2_sfl, target=-1. * y, margin=0.2)
        spk_loss = spk_loss1 + spk_loss2

        vsr_feat = self.cfm(vid_feat, xlens)
        logits = self.fc(self.dropout(vsr_feat))
        log_probs = logits.transpose(0, 1).log_softmax(dim=-1)  # (T, B, V)
        vsr_loss = F.ctc_loss(log_probs, tgts, xlens.reshape(-1), ylens.reshape(-1), zero_infinity=True)

        c1, c2 = vsr_feat.chunk(2, dim=0)  # 对应s1, s2   (N, T, D)
        diff_loss = self.diff_loss(s1.unsqueeze(1), c1) + \
                    self.diff_loss(s2.unsqueeze(1), c2) 
        loss = vsr_loss + spk_loss + diff_loss
        return loss

    def diff_loss(self, x1, x2):  # (B1, D)  (B2, D)
        # nx1 = F.normalize(x1 - torch.mean(x1, 0), dim=-1)
        # nx2 = F.normalize(x2 - torch.mean(x2, 0), dim=-1)
        #nx1 = F.normalize(x1, dim=-1)
        #nx2 = F.normalize(x2, dim=-1)
        #dot_mat = torch.matmul(nx1, nx2.transpose(-1, -2))
        #return torch.sum(dot_mat ** 2)
        return torch.mean(torch.abs(F.cosine_similarity(x1, x2, dim=-1)))


    def greedy_decode(self, vids, lens=None):
        with torch.no_grad():
            vid_feat, _ = self.factorize(vids)
            vsr_feat = self.cfm(vid_feat, lens)
            logits = self.fc(self.dropout(vsr_feat))
            return logits.data.cpu().argmax(dim=-1)

    def beam_decode(self, vids, lens=None):
        res = []
        with torch.no_grad():
            vid_feat, _ = self.factorize(vids)
            vsr_feat = self.cfm(vid_feat, lens)
            logits = self.fc(self.dropout(vsr_feat))
            probs = torch.log_softmax(logits, dim=-1).cpu().numpy()
            for prob in probs:
                pred = ctc_beam_decode3(prob, 10, 0)
                res.append(pred)
            return res

    '''
    def beam_decode(self, vids):
        res = []
        with torch.no_grad():
            logits = self.forward(vids)[0]  # (B, T, V)
            probs = torch.log_softmax(logits, dim=-1).cpu().numpy()
            with Pool(len(probs)) as p:
                res = p.map(ctc_beam_decode3, probs)
                #res.append(pred)
            return res
    '''

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



