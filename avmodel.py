import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ctc_decode import *
from multiprocessing import Pool
from conformer import Conformer
import random
from batch_beam_search import beam_decode
from club import *



def diff_loss(x1, x2):  # (B, D1)  (B, D2)
    if x1.ndim == 3 and x2.ndim == 3:
        x1 = torch.flatten(x1, 0, 1)
        x2 = torch.flatten(x2, 0, 1)
    nx1 = F.normalize(x1 - torch.mean(x1, 0), dim=-1)
    nx2 = F.normalize(x2 - torch.mean(x2, 0), dim=-1)
    #nx1 = F.normalize(x1, dim=-1)
    #nx2 = F.normalize(x2, dim=-1)
    return torch.mean(F.relu(1. - torch.norm(nx1-nx2, p=2, dim=-1)) ** 2)
    #return torch.mean(torch.matmul(nx1.transpose(-1, -2), nx2).pow(2))  # C x C   效果差！
    #return torch.mean(torch.abs(F.cosine_similarity(x1-torch.mean(x1, 0), x2-torch.mean(x2, 0), dim=-1)))  # 效果较差



def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


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



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # PE(pos, 2i)
        pe[:, 1::2] = torch.cos(position * div_term)  # PE(pos, 2i+1)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):  # (B, L, D)
        return self.dropout(x + self.pe[:, :x.size(1)].detach())


class TransDecoder(nn.Module):
    def __init__(self,
                 n_token,
                 d_model,
                 n_layers=3,
                 n_heads=4,
                 ffn_ratio=4,
                 dropout=0.1,
                 max_len=200):
        super(TransDecoder, self).__init__()
        self.d_model = d_model
        self.tok_embedding = nn.Embedding(n_token, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model,
                                                   dim_feedforward=d_model*ffn_ratio,
                                                   nhead=n_heads,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(d_model, n_token-1)  # excluding bos

    def get_mask_from_lens(self, lengths, max_len=None):
        '''
         param:   lengths --- [Batch_size]
         return:  mask --- [Batch_size, max_len]
        '''
        batch_size = lengths.shape[0]
        if max_len is None:
            max_len = torch.max(lengths).item()
        ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(lengths.device)
        mask = ids < lengths.unsqueeze(1).expand(-1, max_len)  # True or False
        return mask

    def forward(self, tgt, src_enc, src_lens=None, tgt_lens=None):
        tgt = self.tok_embedding(tgt) * (self.d_model ** 0.5)
        tgt = self.pos_enc(tgt)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)   # 下三角 (下0上-inf)
        src_padding_mask = ~self.get_mask_from_lens(src_lens, src_enc.size(1))   # True for masking
        tgt_padding_mask = ~self.get_mask_from_lens(tgt_lens, tgt.size(1))   # True for masking
        dec_out = self.decoder(tgt, src_enc, tgt_mask,
                               tgt_key_padding_mask=tgt_padding_mask,
                               memory_key_padding_mask=src_padding_mask)
        return self.fc(dec_out)





class CTCLipModel(nn.Module):
    def __init__(self, vocab_size, se=False):
        super(CTCLipModel, self).__init__()
        # shallow 3DCNN
        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        # resnet18
        self.resnet18 = ResNet(BasicBlock, [2, 2, 2, 2], se=se)

        self.afront = nn.Sequential(
            nn.Conv1d(80, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(128),
            #nn.ReLU(True),
            nn.GELU(),
            #nn.SiLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),   # downsampling
            nn.BatchNorm1d(256),
            #nn.ReLU(True),
            nn.GELU(),
            #nn.SiLU(),
            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),   # downsampling
            nn.BatchNorm1d(512))
        self.scale = 4
        self.am_embed = nn.Parameter(torch.zeros(512))
        self.vm_embed = nn.Parameter(torch.zeros(512))
       
        self.av_mlp = nn.Sequential(
            nn.Linear(512, 512*2),
            #Transpose(1, 2),
            #nn.BatchNorm1d(512*2),
            #Transpose(1, 2),
            nn.ReLU(True),
            nn.Linear(512*2, 512))

        self.spk_id = nn.Sequential(
            nn.Linear(512*2, 512),
            nn.ReLU(True),
            nn.BatchNorm1d(512))
        self.spk_cls = nn.Linear(512, 29)

        # backend: gru/tcn/transformer
        #self.gru = nn.GRU(512, 256, num_layers=3, bidirectional=True, batch_first=True, dropout=0.5)
        self.cfm = Conformer(512, 4, 512*4, 3, 31, 0.1)
        # fc
        self.fc = nn.Linear(512, vocab_size-1)  # including blank label, excluding bos

        self.trans_dec = TransDecoder(vocab_size, 512, 3, 4) 

        # initialize
        #self._initialize_weights()

    def visual_frontend(self, x):
        bs = x.size(0)
        x = x.transpose(1, 2).contiguous()   # (b, c, t, h, w)
        x = self.frontend3D(x)
        x = x.transpose(1, 2).contiguous()
        x = x.reshape(-1, 64, x.size(3), x.size(4))
        x = self.resnet18(x)
        return x.reshape(bs, -1, 512)
    
    def audio_frontend(self, x):  # (b, t, c)
        x = x.transpose(1, 2).contiguous()   # (b, c, t)
        x = self.afront(x)
        x = x.transpose(1, 2).contiguous()
        return x

    def av_fusion(self, a, v):  # (B, Ta, D)  (B, Tv, D)
        av = F.softmax((a @ v.transpose(-1, -2))/(a.size(-1)**0.5), dim=-1) @ v + a
        #va = F.softmax((v @ a.transpose(-1, -2))/(v.size(-1)**0.5), dim=-1) @ a + v
        out = av + self.av_mlp(av)
        return out

    def forward(self, vid, aud, tgt, vid_lens=None, aud_lens=None, tgt_lens=None):  # (b, t, c, h, w)
        b, t = vid.size()[:2]
        src_lens = aud_lens // self.scale   # time downsampling
        vid = self.visual_frontend(vid)
        aud = self.audio_frontend(aud)
        #vid_id = self.spk_id(torch.cat((vid.mean(dim=1), vid.std(dim=1)), dim=-1))   # x-vector
        #aud_id = self.spk_id(torch.cat((aud.mean(dim=1), aud.std(dim=1)), dim=-1))   # x-vector
        vid_id = self.spk_id(torch.cat((vid.mean(dim=1), vid.max(dim=1)[0]), dim=-1))
        aud_id = self.spk_id(torch.cat((aud.mean(dim=1), aud.max(dim=1)[0]), dim=-1))  
        #id_logits = self.spk_cls(vid_id) + self.spk_cls(aud_id)
        vid_id_logits, aud_id_logits = self.spk_cls(vid_id), self.spk_cls(aud_id)

        '''
        inp_feat = self.av_fusion(aud, vid)
        enc_src = self.cfm(inp_feat, src_lens)
        '''
        inp_feat = torch.cat((aud+self.am_embed, vid+self.vm_embed), dim=1)
        enc_src = self.cfm(inp_feat, src_lens+vid_lens)[:, :aud.shape[1]]
        #orth_loss = diff_loss(aud_id.unsqueeze(1).expand_as(enc_src).detach(), enc_src)  # 1.63
        orth_loss = diff_loss(aud_id.unsqueeze(1).expand_as(enc_src).detach(), enc_src) + diff_loss(vid_id.unsqueeze(1).expand_as(enc_src).detach(), enc_src)
        
        ctc_out = self.fc(enc_src)
        dec_out = self.trans_dec(tgt, enc_src, src_lens, tgt_lens)
        #out2 = self.fc(F.normalize(seq_feat1, dim=-1)) + self.fc(F.normalize(seq_feat2, dim=-1))
        #return ctc_out, dec_out, id_logits, orth_loss
        return ctc_out, dec_out, vid_id_logits, aud_id_logits, orth_loss

    def beam_search_decode(self, vid, aud, vid_lens, aud_lens, bos_id, eos_id, max_dec_len=80, pad_id=0):
        with torch.no_grad():
            b, t = vid.size()[:2]
            lens = aud_lens // self.scale   # time downsampling
            vid = self.visual_frontend(vid)
            aud = self.audio_frontend(aud)
            '''
            inp_feat = self.av_fusion(aud, vid)
            enc_src = self.cfm(inp_feat, lens)
            '''
            inp_feat = torch.cat((aud+self.am_embed, vid+self.vm_embed), dim=1)
            enc_src = self.cfm(inp_feat, lens+vid_lens)[:, :aud.shape[1]]
        
            #res = beam_decode(self.trans_dec, enc_src, src_mask, bos_id, eos_id, max_output_length=max_dec_len, beam_size=6)
            res = beam_decode(self.trans_dec, enc_src, lens, bos_id, eos_id, max_output_length=max_dec_len, beam_size=6)
        return res.detach().cpu()

    def ctc_greedy_decode(self, vids, lens=None):
        with torch.no_grad():
            b, t = vids.size()[:2]
            vid_feat = self.visual_frontend(vids)
            seq_feat = self.cfm(vid_feat, lens)
            logits = self.fc(seq_feat)  # (B, T, V)
            return logits.data.cpu().argmax(dim=-1)

    def ctc_beam_decode(self, vids, lens=None):
        res = []
        with torch.no_grad():
            b, t = vids.size()[:2]
            vid_feat = self.visual_frontend(vids)
            seq_feat = self.cfm(vid_feat, lens)
            logits = self.fc(seq_feat)  # (B, T, V)
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



class SpeakerIdentity(nn.Module):
    def __init__(self):
        super(SpeakerIdentity, self).__init__()
        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        self.resnet18 = ResNet(BasicBlock, [2, 2, 2, 2])
        #self.pool = nn.AdaptiveAvgPool2d(1)
        #self.proj = nn.Linear(32768, 256)
        #self.gru = nn.GRU(256, 256 // 2, num_layers=1, bidirectional=True, batch_first=True)
        
        #self.head = nn.Sequential(
        #    nn.Linear(32768, 512),
        #    nn.BatchNorm1d(512),
        #    nn.ReLU(True),
        #    nn.Linear(512, 256))
        self.head = nn.Linear(512, 512)

    def freeze_frontend(self):
        self.frontend3D.requires_grad_(False)
        self.resnet18.requires_grad_(False)
        print('Freeze the weights of visual frontend ...')

    def forward(self, x):  # (b, t, c, h, w)
        b, t = x.shape[:2]
        x = x.transpose(1, 2).contiguous()  # (b, c, t, h, w)
        x = self.frontend3D(x)
        x = x.transpose(1, 2).contiguous()  # (b, t, c, h, w)
        x = x.reshape(-1, 64, x.size(3), x.size(4))
        x = self.resnet18(x)
        #x = x.flatten(0, 1)  # (bt, c, h, w)
        #x = self.pool(x)

        x = x.reshape(b, t, -1)
        #feat = self.gru(self.proj(x))[0]  # (b, t, d)
        feat = x.mean(dim=1)
        #feat = torch.cat((x.mean(dim=1), x.std(dim=1)), dim=-1)   # x-vector
        seq_feat = self.head(feat)  # (b, d)
        return x, seq_feat   # (b, d)


# 适用于ASR，不适用VSR
class SelfAttentivePooling(nn.Module):
    def __init__(self, hid_size, attn_size):
        super(SelfAttentivePooling, self).__init__()
        self.mlp = nn.Sequential(
                nn.Linear(hid_size, attn_size),
                nn.ReLU(True),
                nn.Linear(attn_size, 1, bias=False))

    def forward(self, x):  # (B, L, D)
        attn = self.mlp(x).squeeze(2)  # (B, L)
        attn_weights = F.softmax(attn, dim=1)  # (B, L)
        weighted_inputs = torch.mul(x, attn_weights.unsqueeze(2))  # (B, L, D)
        pooled_output = torch.sum(weighted_inputs, dim=1)  # (B, D)
        return pooled_output



class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return torch.transpose(x, self.dim0, self.dim1)



class SpeakerIdentity2D(nn.Module):
    def __init__(self, num_spk):
        super(SpeakerIdentity2D, self).__init__()
        self.frontend2D = nn.Sequential(
            #nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.resnet18 = ResNet(BasicBlock, [2, 2, 2, 2])
        # self.pool = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 512))
        #self.head = nn.Linear(512, 512)
        #self.head = SelfAttentivePooling(512, 512//2)
        self.spk_cls = nn.Linear(512, num_spk)

    def forward(self, x):  # (b, t, c, h, w)
        b, t = x.shape[:2]
        x = torch.flatten(x, 0, 1)  # (bt, c, h, w)
        x = self.frontend2D(x)
        x = self.resnet18(x)
        # x = self.pool(x)
        x = x.reshape(b, t, -1)
        # feat = x.mean(dim=1)
        # feat = torch.cat((x.mean(dim=1), x.std(dim=1)), dim=-1)  # i-vector
        #seq_feat = self.head(x)  # (b, d)
        seq_feat = self.head(x.mean(dim=1))  # (b, d)
        return x, seq_feat, self.spk_cls(seq_feat)  # (b, t, d)  (b, d)



class DRLModel(nn.Module):
    def __init__(self, vocab_size, num_spk):
        super(DRLModel, self).__init__()
        self.vsr = CTCLipModel(vocab_size)
        #self.spk = SpeakerIdentity()
        self.spk = SpeakerIdentity2D(num_spk)
        self.mi_net = CLUBSample_reshape(512, 512, 512)
        self.tmp = 1.

    def forward(self, vids, auds, tgts, spk_ids, vid_lens, aud_lens, tgt_lens):   # (B, T, C, H, W)
        #ctc_logits, dec_logits, spk_logits, orth_loss = self.vsr(vids, auds, tgts[:, :-1], vid_lens, aud_lens, tgt_lens)[:4]
        #spk_loss = F.cross_entropy(spk_logits, spk_ids)
        ctc_logits, dec_logits, spk_vid_logits, spk_aud_logits, orth_loss = self.vsr(vids, auds, tgts[:, :-1], vid_lens, aud_lens, tgt_lens)[:5]
        spk_loss = F.cross_entropy(spk_vid_logits, spk_ids) + F.cross_entropy(spk_aud_logits, spk_ids)

        ctc_log_probs = ctc_logits.transpose(0, 1).log_softmax(dim=-1)  # (T, B, V)
        ctc_loss = F.ctc_loss(ctc_log_probs, tgts[:, 1:], aud_lens.reshape(-1) // 4, tgt_lens.reshape(-1), zero_infinity=True)  # time downsampling 
        attn_loss = F.cross_entropy(dec_logits.transpose(-1, -2).contiguous(), tgts[:, 1:].long(), ignore_index=0)
        vsr_loss = 0.9*attn_loss + 0.1*ctc_loss
        return vsr_loss + 0.5*spk_loss + 0.5*orth_loss

    def load_pretrain_bert(self, path):
        from transformers import BertTokenizer, BertModel
        self.tokenizer = BertTokenizer.from_pretrained(path)
        self.bert_model = BertModel.from_pretrained(path)
        encoded_input = self.tokenizer(text, return_tensors='pt', padding=True)
        output = self.bert_model(**encoded_input)
        return output[0][:, 0]

    def calc_triplet_loss(self, vids):
        vids = torch.flatten(vids, 0, 1)
        # vids: (2x16, T, C, H, W)
        frame_feat, seq_feat, _ = self.spk(vids)
        spk1, spk2 = frame_feat.chunk(2, dim=0)   # 不相同  (N, T, D)
        L = random.choice(range(1, spk1.shape[1]))  # [1: T-1]
        spk1_sfl = torch.cat((spk1[:, L:, ...], spk1[:, :L, ...]), dim=1).contiguous()
        spk2_sfl = torch.cat((spk2[:, L:, ...], spk2[:, :L, ...]), dim=1).contiguous()
        # pos: (spk1, spk1_sfl)    neg: (spk2, spk1_sfl)
        # dcl_loss = F.relu(0.2 + F.pairwise_distance(spk1, spk1_sfl, p=2) - F.pairwise_distance(spk2, spk1_sfl, p=2))
        # dcl_loss = F.relu(0.2 + torch.norm(spk1-spk1_sfl, dim=-1, p=2) - torch.norm(spk2-spk1_sfl, dim=-1, p=2))
        # dcl_loss = F.relu(0.2 + torch.cdist(spk1, spk1_sfl, p=2) - torch.cdist(spk2, spk1_sfl, p=2))
        labels = torch.ones(spk1.shape[0]*spk1.shape[1], device=vids.device)
        frame_loss1 = F.cosine_embedding_loss(spk1.flatten(0, 1), spk1_sfl.flatten(0, 1), target=labels) + \
                      F.cosine_embedding_loss(spk2.flatten(0, 1), spk1_sfl.flatten(0, 1), target=-1. * labels, margin=0.2)
        frame_loss2 = F.cosine_embedding_loss(spk2.flatten(0, 1), spk2_sfl.flatten(0, 1), target=labels) + \
                      F.cosine_embedding_loss(spk1.flatten(0, 1), spk2_sfl.flatten(0, 1), target=-1. * labels, margin=0.2)
        frame_loss = 0.5 * frame_loss1 + 0.5 * frame_loss2

        s1, s2 = seq_feat.chunk(2, dim=0)  # 不相同  (N, D)
        l = random.choice(range(1, s1.shape[0]))  # [1: T-1]
        s1_sfl = torch.cat((s1[l:, ...], s1[:l, ...]), dim=0).contiguous()
        s2_sfl = torch.cat((s2[l:, ...], s2[:l, ...]), dim=0).contiguous()
        y = torch.ones(s1.shape[0], device=vids.device)
        seq_loss1 = F.cosine_embedding_loss(s1, s1_sfl, target=y) + \
                    F.cosine_embedding_loss(s2, s1_sfl, target=-1. * y, margin=0.2)
        seq_loss2 = F.cosine_embedding_loss(s2, s2_sfl, target=y) + \
                    F.cosine_embedding_loss(s1, s2_sfl, target=-1. * y, margin=0.2)
        seq_loss = 0.5 * seq_loss1 + 0.5 * seq_loss2
        return frame_loss + seq_loss

    
    '''
    def calc_triplet_loss(self, vids):
        vids = torch.flatten(vids, 0, 1)
        # vids: (2x16, T, C, H, W)
        spk_feat = self.spk(vids)[0]
        spk1, spk2 = spk_feat.chunk(2, dim=0)   # 不相同  (N, T, D)
        L = random.choice(range(1, spk1.shape[1]))  # [1: T-1]
        spk1_sfl = torch.cat((spk1[:, L:, ...], spk1[:, :L, ...]), dim=1).contiguous()
        spk2_sfl = torch.cat((spk2[:, L:, ...], spk2[:, :L, ...]), dim=1).contiguous()
        # pos: (spk1, spk1_sfl)    neg: (spk2, spk1_sfl)
        # dcl_loss = F.relu(0.2 + F.pairwise_distance(spk1, spk1_sfl, p=2) - F.pairwise_distance(spk2, spk1_sfl, p=2))
        # dcl_loss = F.relu(0.2 + torch.norm(spk1-spk1_sfl, dim=-1, p=2) - torch.norm(spk2-spk1_sfl, dim=-1, p=2))
        # dcl_loss = F.relu(0.2 + torch.cdist(spk1, spk1_sfl, p=2) - torch.cdist(spk2, spk1_sfl, p=2))
        # F.cosine_similarity()
        labels = torch.ones(spk1.shape[0]*spk1.shape[1], device=vids.device)
        loss1 = F.cosine_embedding_loss(spk1.flatten(0, 1), spk1_sfl.flatten(0, 1), target=labels) + F.cosine_embedding_loss(spk2.flatten(0, 1), spk1_sfl.flatten(0, 1), target=-1.*labels, margin=0.2)
        loss2 = F.cosine_embedding_loss(spk2.flatten(0, 1), spk2_sfl.flatten(0, 1), target=labels) + F.cosine_embedding_loss(spk1.flatten(0, 1), spk2_sfl.flatten(0, 1), target=-1. * labels, margin=0.2)
        dcl_loss = loss1 + loss2
        return dcl_loss
    '''

    def calc_orth_loss(self, vids, tgts, spk_ids, xlens, ylens):
        '''
        vids = torch.flatten(vids, 0, 1)
        tgts = torch.flatten(tgts, 0, 1)
        xlens = torch.flatten(xlens, 0, 1)
        ylens = torch.flatten(ylens, 0, 1)
        '''
        # vids: (2x16, T, C, H, W)
        ## for spk
        _, spk_feat, spk_logits = self.spk(vids)
        spk_loss = F.cross_entropy(spk_logits, spk_ids)
        #s1, s2 = spk_feat.chunk(2, dim=0)  # 不相同  (N, D)
        '''
        l = random.choice(range(1, s1.shape[0]))  # [1: T-1]
        s1_sfl = torch.cat((s1[l:, ...], s1[:l, ...]), dim=0).contiguous()
        s2_sfl = torch.cat((s2[l:, ...], s2[:l, ...]), dim=0).contiguous()
        y = torch.ones(s1.shape[0], device=vids.device)
        spk_loss1 = F.cosine_embedding_loss(s1, s1_sfl, target=y) + \
                    F.cosine_embedding_loss(s2, s1_sfl, target=-1. * y, margin=0.2)
        spk_loss2 = F.cosine_embedding_loss(s2, s2_sfl, target=y) + \
                    F.cosine_embedding_loss(s1, s2_sfl, target=-1. * y, margin=0.2)
        spk_loss = spk_loss1 + spk_loss2
        '''
        ## for vsr
        #logits, _, cont_feat = self.vsr(vids, xlens)
        #log_probs = logits.transpose(0, 1).log_softmax(dim=-1)  # (T, B, V)
        #vsr_loss = F.ctc_loss(log_probs, tgts, xlens.reshape(-1), ylens.reshape(-1), zero_infinity=True)
        ctc_logits, dec_logits, _, cont_feat = self.vsr(vids, tgts[:, :-1], xlens, ylens)
        ctc_log_probs = ctc_logits.transpose(0, 1).log_softmax(dim=-1)  # (T, B, V)
        ctc_loss = F.ctc_loss(ctc_log_probs, tgts[:, 1:], xlens.reshape(-1), ylens.reshape(-1), zero_infinity=True)
        attn_loss = F.cross_entropy(dec_logits.transpose(-1, -2).contiguous(), tgts[:, 1:].long(), ignore_index=0)
        vsr_loss = 0.9*attn_loss + 0.1*ctc_loss
        #c1, c2 = cont_feat.chunk(2, dim=0)  # 对应s1, s2  (N, T, D)
        #diff_loss = diff_loss(s1.unsqueeze(1).detach(), c1) + diff_loss(s2.unsqueeze(1).detach(), c2)    # (N, D)  (N, T, D)
        diff_loss = diff_loss(spk_feat.unsqueeze(1).expand_as(cont_feat).detach(), cont_feat) 
        return vsr_loss + spk_loss + diff_loss

    def calc_orth_loss2(self, vids, tgts, spk_ids, xlens, ylens, opt_mi):
        '''
        vids = torch.flatten(vids, 0, 1)
        tgts = torch.flatten(tgts, 0, 1)
        xlens = torch.flatten(xlens, 0, 1)
        ylens = torch.flatten(ylens, 0, 1)
        '''
        # vids: (2x16, T, C, H, W)
        ## for spk
        _, spk_feat, spk_logits = self.spk(vids)
        spk_loss = F.cross_entropy(spk_logits, spk_ids)
        #s1, s2 = spk_feat.chunk(2, dim=0)  # 不相同  (N, D)
        '''
        l = random.choice(range(1, s1.shape[0]))  # [1: T-1]
        s1_sfl = torch.cat((s1[l:, ...], s1[:l, ...]), dim=0).contiguous()
        s2_sfl = torch.cat((s2[l:, ...], s2[:l, ...]), dim=0).contiguous()
        y = torch.ones(s1.shape[0], device=vids.device)
        spk_loss1 = F.cosine_embedding_loss(s1, s1_sfl, target=y) + \
                    F.cosine_embedding_loss(s2, s1_sfl, target=-1. * y, margin=0.2)
        spk_loss2 = F.cosine_embedding_loss(s2, s2_sfl, target=y) + \
                    F.cosine_embedding_loss(s1, s2_sfl, target=-1. * y, margin=0.2)
        spk_loss = spk_loss1 + spk_loss2
        '''
        ## for vsr
        #logits, _, cont_feat = self.vsr(vids, xlens)
        #log_probs = logits.transpose(0, 1).log_softmax(dim=-1)  # (T, B, V)
        #vsr_loss = F.ctc_loss(log_probs, tgts, xlens.reshape(-1), ylens.reshape(-1), zero_infinity=True)
        ctc_logits, dec_logits, _, cont_feat = self.vsr(vids, tgts[:, :-1], xlens, ylens)
        ctc_log_probs = ctc_logits.transpose(0, 1).log_softmax(dim=-1)  # (T, B, V)
        ctc_loss = F.ctc_loss(ctc_log_probs, tgts[:, 1:], xlens.reshape(-1), ylens.reshape(-1), zero_infinity=True)
        attn_loss = F.cross_entropy(dec_logits.transpose(-1, -2).contiguous(), tgts[:, 1:].long(), ignore_index=0)
        vsr_loss = 0.9*attn_loss + 0.1*ctc_loss
        #diff_loss = diff_loss(spk_feat.unsqueeze(1).expand_as(cont_feat).detach(), cont_feat)
        for _ in range(5):  # 将seq-level换成token-level 
            opt_mi.zero_grad()
            #lld_loss = -self.mi_net.loglikeli(spk_feat.detach(), cont_feat.mean(dim=1).detach())
            lld_loss = -self.mi_net.loglikeli(spk_feat.unsqueeze(1).expand_as(cont_feat).detach(), cont_feat.detach())
            lld_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.mi_net.parameters(), max_norm=1.)
            opt_mi.step()
        #mi_loss = self.mi_net.mi_est(spk_feat, cont_feat.mean(dim=1))
        mi_loss = self.mi_net.mi_est(spk_feat.unsqueeze(1).expand_as(cont_feat), cont_feat)
        return vsr_loss + 0.5 * spk_loss + 0.01 * mi_loss


    def calc_drl_loss(self, vids, tgts, xlens, ylens):
        vids = torch.flatten(vids, 0, 1)
        tgts = torch.flatten(tgts, 0, 1)
        xlens = torch.flatten(xlens, 0, 1)
        ylens = torch.flatten(ylens, 0, 1)
        # vids: (2x16, T, C, H, W)
        ## for spk
        frame_feat, seq_feat = self.spk(vids)
        spk1, spk2 = frame_feat.chunk(2, dim=0)  # 不相同  (N, T, D)
        L = random.choice(range(1, spk1.shape[1]))  # [1: T-1]
        spk1_sfl = torch.cat((spk1[:, L:, ...], spk1[:, :L, ...]), dim=1).contiguous()
        spk2_sfl = torch.cat((spk2[:, L:, ...], spk2[:, :L, ...]), dim=1).contiguous()
        labels = torch.ones(spk1.shape[0] * spk1.shape[1], device=vids.device)
        frame_loss1 = F.cosine_embedding_loss(spk1.flatten(0, 1), spk1_sfl.flatten(0, 1), target=labels) + \
                      F.cosine_embedding_loss(spk2.flatten(0, 1), spk1_sfl.flatten(0, 1), target=-1. * labels,
                                              margin=0.2)
        frame_loss2 = F.cosine_embedding_loss(spk2.flatten(0, 1), spk2_sfl.flatten(0, 1), target=labels) + \
                      F.cosine_embedding_loss(spk1.flatten(0, 1), spk2_sfl.flatten(0, 1), target=-1. * labels,
                                              margin=0.2)
        frame_loss = frame_loss1 + frame_loss2
        s1, s2 = seq_feat.chunk(2, dim=0)  # 不相同  (N, D)
        l = random.choice(range(1, s1.shape[0]))  # [1: T-1]
        s1_sfl = torch.cat((s1[l:, ...], s1[:l, ...]), dim=0).contiguous()
        s2_sfl = torch.cat((s2[l:, ...], s2[:l, ...]), dim=0).contiguous()
        y = torch.ones(s1.shape[0], device=vids.device)
        seq_loss1 = F.cosine_embedding_loss(s1, s1_sfl, target=y) + \
                    F.cosine_embedding_loss(s2, s1_sfl, target=-1. * y, margin=0.2)
        seq_loss2 = F.cosine_embedding_loss(s2, s2_sfl, target=y) + \
                    F.cosine_embedding_loss(s1, s2_sfl, target=-1. * y, margin=0.2)
        seq_loss = seq_loss1 + seq_loss2
        spk_loss = frame_loss + seq_loss
        ## for vsr
        logits, vid_feat, cont_feat = self.vsr(vids, xlens)
        log_probs = logits.transpose(0, 1).log_softmax(dim=-1)  # (T, B, V)
        vsr_loss = F.ctc_loss(log_probs, tgts, xlens.reshape(-1), ylens.reshape(-1), zero_infinity=True)
        ## for drl
        c1, c2 = cont_feat.chunk(2, dim=0)  # 对应s1, s2   (N, T, D)
        diff_loss = diff_loss(s1.unsqueeze(1), c1) + diff_loss(s2.unsqueeze(1), c2) 
        return vsr_loss + spk_loss + diff_loss



    # 低效！
    def calc_cl_loss(self, vids, tgts, xlens, ylens):
        vids = torch.flatten(vids, 0, 1)
        tgts = torch.flatten(tgts, 0, 1)
        xlens = torch.flatten(xlens, 0, 1)
        ylens = torch.flatten(ylens, 0, 1)
        # vids: (16x2, T, C, H, W)
        logits, cont_feat = self.vsr(vids, xlens)
        spk_feat = self.spk(vids)[1]
        ## for drl
        #spk1, spk2 = spk_feat.chunk(2, dim=0)   # 相同
        #cont1, cont2 = cont_feat.chunk(2, dim=0)  # 不同
        spk1, spk2 = spk_feat[0::2], spk_feat[1::2]
        cont1, cont2 = cont_feat[0::2], cont_feat[1::2]
        feat1 = torch.cat((F.normalize(spk1, dim=-1), F.normalize(cont2, dim=-1)), dim=0)
        feat2 = torch.cat((F.normalize(spk2, dim=-1), F.normalize(cont1, dim=-1)), dim=0)
        scores1 = torch.matmul(F.normalize(spk1, dim=-1), feat2.transpose(0, 1)) / self.tmp
        scores2 = torch.matmul(F.normalize(spk2, dim=-1), feat1.transpose(0, 1)) / self.tmp
        drl_loss = F.cross_entropy(scores1, torch.tensor(list(range(len(spk1))), dtype=torch.long, device=vids.device)) + F.cross_entropy(scores2, torch.tensor(list(range(len(spk2))), dtype=torch.long, device=vids.device))
        ## for vsr
        log_probs = logits.transpose(0, 1).log_softmax(dim=-1)  # (T, B, V)
        vsr_loss = F.ctc_loss(log_probs, tgts, xlens.reshape(-1), ylens.reshape(-1), zero_infinity=True)
        return vsr_loss + drl_loss * 0.5

    def greedy_decode(self, vids, lens=None):
        return self.vsr.ctc_greedy_decode(vids, lens)

    def beam_decode(self, vid, aud, vid_lens, aud_lens, bos_id, eos_id, max_dec_len=50, pad_id=0):
        return self.vsr.beam_search_decode(vid, aud, vid_lens, aud_lens, bos_id, eos_id, max_dec_len, pad_id)



