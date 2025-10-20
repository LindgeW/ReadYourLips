import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from avdataset import GRIDDataset, CMLRDataset, BucketBatchSampler
import torch.optim as optim
from asr_model import ASRModel
from vsr_model import VSRModel

from jiwer import cer, wer
import random
import numpy as np
from ctc_decode import ctc_beam_decode
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from constants import *
import sys

DEVICE = torch.device('cuda:0')

Models = {'asr': ASRModel, 'vsr': VSRModel}


def asr_train(train_set, val_set=None, lr=3e-4, epochs=50, batch_size=32, model_path=None):
    model = ASRModel(len(train_set.vocab), len(train_set.spks)).to(DEVICE)
    print('参数量：', sum(param.numel() for param in model.parameters()))
    if model_path is not None:
        checkpoint = torch.load(model_path, map_location='cpu')
        states = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model.load_state_dict(states)
        print('loading weights ...')
    model.train()
    print(model)
    print('ASR training ...')
    data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)   # GRID
    # data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False, collate_fn=CMLRDataset.collate_pad)  # CMLR

    ## 用桶采样的Dataloader
    # bucket_sampler = BucketBatchSampler(train_set, batch_size=batch_size, bucket_boundaries=[50, 100, 150, 200])
    # data_loader = DataLoader(train_set, batch_sampler=bucket_sampler, num_workers=4, pin_memory=False, collate_fn=CMLRDataset.collate_pad)

    #optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-6)
    num_iters = len(data_loader) * epochs
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                   num_warmup_steps=num_iters // 10,
                                                   num_training_steps=num_iters)
    accumulate_steps = 2
    best_wer, best_cer = 1., 1.
    for ep in range(1, 1 + epochs):
        ep_loss = 0.
        for i, batch_data in enumerate(data_loader):  # (B, T, C, H, W)
            aud_inps = batch_data['aud'].to(DEVICE)
            targets = batch_data['txt'].to(DEVICE)
            spk_ids = batch_data['spk_id'].to(DEVICE)
            aud_lens = batch_data['aud_lens'].to(DEVICE)
            target_lens = batch_data['txt_lens'].to(DEVICE)
            ## for GRID
            optimizer.zero_grad()
            losses = model(aud_inps, targets, spk_ids, aud_lens, target_lens)
            loss = losses['asr']
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            ## for CMLR (梯度累积)
            # losses = model(aud_inps, targets, spk_ids, aud_lens, target_lens)
            # loss = losses['asr']
            # loss = loss / accumulate_steps
            # loss.backward()
            # if (i+1) % accumulate_steps == 0:
            #     optimizer.step()
            #     optimizer.zero_grad()
            #     lr_scheduler.step()

            ep_loss += loss.data.item()
            # if (i + 1) % 5 == 0:
            print("Epoch {}, Iteration {}, loss: {:.4f}".format(ep, i + 1, loss.data.item()), flush=True)

        if ep > 20:
            print("Epoch {}, loss: {:.4f}".format(ep, ep_loss), flush=True)
            savename = 'iter_{}.pt'.format(ep)
            savedir = os.path.join('checkpoints', 'asr_only_unseen_grid')
            if not os.path.exists(savedir): os.makedirs(savedir)
            save_path = os.path.join(savedir, savename)
            torch.save({'model': model.state_dict()}, save_path)
            print(f'Saved to {save_path}!!!', flush=True)
            if val_set is not None:
                wer, cer = evaluate('asr', save_path, val_set, batch_size=batch_size)
                print(f'Val WER: {wer}, CER: {cer}', flush=True)
                if wer < best_wer:
                    best_wer, best_cer = wer, cer
                print(f'Best WER: {best_wer}, CER: {best_cer}', flush=True)


def vsr_train(train_set, val_set=None, lr=3e-4, epochs=50, batch_size=32, model_path=None):
    model = VSRModel(len(train_set.vocab), len(train_set.spks)).to(DEVICE)
    print('参数量：', sum(param.numel() for param in model.parameters()))
    if model_path is not None:
        checkpoint = torch.load(model_path, map_location='cpu')
        states = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model.load_state_dict(states)
        print('loading weights ...')
    model.train()
    print(model)
    print('VSR training ...')
    data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)  # GRID
    # data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False, collate_fn=CMLRDataset.collate_pad)  # CMLR

    # bucket_sampler = BucketBatchSampler(train_set, batch_size=batch_size, bucket_boundaries=[50, 100, 150, 200])
    # data_loader = DataLoader(train_set, batch_sampler=bucket_sampler, num_workers=4, pin_memory=False, collate_fn=CMLRDataset.collate_pad)

    #optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-6)
    num_iters = len(data_loader) * epochs
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                   num_warmup_steps=num_iters // 10,
                                                   num_training_steps=num_iters)
    accumulate_steps = 2
    best_wer, best_cer = 1., 1.
    for ep in range(1, 1 + epochs):
        ep_loss = 0.
        for i, batch_data in enumerate(data_loader):  # (B, T, C, H, W)
            vid_inps = batch_data['vid'].to(DEVICE)
            targets = batch_data['txt'].to(DEVICE)
            spk_ids = batch_data['spk_id'].to(DEVICE)
            vid_lens = batch_data['vid_lens'].to(DEVICE)
            target_lens = batch_data['txt_lens'].to(DEVICE)
            ## for GRID
            optimizer.zero_grad()
            losses = model(vid_inps, targets, spk_ids, vid_lens, target_lens)
            loss = losses['vsr']
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            ## for CMLR
            # losses = model(vid_inps, targets, spk_ids, vid_lens, target_lens)
            # loss = losses['vsr']
            # loss = loss / accumulate_steps
            # loss.backward()
            # if (i+1) % accumulate_steps == 0:
            #     optimizer.step()
            #     optimizer.zero_grad()
            #     lr_scheduler.step()

            ep_loss += loss.data.item()
            # if (i + 1) % 5 == 0:
            print("Epoch {}, Iteration {}, loss: {:.4f}".format(ep, i + 1, loss.data.item()), flush=True)

        if ep > 20:
            print("Epoch {}, loss: {:.4f}".format(ep, ep_loss), flush=True)
            savename = 'iter_{}.pt'.format(ep)
            savedir = os.path.join('checkpoints', 'vsr_only_unseen_grid')
            if not os.path.exists(savedir): os.makedirs(savedir)
            save_path = os.path.join(savedir, savename)
            torch.save({'model': model.state_dict()}, save_path)
            print(f'Saved to {save_path}!!!', flush=True)
            if val_set is not None:
                wer, cer = evaluate('vsr', save_path, val_set, batch_size=batch_size)
                print(f'Val WER: {wer}, CER: {cer}', flush=True)
                if wer < best_wer:
                    best_wer, best_cer = wer, cer
                print(f'Best WER: {best_wer}, CER: {best_cer}', flush=True)


@torch.no_grad()
def evaluate(model_type, model_path, dataset, batch_size=32):
    model = Models[model_type.lower()](len(dataset.vocab), len(dataset.spks)).to(DEVICE)
    # checkpoint = torch.load(opt.load, map_location=lambda storage, loc: storage)
    checkpoint = torch.load(model_path, map_location='cpu')
    states = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(states)
    model.eval()
    print(len(dataset), next(model.parameters()).device)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)  # GRID
    # data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=CMLRDataset.collate_pad)  # CMLR
    preds = []
    refs = []
    PAD_ID, BOS_ID, EOS_ID = (dataset.vocab.index(x) for x in [PAD, BOS, EOS])
    for batch_data in data_loader:
        if model_type == 'vsr':
            vid_inp = batch_data['vid'].to(DEVICE)
            tgt_txt = batch_data['txt'].to(DEVICE)
            vid_lens = batch_data['vid_lens'].to(DEVICE)
            output = model.beam_decode(vid_inp, vid_lens, bos_id=BOS_ID, eos_id=EOS_ID, max_dec_len=40)
        elif model_type == 'asr':
            aud_inp = batch_data['aud'].to(DEVICE)
            tgt_txt = batch_data['txt'].to(DEVICE)
            aud_lens = batch_data['aud_lens'].to(DEVICE)
            output = model.beam_decode(aud_inp, aud_lens, bos_id=BOS_ID, eos_id=EOS_ID, max_dec_len=40)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        pred = []
        ref = []
        for out, tgt in zip(output, tgt_txt):
            ## CER
            # pred.append(''.join([dataset.vocab[i] for i in torch.unique_consecutive(out).tolist() if i not in [PAD_ID, BOS_ID, EOS_ID]]))
            # ref.append(''.join([dataset.vocab[i] for i in tgt.tolist() if i not in [PAD_ID, BOS_ID, EOS_ID]]))
            ## WER
            # pred.append(' '.join([dataset.vocab[i] for i in torch.unique_consecutive(out).tolist() if i not in [PAD_ID, BOS_ID, EOS_ID]]))
            pred.append(' '.join([dataset.vocab[i] for i in out.tolist() if i not in [PAD_ID, BOS_ID, EOS_ID]]))
            ref.append(' '.join([dataset.vocab[i] for i in tgt.tolist() if i not in [PAD_ID, BOS_ID, EOS_ID]]))
        preds.extend(pred)
        refs.extend(ref)
    test_wer, test_cer = wer(refs, preds), cer(refs, preds)
    print('JIWER wer: {:.4f}, cer: {:.4f}'.format(test_wer, test_cer))
    return test_wer, test_cer


# 单模态训练
if __name__ == '__main__':
    seed = 1347
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    DEVICE = torch.device('cuda:' + str(sys.argv[1]))
    print('running device:', torch.cuda.get_device_name(), DEVICE)

    data_type = str(sys.argv[2])   # grid or cmlr
    print('using dataset: ', data_type)
    if data_type == 'grid':
        data_root = r'D:\LipData\GRID\LIP_160_80\lip'
        ## 已知说话人
        train_set = GRIDDataset(data_root, r'data\overlap_train.json', phase='train', setting='seen')
        val_set = GRIDDataset(data_root, r'data\overlap_val.json', phase='test', setting='seen')
        vsr_train(train_set, val_set, lr=3e-4, epochs=50, batch_size=32, model_path=None)
        # asr_train(train_set, val_set, lr=3e-4, epochs=50, batch_size=32, model_path=None)
        # 测试
        # test_set = GRIDDataset(data_root, r'data\overlap_val.json', phase='test', setting='seen')
        # evaluate('vsr', 'checkpoints/vsr_only_seen_grid/iter_40.pt', test_set, batch_size=32)
        # evaluate('asr', 'checkpoints/asr_only_seen_grid/iter_40.pt', test_set, batch_size=32)

        ## 未知说话人
        #train_set = GRIDDataset(data_root, r'data\unseen_train.json', phase='train', setting='unseen')
        #val_set = GRIDDataset(data_root, r'data\unseen_val.json', phase='test', setting='unseen')
        # vsr_train(train_set, val_set, lr=3e-4, epochs=50, batch_size=32, model_path=None)
        # asr_train(train_set, val_set, lr=3e-4, epochs=50, batch_size=32, model_path=None)
        # 测试
        #test_set = GRIDDataset(data_root, r'data\unseen_val.json', phase='test', setting='unseen')
        # evaluate('vsr', 'checkpoints/vsr_only_unseen_grid/iter_40.pt', test_set, batch_size=32)
        # evaluate('asr', 'checkpoints/asr_only_unseen_grid/iter_40.pt', test_set, batch_size=32)
    elif data_type == 'cmlr':
        data_root = r'D:\LipData\CMLR'
        ## 已知说话人
        train_set = CMLRDataset(data_root, r'data\train.csv', phase='train', setting='seen')
        val_set = CMLRDataset(data_root, r'data\val.csv', phase='test', setting='seen')
        vsr_train(train_set, val_set, lr=3e-4, epochs=50, batch_size=16, model_path=None)   # for VSR
        # asr_train(train_set, val_set, lr=3e-4, epochs=50, batch_size=16, model_path=None)   # for ASR
        # 测试
        # test_set = CMLRDataset(data_root, r'data\test.csv', phase='test', setting='seen')
        # evaluate('vsr', 'checkpoints/vsr_only_seen_cmlr/iter_49.pt', test_set, batch_size=32)
        # evaluate('asr', 'checkpoints/asr_only_seen_cmlr/iter_49.pt', test_set, batch_size=32)

        ## 未知说话人
        # train_set = CMLRDataset(data_root, r'data\unseen_train.csv', phase='train', setting='unseen')
        # val_set = CMLRDataset(data_root, r'data\unseen_test.csv', phase='test', setting='unseen')
        # vsr_train(train_set, val_set, lr=3e-4, epochs=50, batch_size=16, model_path=None)    # for VSR
        # asr_train(train_set, val_set, lr=3e-4, epochs=50, batch_size=16, model_path=None)   # for ASR
        # 测试
        # test_set = CMLRDataset(data_root, r'data\unseen_test.csv', phase='test', setting='unseen')
        # evaluate('vsr', 'checkpoints/vsr_only_unseen_cmlr/iter_40.pt', test_set, batch_size=32)
        # evaluate('asr', 'checkpoints/asr_only_unseen_cmlr/iter_40.pt', test_set, batch_size=32)
    else:
        raise NotImplementedError('Invalid Dataset!')
