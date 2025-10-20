import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from avdataset import GRIDDataset, CMLRDataset, LRS3Dataset, VSRAppDataset, BucketBatchSampler
import torch.optim as optim
#from avmodel3 import CTCLipModel, DRLModel
#from avmodel_avhubert import CTCLipModel, DRLModel  # avsr
#from amodel_avhubert import CTCLipModel, DRLModel   # asr
from vmodel_avhubert import CTCLipModel, DRLModel  # vsr
from jiwer import cer, wer
import random
import numpy as np
from ctc_decode import ctc_beam_decode
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from constants import *
import sys
from file_io import write_to

DEVICE = torch.device('cuda:0')


# CTC training
def train(train_set, val_set=None, lr=3e-4, epochs=100, batch_size=32, model_path=None):
    model = CTCLipModel(len(train_set.vocab)).to(DEVICE)
    print('参数量：', sum(param.numel() for param in model.parameters()))
    if model_path is not None:
        checkpoint = torch.load(model_path, map_location='cpu')
        states = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model.load_state_dict(states)
        print('loading weights ...')
    model.train()
    print(model)
    print('training ...')

    data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)

    #optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-6)
    num_iters = len(data_loader) * epochs
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                   num_warmup_steps=num_iters // 10,
                                                   num_training_steps=num_iters)
    best_wer, best_cer = 1., 1.
    for ep in range(1, 1 + epochs):
        ep_loss = 0.
        for i, batch_data in enumerate(data_loader):  # (B, T, C, H, W)
            inputs = batch_data['vid'].to(DEVICE)
            targets = batch_data['txt'].to(DEVICE)
            input_lens = batch_data['vid_lens'].to(DEVICE)
            target_lens = batch_data['txt_lens'].to(DEVICE)
            optimizer.zero_grad()
            logits = model(inputs, input_lens)[0]
            logits = logits.transpose(0, 1).log_softmax(dim=-1)  # (T, B, V)
            loss = F.ctc_loss(logits, targets, input_lens.reshape(-1), target_lens.reshape(-1), zero_infinity=True)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            ep_loss += loss.data.item()
            # if (i + 1) % 5 == 0:
            print("Epoch {}, Iteration {}, loss: {:.4f}".format(ep, i + 1, loss.data.item()), flush=True)

        if ep > 20:
            print("Epoch {}, loss: {:.4f}".format(ep, ep_loss), flush=True)
            savename = 'iter_{}.pt'.format(ep)
            savedir = os.path.join('checkpoints', 'vsr_cmlr')
            if not os.path.exists(savedir): os.makedirs(savedir)
            save_path = os.path.join(savedir, savename)
            torch.save({'model': model.state_dict()}, save_path)
            print(f'Saved to {save_path}!!!', flush=True)
            if val_set is not None:
                wer, cer = evaluate(save_path, val_set, batch_size=batch_size)
                print(f'Val WER: {wer}, CER: {cer}', flush=True)
                if wer < best_wer:
                    best_wer, best_cer = wer, cer
                print(f'Best WER: {best_wer}, CER: {best_cer}', flush=True)


# AVSR training
def avtrain(train_set, val_set=None, lr=3e-4, epochs=100, batch_size=32, model_path=None):
    model = DRLModel(len(train_set.vocab), len(train_set.spks)).to(DEVICE)
    print('参数量：', sum(param.numel() for param in model.parameters()))
    if model_path is not None:
        checkpoint = torch.load(model_path, map_location='cpu')
        states = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model.load_state_dict(states)
        print('loading weights ...')
    model.train()
    print(model)
    print('training ...')

    # data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)  
    data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=train_set.collate_pad)  # CMLR
    # bucket_sampler = BucketBatchSampler(train_set, batch_size=batch_size, bucket_boundaries=[50, 100, 150, 200])
    # data_loader = DataLoader(train_set, batch_sampler=bucket_sampler, num_workers=4, pin_memory=False, collate_fn=train_set.collate_pad)

    accumulate_steps = 2
    #optimizer = optim.Adam(model.parameters(), lr=lr)
    #optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-6)
    #optimizer = optim.AdamW([*model.avsr.fc.parameters(), *model.avsr.trans_dec.parameters()], lr=lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-6)
    optimizer = optim.AdamW([{'params': [*model.avsr.fc.parameters(), *model.avsr.trans_dec.parameters()], 'lr': lr, 'weight_decay': 1e-4}, 
                             {'params': model.avsr.avhubert.parameters(), 'lr': lr/2, 'weight_decay': 0.}], betas=(0.9, 0.98), eps=1e-9)
    #num_iters = len(data_loader) * epochs 
    num_iters = len(data_loader) * epochs // accumulate_steps
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_iters // 10, num_training_steps=num_iters)
    #lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_iters // 10, num_training_steps=num_iters)

    best_wer, best_cer = 1., 1.
    for ep in range(1, 1 + epochs):
        ep_loss = 0.
        #if ep == int(epochs*0.2):
        #    print(int(epochs*0.2))
        #    optimizer.add_param_group({'params': model.avsr.avhubert.parameters()})  # 避免重置optimizer状态

        for i, batch_data in enumerate(data_loader):  # (B, T, C, H, W)
            vid_inps = batch_data['vid'].to(DEVICE)
            aud_inps = batch_data['aud'].to(DEVICE)
            targets = batch_data['txt'].to(DEVICE)
            spk_ids = batch_data['spk_id'].to(DEVICE)
            vid_lens = batch_data['vid_lens'].to(DEVICE)
            aud_lens = batch_data['aud_lens'].to(DEVICE)
            target_lens = batch_data['txt_lens'].to(DEVICE)

            ## for GRID
            #optimizer.zero_grad()
            #losses = model(vid_inps, aud_inps, targets, spk_ids, vid_lens, aud_lens, target_lens)
            #loss = losses['vsr'] + losses['spk'] + losses['drl']
            #loss.backward()
            #optimizer.step()
            #lr_scheduler.step()

            ## for CMLR/LRS3 (梯度累积)
            losses = model(vid_inps, aud_inps, targets, spk_ids, vid_lens, aud_lens, target_lens)
            loss = losses['vsr'] 
            #loss = losses['vsr'] + losses['spk'] + losses['drl']
            loss = loss / accumulate_steps
            loss.backward()
            if (i+1) % accumulate_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                #model.zero_grad()
                lr_scheduler.step()

            ep_loss += loss.data.item()
            # if (i + 1) % 5 == 0:
            print("Epoch {}, Iteration {}, lr: {:.6f}, loss: {:.4f}".format(ep, i + 1, optimizer.param_groups[0]['lr'], loss.data.item()), flush=True)

        if ep > 30:
            print("Epoch {}, loss: {:.4f}".format(ep, ep_loss), flush=True)
            savename = 'iter_{}.pt'.format(ep)
            savedir = os.path.join('checkpoints', 'vsr_unseen_cmlr3')
            #savedir = os.path.join('checkpoints', 'avsr_unseen_lrs3')
            #savedir = os.path.join('checkpoints', 'vsr_unseen_lrs3')
            #savedir = os.path.join('checkpoints', 'asr_unseen_lrs3')
            if not os.path.exists(savedir): os.makedirs(savedir)
            save_path = os.path.join(savedir, savename)
            torch.save({'model': model.state_dict()}, save_path)
            print(f'Saved to {save_path}!!!', flush=True)
            if val_set is not None:
                wer, cer = 0, 0
                #wer, cer = evaluate(save_path, val_set, batch_size=batch_size)
                print(f'Val WER: {wer}, CER: {cer}', flush=True)
                if wer < best_wer:
                    best_wer, best_cer = wer, cer
                print(f'Best WER: {best_wer}, CER: {best_cer}', flush=True)


def adapt(train_set, model_path, lr=1e-4, epochs=100, batch_size=32):
    model = DRLModel(len(train_set.vocab), len(train_set.spks)).to(DEVICE)
    print(sum(param.numel() for param in model.parameters()) / 1e6, 'M')
    if model_path is not None:
        checkpoint = torch.load(model_path, map_location='cpu')
        states = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model.load_state_dict(states, strict=True)
        print('loading weights ...')
    # model.model.reset_params()
    model.train()
    print(model)

    data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=train_set.collate_pad)
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-4)
    # optimizer = optim.AdamW([*model.model.adanet.parameters(), model.model.sc], lr=lr, betas=(0.9, 0.98), eps=1e-9)
    # optimizer = optim.AdamW([*model.model.adanet.parameters(), *model.model.adanet2.parameters(), model.model.sc], lr=lr, betas=(0.9, 0.98), eps=1e-9)
    # optimizer = optim.AdamW([*model.model.fc.parameters(), *model.model.gru2.parameters()], lr=lr, betas=(0.9, 0.98), eps=1e-9)
    #num_iters = len(data_loader) * epochs 
    #lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_iters // 10, num_training_steps=num_iters)
    
    for ep in range(epochs):
        for i, batch_data in enumerate(data_loader):
            vid_inps = batch_data['vid'].to(DEVICE)
            aud_inps = batch_data['aud'].to(DEVICE)
            targets = batch_data['txt'].to(DEVICE)
            #spk_ids = batch_data['spk_id'].to(DEVICE)
            vid_lens = batch_data['vid_lens'].to(DEVICE)
            aud_lens = batch_data['aud_lens'].to(DEVICE)
            target_lens = batch_data['txt_lens'].to(DEVICE)
            
            optimizer.zero_grad()
            losses = model(vid_inps, aud_inps, targets, None, vid_lens, aud_lens, target_lens)
            loss = losses['vsr'] 
            loss.backward()
            optimizer.step()
            #lr_scheduler.step()
            print("Epoch {}, Iteration {}, lr: {:.6f}, loss: {:.4f}".format(ep + 1, i + 1, optimizer.param_groups[0]['lr'], loss.data.item()), flush=True)
        if ep < 30: continue
        savename = 'iter_{}.pt'.format(ep + 1)
        savedir = os.path.join('checkpoints', 'vsrapp4')
        if not os.path.exists(savedir): os.makedirs(savedir)
        torch.save({'model': model.state_dict()}, os.path.join(savedir, savename))
        print(f'Saved to {savename}.')



# DRL training for VSR and SV
def drl_train(vsr_set, spk_set, drl_set, val_set=None, lr=1e-4, epochs=100, batch_size=32, model_path=None):
    model = DRLModel(len(vsr_set.vocab), len(vsr_set.spks)).to(DEVICE)
    print(sum(param.numel() for param in model.parameters()))
    if model_path is not None:
        checkpoint = torch.load(model_path, map_location='cpu')
        states = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model.load_state_dict(states, strict=False)
        print('loading weights ...')
    model.train()
    print(model)
    spk_data_loader = DataLoader(spk_set, batch_size=2, shuffle=True, num_workers=2)
    vsr_data_loader = DataLoader(vsr_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)
    drl_data_loader = DataLoader(drl_set, batch_size=2, shuffle=True, num_workers=2)
    #drl_data_loader = DataLoader(drl_set, batch_size=batch_size // 2, shuffle=True, num_workers=6)
    #optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-6)
    #spk_optimizer = optim.AdamW(model.spk.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-6)
    #mi_optimizer = optim.AdamW(model.mi_net.parameters(), lr=3*lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-6)
    vsr_optimizer = optim.AdamW(model.parameters(), lr=3*lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-6)
    num_iters = len(vsr_data_loader) * epochs
    lr_scheduler = get_cosine_schedule_with_warmup(vsr_optimizer, num_warmup_steps=num_iters // 10,
                                                   num_training_steps=num_iters)
    best_wer, best_cer = 1., 1.

    '''
    for ep in range(300):
        for i, batch_data in enumerate(spk_data_loader):  # (2, N, T, C, H, W)
            inputs = batch_data['vid'].to(DEVICE)
            model.zero_grad()
            loss = model.calc_triplet_loss(inputs)
            loss.backward()
            spk_optimizer.step()
            # lr_scheduler.step()
            print("Epoch {}, Iteration {}, sv loss: {:.4f}".format(ep, i + 1, loss.data.item()), flush=True)
    savedir = os.path.join('checkpoints', 'drl_grid2')
    if not os.path.exists(savedir): os.makedirs(savedir)
    save_path = os.path.join(savedir, 'spk.pt')
    torch.save({'model': model.state_dict()}, save_path)
    print(f'Saved to {save_path}!!!', flush=True)
    '''

    '''
    for ep in range(10):
        ep_loss = 0.
        for i, batch_data in enumerate(vsr_data_loader):  # (B, T, C, H, W)
            inputs = batch_data['vid'].to(DEVICE)
            targets = batch_data['txt'].to(DEVICE)
            input_lens = batch_data['vid_lens'].to(DEVICE)
            target_lens = batch_data['txt_lens'].to(DEVICE)
            model.zero_grad()
            loss = model(inputs, targets, input_lens, target_lens)
            loss.backward()
            optimizer.step()
            ep_loss += loss.data.item()
            # lr_scheduler.step()
            print("Epoch {}, Iteration {}, vsr loss: {:.4f}".format(ep, i + 1, loss.data.item()), flush=True)
        if ep % 1 == 0:
            print("Epoch {}, loss: {:.4f}".format(ep, ep_loss), flush=True)
            savename = 'iter_{}.pt'.format(ep)
            savedir = os.path.join('checkpoints', 'drl_grid')
            if not os.path.exists(savedir): os.makedirs(savedir)
            save_path = os.path.join(savedir, savename)
            torch.save({'model': model.state_dict()}, save_path)
            print(f'Saved to {save_path}!!!', flush=True)
            if val_set is not None:
                wer, cer = evaluate(save_path, val_set, batch_size=batch_size)
                print(f'Val WER: {wer}, CER: {cer}', flush=True)
                if wer < best_wer:
                    best_wer, best_cer = wer, cer
                print(f'Best WER: {best_wer}, CER: {best_cer}', flush=True)
    '''

    for ep in range(1, 1 + epochs):
        ep_loss = 0.
        for i, batch_data in enumerate(drl_data_loader):  # (S, 2, T, C, H, W)
            vid_inps = batch_data['vid'].to(DEVICE)
            aud_inps = batch_data['aud'].to(DEVICE)
            targets = batch_data['txt'].to(DEVICE)
            spk_ids = batch_data['spk_id'].to(DEVICE)
            vid_lens = batch_data['vid_lens'].to(DEVICE)
            aud_lens = batch_data['aud_lens'].to(DEVICE)
            target_lens = batch_data['txt_lens'].to(DEVICE)
            model.zero_grad()
            loss = model(vid_inps, aud_inps, targets, spk_ids, vid_lens, aud_lens, target_lens)
            #loss = model.calc_orth_loss(inputs, targets, spk_ids, input_lens, target_lens)
            #loss = model.calc_orth_loss2(inputs, targets, spk_ids, input_lens, target_lens, mi_optimizer)
            #loss = model.calc_drl_loss(inputs, targets, input_lens, target_lens)
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
            vsr_optimizer.step()
            lr_scheduler.step()
            ep_loss += loss.data.item()
            print("Epoch {}, Iteration {}, loss: {:.4f}".format(ep, i + 1, loss.data.item()), flush=True)
        if ep > 20:
            print("Epoch {}, loss: {:.4f}".format(ep, ep_loss), flush=True)
            savename = 'iter_{}.pt'.format(ep)
            savedir = os.path.join('checkpoints', 'drl_grid')
            if not os.path.exists(savedir): os.makedirs(savedir)
            save_path = os.path.join(savedir, savename)
            torch.save({'model': model.state_dict()}, save_path)
            print(f'Saved to {save_path}!!!', flush=True)
            if val_set is not None:
                wer, cer = evaluate(save_path, val_set, batch_size=batch_size)
                print(f'Val WER: {wer}, CER: {cer}', flush=True)
                if wer < best_wer:
                    best_wer, best_cer = wer, cer
                print(f'Best WER: {best_wer}, CER: {best_cer}', flush=True)



@torch.no_grad()
def evaluate(model_path, dataset, batch_size=32):
    model = DRLModel(len(dataset.vocab), len(dataset.spks)).to(DEVICE)
    # checkpoint = torch.load(opt.load, map_location=lambda storage, loc: storage)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    states = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(states)
    model.eval()
    print(len(dataset))
    print('loading checkpoint path: ' + model_path)
    # data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=dataset.collate_pad)
    #data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=LRS3Dataset.collate_pad)
    preds = []
    refs = []
    PAD_ID, BOS_ID, EOS_ID = (dataset.vocab.index(x) for x in [PAD, BOS, EOS])
    n_total, n_correct = 0, 0
    for batch_data in data_loader:
        vid_inp = batch_data['vid'].to(DEVICE)
        aud_inp = batch_data['aud'].to(DEVICE)
        tgt_txt = batch_data['txt'].to(DEVICE)
        vid_lens = batch_data['vid_lens'].to(DEVICE)
        aud_lens = batch_data['aud_lens'].to(DEVICE)
        #output = model.greedy_decode(vid_inp, input_lens)
        output = model.beam_decode(vid_inp, aud_inp, vid_lens, aud_lens, bos_id=BOS_ID, eos_id=EOS_ID, max_dec_len=40)
        #output = model.beam_decode(vid_inp, aud_inp, vid_lens, aud_lens, bos_id=BOS_ID, eos_id=EOS_ID, max_dec_len=150)
        for out, tgt in zip(output, tgt_txt):
            ## CER
            #preds.append(''.join([dataset.vocab[i] for i in torch.unique_consecutive(out).tolist() if i not in [PAD_ID, BOS_ID, EOS_ID]]))
            #preds.append(''.join([dataset.vocab[i] for i in out.tolist() if i not in [PAD_ID, BOS_ID, EOS_ID]]))
            #refs.append(''.join([dataset.vocab[i] for i in tgt.tolist() if i not in [PAD_ID, BOS_ID, EOS_ID]]))
            ## WER
            #preds.append(' '.join([dataset.vocab[i] for i in torch.unique_consecutive(out).tolist() if i not in [PAD_ID, BOS_ID, EOS_ID]]))
            preds.append(' '.join([dataset.vocab[i] for i in out.tolist() if i not in [PAD_ID, BOS_ID, EOS_ID]]).strip())
            refs.append(' '.join([dataset.vocab[i] for i in tgt.tolist() if i not in [PAD_ID, BOS_ID, EOS_ID]]).strip())
            #write_to('cmlr_ref_pred2.txt', refs[-1]+' || '+preds[-1]+' || '+str(refs[-1] == preds[-1]))
            print(refs[-1]+' || '+preds[-1]+' || '+str(refs[-1] == preds[-1]))
            n_total += 1
            n_correct += int(refs[-1] == preds[-1])
    test_wer, test_cer = wer(refs, preds), cer(refs, preds)
    print('JIWER wer: {:.4f}, cer: {:.4f}, acc={}/{}={:.2f}'.format(test_wer, test_cer, n_correct, n_total, n_correct/n_total))
    return test_wer, test_cer


@torch.no_grad()
def predict(model_path, dataset, data_path, batch_size=32):
    model = DRLModel(len(dataset.vocab), len(dataset.spks)).to(DEVICE)
    # checkpoint = torch.load(opt.load, map_location=lambda storage, loc: storage)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    states = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(states)
    model.eval()
    print('loading checkpoint path: ' + model_path)
    # data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    #data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, collate_fn=CMLRDataset.collate_pad)
    #data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=LRS3Dataset.collate_pad)
    
    PAD_ID, BOS_ID, EOS_ID = (dataset.vocab.index(x) for x in [PAD, BOS, EOS])
    
    vids, lens = [], []
    #for fp in sorted(os.listdir(data_path), key=lambda x: int(x.split('.')[0])):
    for fp in sorted([f for f in os.listdir(data_path) if f.endswith('.npy')], key=lambda x: int(x.split('_')[1].split('.')[0])):
        #raw_vid = np.load(os.path.join(data_path, fp))  # (T, H, W)
        #vid = raw_vid[np.arange(len(raw_vid)) % 6 != 0].copy()
        vid = np.load(os.path.join(data_path, fp))  # (T, H, W)
        vids.append(vid[:, None].astype(np.float32) / 255.)  # (T, C, H, W)
        lens.append(len(vid))
        if len(vids) >= 64:
            break

    vid_inp = np.zeros((len(vids), max(lens), *vids[0].shape[-3:]))  # BTCHW
    print(vid_inp.shape)
    for i, (arr, L) in enumerate(zip(vids, lens)):
        vid_inp[i, :L] = arr

    vid_inp = torch.FloatTensor(vid_inp).to(DEVICE)
    vid_lens = torch.LongTensor(lens).to(DEVICE)
    aud_inp, aud_lens = None, None
    
    output = model.beam_decode(vid_inp, aud_inp, vid_lens, aud_lens, bos_id=BOS_ID, eos_id=EOS_ID, max_dec_len=40)
    #preds = []
    for i, out in enumerate(output):
        #preds.append(''.join([dataset.vocab[i] for i in out.tolist() if i not in [PAD_ID, BOS_ID, EOS_ID]]))
        print(i, ' '.join([dataset.vocab[i] for i in out.tolist() if i not in [PAD_ID, BOS_ID, EOS_ID]]))
        #write_to('cmlr_ref_pred.txt', refs[-1]+'||'+preds[-1]+'||'+str(refs[-1] == preds[-1]))
    #print(*preds, sep='\n')
    print('Done')


@torch.no_grad()
def one_infer(model_path, dataset, load_path):
    model = DRLModel(len(dataset.vocab), len(dataset.spks)).to(DEVICE)
    # checkpoint = torch.load(opt.load, map_location=lambda storage, loc: storage)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    states = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(states)
    model.eval()
    print('loading checkpoint path: ' + model_path)
    
    PAD_ID, BOS_ID, EOS_ID = (dataset.vocab.index(x) for x in [PAD, BOS, EOS])
    vid = np.load(load_path)[:, None].astype(np.float32) / 255.  # (T, C, H, W)
    lens = [vid.shape[0]]  # (L, )

    vid_inp = torch.FloatTensor(vid).unsqueeze(0).to(DEVICE)
    vid_lens = torch.LongTensor(lens).to(DEVICE)
    aud_inp, aud_lens = None, None

    output = model.beam_decode(vid_inp, aud_inp, vid_lens, aud_lens, bos_id=BOS_ID, eos_id=EOS_ID, max_dec_len=20)
    res = ' '.join([dataset.vocab[i] for i in output[0].tolist() if i not in [PAD_ID, BOS_ID, EOS_ID]])
    print('Result:', res)
    return res 


@torch.no_grad()
def fast_one_infer(model, dataset, load_path):
    model.eval()
    PAD_ID, BOS_ID, EOS_ID = (dataset.vocab.index(x) for x in [PAD, BOS, EOS])
    vid = np.load(load_path)[:, None].astype(np.float32) / 255.  # (T, C, H, W)
    lens = [vid.shape[0]]  # (L, )

    cur_device = next(model.parameters()).device
    vid_inp = torch.FloatTensor(vid).unsqueeze(0).to(cur_device)
    vid_lens = torch.LongTensor(lens).to(cur_device)
    aud_inp, aud_lens = None, None

    output = model.beam_decode(vid_inp, aud_inp, vid_lens, aud_lens, bos_id=BOS_ID, eos_id=EOS_ID, max_dec_len=20)
    res = ' '.join([dataset.vocab[i] for i in output[0].tolist() if i not in [PAD_ID, BOS_ID, EOS_ID]])
    print('Result:', res)
    return res 


def load_gpt(gpt_path='./gpt2'):
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    lm = GPT2LMHeadModel.from_pretrained(gpt_path).eval()
    lm_tokenizer = GPT2Tokenizer.from_pretrained(gpt_path, use_fast=True)
    lm_tokenizer.pad_token = '<pad>'
    return lm, lm_tokenizer

def get_lm_score(lm, lm_tokenizer, texts, device):
    tokens_tensor = lm_tokenizer.batch_encode_plus(texts, return_tensors="pt", padding=True)
    lm = lm.to(device)
    logits = lm(tokens_tensor['input_ids'].to(device), attention_mask=tokens_tensor['attention_mask'].to(device))[0]
    losses = []
    for logit, m, labels in zip(logits, tokens_tensor['attention_mask'], tokens_tensor['input_ids']):
        y, label = logit[:m.sum() - 1], labels[1:m.sum()]
        if y.shape[0] == label.shape[0] and label.shape[0] > 0:
            loss = F.cross_entropy(y, label.to(device), ignore_index=0)   # ignore padding index 
            losses.append(loss.item())
        else:
            losses.append(100)
    losses = 1./np.exp(np.array(losses)) # higher should be treated as better
    return losses

def minmax_normalize(values):  # [0, 1]
    v = np.array(values)
    v = (v - v.min()) / (v.max() - v.min())
    return v

@torch.no_grad()
def evaluate_gpt(model_path, dataset, batch_size=32):
    model = DRLModel(len(dataset.vocab), len(dataset.spks)).to(DEVICE)
    # checkpoint = torch.load(opt.load, map_location=lambda storage, loc: storage)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    states = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(states)
    model.eval()
    print(len(dataset))
    # data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    #data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=CMLRDataset.collate_pad)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=LRS3Dataset.collate_pad)
    preds = []
    refs = []
    PAD_ID, BOS_ID, EOS_ID = (dataset.vocab.index(x) for x in [PAD, BOS, EOS])
    
    lm, lm_tokenizer = load_gpt("./gpt2/en")
    
    for batch_data in data_loader:
        vid_inp = batch_data['vid'].to(DEVICE)
        aud_inp = batch_data['aud'].to(DEVICE)
        tgt_txt = batch_data['txt'].to(DEVICE)
        vid_lens = batch_data['vid_lens'].to(DEVICE)
        aud_lens = batch_data['aud_lens'].to(DEVICE)
        #output = model.greedy_decode(vid_inp, input_lens)
        #output = model.beam_decode(vid_inp, aud_inp, vid_lens, aud_lens, bos_id=BOS_ID, eos_id=EOS_ID, max_dec_len=40)
        
        batch_beam_outs, batch_beam_scores = model.beam_decode(vid_inp, aud_inp, vid_lens, aud_lens, bos_id=BOS_ID, eos_id=EOS_ID, max_dec_len=150)  # B x n_best x dec_len
        for beam_outs, beam_scores in zip(batch_beam_outs, batch_beam_scores):
            pred_txts = []
            for o in beam_outs:
                pred_txts.append(''.join([dataset.vocab[i] for i in o.cpu().numpy().tolist() if i not in [PAD_ID, BOS_ID, EOS_ID]]))
            lm_scores = get_lm_score(lm, lm_tokenizer, pred_txts, DEVICE)
            lm_scores = minmax_normalize(lm_scores)
            beam_scores = minmax_normalize(beam_scores)
            mix_score = 0.4 * lm_scores + 0.6 * beam_scores
            #mix_score = 0.2 * lm_scores + beam_scores
            pred = ''.join([dataset.vocab[i] for i in beam_outs[mix_score.argmax()].cpu().numpy().tolist() if i not in [PAD_ID, BOS_ID, EOS_ID]])
            preds.append(pred)

        for tgt in tgt_txt:
            ## CER
            refs.append(''.join([dataset.vocab[i] for i in tgt.tolist() if i not in [PAD_ID, BOS_ID, EOS_ID]]))
            ## WER
            #ref.append(' '.join([dataset.vocab[i] for i in tgt.tolist() if i not in [PAD_ID, BOS_ID, EOS_ID]]))
            # write_to('pred-cmlr.txt', ref[-1]+'\t'+pred[-1]+'\t'+str(ref[-1] == pred[-1]))
        #print(pred, gold)
    test_wer, test_cer = wer(refs, preds), cer(refs, preds)
    print('JIWER wer: {:.4f}, cer: {:.4f}'.format(test_wer, test_cer))
    return test_wer, test_cer


if __name__ == '__main__':
    seed = 1337  # 42
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

    data_type = str(sys.argv[2])  # train / adapt 
    print('using dataset: ', data_type)
    if data_type == 'train':
        data_root = r'../LipData/CMLR'
        ## CMLR训练
        #train_set = CMLRDataset(data_root, r'data/unseen_train.csv', phase='train', setting='unseen')
        train_set = CMLRDataset(data_root, r'data/all_cmlr.csv', phase='train', setting='unseen')   # train + dev + test
        val_set = CMLRDataset(data_root, r'data/unseen_test.csv', phase='test', setting='unseen')
        avtrain(train_set, val_set, lr=4e-4, epochs=50, batch_size=20, model_path=None)   # linear warmup + cosine decay
        ## 测试
        test_set = CMLRDataset(data_root, r'data/unseen_test.csv', phase='test', setting='unseen')
        evaluate('cmlr_avg_10.pt', test_set, batch_size=64)
        ## 预测
        #predict('cmlr_avg_10.pt', test_set, 'wx_data', batch_size=64)
    elif data_type == 'adapt':
        ## 医护数据微调
        train_set = VSRAppDataset(r'../LipData/data_160', r'data/vsrapp_train.txt', phase='train', setting='unseen')
        model_path = 'cmlr_avg_10_py_all.pt'
        adapt(train_set, model_path, lr=5e-5, epochs=50, batch_size=16)   # 固定lr
        # 测试
        #test_set = VSRAppDataset(r'../LipData/data_160', r'data/vsrapp_test2.txt', phase='test', setting='unseen')
        test_set = VSRAppDataset(r'../LipData/data_160', r'data/vsrapp_new3.txt', phase='test', setting='unseen')
        evaluate('vsrapp4.pt', test_set, batch_size=64)
    else:
        raise NotImplementedError('Invalid Dataset!')
