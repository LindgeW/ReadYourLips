import torch.nn as nn
import numpy as np


def load_gpt():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    lm = GPT2LMHeadModel.from_pretrained('gpt2').eval()
    lm_tokenizer = GPT2Tokenizer.from_pretrained('gpt2', use_fast=True)
    lm_tokenizer.pad_token = '<pad>'
    return lm, lm_tokenizer


def get_lm_score(lm, lm_tokenizer, texts):
    logloss = nn.CrossEntropyLoss()
    tokens_tensor = lm_tokenizer.batch_encode_plus(texts, return_tensors="pt", padding=True)
    logits = lm(tokens_tensor['input_ids'], attention_mask=tokens_tensor['attention_mask'])[0]
    losses = []
    for logits, m, labels in zip(logits, tokens_tensor['attention_mask'], tokens_tensor['input_ids']):
	loss = logloss(logits[:m.sum() - 1], labels[1:m.sum()])
	losses.append(loss.item())
    losses = 1./np.exp(np.array(losses)) # higher should be treated as better
    return losses


def minmax_normalize(values):
    v = np.array(values)
    v = (v - v.min()) / (v.max() - v.min())
    return v



def infer_wgpt():
preds = []
	for i in range(0, frames.size(2), chunk_frames):
		cur_src = frames[:, :, i : i + chunk_frames].to(args.device)
		cur_src_mask = torch.ones((1, 1, cur_src.size(2))).to(args.device)

		with torch.no_grad():
			with autocast():
				beam_outs, beam_scores = forward_pass(model, cur_src, cur_src_mask)
				beam_outs_f, beam_scores_f = forward_pass(model, augmentor.horizontal_flip(cur_src), cur_src_mask)

				beam_outs = beam_outs[0] + beam_outs_f[0]
				beam_scores = np.array(beam_scores[0] + beam_scores_f[0])

				if lm is not None:
					pred_texts = [dataloader.to_tokens(o.cpu().numpy().tolist()) for o in beam_outs]
					lm_scores = get_lm_score(lm, lm_tokenizer, pred_texts)
					lm_scores = minmax_normalize(lm_scores)
					beam_scores = minmax_normalize(beam_scores)
					beam_scores = args.lm_alpha * lm_scores + (1 - args.lm_alpha) * beam_scores

		best_pred_idx = beam_scores.argmax()
		out = beam_outs[best_pred_idx]
		pred = dataloader.to_tokens(out.cpu().numpy().tolist())
		preds.append(pred)

	pred = ' '.join(preds)
	if display: print(pred)
	return pred
