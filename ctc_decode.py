import numpy as np
import collections
import torch


# 效果差
def ctc_beam_decode(probs, beam_size=10, blank=0, neg_inf=-float("inf")):
    '''
    probs: 概率空间，shape为[sequence_len, vocab_size]的torch tensor
    beam_size: beam_size
    blank: blank index
    '''
    def log_sum_exp(lps):
        if all(lp == neg_inf for lp in lps):
            return neg_inf
        mlp = max(lps)
        return mlp + np.log(sum(np.exp(lp - mlp) for lp in lps))

    seqs = [((idx.item(),), (lp.item(), neg_inf)) if idx.item() != blank
            else (tuple(), (neg_inf, lp.item()))
            for lp, idx in zip(*probs[0].topk(beam_size))]
    for i in range(1, probs.size(0)):
        new_seqs = {}
        for seq, (lps, blps) in seqs:
            last = seq[-1] if len(seq) > 0 else None
            for lp, idx in zip(*probs[i].topk(beam_size)):
                lp = lp.item()
                idx = idx.item()
                if idx == blank:
                    nlps, nblps = new_seqs.get(seq, (neg_inf, neg_inf))
                    new_seqs[seq] = (nlps, log_sum_exp([nblps, lps + lp, blps + lp]))
                elif idx == last:
                    # aa
                    nlps, nblps = new_seqs.get(seq, (neg_inf, neg_inf))
                    new_seqs[seq] = (log_sum_exp([nlps, lps + lp]), nblps)
                    # a-a
                    new_seq = seq + (idx,)
                    nlps, nblps = new_seqs.get(new_seq, (neg_inf, neg_inf))
                    new_seqs[new_seq] = (log_sum_exp([nlps, blps + lp]), nblps)
                else:
                    new_seq = seq + (idx,)
                    nlps, nblps = new_seqs.get(new_seq, (neg_inf, neg_inf))
                    new_seqs[new_seq] = (log_sum_exp([nlps, lps + lp, blps + lp]), nblps)

        new_seqs = sorted(
            new_seqs.items(),
            key=lambda x: log_sum_exp(list(x[1])),
            reverse=True)
        seqs = new_seqs[:beam_size]
    return seqs[0][0]



# prefix beam search
# 效果好，但是非常慢
def ctc_beam_decode2(probs, beam_size=10, blank=0, neg_inf=-float("inf")):
    """
    Performs inference for the given output probabilities.
    Arguments:
        probs: The output probabilities (e.g. post-softmax) for each
          time step. Should be an array of shape (time x output dim).
        beam_size (int): Size of the beam to use during inference.
        blank (int): Index of the CTC blank label.
    Returns the output label sequence and the corresponding negative
    log-likelihood estimated by the decoder.
    :param neg_inf: negative infinity
    """
    def make_new_beam():
        fn = lambda: (neg_inf, neg_inf)
        return collections.defaultdict(fn)

    def log_sum_exp(*args):
        """
        Stable log sum exp.
        """
        if all(a == neg_inf for a in args):
            return neg_inf
        a_max = max(args)
        lsp = np.log(sum(np.exp(a - a_max) for a in args))
        return a_max + lsp

    T, S = probs.shape
    probs = np.log(probs)
    # Elements in the beam are (prefix, (p_blank, p_no_blank))
    # Initialize the beam with the empty sequence, a probability of
    # 1 for ending in blank and zero for ending in non-blank
    # (in log space).
    beam = [(tuple(), (0.0, neg_inf))]
    for t in range(T):  # Loop over time
        # A default dictionary to store the next step candidates.
        next_beam = make_new_beam()
        for s in range(S):  # Loop over vocab
            p = probs[t, s]
            # The variables p_b and p_nb are respectively the
            # probabilities for the prefix given that it ends in a
            # blank and does not end in a blank at this time step.
            for prefix, (p_b, p_nb) in beam:  # Loop over beam
                # If we propose a blank the prefix doesn't change.
                # Only the probability of ending in blank gets updated.
                if s == blank:
                    n_p_b, n_p_nb = next_beam[prefix]
                    n_p_b = log_sum_exp(n_p_b, p_b + p, p_nb + p)
                    next_beam[prefix] = (n_p_b, n_p_nb)
                    continue
                # Extend the prefix by the new character s and add it to
                # the beam. Only the probability of not ending in blank
                # gets updated.
                end_t = prefix[-1] if prefix else None
                n_prefix = prefix + (s,)
                n_p_b, n_p_nb = next_beam[n_prefix]
                if s != end_t:
                    n_p_nb = log_sum_exp(n_p_nb, p_b + p, p_nb + p)
                else:
                    # We don't include the previous probability of not ending
                    # in blank (p_nb) if s is repeated at the end. The CTC
                    # algorithm merges characters not separated by a blank.
                    n_p_nb = log_sum_exp(n_p_nb, p_b + p)
                # *NB* this would be a good place to include an LM score.
                next_beam[n_prefix] = (n_p_b, n_p_nb)
                # If s is repeated at the end we also update the unchanged prefix. This is the merging case.
                if s == end_t:
                    n_p_b, n_p_nb = next_beam[prefix]
                    n_p_nb = log_sum_exp(n_p_nb, p_nb + p)
                    next_beam[prefix] = (n_p_b, n_p_nb)
        # Sort and trim the beam before moving on to the next time-step.
        beam = sorted(next_beam.items(), key=lambda x: log_sum_exp(*x[1]), reverse=True)
        beam = beam[:beam_size]
    best = beam[0]
    #return best[0], -log_sum_exp(*best[1])
    return best[0]



NEG_INF = -float("inf")
#NEG_INF = -torch.inf
def ctc_beam_decode3(log_prob, beam_size=10, blank=0):
    # log_prob: log-softmax-logits    
    def logsumexp(*args):
        if all(a == NEG_INF for a in args):
            return NEG_INF
        a_max = max(args)
        lsp = np.log(sum(np.exp(a - a_max) for a in args))
        return a_max + lsp
    
    #def logsumexp(*args):
    #    if all(a == NEG_INF for a in args):
    #        return NEG_INF
    #    return torch.logsumexp(torch.tensor([a for a in args]), dim=0)
    
    T, V = log_prob.shape
    #log_prob = np.log(prob)
    #log_prob = torch.log(prob)
    beam = [(tuple(), (0, NEG_INF))]  # blank, non-blank
    for t in range(T):  # for every timestep
        new_beam = collections.defaultdict(lambda: (NEG_INF, NEG_INF))
        for prefix, (p_b, p_nb) in beam:
            for i in range(V):  # for every state
                p = log_prob[t, i]
                if i == blank:  # propose a blank
                    new_p_b, new_p_nb = new_beam[prefix]
                    new_p_b = logsumexp(new_p_b, p_b + p, p_nb + p)
                    new_beam[prefix] = (new_p_b, new_p_nb)
                    continue
                else:  # extend with non-blank
                    end_t = prefix[-1] if prefix else None
                    # exntend current prefix
                    new_prefix = prefix + (i,)
                    new_p_b, new_p_nb = new_beam[new_prefix]
                    if i != end_t:
                        new_p_nb = logsumexp(new_p_nb, p_b + p, p_nb + p)
                    else:
                        new_p_nb = logsumexp(new_p_nb, p_b + p)
                    new_beam[new_prefix] = (new_p_b, new_p_nb)
                    # keep current prefix
                    if i == end_t:
                        new_p_b, new_p_nb = new_beam[prefix]
                        new_p_nb = logsumexp(new_p_nb, p_nb + p)
                        new_beam[prefix] = (new_p_b, new_p_nb)
        # top beam_size
        beam = sorted(new_beam.items(), key=lambda x: logsumexp(*x[1]), reverse=True)
        beam = beam[:beam_size]
    return beam[0][0]


