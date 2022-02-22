# Copyright (c) 2020-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
import os
import torch
import numpy as np

from .utils import to_cuda


TOLERANCE_THRESHOLD = 1e-1


logger = getLogger()


def idx_to_infix(env, idx, input=True):
    """
    Convert an indexed prefix expression to SymPy.
    """
    prefix = [env.id2word[wid] for wid in idx]
    infix = env.input_to_infix(prefix) if input else env.output_to_infix(prefix)
    return infix


def check_equilibrium(env, tgt, hyp):
    if len(hyp) == 0 or len(tgt) == 0:
        return -1.0
    if hyp[0] != tgt[0]:
        return -1.0
    if len(tgt) == 1:
        return 0.0
    nr_values_h = hyp.count(env.separator)
    nr_values_t = tgt.count(env.separator)
    if nr_values_h != nr_values_t:
        return -1.0
    h_vals = np.zeros(nr_values_h, dtype=float)
    h = hyp[2:]
    for j in range(nr_values_h):
        val, pos = env.parse_float(h)
        if np.isnan(val):
            return -1.0
        if j < nr_values_h - 1 and (len(h) <= pos or h[pos] != env.separator):
            return -1.0
        h_vals[j] = val
        h = h[pos + 1:]
    # read target
    t = tgt[2:]
    t_vals = np.zeros(nr_values_t, dtype=float)
    for j in range(nr_values_t):
        val, pos = env.parse_float(t)
        t_vals[j] = val
        t = t[pos + 1:]
    num = 0.0
    den = 0.0
    for j in range(nr_values_t):
        num += abs(t_vals[j] - h_vals[j])
        den += abs(t_vals[j])
    if den == 0.0:
        return 0.0 if num == 0.0 else -1.0
    return num / den


def check_hypothesis(eq):
    """
    Check a hypothesis for a given equation and its solution.
    """
    env = Evaluator.ENV

    src = [env.id2word[wid] for wid in eq['src']]
    tgt = [env.id2word[wid] for wid in eq['tgt']]
    hyp = [env.id2word[wid] for wid in eq['hyp']]

    # update hypothesis
    eq['src'] = env.input_to_infix(src)
    eq['tgt'] = tgt
    eq['hyp'] = hyp
    eq['is_valid'] = check_equilibrium(env, tgt, hyp)
    return eq


class Evaluator(object):

    ENV = None

    def __init__(self, trainer):
        """
        Initialize evaluator.
        """
        self.trainer = trainer
        self.modules = trainer.modules
        self.params = trainer.params
        self.env = trainer.env
        Evaluator.ENV = trainer.env

    def run_all_evals(self):
        """
        Run all evaluations.

        """
        params = self.params
        scores = OrderedDict({'epoch': self.trainer.epoch})

        # save statistics about generated data
        if params.export_data:
            scores['total'] = self.trainer.total_samples
            return scores

        with torch.no_grad():
            for data_type in ['valid']:
                for task in params.tasks:
                    if params.beam_eval:
                        self.enc_dec_step_beam(data_type, task, scores)
                    else:
                        self.enc_dec_step(data_type, task, scores)
        return scores

    def enc_dec_step(self, data_type, task, scores):
        """
        Encoding / decoding step.
        """
        params = self.params
        env = self.env
        encoder = self.modules['encoder'].module if params.multi_gpu else self.modules['encoder']
        decoder = self.modules['decoder'].module if params.multi_gpu else self.modules['decoder']
        encoder.eval()
        decoder.eval()
        assert params.eval_verbose in [0, 1]
        assert params.eval_verbose_print is False or params.eval_verbose > 0
        assert task in ['graph']

        # stats
        xe_loss = 0
        n_valid = torch.zeros(10000, dtype=torch.long)
        n_total = torch.zeros(10000, dtype=torch.long)

        # evaluation details
        if params.eval_verbose:
            eval_path = os.path.join(params.dump_path, f"eval.{data_type}.{task}.{scores['epoch']}")
            f_export = open(eval_path, 'w')
            logger.info(f"Writing evaluation results in {eval_path} ...")

        # iterator
        iterator = self.env.create_test_iterator(
            data_type,
            task,
            data_path=self.trainer.data_path,
            batch_size=params.batch_size_eval,
            params=params,
            size=params.eval_size
        )
        eval_size = len(iterator.dataset)

        for (x1, len1), (x2, len2), nb_ops in iterator:

            # print status
            # FC : remove lengthy pacifiers
            # if n_total.sum().item() % 500 < params.batch_size_eval:
            #    logger.info(f"{n_total.sum().item()}/{eval_size}")

            # target words to predict
            alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
            pred_mask = alen[:, None] < len2[None] - 1  # do not predict anything given the last target word
            y = x2[1:].masked_select(pred_mask[:-1])
            assert len(y) == (len2 - 1).sum().item()

            # cuda
            x1_, len1_, x2, len2, y = to_cuda(x1, len1, x2, len2, y)

            # forward / loss
            encoded = encoder('fwd', x=x1_, lengths=len1_, causal=False)
            decoded = decoder('fwd', x=x2, lengths=len2, causal=True, src_enc=encoded.transpose(0, 1), src_len=len1_)
            word_scores, loss = decoder('predict', tensor=decoded, pred_mask=pred_mask, y=y, get_scores=True)

            # correct outputs per sequence / valid top-1 predictions
            t = torch.zeros_like(pred_mask, device=y.device)
            t[pred_mask] += word_scores.max(1)[1] == y
            valid = (t.sum(0) == len2 - 1).cpu().long()

            # export evaluation details
            if params.eval_verbose:
                for i in range(len(len1)):
                    src = idx_to_infix(env, x1[1:len1[i] - 1, i].tolist(), True)
                    tgt = idx_to_infix(env, x2[1:len2[i] - 1, i].tolist(), False)
                    s = f"Equation {n_total.sum().item() + i} ({'Valid' if valid[i] else 'Invalid'})\nsrc={src}\ntgt={tgt}\n"
                    if params.eval_verbose_print:
                        logger.info(s)
                    f_export.write(s + "\n")
                    f_export.flush()

            # stats
            xe_loss += loss.item() * len(y)
            n_valid.index_add_(-1, nb_ops, valid)
            n_total.index_add_(-1, nb_ops, torch.ones_like(nb_ops))

        # evaluation details
        if params.eval_verbose:
            f_export.close()

        # log
        _n_valid = n_valid.sum().item()
        _n_total = n_total.sum().item()
        logger.info(f"{_n_valid}/{_n_total} ({100. * _n_valid / _n_total}%) equations were evaluated correctly.")

        # compute perplexity and prediction accuracy
        assert _n_total == eval_size
        scores[f'{data_type}_{task}_xe_loss'] = xe_loss / _n_total
        scores[f'{data_type}_{task}_acc'] = 100. * _n_valid / _n_total

        # per class perplexity and prediction accuracy
        for i in range(len(n_total)):
            if n_total[i].item() == 0:
                continue
            e = env.decode_class(i)
            scores[f'{data_type}_{task}_acc_{e}'] = 100. * n_valid[i].item() / max(n_total[i].item(), 1)
            if n_valid[i].item() > 0:
                logger.info(f"{e}: {n_valid[i].item()} / {n_total[i].item()} "
                            f"({100. * n_valid[i].item() / max(n_total[i].item(), 1)}%)")

    def enc_dec_step_beam(self, data_type, task, scores, size=None):
        """
        Encoding / decoding step with beam generation and SymPy check.
        """
        params = self.params
        env = self.env
        encoder = self.modules['encoder'].module if params.multi_gpu else self.modules['encoder']
        decoder = self.modules['decoder'].module if params.multi_gpu else self.modules['decoder']
        encoder.eval()
        decoder.eval()
        assert params.eval_verbose in [0, 1, 2]
        assert params.eval_verbose_print is False or params.eval_verbose > 0
        assert task in ['graph']

        # evaluation details
        if params.eval_verbose:
            eval_path = os.path.join(params.dump_path, f"eval.beam.{data_type}.{task}.{scores['epoch']}")
            f_export = open(eval_path, 'w')
            logger.info(f"Writing evaluation results in {eval_path} ...")

        def display_logs(logs, offset):
            """
            Display detailed results about success / fails.
            """
            if params.eval_verbose == 0:
                return
            for i, res in sorted(logs.items()):
                n_valid = sum([int(v) for _, _, v in res['hyps']])
                s = f"Equation {offset + i} ({n_valid}/{len(res['hyps'])})\nsrc={res['src']}\ntgt={res['tgt']}\n"
                for hyp, score, valid in res['hyps']:
                    if score is None:
                        s += f"{int(valid)} {hyp}\n"
                    else:
                        s += f"{int(valid)} {score :.3e} {hyp}\n"
                if params.eval_verbose_print:
                    logger.info(s)
                f_export.write(s + "\n")
                f_export.flush()

        # stats
        xe_loss = 0
        n_valid = torch.zeros(10000, params.beam_size, dtype=torch.long)
        n_total = torch.zeros(10000, dtype=torch.long)
        n_valid_additional = np.zeros(1 + len(env.additional_tolerance), dtype=int)

        # iterator
        iterator = env.create_test_iterator(
            data_type,
            task,
            data_path=self.trainer.data_path,
            batch_size=params.batch_size_eval,
            params=params,
            size=params.eval_size,
        )
        eval_size = len(iterator.dataset)
        n_perfect_match = 0

        for (x1, len1), (x2, len2), nb_ops in iterator:

            # target words to predict
            alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
            pred_mask = alen[:, None] < len2[None] - 1  # do not predict anything given the last target word
            y = x2[1:].masked_select(pred_mask[:-1])
            assert len(y) == (len2 - 1).sum().item()

            # cuda
            x1_, len1_, x2, len2, y = to_cuda(x1, len1, x2, len2, y)
            bs = len(len1)

            # forward
            encoded = encoder('fwd', x=x1_, lengths=len1_, causal=False)
            decoded = decoder('fwd', x=x2, lengths=len2, causal=True, src_enc=encoded.transpose(0, 1), src_len=len1_)
            word_scores, loss = decoder('predict', tensor=decoded, pred_mask=pred_mask, y=y, get_scores=True)

            # correct outputs per sequence / valid top-1 predictions
            t = torch.zeros_like(pred_mask, device=y.device)
            t[pred_mask] += word_scores.max(1)[1] == y
            valid = (t.sum(0) == len2 - 1).cpu().long()
            n_perfect_match += valid.sum().item()

            # save evaluation details
            beam_log = {}
            for i in range(len(len1)):
                src = idx_to_infix(env, x1[1:len1[i] - 1, i].tolist(), True)
                tgt = idx_to_infix(env, x2[1:len2[i] - 1, i].tolist(), False)
                if valid[i]:
                    beam_log[i] = {'src': src, 'tgt': tgt, 'hyps': [(tgt, None, True)]}

            # stats
            xe_loss += loss.item() * len(y)
            n_valid[:, 0].index_add_(-1, nb_ops, valid)
            n_total.index_add_(-1, nb_ops, torch.ones_like(nb_ops))

            # continue if everything is correct. if eval_verbose, perform
            # a full beam search, even on correct greedy generations
            if valid.sum() == len(valid) and params.eval_verbose < 2:
                display_logs(beam_log, offset=n_total.sum().item() - bs)
                continue

            # invalid top-1 predictions - check if there is a solution in the beam
            invalid_idx = (1 - valid).nonzero().view(-1)
            logger.info(f"({n_total.sum().item()}/{eval_size}) Found {bs - len(invalid_idx)}/{bs} "
                        f"valid top-1 predictions. Generating solutions ...")

            # generate
            _, _, generations = decoder.generate_beam(
                encoded.transpose(0, 1),
                len1_,
                beam_size=params.beam_size,
                length_penalty=params.beam_length_penalty,
                early_stopping=params.beam_early_stopping,
                max_len=params.max_len
            )

            # prepare inputs / hypotheses to check
            # if eval_verbose < 2, no beam search on equations solved greedily
            inputs = []
            for i in range(len(generations)):
                if valid[i] and params.eval_verbose < 2:
                    continue
                for j, (score, hyp) in enumerate(sorted(generations[i].hyp, key=lambda x: x[0], reverse=True)):
                    inputs.append({
                        'i': i,
                        'j': j,
                        'score': score,
                        'src': x1[1:len1[i] - 1, i].tolist(),
                        'tgt': x2[1:len2[i] - 1, i].tolist(),
                        'hyp': hyp[1:].tolist(),
                        'task': task,
                    })

            # check hypotheses with multiprocessing
            outputs = []
            if params.windows is True:
                for inp in inputs:
                    outputs.append(check_hypothesis(inp))
            else:
                with ProcessPoolExecutor(max_workers=20) as executor:
                    for output in executor.map(check_hypothesis, inputs, chunksize=1):
                        outputs.append(output)

            # read results
            for i in range(bs):

                # select hypotheses associated to current equation
                gens = sorted([o for o in outputs if o['i'] == i], key=lambda x: x['j'])
                assert (len(gens) == 0) == (valid[i] and params.eval_verbose < 2) and (i in beam_log) == valid[i]
                if len(gens) == 0:
                    continue

                # source / target
                src = gens[0]['src']
                tgt = gens[0]['tgt']
                beam_log[i] = {'src': src, 'tgt': tgt, 'hyps': []}

                # for each hypothesis
                for j, gen in enumerate(gens):

                    # sanity check
                    assert gen['src'] == src and gen['tgt'] == tgt and gen['i'] == i and gen['j'] == j

                    # if hypothesis is correct, and we did not find a correct one before
                    is_valid = gen['is_valid']
                    is_b_valid = (is_valid >= 0.0 and is_valid < env.float_tolerance)
                    if is_valid >= 0.0 and not valid[i]:
                        for k, tol in enumerate(env.additional_tolerance):
                            if is_valid < tol:
                                n_valid_additional[k] += 1
                        if is_valid < env.float_tolerance:
                            n_valid[nb_ops[i], j] += 1
                            valid[i] = 1

                    # update beam log
                    beam_log[i]['hyps'].append((gen['hyp'], gen['score'], is_b_valid))

            # valid solutions found with beam search
            logger.info(f"    Found {valid.sum().item()}/{bs} solutions in beam hypotheses.")

            # export evaluation details
            if params.eval_verbose:
                assert len(beam_log) == bs
                display_logs(beam_log, offset=n_total.sum().item() - bs)

        # evaluation details
        if params.eval_verbose:
            f_export.close()
            logger.info(f"Evaluation results written in {eval_path}")

        # log
        _n_valid = n_valid.sum().item()
        _n_total = n_total.sum().item()
        logger.info(f"{_n_valid}/{_n_total} ({100. * _n_valid / _n_total}%) equations were evaluated correctly.")

        # compute perplexity and prediction accuracy
        assert _n_total == eval_size
        scores[f'{data_type}_{task}_xe_loss'] = xe_loss / _n_total
        scores[f'{data_type}_{task}_beam_acc'] = 100. * _n_valid / _n_total
        for i in range(len(env.additional_tolerance)):
            scores[f'{data_type}_{task}_additional_{i+1}'] = 100. * (n_perfect_match + n_valid_additional[i]) / _n_total

        # per class perplexity and prediction accuracy
        for i in range(len(n_total)):
            if n_total[i].item() == 0:
                continue
            e = env.decode_class(i)
            logger.info(f"{e}: {n_valid[i].sum().item()} / {n_total[i].item()} "
                        f"({100. * n_valid[i].sum().item() / max(n_total[i].item(), 1)}%)")
            scores[f'{data_type}_{task}_beam_acc_{e}'] = 100. * n_valid[i].sum().item() / max(n_total[i].item(), 1)


def convert_to_text(batch, lengths, id2word, params):
    """
    Convert a batch of sequences to a list of text sequences.
    """
    batch = batch.cpu().numpy()
    lengths = lengths.cpu().numpy()

    slen, bs = batch.shape
    assert lengths.max() == slen and lengths.shape[0] == bs
    assert (batch[0] == params.eos_index).sum() == bs
    assert (batch == params.eos_index).sum() == 2 * bs
    sequences = []

    for j in range(bs):
        words = []
        for k in range(1, lengths[j]):
            if batch[k, j] == params.eos_index:
                break
            words.append(id2word[batch[k, j]])
        sequences.append(" ".join(words))
    return sequences
