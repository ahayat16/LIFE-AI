# Copyright (c) 2020-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os
import io
import sys
import math
import numpy as np
import GraphCreationLIFE as gcl
import networkx as nx

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from ..utils import bool_flag


SPECIAL_WORDS = ["<s>", "</s>", "<pad>", "(", ")"]
SPECIAL_WORDS = SPECIAL_WORDS + [f"<SPECIAL_{i}>" for i in range(10)]


logger = getLogger()


class InvalidPrefixExpression(Exception):
    def __init__(self, data):
        self.data = data

    def __str__(self):
        return repr(self.data)


class GraphEnvironment(object):

    TRAINING_TASKS = {"graph"}

    def __init__(self, params):

        self.max_len = params.max_len
        self.min_nodes = params.min_nodes
        self.max_nodes = params.max_nodes
        assert self.min_nodes <= self.max_nodes and self.max_nodes <= 1000
        self.min_edges = params.min_edges
        self.max_edges = params.max_edges
        assert self.min_edges <= self.max_edges
        self.generator = params.generator
        self.redeem_prob = params.redeem_prob
        self.add_random = params.add_random
        self.predict_eq = params.predict_eq
        self.float_precision = params.float_precision
        self.float_tolerance = params.float_tolerance
        self.weight_tokens = params.weight_tokens
        self.additional_tolerance = [
            float(x) for x in params.more_tolerance.split(",") if len(x) > 0
        ]

        self.tokenized_weights = params.tokenized_weights

        self.weighted = params.weighted

        self.proba_rewriting = params.proba_rewriting
        # (
        # self.rng.rand()
        # if params.proba_rewritting is None
        # else params.proba_rewritting
        # )

        # symbols / elements
        self.constants = []
        self.symbols = ["INT+", "INT-", "FLOAT+", "FLOAT-", ".", "10^"]
        # base 10
        self.elements = [str(i) for i in range(10)]
        # hard limit on 1000 nodes FIXME
        self.nodevars = [f"N{i}" for i in range(1000)]
        self.weightvars = [f"W{i}" for i in range(101)] if self.weight_tokens else []

        # vocabulary
        self.words = (
            SPECIAL_WORDS
            + self.constants
            + self.symbols
            + self.elements
            + self.nodevars
            + self.weightvars
        )
        self.id2word = {i: s for i, s in enumerate(self.words)}
        self.word2id = {s: i for i, s in self.id2word.items()}
        assert len(self.words) == len(set(self.words))

        # number of words / indices
        self.n_words = params.n_words = len(self.words)
        self.eos_index = params.eos_index = 0
        self.pad_index = params.pad_index = 1
        self.separator = "<SPECIAL_3>"
        logger.info(f"words: {self.word2id}")

    def batch_sequences(self, sequences):
        """
        Take as input a list of n sequences (torch.LongTensor vectors) and return
        a tensor of size (slen, n) where slen is the length of the longest
        sentence, and a vector lengths containing the length of each sentence.
        """
        lengths = torch.LongTensor([len(s) + 2 for s in sequences])
        sent = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(
            self.pad_index
        )
        assert lengths.min().item() > 2

        sent[0] = self.eos_index
        for i, s in enumerate(sequences):
            sent[1 : lengths[i] - 1, i].copy_(s)
            sent[lengths[i] - 1, i] = self.eos_index

        return sent, lengths

    def write_int(self, val, pad=-1):
        """
        Convert a decimal integer to a representation in the given base.
        The base can be negative.
        In balanced bases (positive), digits range from -(base-1)//2 to (base-1)//2
        """
        base = 10  # self.int_base
        balanced = False  # self.balanced
        res = []
        max_digit = abs(base)
        if balanced:
            max_digit = (base - 1) // 2
        else:
            if base > 0:
                neg = val < 0
                val = -val if neg else val
        while True:
            rem = val % base
            val = val // base
            if rem < 0 or rem > max_digit:
                rem -= base
                val += 1
            res.append(str(rem))
            if val == 0:
                break
        if pad >= 0:
            while len(res) <= pad:
                res.append("0")
        if base < 0 or balanced:
            res.append("INT+")
        else:
            res.append("INT-" if neg else "INT+")
        return res[::-1]

    def parse_int(self, lst):
        """
        Parse a list that starts with an integer.
        Return the integer value, and the position it ends in the list.
        """
        base = 10  # self.int_base
        val = 0
        if (
            len(lst) < 2
            or lst[0] not in ["INT+", "INT-"]
            or not (lst[1].isdigit() or lst[1][0] == "-" and lst[1][1:].isdigit())
        ):
            raise InvalidPrefixExpression("Invalid integer in prefix expression")
        i = 0
        for x in lst[1:]:
            if not (x.isdigit() or x[0] == "-" and x[1:].isdigit()):
                break
            val = val * base + int(x)
            i += 1
        if base > 0 and lst[0] == "INT-":
            val = -val
        return val, i + 1

    def write_float(self, value, precision=None):
        """
        Write a float number.
        """
        precision = self.float_precision if precision is None else precision
        assert value not in [-np.inf, np.inf]
        res = ["FLOAT+"] if value >= 0.0 else ["FLOAT-"]
        m, e = (f"%.{precision}e" % abs(value)).split("e")
        assert e[0] in ["+", "-"]
        e = int(e[1:] if e[0] == "+" else e)
        return res + list(m) + ["10^"] + self.write_int(e)

    def parse_float(self, lst):
        """
        Parse a list that starts with a float.
        Return the float value, and the position it ends in the list.
        """
        if len(lst) < 2 or lst[0] not in ["FLOAT+", "FLOAT-"]:
            return np.nan, 0
        sign = -1 if lst[0] == "FLOAT-" else 1
        if not lst[1].isdigit():
            return np.nan, 1
        mant = 0.0
        i = 1
        for x in lst[1:]:
            if not (x.isdigit()):
                break
            mant = mant * 10.0 + int(x)
            i += 1
        if len(lst) > i and lst[i] == ".":
            i += 1
            mul = 0.1
            for x in lst[i:]:
                if not (x.isdigit()):
                    break
                mant += mul * int(x)
                mul *= 0.1
                i += 1
        mant *= sign
        if len(lst) > i and lst[i] == "10^":
            i += 1
            try:
                exp, offset = self.parse_int(lst[i:])
            except InvalidPrefixExpression:
                return np.nan, i
            i += offset
        else:
            exp = 0
        try:
            value = mant * (10.0 ** exp)
        except Exception:
            return np.nan, i
        return value, i

    def input_to_infix(self, lst):
        pos = 1
        s = lst[0] + "|"

        while pos < len(lst):
            s += "(" + lst[pos] + ", " + lst[pos + 1]
            pos += 2
            if self.weighted:
                if self.tokenized_weights:
                    s += ", " + lst[pos]
                    pos += 1
                else:
                    val, length = self.parse_int(lst[pos:])
                    pos += length
                    s += ", " + str(val)
            s += "),"

        return s

    def output_to_infix(self, lst):
        return lst[0]

    def gen_expr(self, data_type=None):
        """
        Generate pairs of integers/smallest divisors or integers/gcd.
        Encode this as a prefix sentence
        """
        result = 0  # or 1
        nr_nodes = self.rng.randint(self.min_nodes, self.max_nodes + 1)
        nr_edges = round(
            (self.min_edges + self.rng.rand() * (self.max_edges - self.min_edges))
            * nr_nodes
        )
        # HERE BE GENERATION CODE
        if self.generator == "erdos":
            G = nx.gnp_random_graph(
                nr_nodes, nr_edges / (nr_nodes * (nr_nodes - 1)), directed=True
            )  # Erdos random graph.
        elif self.generator == "erdos_gnm":
            G = nx.gnm_random_graph(nr_nodes, nr_edges, directed=True)
            # a graph is chosen uniformly at random from the set of all graphs with
            # $n$ nodes and $m$ edges.
        elif self.generator == "small_world":
            if self.proba_rewriting is None:
                nr_rew = self.rng.rand()
            else:
                nr_rew = self.proba_rewriting
            G = gcl.modified_watts_strogatz_graph(
                nr_nodes, int(2 * nr_edges / nr_nodes), nr_rew
            )
        elif self.generator == "scale_free_directed":
            G = nx.scale_free_graph(
                nr_nodes,
                alpha=0.9 * nr_nodes / nr_edges,
                beta=(nr_edges - nr_nodes) / nr_edges,
                gamma=0.1 * nr_nodes / nr_edges,
                delta_in=0.2,
                delta_out=0,
            )
            # G = nx.neumann_watts_strogatz_graph(
            #     nr_nodes, nr_edges / (nr_nodes * (nr_nodes - 1)), nr_rew
            # )
        else:
            return None

        C = np.array(
            nx.convert_matrix.to_numpy_matrix(G)
        )  # Convert graph to adjacency matrix

        if self.generator == "scale_free_directed":
            for i in range(len(C)):
                C[i][i] = 0
            C1 = []
            for c in C:
                c0 = []
                for s in c:
                    if s != 0:
                        s = 1
                    c0.append(s)
                C1.append(c0)
            C = np.array(C1)

        # ADD IN NODES TO SERVE AS INTAKES AND EXCRETIONS
        # THIS CAN BE DONE WITHOUT FIRST CONVERTING TO MATRIX AND WOULD
        # PROBABLY IMROVE SPEED
        D = np.append(C, np.zeros((1, C.shape[1])), axis=0)
        D = np.append(D, np.zeros((D.shape[0], 1)), axis=1)
        D[self.rng.randint(len(D) - 1), -1] = 1

        D = np.append(np.zeros((1, D.shape[1])), D, axis=0)
        D = np.append(np.zeros((D.shape[0], 1)), D, axis=1)
        D[0, self.rng.randint(1, len(D) - 1)] = 1

        # ADD RANDOM WEIGHTS WITH A VALUE FROM 1 TO 100
        if self.weighted is True:
            D1 = []
            for c in D:
                c0 = []
                for s in c:
                    if s == 1:
                        s = np.random.randint(1, 100)
                    c0.append(s)
                D1.append(c0)
            D = np.array(D1)
            G = nx.convert_matrix.from_numpy_matrix(
                D[1:-1, 1:-1], create_using=nx.DiGraph
            )

        # CREATE A NEW GRAPH USING ADJACENCY WITH INTAKE/EXCRETION
        GG = nx.convert_matrix.from_numpy_matrix(D, create_using=nx.DiGraph)

        # FIND NODES WITH PATH FROM INTAKE WITHOUT PATHS TO EXCRETION
        # PRESENCE OF THESE NODES WILL INDICATE NO EQUILIBRIUM
        non_eq_nodes = gcl.findpaths(GG)

        if not non_eq_nodes:  # If there is a natural equilibrium
            # FIND THE EQUILIBRIUM
            equilibrium = gcl.find_eq(G, D[0, 1:-1], D[1:-1, -1], False)
            # IF DESIRED THE EQUILIBRIUM CAN BE CHECKED
            # if test_the_eq:
            # FC should this be done ?
            # if not gcl.test_eq(D,equilibrium, False):
            #     return None
            result = 1

        else:  # If there is not a natrual equilibrium
            equilibrium = math.nan  # set a value for equilibrium
            # IF THE NUMBER OF FORCED EQUILIBRIUM GRAPHS IS LESS THAN DESIRED,
            # COMPLETE THE GRAPH TO CREATE AN EQUILIBRIUM
            if self.redeem_prob > 0 and self.rng.rand() < self.redeem_prob:
                GG, equilibrium = gcl.create_eq(
                    GG, non_eq_nodes, self.weighted, self.add_random
                )
                # # FC do we need to test?
                # if not gcl.test_eq(nx.adjacency_matrix(GG),equilibrium):
                #     return None
                # ADD THE NEW EDGE ADDED GRAPH WITH AN EQ
                result = 1

            # if creating an equilibrium is not desired add it to the noneqparamslist
            else:
                result = 0
                if self.redeem_prob == -1.0 and self.rng.randint(1, 6) > 1:
                    return None
        # filter
        if result == 0 and self.predict_eq:
            return None

        # encode input
        x = []
        x.append(f"N{nr_nodes}")
        for n in GG.edges(data=True):
            x.append(f"N{n[0]}")
            x.append(f"N{n[1]}")
            if self.weighted:
                w = int(n[2]["weight"])
                if self.tokenized_weights:
                    x.append(f"N{w}")
                else:
                    x.extend(self.write_int(w))

        # encode output
        y = [f"N{result}"]
        if self.predict_eq is True and result == 1:
            for val in equilibrium:
                y.append(self.separator)
                y.extend(self.write_float(val))

        if self.max_len > 0 and (len(x) >= self.max_len or len(y) >= self.max_len):
            return None

        return x, y

    def decode_class(self, i):
        e = "U" if i < 1000 else "S"
        if i >= 1000:
            i -= 1000
        return e + "-" + str(100 * i) + "-" + str(100 * i + 99)

    def code_class(self, xi, yi):
        nre = len(xi) // 200
        result = int(yi[0][1:])
        return result * 1000 + nre

    def create_train_iterator(self, task, data_path, params):
        """
        Create a dataset for this environment.
        """
        logger.info(f"Creating train iterator for {task} ...")

        dataset = EnvDataset(
            self,
            task,
            train=True,
            params=params,
            path=(None if data_path is None else data_path[task][0]),
        )
        return DataLoader(
            dataset,
            timeout=(0 if params.num_workers == 0 else 1800),
            batch_size=params.batch_size,
            num_workers=(
                params.num_workers
                if data_path is None or params.num_workers == 0
                else 1
            ),
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )

    def create_test_iterator(
        self, data_type, task, data_path, batch_size, params, size
    ):
        """
        Create a dataset for this environment.
        """
        assert data_type in ["valid", "test"]
        logger.info(f"Creating {data_type} iterator for {task} ...")

        dataset = EnvDataset(
            self,
            task,
            train=False,
            params=params,
            path=(
                None
                if data_path is None
                else data_path[task][1 if data_type == "valid" else 2]
            ),
            size=size,
            type=data_type,
        )
        return DataLoader(
            dataset,
            timeout=0,
            batch_size=batch_size,
            num_workers=1,
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )

    @staticmethod
    def register_args(parser):
        """
        Register environment parameters.
        """
        parser.add_argument(
            "--eval_size",
            type=int,
            default=10000,
            help="Size of valid and test samples",
        )

        parser.add_argument(
            "--min_nodes", type=int, default=32, help="minimum nodes in graph"
        )
        parser.add_argument(
            "--max_nodes", type=int, default=64, help="maximum nodes in graph"
        )
        parser.add_argument(
            "--min_edges",
            type=float,
            default=2.0,
            help="minimum ratio of edges over nodes in graph",
        )
        parser.add_argument(
            "--max_edges",
            type=float,
            default=4.0,
            help="maximum ration of edges over nodes in graph",
        )
        parser.add_argument(
            "--generator", type=str, default="erdos", help="Type of graph ot generate"
        )
        parser.add_argument(
            "--proba_rewriting",
            type=float,
            default=None,
            help="rewriting proba for small world model",
        )
        parser.add_argument(
            "--add_random", type=bool_flag, default=False, help="Random edge add"
        )
        parser.add_argument(
            "--redeem_prob",
            type=float,
            default=0.4,
            help="probability of stabilizing an unstable graph",
        )
        parser.add_argument(
            "--predict_eq", type=bool_flag, default=False, help="predict equilibrium"
        )
        parser.add_argument(
            "--weighted", type=bool_flag, default=False, help="weighted graph"
        )
        parser.add_argument(
            "--weight_tokens",
            type=bool_flag,
            default=False,
            help="special tokens for weights",
        )
        parser.add_argument(
            "--tokenized_weights",
            type=bool_flag,
            default=True,
            help="weights as symbolic tokens",
        )
        parser.add_argument(
            "--float_precision", type=int, default=2, help="precisions of floats"
        )
        parser.add_argument(
            "--float_tolerance",
            type=float,
            default=0.1,
            help="error tolerance for float results",
        )
        parser.add_argument(
            "--more_tolerance", type=str, default="", help="additional tolerance limits"
        )


class EnvDataset(Dataset):
    def __init__(self, env, task, train, params, path, size=None, type=None):
        super(EnvDataset).__init__()
        self.env = env
        self.train = train
        self.task = task
        self.batch_size = params.batch_size
        self.env_base_seed = params.env_base_seed
        self.path = path
        self.global_rank = params.global_rank
        self.count = 0
        self.type = type
        assert task in GraphEnvironment.TRAINING_TASKS
        assert size is None or not self.train
        assert not params.batch_load or params.reload_size > 0

        # batching
        self.num_workers = params.num_workers
        self.batch_size = params.batch_size

        self.batch_load = params.batch_load
        self.reload_size = params.reload_size
        self.local_rank = params.local_rank
        self.n_gpu_per_node = params.n_gpu_per_node

        self.basepos = 0
        self.nextpos = 0
        self.seekpos = 0

        # generation, or reloading from file
        if path is not None:
            assert os.path.isfile(path)
            if params.batch_load and self.train:
                self.load_chunk()
            else:
                logger.info(f"Loading data from {path} ...")
                with io.open(path, mode="r", encoding="utf-8") as f:
                    # either reload the entire file, or the first N lines
                    # (for the training set)
                    if not train:
                        lines = [line.rstrip().split("|") for line in f]
                    else:
                        lines = []
                        for i, line in enumerate(f):
                            if i == params.reload_size:
                                break
                            if i % params.n_gpu_per_node == params.local_rank:
                                lines.append(line.rstrip().split("|"))
                self.data = [xy.split("\t") for _, xy in lines]
                self.data = [xy for xy in self.data if len(xy) == 2]
                logger.info(f"Loaded {len(self.data)} equations from the disk.")

        # dataset size: infinite iterator for train, finite for valid / test
        # (default of 5000 if no file provided)
        if self.train:
            self.size = 1 << 60
        elif size is None:
            self.size = 10000 if path is None else len(self.data)
        else:
            assert size > 0
            self.size = size

    def load_chunk(self):
        self.basepos = self.nextpos
        logger.info(
            f"Loading data from {self.path} ... seekpos {self.seekpos}, basepos {self.basepos}"
        )
        endfile = False
        with io.open(self.path, mode="r", encoding="utf-8") as f:
            f.seek(self.seekpos, 0)
            lines = []
            for i in range(self.reload_size):
                line = f.readline()
                if not line:
                    endfile = True
                    break
                if i % self.n_gpu_per_node == self.local_rank:
                    lines.append(line.rstrip().split("|"))
            self.seekpos = 0 if endfile else f.tell()

        self.data = [xy.split("\t") for _, xy in lines]
        self.data = [xy for xy in self.data if len(xy) == 2]
        self.nextpos = self.basepos + len(self.data)
        logger.info(
            f"Loaded {len(self.data)} equations from the disk. seekpos {self.seekpos}, nextpos {self.nextpos}"
        )
        if len(self.data) == 0:
            self.load_chunk()

    def collate_fn(self, elements):
        """
        Collate samples into a batch.
        """
        x, y = zip(*elements)
        nb_eqs = [self.env.code_class(xi, yi) for xi, yi in zip(x, y)]
        x = [torch.LongTensor([self.env.word2id[w] for w in seq]) for seq in x]
        y = [torch.LongTensor([self.env.word2id[w] for w in seq]) for seq in y]
        x, x_len = self.env.batch_sequences(x)
        y, y_len = self.env.batch_sequences(y)
        return (x, x_len), (y, y_len), torch.LongTensor(nb_eqs)

    def init_rng(self):
        """
        Initialize random generator for training.
        """
        if hasattr(self.env, "rng"):
            return
        if self.train:
            worker_id = self.get_worker_id()
            self.env.worker_id = worker_id
            self.env.rng = np.random.RandomState(
                [worker_id, self.global_rank, self.env_base_seed]
            )
            logger.info(
                f"Initialized random generator for worker {worker_id}, with seed "
                f"{[worker_id, self.global_rank, self.env_base_seed]} (base seed={self.env_base_seed})."
            )
        else:
            self.env.rng = np.random.RandomState(None if self.type == "valid" else 0)

    def get_worker_id(self):
        """
        Get worker ID.
        """
        if not self.train:
            return 0
        worker_info = torch.utils.data.get_worker_info()
        assert (worker_info is None) == (self.num_workers == 0)
        return 0 if worker_info is None else worker_info.id

    def __len__(self):
        """
        Return dataset size.
        """
        return self.size

    def __getitem__(self, index):
        """
        Return a training sample.
        Either generate it, or read it from file.
        """
        self.init_rng()
        if self.path is None:
            return self.generate_sample()
        else:
            return self.read_sample(index)

    def read_sample(self, index):
        """
        Read a sample.
        """
        idx = index
        if self.train:
            if self.batch_load:
                if index >= self.nextpos:
                    self.load_chunk()
                idx = index - self.basepos
            else:
                index = self.env.rng.randint(len(self.data))
                idx = index
        x, y = self.data[idx]
        x = x.split()
        y = y.split()
        if self.env.weight_tokens:
            x = ["W" + t[1:] if i > 0 and i % 3 == 0 else t for i, t in enumerate(x)]
        assert len(x) >= 1 and len(y) >= 1
        return x, y

    def generate_sample(self):
        """
        Generate a sample.
        """
        while True:
            try:
                if self.task == "graph":
                    xy = self.env.gen_expr(self.type)
                else:
                    raise Exception(f"Unknown data type: {self.task}")
                if xy is None:
                    continue
                x, y = xy
                break
            except Exception as e:
                logger.error(
                    'An unknown exception of type {0} occurred for worker {4} in line {1} for expression "{2}". Arguments:{3!r}.'.format(
                        type(e).__name__,
                        sys.exc_info()[-1].tb_lineno,
                        "F",
                        e.args,
                        self.get_worker_id(),
                    )
                )
                continue
        self.count += 1

        return x, y
