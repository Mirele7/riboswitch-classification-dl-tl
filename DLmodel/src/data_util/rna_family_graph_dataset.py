from torch_geometric.data import InMemoryDataset, Data
import pickle
import numpy as np
import torch
from Bio import SeqIO

from src.data_util.data_processing import prepare_sequence
from src.data_util.data_constants import word_to_ix, tag_to_ix, families
from src.util.util import dotbracket_to_graph


def paired_mask_from_dotbracket(db: str) -> torch.Tensor:
    """True = posição pareada ( '(' ou ')' ), False = '.'."""
    if db is None:
        return None
    db = db.strip()
    if len(db) == 0:
        return None
    return torch.tensor([ch != '.' for ch in db], dtype=torch.bool)


class RNAFamilyGraphDataset(InMemoryDataset):
    def __init__(self, file_path, foldings_path, transform=None, pre_transform=None,
                 seq_max_len=10000, seq_min_len=1, n_samples=None):
        super(RNAFamilyGraphDataset, self).__init__(file_path, transform, pre_transform)

        with open(file_path, "r") as handle:
            records = list(SeqIO.parse(handle, "fasta"))

        foldings = pickle.load(open(foldings_path, "rb"))

        # mantém ordem do FASTA (não embaralhar)
        records = [x for x in records if seq_min_len <= len(str(x.seq)) <= seq_max_len]
        records = records if not n_samples else records[:n_samples]

        lengths = [len(str(x.seq)) for x in records]
        print("{} sequences found at path {} with max length {}, average length of {}, "
              "and median length of {}".format(len(lengths), file_path, np.max(lengths),
                                               np.mean(lengths), np.median(lengths)))

        data_list = []

        for rec in records:
            sequence_string = str(rec.seq)

            # x = token por nó (LongTensor [L])
            x_seq = prepare_sequence(sequence_string, word_to_ix)
            if not torch.is_tensor(x_seq):
                x_seq = torch.tensor(x_seq, dtype=torch.long)
            else:
                x_seq = x_seq.long()

            family = rec.description.split()[-1]

            dot_bracket_string = foldings[sequence_string][0]

            # grafo do dot-bracket
            g = dotbracket_to_graph(dot_bracket_string)
            edges = list(g.edges(data=True))

            # edge_attr: adjacent -> [0,1] else -> [1,0]
            edge_attr = torch.tensor(
                [[0, 1] if e[2].get('edge_type') == 'adjacent' else [1, 0] for e in edges],
                dtype=torch.float
            )
            edge_index = torch.LongTensor(list(g.edges())).t().contiguous()

            y = self.get_family_idx(family)

            data = Data(x=x_seq, edge_index=edge_index, edge_attr=edge_attr, y=y)

            #  posição real na sequência
            L = int(x_seq.numel())
            data.pos = torch.arange(L, dtype=torch.long)

            # paired/unpaired do dot-bracket
            pm = paired_mask_from_dotbracket(dot_bracket_string)
            if pm is not None and pm.numel() == L:
                data.paired = pm
            else:
                data.paired = None  # evaluate coloca NaN

            # NÃO adicionar strings ao Data (quebra o collate)
            data_list.append(data)

        self.data, self.slices = self.collate(data_list)

    def download(self):
        pass

    def process(self):
        pass

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    @staticmethod
    def get_family_idx(family):
        if family not in families:
            raise Exception("Family not in list")
        return torch.LongTensor([families.index(family)])

