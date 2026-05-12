import os
import sys
sys.path.append(os.getcwd().split('src')[0])

import argparse
import pickle
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef,
    roc_auc_score, average_precision_score
)

from src.data_util.data_constants import families, word_to_ix
from src.data_util.rna_family_graph_dataset import RNAFamilyGraphDataset
from src.model.gcn import GCN


def read_fasta_headers_in_order(fasta_path: str):
    headers = []
    with open(fasta_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(">"):
                headers.append(line[1:].strip())
    return headers


def extract_id_and_group_from_header(full_header: str):
    first_token = str(full_header).split()[0]
    parts = first_token.split("_")
    id_global = parts[0] if len(parts) >= 1 else ""
    group_id = parts[1] if len(parts) >= 2 else ""
    return id_global, group_id


def per_graph_pos_from_batch(batch_vec: torch.Tensor) -> torch.Tensor:
    pos = torch.empty_like(batch_vec, dtype=torch.long)
    if batch_vec.numel() == 0:
        return pos
    num_graphs = int(batch_vec.max().item()) + 1
    for g in range(num_graphs):
        idx = (batch_vec == g).nonzero(as_tuple=False).view(-1)
        pos[idx] = torch.arange(idx.numel(), device=batch_vec.device, dtype=torch.long)
    return pos


def region_sums(node_imp: torch.Tensor, pos_local: torch.Tensor, batch_vec: torch.Tensor):
    if batch_vec.numel() == 0:
        return [], [], [], []
    num_graphs = int(batch_vec.max().item()) + 1
    begin_s, mid_s, end_s, Ls = [], [], [], []
    for g in range(num_graphs):
        mask_g = (batch_vec == g)
        pos_g = pos_local[mask_g]
        imp_g = node_imp[mask_g]
        if pos_g.numel() == 0:
            begin_s.append(0.0); mid_s.append(0.0); end_s.append(0.0); Ls.append(0)
            continue

        L = int(pos_g.max().item()) + 1
        t1 = int(0.33 * L)
        t2 = int(0.66 * L)

        begin = imp_g[(pos_g < t1)].sum().item()
        mid   = imp_g[(pos_g >= t1) & (pos_g < t2)].sum().item()
        end   = imp_g[(pos_g >= t2)].sum().item()

        begin_s.append(float(begin))
        mid_s.append(float(mid))
        end_s.append(float(end))
        Ls.append(int(L))
    return begin_s, mid_s, end_s, Ls


def paired_unpaired_sums(node_imp: torch.Tensor, paired_mask: torch.Tensor, batch_vec: torch.Tensor):
    if batch_vec.numel() == 0:
        return [], []
    num_graphs = int(batch_vec.max().item()) + 1
    paired_s, unpaired_s = [], []
    for g in range(num_graphs):
        mask_g = (batch_vec == g)
        imp_g = node_imp[mask_g]
        pm_g = paired_mask[mask_g]
        paired_s.append(float(imp_g[pm_g].sum().item()))
        unpaired_s.append(float(imp_g[~pm_g].sum().item()))
    return paired_s, unpaired_s


@torch.no_grad()
def predict_all(model, loader, ribo_idx, device):
    y_true, y_pred, p_ribo = [], [], []
    for data in loader:
        data = data.to(device)
        out = model(data)      # log_softmax
        probs = out.exp()      # correto
        pred = torch.argmax(probs, dim=1)
        y_true.extend(data.y.cpu().numpy().tolist())
        y_pred.extend(pred.cpu().numpy().tolist())
        p_ribo.extend(probs[:, ribo_idx].cpu().numpy().tolist())
    return np.array(y_true, dtype=int), np.array(y_pred, dtype=int), np.array(p_ribo, dtype=float)


def explain_saliency(model, loader, ribo_idx, device, max_batches=None):
    model.eval()
    region_sums_all, paired_sums_all, node_imp_all = [], [], []

    for b_idx, data in enumerate(loader):
        if max_batches is not None and b_idx >= max_batches:
            break

        data = data.to(device)
        model.zero_grad(set_to_none=True)

        logits, x_emb, batch_vec = model(data, return_node_emb=True)
        x_emb.retain_grad()

        target = logits[:, ribo_idx].sum()
        target.backward()

        node_imp = (x_emb.grad * x_emb).abs().sum(dim=1)

        pos_attr = getattr(data, "pos", None)
        if pos_attr is not None:
            pos_local = pos_attr.to(node_imp.device)
        else:
            pos_local = per_graph_pos_from_batch(batch_vec)

        begin_s, mid_s, end_s, Ls = region_sums(node_imp, pos_local, batch_vec)

        paired_attr = getattr(data, "paired", None)
        if paired_attr is not None:
            paired_mask = paired_attr.bool().to(node_imp.device)
            paired_s, unpaired_s = paired_unpaired_sums(node_imp, paired_mask, batch_vec)
        else:
            num_graphs = int(batch_vec.max().item()) + 1 if batch_vec.numel() else 0
            paired_s = [np.nan] * num_graphs
            unpaired_s = [np.nan] * num_graphs

        num_graphs = int(batch_vec.max().item()) + 1 if batch_vec.numel() else 0
        for g in range(num_graphs):
            region_sums_all.append((begin_s[g], mid_s[g], end_s[g], Ls[g]))
            paired_sums_all.append((paired_s[g], unpaired_s[g]))
            mask_g = (batch_vec == g)
            node_imp_all.append(node_imp[mask_g].detach().cpu())

    return region_sums_all, paired_sums_all, node_imp_all


def _print_region_summary(title: str, subdf: pd.DataFrame):
    if len(subdf) == 0:
        print(f"\n--- {title} (vazio) ---")
        return
    print(f"\n--- {title} (n={len(subdf)}) ---")
    print("Mean frac_begin:", float(subdf["frac_begin"].mean()))
    print("Mean frac_mid  :", float(subdf["frac_mid"].mean()))
    print("Mean frac_end  :", float(subdf["frac_end"].mean()))
    if subdf.get("frac_paired") is not None and subdf["frac_paired"].notna().any():
        print("Mean frac_paired   :", float(subdf["frac_paired"].dropna().mean()))
        print("Mean frac_unpaired :", float(subdf["frac_unpaired"].dropna().mean()))

   
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default="test", help='model name (folder name)')
    parser.add_argument('--test_dataset',
                        default='/home/Mirele/Documentos/0_0_DOUTORADO/0_DISTANCE_EVOLUTIVE/test/janelas_deslizantes/8_riboswitch/riboswitch_windows_dual.fa',
                        help='Path to test dataset (.fa)')
    parser.add_argument('--foldings_dataset',
                        default='/home/Mirele/Documentos/0_0_DOUTORADO/0_DISTANCE_EVOLUTIVE/test/janelas_deslizantes/8_riboswitch/riboswitch_windows_dual.pkl',
                        help='Path to foldings dataset (.pkl)')
    parser.add_argument('--checkpoint',
                        default=None,
                        help=('Path to model checkpoint '
                              'If not set, uses ../models/<model_name>/ML_set1_model_full.pt'))
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--out_csv',
                        default='../results_model/janelas_deslizantes/8_riboswitch/detection_results_test_set1.csv')
    
    
    parser.add_argument('--out_txt_all', default=None)
    parser.add_argument('--out_txt_hits_by_group', default=None)
    parser.add_argument('--explain', action='store_true')
    parser.add_argument('--explain_out_csv', default=None)
    parser.add_argument('--explain_topk', type=int, default=200)
    parser.add_argument('--explain_max_batches', type=int, default=None)

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=== Class index map (families) ===")
    for i, name in enumerate(families):
        print(f"{i}\t{name}")

    if "riboswitch" not in families or "genomicbackground" not in families:
        raise ValueError("Esperado 'riboswitch' e 'genomicbackground' em families.")

    ribo_idx = families.index("riboswitch")
    bg_idx = families.index("genomicbackground")
    print(f"\nIndex(riboswitch) = {ribo_idx}")
    print(f"Index(genomicbackground) = {bg_idx}")

    test_set = RNAFamilyGraphDataset(args.test_dataset, args.foldings_dataset)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    opt = pickle.load(open(f'../results_model/{args.model_name}/hyperparams.pkl', "rb"))
    opt.device = device

    model = GCN(
        n_features=opt.embedding_dim,
        hidden_dim=opt.hidden_dim,
        n_classes=len(families),
        n_conv_layers=opt.n_conv_layers,
        dropout=opt.dropout,
        batch_norm=opt.batch_norm,
        num_embeddings=len(word_to_ix),
        embedding_dim=opt.embedding_dim,
        node_classification=False,
        set2set_pooling=opt.set2set_pooling
    ).to(device)

    checkpoint_path = args.checkpoint or f'../models/{args.model_name}/ML_set1_model_full.pt'
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    y_true, y_pred, p_ribo = predict_all(model, test_loader, ribo_idx, device)

    # métricas binárias
    mask_bin = np.isin(y_true, [bg_idx, ribo_idx])
    y_true_bin = (y_true[mask_bin] == ribo_idx).astype(int)
    y_pred_bin = (y_pred[mask_bin] == ribo_idx).astype(int)
    p_ribo_bin = p_ribo[mask_bin]

    print("\n=== Binary metrics (riboswitch vs genomicbackground) ===")
    acc = accuracy_score(y_true_bin, y_pred_bin)
    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    bacc = balanced_accuracy_score(y_true_bin, y_pred_bin)
    mcc = matthews_corrcoef(y_true_bin, y_pred_bin) if len(np.unique(y_true_bin)) > 1 else float("nan")
    print(f"Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}  BAcc={bacc:.4f}  MCC={mcc:.4f}" if not np.isnan(mcc) else "MCC=NA")

    if len(np.unique(y_true_bin)) == 2:
        print(f"ROC-AUC={roc_auc_score(y_true_bin, p_ribo_bin):.4f}")
        print(f"PR-AUC={average_precision_score(y_true_bin, p_ribo_bin):.4f}")

    # report
    present_labels = sorted(np.unique(y_true).tolist())
    present_names = [families[i] for i in present_labels]
    print("\n=== Classification report ===")
    print(classification_report(y_true, y_pred, labels=present_labels, target_names=present_names, digits=4, zero_division=0))

    # dataframe base (IDs via FASTA)
    fasta_headers = read_fasta_headers_in_order(args.test_dataset)
    if len(fasta_headers) != len(y_true):
        raise ValueError(f"FASTA headers ({len(fasta_headers)}) != predictions ({len(y_true)}).")

    df = pd.DataFrame({
        "qseqid_full": fasta_headers,
        "y_true_idx": y_true,
        "y_true_name": [families[i] for i in y_true],
        "y_pred_idx": y_pred,
        "y_pred_name": [families[i] for i in y_pred],
        "p_riboswitch": p_ribo
    })

    id_group = df["qseqid_full"].apply(extract_id_and_group_from_header)
    df["id_global"] = [x[0] for x in id_group]
    df["group_id"] = [x[1] for x in id_group]
    df["qid_short"] = df["id_global"].astype(str) + "_" + df["group_id"].astype(str)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved per-sequence results to: {args.out_csv}")

    out_txt_all = args.out_txt_all or args.out_csv.replace(".csv", "_queries_with_p.txt")
    with open(out_txt_all, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(f">{row['qid_short']}\tp_riboswitch={row['p_riboswitch']:.6f}\n")
    print(f"Saved all-window query+p_riboswitch TXT to: {out_txt_all}")

    out_txt_hits = args.out_txt_hits_by_group or args.out_csv.replace(".csv", "_hits_by_group.txt")
    with open(out_txt_hits, "w", encoding="utf-8") as f:
        unique_groups = sorted(df["group_id"].unique(), key=lambda x: int(x) if str(x).isdigit() else str(x))
        for gid in unique_groups:
            sub = df[df["group_id"] == gid]
            hits = sub[sub["y_pred_name"] == "riboswitch"]
            f.write(f">Grupo_{gid} Identificou {len(hits)}:\n")
            for _, row in hits.iterrows():
                f.write(f"{row['qid_short']}\tp_riboswitch={row['p_riboswitch']:.6f}\n")
            f.write("\n")
    print(f"Saved hits-by-group TXT to: {out_txt_hits}")

    # explain
    if args.explain:
        print("\n=== Explainability: computing saliency (embedding-grad) ===")
        region_sums_all, paired_sums_all, node_imp_all = explain_saliency(
            model=model, loader=test_loader, ribo_idx=ribo_idx, device=device, max_batches=args.explain_max_batches
        )

        if len(region_sums_all) != len(df):
            raise ValueError(f"Explain count ({len(region_sums_all)}) != predictions ({len(df)}).")

        df["imp_begin"] = [t[0] for t in region_sums_all]
        df["imp_mid"]   = [t[1] for t in region_sums_all]
        df["imp_end"]   = [t[2] for t in region_sums_all]
        df["seq_len_nodes"] = [t[3] for t in region_sums_all]

        denom = (df["imp_begin"] + df["imp_mid"] + df["imp_end"]).replace(0, np.nan)
        df["frac_begin"] = (df["imp_begin"] / denom).fillna(0.0)
        df["frac_mid"]   = (df["imp_mid"]   / denom).fillna(0.0)
        df["frac_end"]   = (df["imp_end"]   / denom).fillna(0.0)

        df["imp_paired"] = [t[0] for t in paired_sums_all]
        df["imp_unpaired"] = [t[1] for t in paired_sums_all]
        denom2 = (df["imp_paired"] + df["imp_unpaired"]).replace(0, np.nan)
        df["frac_paired"] = (df["imp_paired"] / denom2).fillna(np.nan)
        df["frac_unpaired"] = (df["imp_unpaired"] / denom2).fillna(np.nan)

        # top positions (top-K por p_riboswitch)
        topk = int(max(0, args.explain_topk))
        top_pos_str = [""] * len(df)
        if topk > 0:
            top_idx = np.argsort(-df["p_riboswitch"].values)[:min(topk, len(df))]
            for i in top_idx:
                imp = node_imp_all[i].numpy()
                if imp.size == 0:
                    continue
                k = min(15, imp.size)
                pos = np.argsort(-imp)[:k]
                top_pos_str[i] = ",".join([str(int(p)) for p in pos])
        df["top_positions_by_saliency"] = top_pos_str

        explain_out = args.explain_out_csv or args.out_csv.replace(".csv", "_explain.csv")
        df.to_csv(explain_out, index=False)
        print(f"Saved explainability CSV to: {explain_out}")

        # prints extras
        k = min(200, len(df))
        _print_region_summary("Top p_riboswitch", df.sort_values("p_riboswitch", ascending=False).head(k))
        _print_region_summary("Bottom p_riboswitch", df.sort_values("p_riboswitch", ascending=True).head(k))

        pred_ribo = df[df["y_pred_name"] == "riboswitch"]
        pred_bg = df[df["y_pred_name"] == "genomicbackground"]
        _print_region_summary("Predito = riboswitch", pred_ribo)
        _print_region_summary("Predito = background", pred_bg)

        tp = df[(df["y_true_name"] == "riboswitch") & (df["y_pred_name"] == "riboswitch")]
        fp = df[(df["y_true_name"] == "genomicbackground") & (df["y_pred_name"] == "riboswitch")]
        _print_region_summary("TP (true ribo, pred ribo)", tp)
        _print_region_summary("FP (true bg, pred ribo)", fp)

    # sobrescreve out_csv (inclui colunas extras se --explain)
    df.to_csv(args.out_csv, index=False)
    print(opt.embedding_dim)

if __name__ == "__main__":
    main()

