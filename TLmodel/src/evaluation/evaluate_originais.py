#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.getcwd().split('src')[0])

import argparse
import pickle
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef,
    roc_auc_score, average_precision_score
)

from src.data_util.data_constants import families, word_to_ix
from src.data_util.rna_family_graph_dataset import RNAFamilyGraphDataset
from src.model.gcn import TransferGCN  # <<< TL model


def read_fasta_headers_in_order(fasta_path: str):
    """Return FASTA headers (without '>') in file order."""
    headers = []
    with open(fasta_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(">"):
                headers.append(line[1:].strip())
    return headers


def extract_id_and_group_from_header(full_header: str):
    """
    full_header example:
      "10_1_U00096-..._Porcentagem_identificacao_100 riboswitch"
    We want:
      id_global = "10"
      group_id  = "1"
    """
    first_token = full_header.split()[0]  # remove trailing label after space
    parts = first_token.split("_")
    id_global = parts[0] if len(parts) >= 1 else ""
    group_id = parts[1] if len(parts) >= 2 else ""
    return id_global, group_id


def logits_to_probs_safely(out: torch.Tensor) -> torch.Tensor:
    """
    TransferGCN frequentemente retorna log-probabilidades (log_softmax).
    Este helper tenta detectar isso e retornar probabilidades corretas:
      - se exp(out) soma ~1, assume log-probs -> probs = exp(out)
      - senão, assume logits -> probs = softmax(out)
    """
    with torch.no_grad():
        s_exp = out.exp().sum(dim=1).mean().item()
        # softmax sempre soma 1, então usamos exp-sum como detecção
        if 0.98 <= s_exp <= 1.02:
            return out.exp()
        return torch.softmax(out, dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default="test", help='model name (folder name)')
    parser.add_argument('--test_dataset',
                        default='../data/test_riboswitches_E_coli.fa',
                        help='Path to test dataset (.fa)')
    parser.add_argument('--foldings_dataset', default='../data/test_sem_200.pkl',
                        help='Path to foldings dataset (.pkl)')
    parser.add_argument('--checkpoint',
                        default=None,
                        help=('Path to model checkpoint (optional). '
                              'If not set, uses ../models/<model_name>/final_model_full.pt'))
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--out_csv', default='../results_model/janelas_originais/detection_results_test_set1_originais.csv')

    # outputs adicionais
    parser.add_argument('--out_txt_all', default=None,
                        help='(optional) TXT with all windows: >id_global_group  p_riboswitch=...')
    parser.add_argument('--out_txt_hits_by_group', default=None,
                        help='(optional) TXT summary by group listing only predicted riboswitch windows')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- Basic sanity: class mapping ---
    print("=== Class index map (families) ===")
    for i, name in enumerate(families):
        print(f"{i}\t{name}")

    if "riboswitch" not in families or "genomicbackground" not in families:
        raise ValueError(f"Expected both 'riboswitch' and 'genomicbackground' in families, got: {families}")

    ribo_idx = families.index("riboswitch")
    bg_idx = families.index("genomicbackground")
    print(f"\nIndex(riboswitch) = {ribo_idx}")
    print(f"Index(genomicbackground) = {bg_idx}")

    # --- Load test set ---
    test_set = RNAFamilyGraphDataset(args.test_dataset, args.foldings_dataset)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # --- Load hyperparams ---
    opt_path = f'../results_model/{args.model_name}/hyperparams.pkl'
    opt = pickle.load(open(opt_path, "rb"))
    opt.device = device  # force consistent device

    # --- Build TL model (TransferGCN) ---
    model = TransferGCN(
        n_features=opt.embedding_dim,
        hidden_dim=opt.hidden_dim,
        n_classes=len(families),
        n_conv_layers=opt.n_conv_layers,
        dropout=opt.dropout,
        batch_norm=opt.batch_norm,
        num_embeddings=len(word_to_ix),
        embedding_dim=opt.embedding_dim,
        node_classification=False,
        residuals=getattr(opt, "residuals", False),
        device=opt.device,
        set2set_pooling=getattr(opt, "set2set_pooling", False),
        conv_type=getattr(opt, "conv_type", "MPNN"),
        pretrained_modelo_path=getattr(opt, "pretrained_modelo_path", None),
    ).to(device)

    checkpoint_path = args.checkpoint or f'../models/{args.model_name}/final_model_full.pt'
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nThe model has {n_params} trainable parameters")
    print(f"Using device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Hyperparams: {opt_path}")
    print(f"Test dataset: {args.test_dataset}")
    print(f"Foldings dataset: {args.foldings_dataset}")

    # --- Collect predictions ---
    y_true, y_pred = [], []
    p_ribo = []
    q_ids = []

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device)
            out = model(data)  # logits OU log-probs (depende da implementação)
            probs = logits_to_probs_safely(out)
            pred = torch.argmax(probs, dim=1)

            y_true.extend(data.y.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())
            p_ribo.extend(probs[:, ribo_idx].cpu().numpy().tolist())

            if hasattr(data, "seq_id"):
                q_ids.extend(list(data.seq_id))
            else:
                q_ids.extend([f"query_{len(q_ids)+i+1}" for i in range(len(pred))])

            # debug 1x: ajuda a confirmar logits vs log-probs
            if batch_idx == 0:
                s_exp = out.exp().sum(dim=1).mean().item()
                print(f"\n[Debug] mean sum(exp(out)) = {s_exp:.4f}  (≈1.0 sugere log-probs)")
                print(f"[Debug] first probs row sum = {probs[0].sum().item():.4f}")

    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    p_ribo = np.array(p_ribo, dtype=float)

    # --- Binary view: riboswitch (positive) vs background (negative) ---
    mask_bin = np.isin(y_true, [bg_idx, ribo_idx])
    y_true_bin = (y_true[mask_bin] == ribo_idx).astype(int)  # 1=ribo, 0=bg
    y_pred_bin = (y_pred[mask_bin] == ribo_idx).astype(int)
    p_ribo_bin = p_ribo[mask_bin]

    print("\n=== Binary metrics (riboswitch vs genomicbackground) ===")
    print(f"Total (binário) = {len(y_true_bin)}  |  Positivos (ribo) = {int(y_true_bin.sum())}  |  Negativos (bg) = {int((1-y_true_bin).sum())}")

    acc = accuracy_score(y_true_bin, y_pred_bin)
    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)  # sensibilidade / TPR
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    bacc = balanced_accuracy_score(y_true_bin, y_pred_bin)
    mcc = matthews_corrcoef(y_true_bin, y_pred_bin) if len(np.unique(y_true_bin)) > 1 else float("nan")

    print(f"Accuracy           = {acc:.4f}")
    print(f"Precision (PPV)    = {prec:.4f}")
    print(f"Recall (TPR)       = {rec:.4f}")
    print(f"F1                 = {f1:.4f}")
    print(f"Balanced Accuracy  = {bacc:.4f}")
    print(f"MCC                = {mcc:.4f}" if not np.isnan(mcc) else "MCC = NA (apenas uma classe presente em y_true)")

    if len(np.unique(y_true_bin)) == 2:
        rocauc = roc_auc_score(y_true_bin, p_ribo_bin)
        prauc = average_precision_score(y_true_bin, p_ribo_bin)  # PR-AUC (Average Precision)
        print(f"ROC-AUC            = {rocauc:.4f}")
        print(f"PR-AUC (AP)        = {prauc:.4f}")
    else:
        print("ROC-AUC = NA (precisa de positivos e negativos em y_true)")
        print("PR-AUC (AP) = NA (precisa de positivos e negativos em y_true)")

    # --- Label distributions ---
    print("\n=== Label distributions ===")
    print("y_true counts:", np.unique(y_true, return_counts=True))
    print("y_pred counts:", np.unique(y_pred, return_counts=True))

    # --- Confusion matrix (restricted to bg/ribo indices) ---
    cm = confusion_matrix(y_true, y_pred, labels=[bg_idx, ribo_idx])
    TN = int(cm[0, 0])
    FP = int(cm[0, 1])
    FN = int(cm[1, 0])
    TP = int(cm[1, 1])

    print("\n=== Confusion matrix (rows=true, cols=pred) for [genomicbackground, riboswitch] ===")
    print(cm)
    print(f"TN={TN}  FP={FP}  FN={FN}  TP={TP}")

    # --- Detection metrics ---
    P = int(np.sum(y_true == ribo_idx))
    N = int(np.sum(y_true == bg_idx))

    sensitivity = TP / P if P else float("nan")   # TPR
    specificity = TN / N if N else float("nan")   # TNR
    fpr = FP / N if N else float("nan")           # false positive rate

    print("\n=== Detection metrics ===")
    print(f"P (riboswitch true) = {P}")
    print(f"N (background true) = {N}")
    print(f"Sensitivity / TPR   = {sensitivity:.4f}" if not np.isnan(sensitivity) else "Sensitivity / TPR = NA (no positives)")
    print(f"Specificity / TNR   = {specificity:.4f}" if not np.isnan(specificity) else "Specificity / TNR = NA (no negatives)")
    print(f"FPR                 = {fpr:.4f}" if not np.isnan(fpr) else "FPR = NA (no negatives)")

    # --- Classification report (labels present in y_true only) ---
    present_labels = sorted(np.unique(y_true).tolist())
    present_names = [families[i] if i < len(families) else f"class_{i}" for i in present_labels]

    print("\n=== Classification report (labels present in y_true only) ===")
    print(classification_report(
        y_true, y_pred,
        labels=present_labels,
        target_names=present_names,
        digits=4,
        zero_division=0
    ))

    # --- Save per-window results to CSV ---
    df = pd.DataFrame({
        "qseqid": q_ids,
        "y_true_idx": y_true,
        "y_true_name": [families[i] for i in y_true],
        "y_pred_idx": y_pred,
        "y_pred_name": [families[i] for i in y_pred],
        "p_riboswitch": p_ribo
    })

    # recover FASTA header for each prediction (order-preserving)
    fasta_headers = read_fasta_headers_in_order(args.test_dataset)
    if len(fasta_headers) != len(df):
        raise ValueError(
            f"FASTA headers ({len(fasta_headers)}) != predictions ({len(df)}). "
            "Isso indica que a ordem/tamanho não bate (shuffle, dataset, ou arquivo diferente)."
        )

    df["qseqid_full"] = fasta_headers  # header completo (sem '>')
    id_group = df["qseqid_full"].apply(extract_id_and_group_from_header)
    df["id_global"] = [x[0] for x in id_group]
    df["group_id"] = [x[1] for x in id_group]
    df["qid_short"] = df["id_global"].astype(str) + "_" + df["group_id"].astype(str)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved per-sequence results to: {args.out_csv}")

    print("\n=== First 10 predictions ===")
    print(df.head(10)[
        ["qid_short", "qseqid_full", "y_true_name", "y_pred_name", "p_riboswitch"]
    ].to_string(index=False))

    # TXT: all windows
    out_txt_all = args.out_txt_all or args.out_csv.replace(".csv", "_queries_with_p.txt")
    with open(out_txt_all, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(f">{row['qid_short']}\tp_riboswitch={row['p_riboswitch']:.6f}\n")
    print(f"\nSaved all-window query+p_riboswitch TXT to: {out_txt_all}")

    # TXT: hits by group (predicted riboswitch)
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


if __name__ == "__main__":
    main()

