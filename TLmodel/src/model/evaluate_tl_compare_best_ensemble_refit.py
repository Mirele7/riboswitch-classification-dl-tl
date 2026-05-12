import os
import sys
import glob
import re
import time
sys.path.append(os.getcwd().split('src')[0])

import pickle
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, roc_curve, auc

from src.data_util.data_constants import families, word_to_ix
from src.data_util.rna_family_graph_dataset import RNAFamilyGraphDataset
from torch_geometric.data import DataLoader
from src.model.gcn import TransferGCN


# =====================
# Helpers
# =====================

def build_model(opt, n_classes, device):
    model = TransferGCN(
        n_features=opt.embedding_dim,
        hidden_dim=opt.hidden_dim,
        n_classes=n_classes,
        n_conv_layers=opt.n_conv_layers,
        dropout=opt.dropout,
        batch_norm=opt.batch_norm,
        num_embeddings=len(word_to_ix),
        embedding_dim=opt.embedding_dim,
        node_classification=False,
        residuals=opt.residuals,
        device=opt.device,
        set2set_pooling=opt.set2set_pooling,
        conv_type=getattr(opt, 'conv_type', 'MPNN'),
        pretrained_modelo_path=opt.pretrained_modelo_path,
    ).to(device)
    return model


def find_numbered_fold_checkpoints(model_dir, pattern='model_100_fold_*.pt'):
    paths = glob.glob(os.path.join(model_dir, pattern))
    numbered = []
    for p in paths:
        m = re.search(r"fold_(\d+)\.pt$", os.path.basename(p))
        if m:
            numbered.append((int(m.group(1)), p))
    if not numbered:
        raise FileNotFoundError(
            f"Nenhum checkpoint de fold numerado encontrado com pattern {os.path.join(model_dir, pattern)}"
        )
    numbered.sort(key=lambda t: t[0])
    return [p for _, p in numbered]


def pick_best_checkpoint(model_dir: str, results_dir: str, pattern_base: str = 'model_100_fold_{}.pt'):
    best_model_path = os.path.join(model_dir, 'best_model.pt')
    if os.path.isfile(best_model_path):
        return best_model_path, 'best'
    # tenta pelos scores
    candidates = []
    for i in range(50):
        pkl_path = os.path.join(results_dir, f'scores_fold_{i}.pkl')
        if os.path.isfile(pkl_path):
            with open(pkl_path, 'rb') as f:
                s = pickle.load(f)
            max_val = max(s.get('val_accuracies', [-1])) if s.get('val_accuracies') else -1
            candidates.append((i, max_val))
    if candidates:
        candidates.sort(key=lambda t: t[1], reverse=True)
        best_i = candidates[0][0]
        ckpt_path = os.path.join(model_dir, pattern_base.format(best_i))
        if os.path.isfile(ckpt_path):
            return ckpt_path, best_i
    # fallback
    fallback = os.path.join(model_dir, pattern_base.format(0))
    return fallback, 0


def run_single(model, loader, device):
    model.eval()
    probs_batches = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)                 # log-probs (TransferGCN)
            probs = out.exp().cpu().numpy()    # -> probs
            probs_batches.append(probs)
    probs_all = np.vstack(probs_batches)
    y_pred = probs_all.argmax(axis=1)
    return probs_all, y_pred


def metrics_block(y_true, y_pred, probs_all, title_prefix):
    report = classification_report(y_true, y_pred, target_names=families, digits=4)
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    curves = None
    if probs_all.shape[1] == 2:
        y_score = probs_all[:, 1]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        curves = (fpr, tpr, roc_auc)

    print(f"\n=== {title_prefix} REPORT (test set) ===")
    print(report)
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall    (weighted): {recall:.4f}")
    print(f"F1-score  (weighted): {f1:.4f}")
    return acc, f1, curves


def plot_rocs(curves_dict, save_path, title='Best vs Ensemble vs Refit — ROC (test set)'):
    if not curves_dict:
        print('ROC não disponível (n_classes != 2). Pulando plot.')
        return
    plt.figure()
    for label, (fpr, tpr, auc_val) in curves_dict.items():
        plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC={auc_val:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Saved ROC comparison to: {save_path}")


# =====================
# Main
# =====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='transfer_unfreeze_early')
    parser.add_argument('--test_dataset', default='../data/100/test_100.fa')
    parser.add_argument('--foldings_dataset', default='../data/riboswitches_100.pkl')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--save_dir', default='../results_TL_model')
    parser.add_argument('--model_dir_root', default='../TL_model')

    # patterns para 100 ou 200 (ajuste conforme seus checkpoints)
    parser.add_argument('--pattern', default='model_100_fold_*.pt')
    parser.add_argument('--pattern_base', default='model_100_fold_{}.pt')

    # quais avaliar
    parser.add_argument('--eval_best', action='store_true')
    parser.add_argument('--eval_ensemble', action='store_true')
    parser.add_argument('--eval_refit', action='store_true')

    # se nenhum for passado, ativar todos
    parser.set_defaults(eval_best=None, eval_ensemble=None, eval_refit=None)

    # força um fold específico como "best"
    parser.add_argument('--force_fold_index', type=int, default=None)

    args = parser.parse_args()
    device = args.device

    # ativa todos por padrão
    if args.eval_best is None and args.eval_ensemble is None and args.eval_refit is None:
        eval_best = eval_ensemble = eval_refit = True
    else:
        eval_best = bool(args.eval_best)
        eval_ensemble = bool(args.eval_ensemble)
        eval_refit = bool(args.eval_refit)

    # hyperparams do treino
    opt_path = os.path.join(args.save_dir, args.model_name, 'hyperparams.pkl')
    with open(opt_path, 'rb') as f:
        opt = pickle.load(f)

    n_classes = 2  # seu setup binário

    # dados
    test_set = RNAFamilyGraphDataset(args.test_dataset, args.foldings_dataset)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    y_true = []
    for batch in test_loader:
        y_true += list(batch.y.numpy())
    y_true = np.array(y_true)

    model_dir = os.path.join(args.model_dir_root, args.model_name)
    results_dir = os.path.join(args.save_dir, args.model_name)
    os.makedirs(results_dir, exist_ok=True)

    curves = {}

    # ------------- BEST -------------
    if eval_best:
        if args.force_fold_index is not None:
            best_ckpt = os.path.join(model_dir, args.pattern_base.format(args.force_fold_index))
            best_tag = f'fold_{args.force_fold_index}'
        else:
            best_ckpt, which = pick_best_checkpoint(model_dir, results_dir, args.pattern_base)
            best_tag = f'best({which})'
        if not os.path.isfile(best_ckpt):
            raise FileNotFoundError(f'best checkpoint não encontrado: {best_ckpt}')

        model_best = build_model(opt, n_classes, device)
        state = torch.load(best_ckpt, map_location=device)
        model_best.load_state_dict(state, strict=False)

        t0 = time.time()
        probs_best, ypred_best = run_single(model_best, test_loader, device)
        t_best = time.time() - t0
        acc_best, f1_best, roc_best = metrics_block(y_true, ypred_best, probs_best, 'BEST FOLD')
        if roc_best is not None:
            curves[f'Best {best_tag}'] = roc_best
    else:
        probs_best = ypred_best = acc_best = f1_best = t_best = None

    # ------------- ENSEMBLE -------------
    if eval_ensemble:
        ckpts = find_numbered_fold_checkpoints(model_dir, args.pattern)
        print(f"Using {len(ckpts)} checkpoints for ensemble")
        probs_accum = [None for _ in test_loader]
        t0 = time.time()
        with torch.no_grad():
            for ckpt in ckpts:
                model = build_model(opt, n_classes, device)
                state = torch.load(ckpt, map_location=device)
                model.load_state_dict(state, strict=False)
                model.eval()
                for i, batch in enumerate(test_loader):
                    batch = batch.to(device)
                    out = model(batch)
                    p = out.exp().cpu().numpy()
                    probs_accum[i] = p if probs_accum[i] is None else (probs_accum[i] + p)
        for i in range(len(probs_accum)):
            probs_accum[i] /= len(ckpts)
        probs_ens = np.vstack(probs_accum)
        ypred_ens = probs_ens.argmax(axis=1)
        t_ens = time.time() - t0
        acc_ens, f1_ens, roc_ens = metrics_block(y_true, ypred_ens, probs_ens, 'ENSEMBLE')
        if roc_ens is not None:
            curves[f'Ensemble (k={len(ckpts)})'] = roc_ens
    else:
        probs_ens = ypred_ens = acc_ens = f1_ens = t_ens = None

    # ------------- REFIT (final_model_full.pt) -------------
    if eval_refit:
        refit_ckpt = os.path.join(model_dir, 'final_model_full.pt')
        if not os.path.isfile(refit_ckpt):
            print('[Refit] final_model_full.pt não encontrado — pulando avaliação do refit.')
            probs_ref = ypred_ref = acc_ref = f1_ref = t_ref = roc_ref = None
        else:
            model_ref = build_model(opt, n_classes, device)
            state = torch.load(refit_ckpt, map_location=device)
            model_ref.load_state_dict(state, strict=False)
            t0 = time.time()
            probs_ref, ypred_ref = run_single(model_ref, test_loader, device)
            t_ref = time.time() - t0
            acc_ref, f1_ref, roc_ref = metrics_block(y_true, ypred_ref, probs_ref, 'REFIT (final_model_full)')
            if roc_ref is not None:
                curves['Refit (full)'] = roc_ref
    else:
        probs_ref = ypred_ref = acc_ref = f1_ref = t_ref = roc_ref = None

    # ------------- SAVE & PLOT -------------
    if curves:
        plot_rocs(curves, os.path.join(results_dir, 'tl_best_ens_refit_roc.png'))

    npz_path = os.path.join(results_dir, 'best_ens_refit_predictions.npz')
    np.savez(npz_path,
             y_true=y_true,
             probs_best=probs_best, ypred_best=ypred_best,
             probs_ens=probs_ens,  ypred_ens=ypred_ens,
             probs_ref=probs_ref,  ypred_ref=ypred_ref,
             acc_best=acc_best, f1_best=f1_best,
             acc_ens=acc_ens,   f1_ens=f1_ens,
             acc_ref=acc_ref,   f1_ref=f1_ref,
             t_best=t_best, t_ens=t_ens, t_ref=t_ref)
    print(f"Saved comparison to: {npz_path}")


if __name__ == '__main__':
    from sklearn.metrics import roc_curve, auc
    main()
