import os
import sys
sys.path.append(os.getcwd().split('src')[0])

import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import time
import argparse
import numpy as np
from torch_geometric.data import DataLoader
from sklearn.model_selection import KFold, StratifiedShuffleSplit

from src.model.gnn import GNN, TransferGNN
from src.data_util.rna_family_graph_dataset import RNAFamilyGraphDataset
from src.util.visualization_util import plot_loss
from src.data_util.data_constants import word_to_ix, families
from src.evaluation.evaluation_util import evaluate_family_classifier, compute_metrics_family

# ----------------------
# Seeds & CUDNN
# ----------------------
torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----------------------
# Args
# ----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default="test", help='model name')
parser.add_argument('--device', default="cpu", help='cpu or cuda')
parser.add_argument('--n_samples', type=int, default=None, help='Number of samples to train on')
parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs to train on')
parser.add_argument('--embedding_dim', type=int, default=20, help='Dimension of nucleotide embeddings')
parser.add_argument('--hidden_dim', type=int, default=80, help='Dimension of hidden representations of convolutional layers')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--learning_rate', type=float, default=0.0004, help='Learning rate')
parser.add_argument('--seq_max_len', type=int, default=10000, help='Maximum length of sequences used for training and testing')
parser.add_argument('--seq_min_len', type=int, default=1, help='Minimum length of sequences used for training and testing')
parser.add_argument('--n_conv_layers', type=int, default=5, help='Number of convolutional layers')
parser.add_argument('--conv_type', type=str, default="MPNN", help='Type of convolutional layers')
parser.add_argument('--dropout', type=float, default=0.1, help='Amount of dropout')
parser.add_argument('--batch_norm', dest='batch_norm', action='store_true')
parser.add_argument('--no_batch_norm', dest='batch_norm', action='store_false')
parser.set_defaults(batch_norm=True)
parser.add_argument('--residuals', type=bool, default=False, help='Whether to use residuals')
parser.add_argument('--set2set_pooling', type=bool, default=True, help='Whether to use set2set pooling')
parser.add_argument('--early_stopping', type=int, default=30, help='Number of epochs for early stopping')
parser.add_argument('--verbose', type=bool, default=False, help='Verbosity')
parser.add_argument('--foldings_dataset', type=str, default='../data/train_DISTANCE.pkl', help='Path to foldings')
parser.add_argument('--train_dataset', type=str, default='../data/train_DISTANCE.fa', help='Path to training dataset')
parser.add_argument('--pretrained_modelo_path', type=str, default='../data/pre_trained_model_7_classes.pt', help='Path to the pretrained model with 7 classes')

# NEW FLAGS
parser.add_argument('--k_folds', type=int, default=5, help='Number of CV folds')
parser.add_argument('--refit_full', action='store_true', help='After CV, refit a final model on 100% of data')
parser.add_argument('--internal_val_fraction', type=float, default=0.1, help='Internal validation fraction used only for refit_full early stopping')

opt = parser.parse_args()
opt.k_folds = 5
opt.refit_full = True
opt.internal_val_fraction = 0.1
print(opt)

# ----------------------
# CV setup
# ----------------------
k_folds = opt.k_folds
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# ----------------------
# Data Loading
# ----------------------
n_classes = 2  # Ajuste se necessário
n_samples = opt.n_samples if opt.n_samples else None
dataset = RNAFamilyGraphDataset(
    opt.train_dataset,
    opt.foldings_dataset,
    seq_max_len=opt.seq_max_len,
    seq_min_len=opt.seq_min_len,
    n_samples=n_samples
)

# ----------------------
# Loss (global, stateless)
# ----------------------
loss_function = nn.NLLLoss()

# ----------------------
# Helpers
# ----------------------
def make_model_and_optim():
    
    model = TransferGNN(
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
        pretrained_modelo_path=opt.pretrained_modelo_path
    ).to(opt.device)
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
    return model, optimizer


def unfreeze_layers(model, layer_keywords):
    """
    Active requires_grad=True
    """
    for name, param in model.named_parameters():
        if any(keyword in name for keyword in layer_keywords):
            if not param.requires_grad:
                print(f"Unfreezing layer: {name}")
            param.requires_grad = True


def train_epoch(model, optimizer, train_loader):
    model.train()
    total_loss = 0.0
    correct = 0
    total_examples = 0

    for batch_idx, data in enumerate(train_loader):
        data = data.to(opt.device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, data.y)
        loss.backward()
        optimizer.step()

        num_graphs = getattr(data, 'num_graphs', data.y.size(0))
        total_loss += loss.item() * num_graphs
        total_examples += num_graphs

        pred = output.max(1)[1]
        correct += pred.eq(data.y).sum().item()

    train_loss = total_loss / max(1, total_examples)
    train_accuracy = correct / max(1, total_examples)
    print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
    return train_loss, train_accuracy


def validate_epoch(model, val_loader):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    accuracies = []

    with torch.no_grad():
        for data in val_loader:
            data = data.to(opt.device)
            out = model(data)
            loss = loss_function(out, data.y)
            num_graphs = getattr(data, 'num_graphs', data.y.size(0))
            total_loss += loss.item() * num_graphs
            total_examples += num_graphs

            pred = out.max(1)[1]
            accuracy = compute_metrics_family(data.y, pred)  # assume retorna escalar [0,1]
            accuracies.append(accuracy)

    avg_loss = total_loss / max(1, total_examples)
    avg_accuracy = float(np.mean(accuracies)) if len(accuracies) > 0 else 0.0

    print(f"Validation loss: {avg_loss:.4f}")
    print(f"Validation accuracy: {avg_accuracy:.4f}")
    return avg_loss, avg_accuracy


def run(model, optimizer, n_epochs, train_loader, val_loader, fold, results_dir, model_dir):
    print(f"Fold {fold}: The model contains {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters")

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    best_val_acc = -1.0
    best_path = os.path.join(model_dir, f'TL_model_DISTANCE_fold_{fold}.pt')

    for epoch in range(n_epochs):
        start = time.time()
        print(f"\nEpoch {epoch + 1}/{n_epochs}")

        #Progressive thawing of layers
        if epoch == 1:
            print("Descongelando convs.3 e convs.4 (parcial)")
            unfreeze_layers(model, ['convs.3', 'convs.4'])
        if epoch == 5:
            print("Descongelando todas as camadas convolucionais (total)")
            unfreeze_layers(model, ['convs'])

        train_loss, train_accuracy = train_epoch(model, optimizer, train_loader)
        val_loss, val_accuracy = validate_epoch(model, val_loader)

        elapsed = time.time() - start
        print(f"Epoch took {elapsed:.2f} seconds")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        plot_loss(train_losses, val_losses, file_name=os.path.join(results_dir, f'TL_DISTANCE_loss_fold_{fold}.jpg'))
        plot_loss(train_accuracies, val_accuracies, file_name=os.path.join(results_dir, f'TL_DISTANCE_acc_fold_{fold}.jpg'), y_label='accuracy')

        with open(os.path.join(results_dir, f'TL_DISTANCE_scores_fold_{fold}.pkl'), 'wb') as f:
            pickle.dump({
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accuracies': train_accuracies,
                'val_accuracies': val_accuracies,
            }, f)

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), best_path)
            print(f"Model salve: fold {fold} (val_acc={best_val_acc:.4f})")

        # Early stopping
        if len(val_accuracies) > opt.early_stopping and max(val_accuracies[-opt.early_stopping:]) < max(val_accuracies):
            print("Training interrupted by early stopping")
            break

    return best_path


def get_labels_for_indices(indices):
    ys = []
    for i in indices:
        data = dataset[int(i)]
        y = int(data.y) if hasattr(data, 'y') else int(data.y.item())
        ys.append(y)
    return np.array(ys)


def main():
    results_dir = os.path.join('..', 'results_model', opt.model_name)
    model_dir = os.path.join('..', 'models', opt.model_name)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(results_dir, 'DISTANCE_hyperparams.txt'), 'w') as f:
        f.write(str(opt))
    with open(os.path.join(results_dir, 'DISTANCE_hyperparams.pkl'), 'wb') as f:
        pickle.dump(opt, f)

    # ----------------------
    # K-Fold Cross-Validation
    # ----------------------
    best_model_paths = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f'Fold {fold + 1}/{k_folds}')
        train_idx = [int(i) for i in train_idx]
        val_idx = [int(i) for i in val_idx]

        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=opt.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=opt.batch_size, shuffle=False)

        model, optimizer = make_model_and_optim()

        best_model_path = run(model, optimizer, opt.n_epochs, train_loader, val_loader, fold, results_dir, model_dir)
        best_model_paths.append(best_model_path)

    # ----------------------
    #Best fold
    # ----------------------
    best_model_index = -1
    best_val_acc = -1.0

    for i, path in enumerate(best_model_paths):
        scores_path = os.path.join(results_dir, f'TL_DISTANCE_scores_fold_{i}.pkl')
        with open(scores_path, 'rb') as f:
            scores = pickle.load(f)
        max_val_acc = max(scores['val_accuracies']) if len(scores['val_accuracies']) > 0 else -1.0
        print(f"Fold {i}: Best validation accuracy = {max_val_acc:.4f}")
        if max_val_acc > best_val_acc:
            best_val_acc = max_val_acc
            best_model_index = i

    best_model_path = best_model_paths[best_model_index]
    
    import shutil
    final_model_path = os.path.join(model_dir, 'TL_DISTANCE_best_model.pt')
    shutil.copy(best_model_path, final_model_path)
    

    # ----------------------
    # Refit
    # ----------------------
    if opt.refit_full:
        
        labels = get_labels_for_indices(range(len(dataset)))
        sss = StratifiedShuffleSplit(n_splits=1, test_size=opt.internal_val_fraction, random_state=123)
        train_idx_full, val_idx_full = next(sss.split(np.arange(len(dataset)), labels))

        train_subset_full = torch.utils.data.Subset(dataset, train_idx_full.tolist())
        val_subset_full = torch.utils.data.Subset(dataset, val_idx_full.tolist())
        train_loader_full = DataLoader(train_subset_full, batch_size=opt.batch_size, shuffle=True)
        val_loader_full = DataLoader(val_subset_full, batch_size=opt.batch_size, shuffle=False)

        model_full, optim_full = make_model_and_optim()
        best_full_path = run(model_full, optim_full, opt.n_epochs, train_loader_full, val_loader_full, 'full', results_dir, model_dir)

        final_full_path = os.path.join(model_dir, 'TL_DISTANCE_model_full.pt')
        shutil.copy(best_full_path, final_full_path)
        print(f"[Refit] model save in: {TL_DISTANCE_model_full}")


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"\nTotal training time: {(end-start)/60:.2f} minutes")
