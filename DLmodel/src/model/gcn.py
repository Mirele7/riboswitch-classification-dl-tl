import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, NNConv, GINConv, GATConv, global_add_pool, Set2Set



class TransferGCN(nn.Module):
    def __init__(self, n_features, hidden_dim, n_classes, n_conv_layers=3, dropout=0,
                 conv_type="MPNN", set2set_pooling=False, node_classification=True, softmax=False,
                 probability=True, batch_norm=True, num_embeddings=None, embedding_dim=20,
                 residuals=False, device="cpu", pretrained_modelo_path='None'):
        super().__init__()

        self.pretrained_model = None
        if pretrained_modelo_path:
            pretrained_model = torch.load(pretrained_modelo_path)
            pretrained_model['fc.weight'] = pretrained_model['fc.weight'][:2]  # Ajuste o peso
            pretrained_model['fc.bias'] = pretrained_model['fc.bias'][:2]  # Ajuste o viés

            self.pretrained_model = GCN(n_features, hidden_dim, n_classes, n_conv_layers, dropout,
                                        conv_type, set2set_pooling, node_classification, softmax, probability,
                                        batch_norm, num_embeddings, embedding_dim, residuals, device)

            self.pretrained_model.load_state_dict(pretrained_model, strict=False)

            #Congela todas as camadas, exceto a nova fc
            for name, param in self.pretrained_model.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False
                    print(f"Layer frozen: {name}")  # Opcional para debug

            #Garante que a fc antiga esteja congelada também (por segurança)
            self.pretrained_model.fc.requires_grad = False

            #Substitui a fc para ter saída com 2 classes
            in_features = self.pretrained_model.fc.in_features
            self.pretrained_model.fc = nn.Linear(in_features, 2)
            self.pretrained_model.device = device

        else:
            if num_embeddings:
                self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

            self.convs = nn.ModuleList()
            self.batch_norms = nn.ModuleList()

            self.convs.append(self.get_conv_layer(n_features, hidden_dim, conv_type=conv_type))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

            for i in range(n_conv_layers - 1):
                self.convs.append(self.get_conv_layer(hidden_dim, hidden_dim, conv_type=conv_type))
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

            self.fc = nn.Linear(hidden_dim, n_classes)

            if set2set_pooling:
                self.fc = nn.Linear(2 * hidden_dim, n_classes)
                self.pooling = Set2Set(hidden_dim, processing_steps=10)

            self.dropout = nn.Dropout(dropout)
            self.conv_type = conv_type
            self.node_classification = node_classification
            self.softmax = softmax
            self.probability = probability
            self.batch_norm = batch_norm
            self.num_embeddings = num_embeddings
            self.set2set_pooling = set2set_pooling
            self.residuals = residuals
            self.device = device

    def reset_parameters(self):
        if self.pretrained_model is not None:
            self.pretrained_model.reset_parameters()
            in_features = self.pretrained_model.fc.in_features
            self.pretrained_model.fc = nn.Linear(in_features, 2)
        else:
            if hasattr(self, 'embedding'):
                self.embedding.reset_parameters()
            for conv in self.convs:
                conv.reset_parameters()
            for bn in self.batch_norms:
                bn.reset_parameters()
            self.fc.reset_parameters()

    def forward(self, data):
        if self.pretrained_model is not None:
            return self.pretrained_model(data)
        else:
            x, adj, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
            if self.num_embeddings:
                x = self.embedding(x)
                x = self.dropout(x)

            for i, conv in enumerate(self.convs):
                x = self.apply_conv_layer(conv, x, adj, edge_attr, conv_type=self.conv_type)
                x = self.batch_norms[i](x) if self.batch_norm else x
                x = nn.functional.leaky_relu(x)
                x = self.dropout(x)

            if not self.node_classification:
                if self.set2set_pooling:
                    x = self.pooling(x, batch)
                else:
                    x = global_add_pool(x, batch)
                x = self.dropout(x)

            x = self.fc(x)
            return F.log_softmax(x, dim=1) if not self.softmax else F.softmax(x, dim=1)



class GCN(nn.Module):
    def __init__(self, n_features, hidden_dim, n_classes, n_conv_layers=3, dropout=0,
                 conv_type="MPNN", set2set_pooling=False, node_classification=True, softmax=False,
                 probability=True, batch_norm=True, num_embeddings=None, embedding_dim=20,
                 residuals=False, device="cpu"):
        super(GCN, self).__init__()
        if num_embeddings:
            self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # First layer
        self.convs.append(self.get_conv_layer(n_features, hidden_dim, conv_type=conv_type))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Hidden layers
        for i in range(n_conv_layers - 1):
            self.convs.append(self.get_conv_layer(hidden_dim, hidden_dim, conv_type=conv_type))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, n_classes)

        if set2set_pooling:
            self.fc = nn.Linear(2 * hidden_dim, n_classes)
            self.pooling = Set2Set(hidden_dim, processing_steps=10)

        self.dropout = nn.Dropout(dropout)

        self.conv_type = conv_type
        self.node_classification = node_classification
        self.softmax = softmax
        self.probability = probability
        self.batch_norm = batch_norm
        self.num_embeddings = num_embeddings
        self.set2set_pooling = set2set_pooling
        self.residuals = residuals
        self.device = device

    def reset_parameters(self):
        if self.num_embeddings:
            self.embedding.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.batch_norms:
            bn.reset_parameters()
        self.fc.reset_parameters()
        if self.set2set_pooling:
            self.pooling.reset_parameters()

    # no GCN.forward
    def forward(self, data, return_node_emb=False):
        x, adj, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        if self.num_embeddings:
            x = self.embedding(x)
            x = self.dropout(x)

        if return_node_emb:
            x.retain_grad()
            x_emb = x

        for i, conv in enumerate(self.convs):
            x = self.apply_conv_layer(conv, x, adj, edge_attr, conv_type=self.conv_type)
            x = self.batch_norms[i](x) if self.batch_norm else x
            x = F.leaky_relu(x)
            x = self.dropout(x)

        if not self.node_classification:
            if self.set2set_pooling:
                x = self.pooling(x, batch)
            else:
                x = global_add_pool(x, batch)

        logits = self.fc(x)   # <-- logits (melhor para explain)
        if return_node_emb:
            return logits, x_emb, batch
        return F.log_softmax(logits, dim=1)


    @staticmethod
    def get_conv_layer(n_input_features, n_output_features, conv_type="GCN"):
        if conv_type == "GCN":
            return GCNConv(n_input_features, n_output_features)
        elif conv_type == "GAT":
            return GATConv(n_input_features, n_output_features)
        elif conv_type == "MPNN":
            net = nn.Sequential(nn.Linear(2, 10), nn.ReLU(), nn.Linear(10, n_input_features * n_output_features))
            return NNConv(n_input_features, n_output_features, net)
        elif conv_type == "GIN":
            net = nn.Sequential(nn.Linear(n_input_features, n_output_features), nn.ReLU(),
                                nn.Linear(n_output_features, n_output_features))
            return GINConv(net)
        else:
            raise Exception("{} convolutional layer is not supported.".format(conv_type))

    @staticmethod
    def apply_conv_layer(conv, x, adj, edge_attr, conv_type="GCN"):
        if conv_type in ["GCN", "GAT", "GIN"]:
            return conv(x, adj)
        elif conv_type in ["MPNN"]:
            return conv(x, adj, edge_attr)
        else:
            raise Exception("{} convolutional layer is not supported.".format(conv_type))

