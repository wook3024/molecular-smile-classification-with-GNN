import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class SkipConnection(nn.Module):
    '''Used to obtain the same effect as the skip connection used in the existing CNN
    https://github.com/heartcored98/Standalone-DeepLearning-Chemistry
    '''
    def __init__(self, in_dim, out_dim):
        super(SkipConnection, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim, bias=False)        
        
    def forward(self, in_x, out_x):
        if (self.in_dim != self.out_dim):
            in_x = self.linear(in_x)
        out = in_x + out_x
        return out


class GatedSkipConnection(nn.Module):
    '''Residual connections at appropriate rates using learnable parametes
    https://github.com/heartcored98/Standalone-DeepLearning-Chemistry
    '''
    def __init__(self, in_dim, out_dim):
        super(GatedSkipConnection, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.linear_coef_in = nn.Linear(out_dim, out_dim)
        self.linear_coef_out = nn.Linear(out_dim, out_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, in_x, out_x):
        if (self.in_dim != self.out_dim):
            in_x = self.linear(in_x)
        z = self.gate_coefficient(in_x, out_x)
        out = torch.mul(z, out_x) + torch.mul(1.0-z, in_x)
        return out
            
    def gate_coefficient(self, in_x, out_x):
        x1 = self.linear_coef_in(in_x)
        x2 = self.linear_coef_out(out_x)
        return self.sigmoid(x1+x2)


class GraphNorm(nn.Module):
    ''' https://github.com/lsj2408/GraphNorm '''
    def __init__(self, num_features, eps=1e-5, affine=True, is_node=True):
        super().__init__()
        self.eps = eps
        self.num_features = num_features
        self.affine = affine
        self.is_node = is_node

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(self.num_features))
            self.beta = nn.Parameter(torch.zeros(self.num_features))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

    def norm(self, x):
        mean = x.mean(dim = 0, keepdim = True)
        var = x.std(dim = 0, keepdim = True)
        x = (x - mean) / (var + self.eps)
        return x

    def forward(self, g, x):
        graph_size  = len(g)
        x_list = torch.split(x, graph_size)
        norm_list = []
        for x in x_list:
            norm_list.append(self.norm(x))
        norm_x = torch.cat(norm_list, 0)

        if self.affine:
            return self.gamma * norm_x + self.beta
        else:
            return norm_x


class GNNLayer(nn.Module):
    '''https://github.com/heartcored98/Standalone-DeepLearning-Chemistry'''
    def __init__(self, in_dim, out_dim, device, act=None, norm_type='no'):
        super(GNNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm_type = norm_type
        self.activation = act
        self.device = device

        self.norm = nn.BatchNorm1d(out_dim)
        if norm_type == 'gn':
            self.norm = GraphNorm(out_dim)
        elif norm_type == 'ln':
            self.norm == nn.LayerNorm(out_dim)
        elif norm_type == 'in':
            self.norm = nn.InstanceNorm1d(out_dim)

        
    def forward(self, x, adj):
        out = self.linear(x)
        out = torch.matmul(adj, out)
        if self.norm_type == 'gn':
            out = self.norm(adj, out)
        elif self.norm_type =='in' or self.norm_type =='ln' or self.norm_type =='bn':
            out = self.norm(out)
        elif self.norm_type == 'no':
            out = F.normalize(out, 2, 1)
        if self.activation != None:
            out = self.activation(out)
        return out, adj


class GNNBlock(nn.Module):
    '''https://github.com/heartcored98/Standalone-DeepLearning-Chemistry'''
    def __init__(self, n_layer, in_dim, hidden_dim, out_dim, device, norm_type='no', sc='sc'):
        super(GNNBlock, self).__init__()
        self.layers = nn.ModuleList()
        self.relu = nn.ReLU()
        for i in range(n_layer):
            self.layers.append(GNNLayer(in_dim if i==0 else hidden_dim,
                                        out_dim if i==n_layer-1 else hidden_dim, device,
                                        nn.ReLU() if i!=n_layer-1 else None,
                                        norm_type))

        if sc=='gsc':
            self.sc = GatedSkipConnection(in_dim, out_dim)
        elif sc=='sc':
            self.sc = SkipConnection(in_dim, out_dim)
        elif sc=='no':
            self.sc = None
        else:
            assert False, "Wrong sc type."
        
    def forward(self, x, adj):
        residual = x
        for i, layer in enumerate(self.layers):
            x, adj = layer(x, adj)
        if self.sc != None:
            x = self.sc(residual, x)
        out = self.relu(x)
        return out


class MolGNN(nn.Module):
    def __init__(self, N_fingerprints, in_dim, hidden_dim, out_dim, 
        layer_hidden, layer_output, device, loss_type, norm_type, skip_connection_type):
        super(MolGNN, self).__init__()
        self.embed_fingerprint = nn.Embedding(N_fingerprints, in_dim)

        self.blocks = nn.ModuleList()
        for i in range(layer_hidden):
            self.blocks.append(GNNBlock(3, in_dim if i==0 else hidden_dim, 
                hidden_dim, hidden_dim, device, norm_type, skip_connection_type))
       
        self.W_output = nn.ModuleList()
        for i in range(layer_output):
            self.W_output.append(nn.Linear(hidden_dim if i==0 else out_dim, out_dim))

        self.layer_hidden = layer_hidden
        self.layer_output = layer_output
        self.device = device
        self.W_property = nn.Linear(out_dim, 1)


    def pad(self, matrices, pad_value):
        """Pad the list of matrices. For example, given a list of matrices [A, B, C],
        we obtain a new matrix [A00, 0B0, 00C], where 0 is the zero (i.e., pad value) matrix.
        """
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        zeros = torch.FloatTensor(np.zeros((M, N))).to(self.device)
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i:i+m, j:j+n] = matrix
            i += m
            j += n
        return pad_matrices


    def forward(self, train_data, train):
        fingerprints, adjacencies, molecular_sizes = train_data[:-1]
        correct_labels = torch.cat(train_data[-1]).view(-1, 1)

        '''Preprocessing input data'''
        fingerprints = torch.cat(fingerprints)
        adjacencies = self.pad(adjacencies, 0)
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        
        '''Hidden layers'''
        for i, block in enumerate(self.blocks):
            fingerprint_vectors = block(fingerprint_vectors, adjacencies)
        
        cal_vectors = [torch.mean(v, 0) for v in torch.split(fingerprint_vectors, molecular_sizes)]
        vectors = torch.stack(cal_vectors)

        '''Output layers'''
        for l in range(self.layer_output):
            vectors = torch.relu(self.W_output[l](vectors))
        predicted_values = self.W_property(vectors)

        return predicted_values, correct_labels