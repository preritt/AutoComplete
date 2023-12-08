
#%%
import torch
import numpy as np
import torch.nn as nn

parseFloat = lambda raw: float(raw[0] + '.'+raw[1:])
getconf = lambda tags, name: tags.split(name)[1].split('_')[0]

class AutoComplete(nn.Module):
    def __init__(self,
			indim=80, # input data dimension
			width=10, # encoding dim ratio; 10=x1.0, 20=x0.5
			n_depth=4, # number of layers between input layer & encoding layer
			n_multiples=0, # repeated layers of same dim per layer
			nonlin=lambda dim: torch.nn.LeakyReLU(inplace=True), # the nonlinearity
			verbose=False
		):
        super().__init__()

        outdim = indim

        if verbose:
            print('WIDTH', width)
            print('DEPTH', n_depth)
            print('MULT', n_multiples)
            print('NONLIN', nonlin)
            print('In D', indim)
            print('OutD', outdim)

        spec = []
        zdim = int(indim/width)
        zlist = list(np.linspace(indim, zdim, n_depth+1).astype(int))
        if verbose: print('Encoding progression:', zlist)

        for li in range(n_depth):
            dnow = zlist[li]
            dnext = zlist[li+1]
            spec += [(dnow, dnext)]
            if li != n_depth-1:
                for mm in range(n_multiples):
                    spec += [(dnext, dnext)]

        if verbose: print('Fc layers spec:', spec)

        layers = []
        for si, (d1, d2) in enumerate(spec):
            layers += [nn.Linear(d1, d2)]
            layers += [nonlin(d2)]

        for si, (d2, d1) in enumerate(spec[::-1]):
            d2 = outdim if si == len(spec)-1 else d2
            layers += [nn.Linear(d1, d2)]
            if si != len(spec)-1:
                layers += [nonlin(d2)]

        self.net = nn.Sequential(*layers)

        if verbose: print('Zdim:', zlist[-1])

    def forward(self, x):
        x = self.net(x)
        return x


# %%
class AutoCompleteWithMissingMask(nn.Module):
    def __init__(self,
			indim=80, # input data dimension
			width=10, # encoding dim ratio; 10=x1.0, 20=x0.5
			n_depth=4, # number of layers between input layer & encoding layer
			n_multiples=0, # repeated layers of same dim per layer
			nonlin=lambda dim: torch.nn.LeakyReLU(inplace=True), # the nonlinearity
			verbose=False
		):
        super().__init__()

        outdim = indim

        if verbose:
            print('WIDTH', width)
            print('DEPTH', n_depth)
            print('MULT', n_multiples)
            print('NONLIN', nonlin)
            print('In D', indim)
            print('OutD', outdim)

        spec = []
        zdim = int(indim/width)
        zlist = list(np.linspace(indim, zdim, n_depth+1).astype(int))
        zlist[0] = indim*2
        if verbose: print('Encoding progression:', zlist)

        for li in range(n_depth):
            dnow = zlist[li]
            dnext = zlist[li+1]
            spec += [(dnow, dnext)]
            if li != n_depth-1:
                for mm in range(n_multiples):
                    spec += [(dnext, dnext)]

        if verbose: print('Fc layers spec:', spec)

        layers = []
        for si, (d1, d2) in enumerate(spec):
            layers += [nn.Linear(d1, d2)]
            layers += [nonlin(d2)]

        for si, (d2, d1) in enumerate(spec[::-1]):
            d2 = outdim if si == len(spec)-1 else d2
            layers += [nn.Linear(d1, d2)]
            if si != len(spec)-1:
                layers += [nonlin(d2)]

        self.net = nn.Sequential(*layers)

        if verbose: print('Zdim:', zlist[-1])

    def forward(self, x):

        # Assuming x is a tensor on CUDA
        y = torch.where(x == 0, torch.zeros_like(x), torch.ones_like(x))
        concatenated_data = torch.cat((x, y), dim=1)
        x = self.net(concatenated_data)
        return x
    
# %%
class AutoCompleteVAE(nn.Module):
    def __init__(self,
                 indim=80,  # input data dimension
                 width=10,  # encoding dim ratio; 10=x1.0, 20=x0.5
                 n_depth=4,  # number of layers between input layer & encoding layer
                 n_multiples=0,  # repeated layers of same dim per layer
                 nonlin=lambda dim: torch.nn.LeakyReLU(inplace=True),  # the nonlinearity
                 verbose=False
                 ):
        super().__init__()

        outdim = indim

        if verbose:
            print('WIDTH', width)
            print('DEPTH', n_depth)
            print('MULT', n_multiples)
            print('NONLIN', nonlin)
            print('In D', indim)
            print('OutD', outdim)

        spec = []
        zdim = int(indim / width)
        zlist = list(np.linspace(indim, zdim, n_depth + 1).astype(int))
        if verbose:
            print('Encoding progression:', zlist)

        for li in range(n_depth):
            dnow = zlist[li]
            dnext = zlist[li + 1]
            spec += [(dnow, dnext)]
            if li != n_depth - 1:
                for mm in range(n_multiples):
                    spec += [(dnext, dnext)]

        if verbose:
            print('Fc layers spec:', spec)

        # Encoder layers
        encoder_layers = []
        for si, (d1, d2) in enumerate(spec):
            encoder_layers += [nn.Linear(d1, d2)]
            encoder_layers += [nonlin(d2)]

        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space layers
        self.mean_layer = nn.Linear(zlist[-1], zlist[-1])
        self.logvar_layer = nn.Linear(zlist[-1], zlist[-1])

        # Decoder layers
        decoder_layers = []
        for si, (d2, d1) in enumerate(spec[::-1]):
            d2 = outdim if si == len(spec) - 1 else d2
            decoder_layers += [nn.Linear(d1, d2)]
            if si != len(spec) - 1:
                decoder_layers += [nonlin(d2)]

        self.decoder = nn.Sequential(*decoder_layers)

        if verbose:
            print('Zdim:', zlist[-1])

    def encode(self, x):
        x = self.encoder(x)
        return self.mean_layer(x), self.logvar_layer(x)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar
# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
# %%
class TransformerNoPosAutoCompleteWithoutMissingMask(nn.Module):
    def __init__(self,
                 indim=80,  # input data dimension
                 n_layers=4,  # number of layers in the transformer encoder
                 n_head=2,  # number of attention heads in the transformer encoder
                 d_model=64,  # dimension of the transformer encoder input and output
                 dim_feedforward=256,  # dimension of the feedforward network in the transformer encoder
                 dropout=0.1,  # dropout rate in the transformer encoder
                 verbose=False):
        super().__init__()

        encoder_layers = TransformerEncoderLayer(indim, n_head, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)

        self.fc = nn.Linear(indim, indim)

        if verbose:
            print('In D', indim)
            print('Out D', indim)

    def forward(self, x):
        
        x = self.transformer_encoder(x)
        x = self.fc(x)
        return x
# %%
# %%
class TransformerNoPosAutoCompleteWithMissingMask(nn.Module):
    def __init__(self,
                 indim=80,  # input data dimension
                 n_layers=4,  # number of layers in the transformer encoder
                 n_head=2,  # number of attention heads in the transformer encoder
                 d_model=64,  # dimension of the transformer encoder input and output
                 dim_feedforward=256,  # dimension of the feedforward network in the transformer encoder
                 dropout=0.1,  # dropout rate in the transformer encoder
                 verbose=False):
        super().__init__()

        encoder_layers = TransformerEncoderLayer(indim*2, n_head, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)

        self.fc = nn.Linear(indim*2, indim)

        if verbose:
            print('In D', indim)
            print('Out D', indim)

    def forward(self, x):
        # Assuming x is a tensor on CUDA
        y = torch.where(x == 0, torch.zeros_like(x), torch.ones_like(x))
        concatenated_data = torch.cat((x, y), dim=1)
        x = self.transformer_encoder(concatenated_data)
        x = self.fc(x)
        return x
# %%
class TransformerAutoCompleteWithMissingMask(nn.Module):
    def __init__(self,
                 indim=80,  # input data dimension
                 n_layers=4,  # number of layers in the transformer encoder
                 n_head=2,  # number of attention heads in the transformer encoder
                 d_model=64,  # dimension of the transformer encoder input and output
                 dim_feedforward=256,  # dimension of the feedforward network in the transformer encoder
                 dropout=0.1,  # dropout rate in the transformer encoder
                 verbose=False):
        super().__init__()

        self.embedding = nn.Linear(indim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layers = TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)

        self.fc = nn.Linear(d_model, indim)

        if verbose:
            print('In D', indim)
            print('Out D', indim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.fc(x)
        return x

#%%
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
# %%
class TransformerNoPosAutoCompleteWithoutMissingMaskV2(nn.Module):
    def __init__(self,
                 indim=80,  # input data dimension
                 n_layers=2,  # number of layers in the transformer encoder
                 n_head=1,  # number of attention heads in the transformer encoder
                 d_model=64,  # dimension of the transformer encoder input and output
                 dim_feedforward=256,  # dimension of the feedforward network in the transformer encoder
                 dropout=0.1,  # dropout rate in the transformer encoder
                 verbose=False):
        super().__init__()

        encoder_layers = TransformerEncoderLayer(1, n_head, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)

        # self.fc = nn.Linear(indim, indim)
        # self.fc = nn.Linear(1, dim_feedforward)
        # Add a feedforward layer
        self.feedforward = nn.Sequential(
            nn.Linear(1, dim_feedforward),
            nn.Linear(dim_feedforward, 1)  # Adjust the output size to 1
        )
        if verbose:
            print('In D', indim)
            print('Out D', indim)

    def forward(self, x):
        x = x.unsqueeze(-1)
        # x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = self.feedforward(x)
        x = x.squeeze(-1) 
        return x
# %%
class TransformerNoPosAutoCompleteWithoutMissingWithMaskV2(nn.Module):
    def __init__(self,
                 indim=80,  # input data dimension
                 n_layers=2,  # number of layers in the transformer encoder
                 n_head=2,  # number of attention heads in the transformer encoder
                 d_model=64,  # dimension of the transformer encoder input and output
                 dim_feedforward=256,  # dimension of the feedforward network in the transformer encoder
                 dropout=0.1,  # dropout rate in the transformer encoder
                 dim_ff_penultimate=32,
                 verbose=False):
        super().__init__()

        encoder_layers = TransformerEncoderLayer(2, n_head, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)

        # self.fc = nn.Linear(indim, indim)
        # self.fc = nn.Linear(1, dim_feedforward)
        # Add a feedforward layer
        self.feedforward = nn.Sequential(
            nn.Linear(2, dim_ff_penultimate),
            nn.ReLU(),
            nn.Linear(dim_ff_penultimate, 2),  # Adjust the output size to 1
            nn.ReLU()
        )
        self.final_fc = nn.Linear(indim*2, indim)
        if verbose:
            print('In D', indim)
            print('Out D', indim)

    def forward(self, x):
        y = torch.where(x == 0, torch.zeros_like(x), torch.ones_like(x))
        # concatenated_data = torch.cat((x, y), dim=0)
        tensor = torch.cat((x.unsqueeze(-1), y.unsqueeze(-1)), dim=-1)
        # x = concatenated_data.unsqueeze(-1)
        # x = x.permute(1, 0, 2)
        x = self.transformer_encoder(tensor)
        
        # x = self.feedforward(x)
        # reshape x to be of batch size x 
        x = x.reshape(x.shape[0], -1)

        x = self.final_fc(x)
        return x
    
# %%
class TransformerNoPosAutoCompleteWithoutMissingMaskAttention(nn.Module):
    def __init__(self,
                 indim=80,  # input data dimension
                 n_layers=4,  # number of layers in the transformer encoder
                 n_head=2,  # number of attention heads in the transformer encoder
                 d_model=64,  # dimension of the transformer encoder input and output
                 dim_feedforward=256,  # dimension of the feedforward network in the transformer encoder
                 dropout=0.1,  # dropout rate in the transformer encoder
                 verbose=False):
        super().__init__()

        # Create the transformer encoder with multi-head attention
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=indim,
                nhead=n_head,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            ),
            num_layers=n_layers
        )

        # Linear layer
        self.fc = nn.Linear(indim, indim)

        if verbose:
            print('In D', indim)
            print('Out D', indim)

    def forward(self, x):
        # Forward pass through the transformer encoder
        attention_scores = []  # List to store attention scores for each layer

        for layer in self.transformer_encoder.layers:
            # Get attention scores from each layer
            x, attention_score = layer.self_attn(x, x, x)
            attention_scores.append(attention_score)

        # Apply linear layer
        x = self.fc(x)

        return x, attention_scores
# %%
class TransformerNoPosAutoCompleteWithoutMissingWithMaskV2(nn.Module):
    def __init__(self,
                 indim=80,  # input data dimension
                 n_layers=2,  # number of layers in the transformer encoder
                 n_head=2,  # number of attention heads in the transformer encoder
                 d_model=64,  # dimension of the transformer encoder input and output
                 dim_feedforward=256,  # dimension of the feedforward network in the transformer encoder
                 dropout=0.1,  # dropout rate in the transformer encoder
                 dim_ff_penultimate=32,
                 verbose=False):
        super().__init__()

        encoder_layers = TransformerEncoderLayer(2, n_head, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)

        # self.fc = nn.Linear(indim, indim)
        # self.fc = nn.Linear(1, dim_feedforward)
        # Add a feedforward layer
        self.feedforward = nn.Sequential(
            nn.Linear(2, dim_ff_penultimate),
            nn.ReLU(),
            nn.Linear(dim_ff_penultimate, 2),  # Adjust the output size to 1
            nn.ReLU()
        )
        self.final_fc = nn.Linear(indim*2, indim)
        if verbose:
            print('In D', indim)
            print('Out D', indim)

    def forward(self, x):
        y = torch.where(x == 0, torch.zeros_like(x), torch.ones_like(x))
        # concatenated_data = torch.cat((x, y), dim=0)
        tensor = torch.cat((x.unsqueeze(-1), y.unsqueeze(-1)), dim=-1)
        # x = concatenated_data.unsqueeze(-1)
        # x = x.permute(1, 0, 2)
        x = self.transformer_encoder(tensor)
        
        # x = self.feedforward(x)
        # reshape x to be of batch size x 
        x = x.reshape(x.shape[0], -1)

        x = self.final_fc(x)
        return x
# %%
####### NEw code based on the ATM method
class TransformerCompatibleInput(nn.Module):
	# make transformer compatible input
    """
    A PyTorch module that creates a weight vector and multiplies it with the input tensor.

    Args:
        output_size (int): The size of the output tensor.

    Returns:
        torch.Tensor: The output tensor after being multiplied by the weight vector.
    """
    def __init__(self, input_dim, output_size):
        super().__init__()
        self.output_size = output_size

        # create the weight vector and register it as a parameter
        self.weight = nn.Parameter(torch.randn(input_dim, output_size))

    def forward(self, x):
        """
        x is a 1D tensor of size (batch_size, input_size, 1)
        The forward pass multiplies the input tensor by the weight vector to produce the output tensor.
        of size (batch_size, input_size, output_size)
        """
        # multiply the input tensor by the weight vector
        x = torch.matmul(x, self.weight)

        return x
#%%



# Define the input tensors
# input_tensor = torch.zeros(32, 300, 2)

# transformer_input = TransformerCompatibleInput(input_dim=2, output_size=64)

# # Pass the input tensor through the TransformerCompatibleInput module
# output = transformer_input(input_tensor)

# # Print the output and its shape
# print(output)
# print(output.shape)
    
# %%
class TransformerAdaptInput(nn.Module):
    def __init__(self,
                 indim=2,  # input data dimension
                 n_layers=2,  # number of layers in the transformer encoder
                 n_head=8,  # number of attention heads in the transformer encoder
                 d_model=32,  # dimension of the transformer encoder input and output
                 dim_feedforward=128,  # dimension of the feedforward network in the transformer encoder
                 dropout=0.1,  # dropout rate in the transformer encoder
                 verbose=False):
        super().__init__()

        # make a transformer compatible input
        self.transformer_input = TransformerCompatibleInput(indim, d_model)
        # Create the transformer encoder with multi-head attention
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first = True
            ),
            num_layers=n_layers
        )

        # Linear layer
        self.fc = nn.Linear(d_model, 1)

        if verbose:
            print('In D', indim)
            print('Out D', indim)

    def forward(self, x):

        y = torch.where(x == 0, torch.zeros_like(x), torch.ones_like(x))
        # concatenated_data = torch.cat((x, y), dim=0)
        tensor = torch.cat((x.unsqueeze(-1), y.unsqueeze(-1)), dim=-1)
        # make this to be transformer compatible
        tensor = self.transformer_input(tensor)

        # x = concatenated_data.unsqueeze(-1)
        # x = x.permute(1, 0, 2)
        x = self.transformer_encoder(tensor)

        # Forward pass through the transformer encoder
        attention_scores = []  # List to store attention scores for each layer

        for layer in self.transformer_encoder.layers:
            # Get attention scores from each layer
            x, attention_score = layer.self_attn(x, x, x)
            attention_scores.append(attention_score)

        # Apply linear layer
        x = self.fc(x)
        # remove the last dimension
        x = x.squeeze(-1)

        return x, attention_scores
    
# %%
import math
class PositionalEncoding(nn.Module):
	# define positional encoding
    """
    This class implements the positional encoding for a transformer model.

    Args:
        d_model (int): The number of expected features in the input.
        max_seq_len (int): The maximum sequence length.

    Attributes:
        pe (torch.Tensor): The positional encoding matrix.

    """
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # create the positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register the positional encoding matrix as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Adds the positional encoding to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The input tensor with the positional encoding added.

        """
        return self.pe[:, :x.size(1)]

# %%
class TransformerAdaptInputWithPosition(nn.Module):
    def __init__(self,
                 indim=2,  # input data dimension
                 n_layers=2,  # number of layers in the transformer encoder
                 n_head=8,  # number of attention heads in the transformer encoder
                 d_model=32,  # dimension of the transformer encoder input and output
                 dim_feedforward=128,  # dimension of the feedforward network in the transformer encoder
                 dropout=0.1,  # dropout rate in the transformer encoder
                 verbose=False,
                 positional_encoding=PositionalEncoding,
                 max_seq_len=1028):
        super().__init__()

        # make a transformer compatible input
        self.transformer_input = TransformerCompatibleInput(indim, d_model)
        self.positional_encoding = positional_encoding(d_model, max_seq_len)
        # Create the transformer encoder with multi-head attention
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first = True
            ),
            num_layers=n_layers
        )

        # Linear layer
        self.fc = nn.Linear(d_model, 1)
        self.d_model = d_model
        # add a layer normalization layer
        self.layer_norm = nn.LayerNorm(d_model)

        if verbose:
            print('In D', indim)
            print('Out D', indim)

    def forward(self, x):

        y = torch.where(x == 0, torch.zeros_like(x), torch.ones_like(x))
        # concatenated_data = torch.cat((x, y), dim=0)
        tensor = torch.cat((x.unsqueeze(-1), y.unsqueeze(-1)), dim=-1)
        # make this to be transformer compatible
        tensor = self.transformer_input(tensor)*math.sqrt(self.d_model)
        tensor = self.layer_norm(tensor)
        tensor = tensor + self.positional_encoding(tensor)
        

        # x = concatenated_data.unsqueeze(-1)
        # x = x.permute(1, 0, 2)
        x = self.transformer_encoder(tensor)

        # Forward pass through the transformer encoder
        attention_scores = []  # List to store attention scores for each layer

        for layer in self.transformer_encoder.layers:
            # Get attention scores from each layer
            x, attention_score = layer.self_attn(x, x, x)
            attention_scores.append(attention_score)

        # Apply linear layer
        x = self.fc(x)
        # remove the last dimension
        x = x.squeeze(-1)

        return x, attention_scores

# %%
class HybridTransformerAutoencoder(nn.Module):
    def __init__(self,
                 indim=2,  # input data dimension
                 n_layers=2,  # number of layers in the transformer encoder
                 n_head=8,  # number of attention heads in the transformer encoder
                 d_model=32,  # dimension of the transformer encoder input and output
                 dim_feedforward=128,  # dimension of the feedforward network in the transformer encoder
                 dropout=0.1,  # dropout rate in the transformer encoder
                 verbose=False,
                 positional_encoding=PositionalEncoding,
                 max_seq_len=1028,
                number_continuous_features = 100,
                number_categorical_features = 10):
        super().__init__()

        # make a transformer compatible input
        self.transformer_input = TransformerCompatibleInput(indim, d_model)
        self.positional_encoding = positional_encoding(d_model, max_seq_len)
        self.number_continuous_features = number_continuous_features
        self.number_categorical_features = number_categorical_features
        # Create the transformer encoder with multi-head attention
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first = True
            ),
            num_layers=n_layers
        )

        # Linear layer
        # self.fc = nn.Linear(d_model, 1)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 1),
            # nn.LeakyReLU()
        )
        self.d_model = d_model
        # add a layer normalization layer
        self.layer_norm = nn.LayerNorm(d_model)

        # define the autoencoder with mask AutoCompleteWithMissingMask
        self.autoencoder = AutoCompleteWithMissingMask(indim=number_categorical_features, 
                                                       width=1, n_depth=1, n_multiples=0,
                                                        nonlin=lambda dim: torch.nn.LeakyReLU(inplace=True), verbose=False)

        self.fc_all = nn.Sequential(
            nn.Linear(number_continuous_features + number_categorical_features, 
                      number_continuous_features + number_categorical_features),
            nn.LeakyReLU(),
            nn.Linear(number_continuous_features + number_categorical_features, 
                      number_continuous_features + number_categorical_features),
        )

        if verbose:
            print('In D', indim)
            print('Out D', indim)

    def forward(self, x):
        # get the continuous and categorical features
        x_continuous = x[:, :self.number_continuous_features]
        x_categorical = x[:, self.number_continuous_features:]
        # get location of missing values
        y_continuous = torch.where(x_continuous == 0, torch.zeros_like(x_continuous), torch.ones_like(x_continuous))
        # y_categorical = torch.where(x_categorical == 0, torch.zeros_like(x_categorical), torch.ones_like(x_categorical))

        discrete_output = self.autoencoder(x_categorical)

        # y = torch.where(x_continuous == 0, torch.zeros_like(x_continuous), torch.ones_like(x_continuous))
        # concatenated_data = torch.cat((x, y), dim=0)
        tensor = torch.cat((x_continuous.unsqueeze(-1), y_continuous.unsqueeze(-1)), dim=-1)
        # make this to be transformer compatible
        tensor = self.transformer_input(tensor)*math.sqrt(self.d_model)
        tensor = self.layer_norm(tensor)
        tensor = tensor + self.positional_encoding(tensor)
        
        # x = concatenated_data.unsqueeze(-1)
        # x = x.permute(1, 0, 2)
        x = self.transformer_encoder(tensor)

        # Forward pass through the transformer encoder
        attention_scores = []  # List to store attention scores for each layer

        for layer in self.transformer_encoder.layers:
            # Get attention scores from each layer
            x, attention_score = layer.self_attn(x, x, x)
            attention_scores.append(attention_score)

        # Apply linear layer
        x = self.fc(x)
        # remove the last dimension
        x = x.squeeze(-1)

        # concatenate x with discrete_output
        x = torch.cat((x, discrete_output), dim=1)
        # x = self.fc_all(x)


        return x, attention_scores