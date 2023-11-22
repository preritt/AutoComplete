
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