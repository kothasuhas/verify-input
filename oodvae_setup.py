from crown.model_utils import load_model, _simplify_network
import torch
import torch.nn as nn

vae = load_model("vae", "vae.pt")
decoder = vae.decoder

mlp = load_model("mlp", "mlp_vanilla.pt")

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

decoder_mlp = _simplify_network(nn.Sequential(Flatten()) + decoder + mlp)
print(decoder_mlp)

torch.save(decoder_mlp.state_dict(), 'decoder_mlp.pt')
