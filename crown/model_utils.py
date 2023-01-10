import torch
import core.trainer as trainer

def _encode_constraints(model: trainer.nn.Sequential, H: torch.Tensor, d: torch.Tensor):
    num_constraints: int = H.size(0)
    last_layer = model[-1]
    assert isinstance(last_layer, trainer.nn.Linear)
    constraint_layer = trainer.nn.Linear(in_features=last_layer.in_features, out_features=num_constraints)
    constraint_layer.bias.data = H.matmul(last_layer.bias.data) + d
    constraint_layer.weight.data = H.matmul(last_layer.weight.data)

    relu_layer = trainer.nn.ReLU()

    combined_constraint_layer = trainer.nn.Linear(in_features=num_constraints, out_features=1)
    combined_constraint_layer.bias.data = torch.Tensor([0.])
    combined_constraint_layer.weight.data = torch.ones((1, num_constraints))

    model = model[:-1]
    model.append(constraint_layer)
    model.append(relu_layer)
    model.append(combined_constraint_layer)

    H = torch.Tensor([[1.0]])
    d = torch.Tensor([0.0])

    return model, H, d

def _simplify_network(all_layers):
    new_all_layers = [all_layers[0]]
    for layer in all_layers[1:]:
        if (type(layer) is not torch.nn.Linear) and (type(layer) is not torch.nn.ReLU):
            continue
        elif (type(layer) is torch.nn.Linear) and (type(new_all_layers[-1]) is torch.nn.Linear):
            # We can fold together those two layers
            prev_layer = new_all_layers.pop()

            joint_weight = torch.mm(layer.weight.data, prev_layer.weight.data)
            if prev_layer.bias is not None:
                joint_bias = layer.bias.data + torch.mv(layer.weight.data, prev_layer.bias.data)
            else:
                joint_bias = layer.bias.data

            joint_out_features = layer.out_features
            joint_in_features = prev_layer.in_features

            joint_layer = torch.nn.Linear(joint_in_features, joint_out_features)
            joint_layer.bias.data.copy_(joint_bias)
            joint_layer.weight.data.copy_(joint_weight)
            new_all_layers.append(joint_layer)
        else:
            new_all_layers.append(layer)
    return torch.nn.Sequential(*new_all_layers)

def load_model(model_name, file_name, H: torch.Tensor, d: torch.Tensor, stack_n_times: int = 1) -> trainer.nn.Sequential:
    class args():
        def __init__(self):
            self.model = model_name
            self.num_epochs = 1
            self.lr = 0.1

    t = trainer.Trainer(args())
    t.load_model(file_name)
    t.model.eval()

    model = _simplify_network(t.model * stack_n_times)
    return model, H, d


def get_num_layers(model: trainer.nn.Sequential):
    layers = len(model) // 2
    assert layers * 2 == len(model), "Model should have an even number of entries"
    return layers

def get_num_neurons(model: trainer.nn.Sequential, layer: int):
    return model[layer*2+1].weight.detach().numpy().shape[1]

def get_direction_layer_pairs(model: trainer.nn.Sequential):
    num_layers = get_num_layers(model)
    return [(direction, layer) for layer in range(num_layers-1, -1, -1) for direction in ["ubs", "lbs"]]