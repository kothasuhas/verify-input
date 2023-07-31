import torch
from torch import nn

def _keras_layer(i, keras_model):
    if i % 2 == 0:
        return torch.nn.Parameter(torch.Tensor(keras_model.weights[i].numpy()).transpose(0, 1))
    else:
        return torch.nn.Parameter(torch.Tensor(keras_model.weights[i].numpy()))
    
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def construct_full_step(keras_model, orig_model, full_step, u_limits, u_lb, u_ub, A, B,
                        STATE_DIM, HIDDEN1_DIM, HIDDEN2_DIM, POLICY_DIM):
    
    assert u_limits == (u_lb is not None)
    assert u_limits == (u_ub is not None)

    policy_base = nn.Sequential(
        Flatten(),
        nn.Linear(STATE_DIM, HIDDEN1_DIM),
        nn.ReLU(),
        nn.Linear(HIDDEN1_DIM, HIDDEN2_DIM),
        nn.ReLU(),
        nn.Linear(HIDDEN2_DIM, STATE_DIM)
    )

    with torch.no_grad():
        policy_base[1].weight = _keras_layer(0, keras_model)
        policy_base[1].bias   = _keras_layer(1, keras_model)
        policy_base[3].weight = _keras_layer(2, keras_model)
        policy_base[3].bias   = _keras_layer(3, keras_model)
        # here, we assume no u_limits are used. If they are, this layer will be replaced
        policy_base[5].weight = torch.nn.Parameter(B.matmul(_keras_layer(4, keras_model)))
        policy_base[5].bias   = torch.nn.Parameter(B.matmul(_keras_layer(5, keras_model)))

        orig_model[1].weight = _keras_layer(0, keras_model)
        orig_model[1].bias   = _keras_layer(1, keras_model)
        orig_model[3].weight = _keras_layer(2, keras_model)
        orig_model[3].bias   = _keras_layer(3, keras_model)
        orig_model[5].weight = _keras_layer(4, keras_model)
        orig_model[5].bias   = _keras_layer(5, keras_model)
    torch.save(orig_model.state_dict(), 'doubleintegrator_orig.pt')

    if u_limits:
        policy_base = policy_base[:-1]  # the last layer already incorporated B
        layer = nn.Linear(policy_base[-2].out_features, POLICY_DIM, True)
        layer.weight = _keras_layer(4, keras_model)
        layer.bias = _keras_layer(5, keras_model)
        policy_base.append(layer)  # now the model outputs u, not Bu

        # Last b -= u_min
        policy_base[-1].bias.data -= u_lb
        # ReLU
        policy_base.append(nn.ReLU())
        # W = -I, b = u_max - u_min
        layer = nn.Linear(POLICY_DIM, POLICY_DIM, bias=True)
        layer.weight.data = -torch.eye(POLICY_DIM)
        layer.bias.data = torch.Tensor(u_ub - u_lb)
        policy_base.append(layer)
        # ReLU
        policy_base.append(nn.ReLU())
        # W = -I, b = u_max
        layer = nn.Linear(POLICY_DIM, POLICY_DIM, bias=True)
        layer.weight.data = torch.nn.Parameter(-B)  # u is clipped, so B can be used now
        layer.bias.data = u_ub.matmul(B.T)
        policy_base.append(layer)

    with torch.no_grad():
        for layer in full_step:
            if not isinstance(layer, nn.Linear):
                continue
            else:
                layer.weight = torch.nn.Parameter(torch.zeros_like(layer.weight))
                layer.bias = torch.nn.Parameter(torch.zeros_like(layer.bias))

        full_step[1].weight[:HIDDEN1_DIM] = policy_base[1].weight
        full_step[1].weight[-STATE_DIM:] = torch.nn.Parameter(torch.eye(STATE_DIM))
        full_step[1].bias[:HIDDEN1_DIM] = policy_base[1].bias
        full_step[1].bias[-STATE_DIM:] = torch.nn.Parameter(torch.Tensor([10.0 for _ in range(STATE_DIM)])) # TODO

        full_step[3].weight[:HIDDEN2_DIM,:HIDDEN1_DIM] = policy_base[3].weight
        full_step[3].weight[-STATE_DIM:,-STATE_DIM:] = torch.nn.Parameter(torch.eye(STATE_DIM))
        full_step[3].bias[:HIDDEN2_DIM] = policy_base[3].bias

        if u_limits:
            full_step[5].weight[:POLICY_DIM,:HIDDEN2_DIM] = policy_base[5].weight
            full_step[5].weight[-STATE_DIM:,-STATE_DIM:] = torch.nn.Parameter(torch.eye(STATE_DIM))
            full_step[5].bias[:POLICY_DIM] = policy_base[5].bias

            full_step[7].weight[:POLICY_DIM,:POLICY_DIM] = policy_base[7].weight
            full_step[7].weight[-STATE_DIM:,-STATE_DIM:] = torch.nn.Parameter(torch.eye(STATE_DIM))
            full_step[7].bias[:POLICY_DIM] = policy_base[7].bias

            full_step[9].weight[:,:POLICY_DIM] = policy_base[9].weight
            full_step[9].weight[:,-STATE_DIM:] = torch.nn.Parameter(A)
            full_step[9].bias = torch.nn.Parameter(policy_base[9].bias - A.matmul(torch.Tensor([10.0 for _ in range(STATE_DIM)])))
        else:
            full_step[5].weight[:,:HIDDEN2_DIM] = policy_base[5].weight
            full_step[5].weight[:,-STATE_DIM:] = torch.nn.Parameter(A)
            full_step[5].bias = torch.nn.Parameter(policy_base[5].bias - A.matmul(torch.Tensor([10.0 for _ in range(STATE_DIM)])))

    def forward(x, orig_model):
        policy_output = orig_model(x)
        if u_limits:
            policy_output = torch.clip(policy_output, min=u_lb, max=u_ub) # TODO: generalize to more dims
            return nn.functional.linear(x, A, bias=None) + nn.functional.linear(policy_output, B, bias=None)
        else:
            return nn.functional.linear(x, A, bias=None) + nn.functional.linear(policy_output, B, bias=None)

    # Ensure the model combining Ax+Bu is equivalent
    for _ in range(100):
        input_tensor = torch.randn(1, STATE_DIM)
        assert ((forward(input_tensor, orig_model) - full_step(input_tensor)) < 1e-5).all()

    return full_step