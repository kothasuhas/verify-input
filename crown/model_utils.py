import core.trainer as trainer

def load_model(model_name, file_name):
    class args():
        def __init__(self):
            self.model = model_name
            self.num_epochs = 1
            self.lr = 0.1

    t = trainer.Trainer(args())
    t.load_model(file_name)
    t.model.eval()
    return t.model

def get_num_layers(model: trainer.nn.Sequential):
    layers = len(model) // 2
    assert layers * 2 == len(model), "Model should have an even number of entries"
    return layers

def get_num_neurons(model: trainer.nn.Sequential, layer: int):
    return model[layer*2+1].weight.detach().numpy().shape[1]

def get_direction_layer_pairs(model: trainer.nn.Sequential):
    num_layers = get_num_layers(model)
    return [(direction, layer) for layer in range(num_layers-1, -1, -1) for direction in ["ubs", "lbs"]]