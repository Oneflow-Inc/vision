import oneflow as flow
import torch


def convert_torch2oneflow(model, torch_weight_path, save_path):
    torch_params = torch.load(torch_weight_path)
    params = list()
    for name, weights in torch_params.items():
        if 'num_batches_tracked' not in name:
            params.append(weights.detach().cpu().numpy())
    # solve different names problem
    flow_params = model.state_dict()
    for i, (name, weights) in enumerate(flow_params.items()):
        flow_params[name] = params[i]
    model.load_state_dict(flow_params)
    flow.save(model.state_dict(), save_path)

if __name__ == '__main__':
    from network.nets import Vgg16
    vgg_flow = Vgg16()
    convert_torch2oneflow(vgg_flow, '/home/kaijie/Documents/checkpoints/unit/vgg16.pth', './vgg16_flow')
