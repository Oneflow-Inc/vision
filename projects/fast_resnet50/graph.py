import oneflow as flow

def make_grad_scaler():
    return flow.amp.GradScaler(
        init_scale=2 ** 30, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000,
    )

class TrainGraph(flow.nn.Graph):
    def __init__(self, model, loss, optimizer, lr_scheduler, data_loader, config):
        super().__init__()
        if config.use_fp16:
            # 使用 nn.Graph 的自动混合精度训练
            self.config.enable_amp(True)
            self.set_grad_scalar(
                flow.amp.GradScaler(
                    init_scale=2 ** 30, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000,
                )
            )
        elif config.scale_grad:
            self.set_grad_scaler(
                flow.amp.StaticGradScaler(flow.env.get_world_size())
            )

        if config.fuse_add_to_output:
            # 使用 nn.Graph 的add算子融合
            self.config.allow_fuse_add_to_output(True)

        if config.fuse_model_update_ops:
            self.config.allow_fuse_model_update_ops(True)

        if config.conv_try_run:
            # 使用 nn.Graph 的卷积试跑优化
            self.config.enable_cudnn_conv_heuristic_search_algo(False)
        
        if config.fuse_pad_to_conv:
            # 使用 nn.Graph 的pad算子融合
            self.config.allow_fuse_pad_to_conv(True)
        

        self.model = model
        self.loss = loss
        self.add_optimizer(optimizer, lr_sch=lr_scheduler)
        self.data_loader = data_loader
    
    def build(self):
        image, label = self.data_loader()
        image = image.to("cuda")
        label = label.to("cuda")
        logits = self.model(image)
        loss = self.cross_entropy(logits, label)
        loss.backward()
        return loss


class EvalGraph(flow.nn.Graph):
    def __init__(self, model, data_loader, config):
        super().__init__()

        if config.use_fp16:
            # 使用 nn.Graph 的自动混合精度训练
            self.config.enable_amp(True)

        if config.fuse_add_to_output:
            # 使用 nn.Graph 的add算子融合
            self.config.allow_fuse_add_to_output(True)

        self.data_loader = data_loader
        self.model = model

    def build(self):
        image, label = self.data_loader()
        image = image.to("cuda")
        label = label.to("cuda")
        logits = self.model(image)
        pred = logits.softmax()
        return pred, label