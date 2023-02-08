from torch import nn

from .MLP import MLP
from .Transformer import PCTransformer


class BackBone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.knn_layer = config.knn_layer

        self.transformer = PCTransformer(in_chans=config.n_channels,
                                         embed_dim=config.embed_dim,
                                         depth=config.encoder_depth,
                                         mlp_ratio=config.mlp_ratio,
                                         qkv_bias=config.qkv_bias,
                                         knn_layer=config.knn_layer,
                                         num_heads=config.num_heads,
                                         attn_drop_rate=config.attn_drop_rate,
                                         drop_rate=config.drop_rate,
                                         qk_scale=config.qk_scale,
                                         out_size=config.out_size)

        # Select between deep feature extractor and not
        if config.use_deep_weights_generator:
            generator = MLP
        else:
            generator = nn.Linear

        # Select the right dimension for linear layers
        global_size = config.out_size
        if config.use_object_id:
            global_size = config.out_size * 2

        # Generate first weight, bias and scale of the input layer of the implicit function
        self.output = nn.ModuleList([nn.ModuleList([
            generator(global_size, config.hidden_dim * 3),
            generator(global_size, config.hidden_dim),
            generator(global_size, config.hidden_dim)])])

        # Generate weights, biases and scales of the hidden layers of the implicit function
        for _ in range(config.depth):
            self.output.append(nn.ModuleList([
                generator(global_size, config.hidden_dim * config.hidden_dim),
                generator(global_size, config.hidden_dim),
                generator(global_size, config.hidden_dim)
            ]))
        # Generate weights, biases and scales of the output layer of the implicit function
        self.output.append(nn.ModuleList([
            generator(global_size, config.hidden_dim),
            generator(global_size, 1),
            generator(global_size, 1),
        ]))

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm) or isinstance(m, nn.GroupNorm) or isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data, gain=1)

            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, xyz, object_id=None):

        global_feature = self.transformer(xyz)  # B M C and B M 3

        fast_weights = []
        for layer in self.output:
            fast_weights.append([ly(global_feature) for ly in layer])

        # return fast_weights, global_feature
        return fast_weights, global_feature


