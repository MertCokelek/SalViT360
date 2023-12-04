from model.utils.projection import Equi2Pers, Pers2Equi
from model.blocks import Posemb_LW, UpsampleBlock3D
from model.VSTA import SphericalVideoTransformer

from tqdm import tqdm
from dotmap import DotMap
from torch import nn
from torchvision.models import resnet18
from einops import rearrange

import torch
import torch.nn.functional as F
import torchvision.transforms as tf

SPH_EMB_SIZE = 56


class SalViT360_VSTA(nn.Module):
    def __init__(self, config: DotMap = None, erp_size=(960, 1920)):
        super().__init__()

        network_config = config.network
        tangent_config = config.tangent_images

        self.nrows = tangent_config.nrows
        self.fov = tangent_config.fov
        self.patch_size_e2p = tangent_config.patch_size.e2p[0]

        # Encoder
        self.feature_extractor = torch.nn.Sequential(*list(resnet18(pretrained=True).eval().children())[:-2])

        self.resnet_layernorm = nn.LayerNorm([512, 7, 7])

        self.e2p = Equi2Pers(erp_size=erp_size, fov=self.fov[0], nrows=self.nrows[0], patch_size=self.patch_size_e2p)
        self.p2e = Pers2Equi(erp_size=(240, 480), fov=self.fov[0], nrows=self.nrows[0], patch_size=(28, 28))
        self.sph_coords = F.adaptive_avg_pool2d(
            Equi2Pers(erp_size, self.fov[0], self.nrows[0], SPH_EMB_SIZE).get_spherical_embeddings(), (14, 14))
        self.sph_coord_dim = 14 * 14 * 5  # Hard-coded
        self.posemb = Posemb_LW()

        # Decoder
        self.set_tangent_decoder(config)

        self.down = nn.AdaptiveAvgPool2d(output_size=(1, 1))  # Out size: [b t 512 1 1]

        config_transformer = network_config.transformer
        
        transformer_hparams = {'emb_dim': 512,
                               'sph_input_dim': self.sph_coord_dim,
                               'depth': config_transformer.depth,
                               'num_heads': config_transformer.num_heads,
                               'mlp_mult': config_transformer.mlp_dim,
                               'ffdropout': config_transformer.ff_dropout,
                               'attn_dropout': config_transformer.attn_dropout,
                               }

        self.transformer = SphericalVideoTransformer(**transformer_hparams)



    def forward(self, x):
        with torch.no_grad():
            x_tang = self.videoproject_singlescale(x)  # BS, F=8, C=3, H=224, W=224, T=18
            BS, F, _, _, _, _ = x_tang.shape

            # Encoder
            x_tang = rearrange(x_tang, 'b f c h w t -> (b f t) c h w', b=BS, f=F)
            resnet_features = self.feature_extractor(x_tang)  # (b f t) c h w
            features = self.down(resnet_features)

            features = rearrange(features, '(b f t) c h w -> b f t (c h w)', b=BS, f=F)

            # Spherical Position Embedding
            sph_coords = self.sph_coords.to(x_tang.device)  # T, 5, 14, 14
            sph_emb = rearrange(sph_coords, 't c h w -> t (c h w)')  # T, 5*14*14=980

        # Transformer
        features = self.transformer(features, sph_emb)  # BS, F, T, D=512
        features = rearrange(features[:, -1], 'b t d -> b d 1 1 t')
        resnet_features = self.resnet_layernorm(resnet_features)
        resnet_features = rearrange(resnet_features, '(b f t) c h w -> b f c h w t', b=BS, f=F)[:, -1]

        decoder_in = features + resnet_features
        decoder_out = self.decoder(decoder_in)[:, :1]
        decoder_out = self.p2e(decoder_out)

        # 240x480 -> 480x960
        pred = nn.functional.interpolate(decoder_out, size=(480, 960), mode='bilinear', align_corners=False)
        return pred


    @torch.no_grad()
    @torch.inference_mode()
    def inference(self, video):
        """
        video: list of Tensors with shape: [C, H, W]
        
        return: Saliency predictions with shape: [F, 1, H, W]
        """
        transform = tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        n_frames = len(video)

        preds = []
        
        for st in tqdm(range(n_frames - 8)):
            clip = video[st:st + 8]  # 8 frames
            clip = transform(torch.stack(clip).float() / 255.).unsqueeze(0)
            
            pred = self.forward(clip.cuda())
            pred = pred.squeeze(1).detach().cpu().numpy()
            preds.append(pred)
        
        return preds



    def set_tangent_decoder(self, config):
        """
        Full tangent decoder.
        Its architecture is simplified (channel dims and n_blocks) V1-Decoder (with optional Blur and LayerNorm)
        input: Tangent Feature Maps [B, C=512, H=7, W=7, T]
        """
        norm_layer = config.network.decoder.norm_layer
        apply_blur = config.network.decoder.apply_blur

        decoder_out_dim = 2
        if norm_layer == 'layernorm':
            t = 18
            norm_layers = [
                nn.LayerNorm([128, 7, 7, t]),
                nn.LayerNorm([32, 7, 7, t]),
                nn.LayerNorm([8, 14, 14, t]),
                nn.LayerNorm([decoder_out_dim, 28, 28, t])
            ]
        elif norm_layer == 'batchnorm':
            norm_layers = [nn.BatchNorm3d(128), nn.BatchNorm3d(32), nn.BatchNorm3d(8), nn.BatchNorm3d(decoder_out_dim)]
        else:
            norm_layers = [nn.Identity() for _ in range(4)]

        self.decoder = nn.Sequential(
            UpsampleBlock3D(512, 128, norm_layers[0], apply_blur=apply_blur, upsample=False),
            UpsampleBlock3D(128, 32, norm_layers[1], apply_blur=apply_blur),
            UpsampleBlock3D(32, 8, norm_layers[2], apply_blur=apply_blur),
            UpsampleBlock3D(8, decoder_out_dim, norm_layers[3], apply_blur=apply_blur),
            nn.ReLU(inplace=True)
        )


    def freeze_resnet(self):
        for param in self.parameters():
            param.requires_grad = True

        for param in self.feature_extractor.parameters():
            param.requires_grad = False


    def videoproject_singlescale(self, x):
        return self.e2p.project_clip(x)
