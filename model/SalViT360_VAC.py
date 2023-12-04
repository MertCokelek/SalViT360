from model.utils.projection import Equi2Pers, Pers2Equi
from model.blocks import Posemb_LW, UpsampleBlock3D
from model.VSTA import SphericalVideoTransformer
from utils.saliency_losses import VACLoss

from tqdm import tqdm
from dotmap import DotMap
from torch import nn
from torchvision.models import resnet18
from einops import rearrange

import torch
import torch.nn.functional as F
import torchvision.transforms as tf

SPH_EMB_SIZE = 56


class SalViT360_VAC(nn.Module):
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

        nrows = tangent_config.nrows
        patch_sizes = tangent_config.patch_size.e2p
        fovs = tangent_config.fov
        self.n_patches = tangent_config.npatches

        self.E2P, self.P2E, self.sph_coords = [], [], []
        tangent_configs = [
            {
                'nrows': nrows[i],
                'patch_size': patch_sizes[i],
                'fov': fovs[i]
            }
            for i in range(len(nrows))
        ]
        shift = [False, True]

        for tangent_conf in tangent_configs:
            shift_ = shift.pop(0)
            # print('Using shift:', shift_)
            self.E2P.append(Equi2Pers(erp_size=erp_size, shift=shift_, **tangent_conf))
            self.P2E.append(Pers2Equi(erp_size=(240, 480), fov=tangent_conf['fov'], nrows=tangent_conf['nrows'],
                                      shift=shift_, patch_size=(28, 28)))
            self.sph_coords.append(
                F.adaptive_avg_pool2d(
                    Equi2Pers(erp_size, fov=tangent_conf['fov'], nrows=tangent_conf['nrows'], shift=shift_,
                              patch_size=SPH_EMB_SIZE).get_spherical_embeddings(),
                    (14, 14))
            )

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

        self.VACLoss = VACLoss()

    def forward(self, x, return_p0=False):
        preds = []

        for scale in range(len(self.E2P)):
            with torch.no_grad():
                x_tang = self.E2P[scale].project_clip(x)  # BS, F=8, C=3, H=224, W=224, T=18
                # if scale == 1:  # Don't train the second scale
                #     x_tang = x_tang.detach()
                BS, F, _, _, _, _ = x_tang.shape

                # Encoder
                x_tang = rearrange(x_tang, 'b f c h w t -> (b f t) c h w', b=BS, f=F)
                resnet_features = self.feature_extractor(x_tang)  # (b f t) c h w
                features = self.down(resnet_features)

                features = rearrange(features, '(b f t) c h w -> b f t (c h w)', b=BS, f=F)

                # Spherical Position Embedding
                sph_coords = self.sph_coords[scale].to(x_tang.device)  # T, 5, 14, 14
                sph_emb = rearrange(sph_coords, 't c h w -> t (c h w)')  # T, 5*14*14=980

            # Transformer
            features = self.transformer(features, sph_emb)  # BS, F, T, D=512
            features = rearrange(features[:, -1], 'b t d -> b d 1 1 t')
            resnet_features = self.resnet_layernorm(resnet_features)
            resnet_features = rearrange(resnet_features, '(b f t) c h w -> b f c h w t', b=BS, f=F)[:, -1]

            decoder_in = features + resnet_features 
            decoder_out = self.decoders[scale](decoder_in)[:, :1]  
            # decoder_out = self.decoder(decoder_in)[:, :1]  # You may use a single decoder too. 
            decoder_out = self.P2E[scale](decoder_out)

            preds.append(decoder_out)

        p0 = nn.functional.interpolate(preds[0], size=(480, 960), mode='bilinear', align_corners=False)
        p1 = nn.functional.interpolate(preds[1], size=(480, 960), mode='bilinear', align_corners=False)

        vac_loss = self.VACLoss(p0, p1.detach(), overlap_mask=torch.ones([1, 480, 960]).cuda())

        if return_p0:  # Use only one stream for backprop 
            return p0, vac_loss
        
        pred = 0.5 * (p0 + p1)  # Backpropagate two streams
        return pred, vac_loss
    

    @torch.no_grad()
    @torch.inference_mode()
    def _inference_with_LF(self, x):
        # Use both tangent scales and apply Late Fusion
        preds = []
        for scale in range(len(self.E2P)):
            x_tang = self.E2P[scale].project_clip(x)  # BS, F=8, C=3, H=224, W=224, T=18
            BS, F, _, _, _, _ = x_tang.shape

            # Encoder
            x_tang = rearrange(x_tang, 'b f c h w t -> (b f t) c h w', b=BS, f=F)
            resnet_features = self.feature_extractor(x_tang)  # (b f t) c h w
            features = self.down(resnet_features)

            features = rearrange(features, '(b f t) c h w -> b f t (c h w)', b=BS, f=F)

            # Spherical Position Embedding
            sph_coords = self.sph_coords[scale].to(x_tang.device)  # T, 5, 14, 14
            sph_emb = rearrange(sph_coords, 't c h w -> t (c h w)')  # T, 5*14*14=980

            # Transformer
            features = self.transformer(features, sph_emb)  # BS, F, T, D=512
            features = rearrange(features[:, -1], 'b t d -> b d 1 1 t')
            resnet_features = self.resnet_layernorm(resnet_features)
            resnet_features = rearrange(resnet_features, '(b f t) c h w -> b f c h w t', b=BS, f=F)[:, -1]

            decoder_in = features + resnet_features  # Should we do a division by 2 here?  # features: 0-mean, 1-std. resnet: 0 min, 30 max
            decoder_out = self.decoders[scale](decoder_in)[:, :1]  # Was commented on 4Dec
            decoder_out = self.P2E[scale](decoder_out)

            preds.append(decoder_out)

        p0 = nn.functional.interpolate(preds[0], size=(480, 960), mode='bilinear', align_corners=False)
        p1 = nn.functional.interpolate(preds[1], size=(480, 960), mode='bilinear', align_corners=False)

        pred = p0 * p1 
        return pred
    

    def _inference_single_scale(self, x):
        # Use only first tangent scale

        scale = 0
        x_tang = self.E2P[scale].project_clip(x)  # BS, F=8, C=3, H=224, W=224, T=18
        BS, F, _, _, _, _ = x_tang.shape

        # Encoder
        x_tang = rearrange(x_tang, 'b f c h w t -> (b f t) c h w', b=BS, f=F)
        resnet_features = self.feature_extractor(x_tang)  # (b f t) c h w
        features = self.down(resnet_features)

        features = rearrange(features, '(b f t) c h w -> b f t (c h w)', b=BS, f=F)

        # Spherical Position Embedding
        sph_coords = self.sph_coords[scale].to(x_tang.device)  # T, 5, 14, 14
        sph_emb = rearrange(sph_coords, 't c h w -> t (c h w)')  # T, 5*14*14=980

        # Transformer
        features = self.transformer(features, sph_emb)  # BS, F, T, D=512
        features = rearrange(features[:, -1], 'b t d -> b d 1 1 t')
        resnet_features = self.resnet_layernorm(resnet_features)
        resnet_features = rearrange(resnet_features, '(b f t) c h w -> b f c h w t', b=BS, f=F)[:, -1]

        decoder_in = features + resnet_features  # Should we do a division by 2 here?  # features: 0-mean, 1-std. resnet: 0 min, 30 max
        decoder_out = self.decoders[scale](decoder_in)[:, :1]  # Was commented on 4Dec
        decoder_out = self.P2E[scale](decoder_out)

        pred = nn.functional.interpolate(decoder_out[0], size=(480, 960), mode='bilinear', align_corners=False)
        return pred


    @torch.no_grad()
    @torch.inference_mode()
    def inference(self, video, late_fusion=False):
        """
        video (list): [F, C, H, W]
        """
        transform = tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        n_frames = len(video)

        preds = []
        
        for st in tqdm(range(n_frames - 8)):
            clip = video[st:st + 8]
            clip = transform(torch.stack(clip).float() / 255.).unsqueeze(0)
            if late_fusion:
                pred = self._inference_with_LF(clip.cuda())
            else:
                pred, _ = self.forward(clip.cuda())
            pred = pred.squeeze(1).detach().cpu().numpy()
            preds.append(pred)
        
        return preds # [F, 1, 480, 960]


    def set_tangent_decoder(self, config):
        """
        Full tangent decoder.
        Its architecture is simplified (channel dims and n_blocks) V1-Decoder (with optional Blur and LayerNorm)
        input: Tangent Feature Maps [B, C=512, H=7, W=7, T]
        """
        apply_blur = config.network.decoder.apply_blur

        decoder_out_dim = 2
        self.decoders = nn.ModuleList()

        for decoder_id in range(2):
            t = self.n_patches[decoder_id]
            norm_layers = [
                nn.LayerNorm([128, 7, 7, t]),
                nn.LayerNorm([32, 7, 7, t]),
                nn.LayerNorm([decoder_out_dim, 14, 14, t])
            ]

            self.decoders.append(nn.Sequential(
                UpsampleBlock3D(512, 128, norm_layers[0], apply_blur=apply_blur, upsample=False),
                UpsampleBlock3D(128, 32, norm_layers[1], apply_blur=apply_blur),
                UpsampleBlock3D(32, decoder_out_dim, norm_layers[2], apply_blur=apply_blur),
                nn.ReLU(inplace=True))
            )


    def freeze_resnet(self):
        for param in self.parameters():
            param.requires_grad = True

        for param in self.feature_extractor.parameters():
            param.requires_grad = False


    def videoproject_singlescale(self, x):
        return self.e2p.project_clip(x)
