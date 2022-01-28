import logging
import os
import pickle

import coloredlogs
import torch
from einops import rearrange
from torch import nn

from common.losses import create_loss
from models.feature_transformer import FeatureTransformer
from models.networks import make_network, get_output_channels

MIN_NUM_PATCHES = 0

coloredlogs.install()


class CLIMAT(nn.Module):
    def __init__(self, cfg, device, pn_weights=None, y0_weights=None):
        super().__init__()

        self.device = device
        self.cfg = cfg
        self.n_meta_out_features = 0
        self.n_meta_features = cfg.n_meta_features
        self.input_data = cfg.parser.metadata
        self.pn_weights = torch.tensor(pn_weights) if pn_weights is not None else pn_weights

        if "AGE" in self.input_data:
            self.age_ft = self.create_metadata_layers(4, self.n_meta_features)
            self.n_meta_out_features += self.n_meta_features

        if "WOMAC" in self.input_data:
            self.womac_ft = self.create_metadata_layers(4, self.n_meta_features)
            self.n_meta_out_features += self.n_meta_features

        if "BMI" in self.input_data:
            self.bmi_ft = self.create_metadata_layers(4, self.n_meta_features)
            self.n_meta_out_features += self.n_meta_features

        if "SEX" in self.input_data:
            self.sex_ft = self.create_metadata_layers(2, self.n_meta_features)
            self.n_meta_out_features += self.n_meta_features

        if "INJ" in self.input_data:
            self.inj_ft = self.create_metadata_layers(2, self.n_meta_features)
            self.n_meta_out_features += self.n_meta_features

        if "SURG" in self.input_data:
            self.surg_ft = self.create_metadata_layers(2, self.n_meta_features)
            self.n_meta_out_features += self.n_meta_features

        if "KL" in self.input_data:
            self.kl_ft = self.create_metadata_layers(cfg.n_pn_classes, self.n_meta_features)
            self.n_meta_out_features += self.n_meta_features

        self.n_all_features = self.n_meta_out_features
        if "IMG" in self.input_data:
            if cfg.max_depth < 1 or cfg.max_depth > 5:
                logging.fatal('Max depth must be in [1, 5].')
                assert False

            self.n_input_imgs = cfg.n_input_imgs

            self.feature_extractor = make_network(name=cfg.backbone_name, pretrained=cfg.pretrained,
                                                  input_3x3=cfg.input_3x3)

            self.blocks = []
            for i in range(cfg.max_depth):
                if hasattr(self.feature_extractor, f"layer{i}"):
                    self.blocks.append(getattr(self.feature_extractor, f"layer{i}"))
                elif i == 0 and 'resnet' in cfg.backbone_name:
                    self.blocks.append([self.feature_extractor.conv1, self.feature_extractor.bn1,
                                        self.feature_extractor.relu, self.feature_extractor.maxpool])

            if self.cfg.dataset == "toy":
                self.sz_list = [32, 16, 8]  # 32x32
                self.n_last_img_features = 64
                self.n_last_img_ft_size = 8
            elif self.cfg.dataset == "mnist3x3":
                self.sz_list = [64, 32, 16, 8, 4]  # 128x128
                self.n_last_img_features = get_output_channels(self.blocks[-1], cfg.max_depth)
                self.n_last_img_ft_size = self.sz_list[cfg.max_depth - 1]
            else:
                # self.sz_list = [75, 75, 38, 19, 10] # 300x300
                self.sz_list = [64, 64, 32, 16, 8]  # 256x256
                self.n_last_img_features = get_output_channels(self.blocks[-1], cfg.max_depth)
                self.n_last_img_ft_size = self.sz_list[cfg.max_depth - 1]

            self.n_img_features = cfg.n_img_features

            if self.n_img_features <= 0:
                self.img_ft_projection = nn.Identity()
            else:
                self.img_ft_projection = nn.Linear(self.n_last_img_features, cfg.n_img_features, bias=True)
                self.n_last_img_features = cfg.n_img_features

            self.n_all_features += self.n_last_img_features
            logging.info(f'[INFO] Num of blocks: {len(self.blocks)}')

            self.n_patches = self.n_last_img_ft_size * self.n_last_img_ft_size
        else:
            self.n_patches = self.cfg.seq_len

        self.dropout = nn.Dropout(p=cfg.drop_rate)
        self.dropout_between = nn.Dropout(cfg.drop_rate_between)

        self.n_classes = cfg.n_pn_classes

        self.feat_dim = cfg.feat_dim = cfg.feat_dim if isinstance(cfg.feat_dim, int) and cfg.feat_dim > 0 \
            else self.n_meta_features + self.n_last_img_features

        if hasattr(cfg, "num_cls_num"):
            self.num_cls_num = cfg.num_cls_num
        else:
            self.num_cls_num = cfg.seq_len

        self.feat_kl_dim = cfg.feat_kl_dim = cfg.feat_kl_dim if isinstance(cfg.feat_kl_dim,
                                                                           int) and cfg.feat_kl_dim > 0 else self.n_last_img_features

        # Fusion
        self.n_metadata = len(self.input_data) - 1  # Remove current KL (y_0)
        self.feat_fusion = FeatureTransformer(num_patches=self.n_metadata, with_cls=True, num_cls_num=1,
                                              patch_dim=self.n_meta_features,
                                              num_classes=0, dim=self.n_meta_features, depth=cfg.feat_fusion_depth,
                                              heads=cfg.feat_fusion_heads, mlp_dim=cfg.feat_fusion_mlp_dim,
                                              dropout=cfg.drop_rate,
                                              emb_dropout=cfg.feat_fusion_emb_drop_rate, n_outputs=0)

        self.feat_kl = FeatureTransformer(num_patches=self.n_patches, with_cls=True, num_cls_num=1,
                                          patch_dim=self.feat_kl_dim,
                                          num_classes=cfg.n_pn_classes, dim=self.feat_kl_dim, depth=cfg.feat_kl_depth,
                                          heads=cfg.feat_kl_heads, mlp_dim=cfg.feat_kl_mlp_dim, dropout=cfg.drop_rate,
                                          emb_dropout=cfg.feat_kl_emb_drop_rate, n_outputs=cfg.feat_kl_n_outputs)

        self.feat_prognosis = FeatureTransformer(num_patches=self.n_patches + 1, with_cls=True,
                                                 num_cls_num=self.num_cls_num,
                                                 patch_dim=self.feat_dim,
                                                 num_classes=self.n_classes, dim=self.feat_dim, depth=cfg.feat_depth,
                                                 heads=cfg.feat_heads, mlp_dim=cfg.feat_mlp_dim, dropout=cfg.drop_rate,
                                                 emb_dropout=cfg.feat_emb_drop_rate, n_outputs=cfg.feat_n_outputs)

        self.use_tensorboard = False
        if self.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter()

        self.pn_weights = torch.tensor(pn_weights) if pn_weights is not None and cfg.prognosis_coef > 0 else None
        self.y0_weights = torch.tensor(y0_weights) if y0_weights is not None and cfg.kl_coef > 0 else None
        self.configure_loss_coefs(cfg)
        self.configure_crits()
        self.configure_optimizers()
        self.batch_ind = 0
        self.to(self.device)

    def configure_loss_coefs(self, cfg):
        # alpha
        self.y0_init_power = cfg.y0_init_power
        self.pn_init_power = cfg.pn_init_power

        if cfg.kl_coef > 0:
            self.alpha_power_y0 = torch.tensor(self.y0_init_power, dtype=torch.float32)
        if cfg.prognosis_coef > 0:
            self.alpha_power_pn = torch.tensor([self.pn_init_power] * cfg.seq_len, dtype=torch.float32)

        # Show class weights
        if self.y0_weights is not None and self.y0_init_power is not None:
            _y0_weights = self.y0_weights ** self.y0_init_power
        else:
            _y0_weights = None
        if self.pn_weights is not None and self.pn_init_power is not None:
            _pn_weights = self.pn_weights ** self.pn_init_power
        else:
            _pn_weights = None

        print(f'{cfg.grading} weights:\n{_y0_weights}')
        print(f'PN weights:\n{_pn_weights}')

    def configure_crits(self):
        self.crit_pn = create_loss(loss_name=self.cfg.loss_name,
                                   normalized=False,
                                   reduction='mean').to(self.device)
        self.crit_kl = create_loss(loss_name=self.cfg.loss_name,
                                   normalized=False,
                                   reduction='mean').to(self.device)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg['lr'],
                                          betas=(self.cfg['beta1'], self.cfg['beta2']))

    def has_y0(self):
        return self.cfg.kl_coef > 0

    def has_pn(self):
        return self.cfg.prognosis_coef > 0

    def _to_numpy(self, x):
        return x.to('cpu').detach().numpy()

    def _compute_probs(self, x, tau=1.0, dim=-1, to_numpy=True):
        tau = tau if tau is not None else 1.0

        probs = torch.softmax(x * tau, dim=dim)

        if to_numpy:
            probs = self._to_numpy(probs)
        return probs

    def forward(self, input, batch_i=None, target=None):
        meta_features = []
        img_features = None
        for input_type in self.input_data:
            if input_type.lower() == "img":
                img_features = self.forward_img(input[input_type.upper()])
            else:
                _ft = getattr(self, f"{input_type.lower()}_ft")(input[input_type.upper()])
                _ft = torch.unsqueeze(_ft, 1)
                _ft = self.dropout_between(_ft)
                meta_features.append(_ft)

        preds, kl_preds, d_attn, f_attn, p_attn = self.predict_prognosis(img_features, meta_features)

        # Save image, metadata names, and attention map into pkl files
        if self.cfg.save_attn:
            self.save_attentions(batch_i, root=self.cfg.log_dir, img=input['IMG'] if "IMG" in self.input_data else None,
                                 metadata_names=self.input_data, diags=kl_preds,
                                 d_attn=d_attn, f_attn=f_attn, p_attn=p_attn, preds=preds, targets=target)

        return preds, kl_preds

    def save_attentions(self, batch_i, root, img, metadata_names, d_attn, f_attn, p_attn, preds, targets, diags):
        data = {'img': img.to('cpu').detach().numpy(),
                'metadata': metadata_names,
                'D': d_attn.to('cpu').detach().numpy(),
                'F': f_attn.to('cpu').detach().numpy(),
                'P': p_attn.to('cpu').detach().numpy(),
                'preds': preds.to('cpu').detach().numpy(),
                'targets': targets['prognosis'].to('cpu').detach().numpy(),
                'mask': targets['prognosis_mask'].to('cpu').detach().numpy(),
                'diags': diags.to('cpu').detach().numpy(),
                'y0': targets[f'current_{self.cfg.grading}'].to('cpu').detach().numpy()}

        os.makedirs(os.path.join(root, "attn"), exist_ok=True)
        attn_fullname = os.path.join(root, "attn", f"batch_{self.cfg.site}_{batch_i}.pkl")
        print(attn_fullname)
        with open(attn_fullname, 'wb') as f:
            pickle.dump(data, f, 4)

    def predict_prognosis(self, img_features, meta_features):
        has_img = img_features is not None
        has_meta = meta_features != []

        if has_img:
            img_features = rearrange(img_features, 'b c h w -> b (h w) c')

        kl_preds, img_descs, d_attns = self.feat_kl(img_features)
        kl_preds = kl_preds.squeeze(1)

        if has_meta:
            meta_features = torch.cat(meta_features, 1)
            # Apply Fusion transformer
            _, fusion_features, f_attns = self.feat_fusion(meta_features)
            meta_features = fusion_features[:, 0:1, :]

            meta_features = meta_features.repeat(1, img_descs.shape[1], 1)
            meta_features = self.dropout(meta_features)
            meta_features = torch.cat((img_descs, meta_features), dim=-1)
        else:
            meta_features = img_descs
            f_attns = [None]

        preds, _, p_attns = self.feat_prognosis(meta_features)

        return preds, kl_preds, d_attns[-1], f_attns[-1], p_attns[-1]

    def create_metadata_layers(self, n_input_dim, n_output_dim):
        return nn.Sequential(
            nn.Linear(n_input_dim, n_output_dim, bias=True),
            nn.ReLU(),
            nn.LayerNorm(n_output_dim)
        )

    def forward_img(self, input):
        features = []
        if isinstance(input, torch.Tensor):
            input = (input,)

        for x in input:
            for block in self.blocks:
                if isinstance(block, list) or isinstance(block, tuple):
                    for sub_block in block:
                        x = sub_block(x)
                else:
                    x = block(x)
                x = self.dropout_between(x)

            if self.cfg.feat_use:
                x = x.permute(0, 2, 3, 1)
                img_ft = self.img_ft_projection(x)
                img_ft = self.dropout(img_ft)
                img_ft = img_ft.permute(0, 3, 1, 2)
            else:
                img_ft = self.gap(x)
                img_ft = img_ft.squeeze(-1).squeeze(-1)
            features.append(img_ft)

        features = torch.cat(features, 1)
        return features

    def fit(self, input, target, batch_i, n_iters, epoch_i, stage="train"):
        grading_mask = target[f'current_{self.cfg.grading}_mask']

        pn_target = target['prognosis']
        pn_masks = target['prognosis_mask']

        preds, kl_preds = self.forward(input, batch_i, target)

        outputs = {'pn': {'prob': [], 'label': []}}

        # Current KL
        if self.has_y0():
            outputs[self.cfg.grading] = {'prob': None, 'label': None}

            grading_preds_mask = kl_preds[grading_mask, :]
            grading_target_mask = target[f'current_{self.cfg.grading}'][grading_mask]

            outputs[self.cfg.grading]['prob'] = self._compute_probs(grading_preds_mask, to_numpy=True)
            outputs[self.cfg.grading]['label'] = self._to_numpy(grading_target_mask)
            if grading_target_mask.nelement() > 0:
                if self.y0_weights is None or self.alpha_power_y0 is None:
                    y0_pw_weights = None
                else:
                    self.y0_weights = self.y0_weights.to(self.alpha_power_y0.device)
                    y0_pw_weights = self.y0_weights ** self.alpha_power_y0

                cur_kl_loss = self.crit_kl(grading_preds_mask, grading_target_mask, alpha=y0_pw_weights)
            else:
                cur_kl_loss = torch.tensor(0.0, requires_grad=True)

        T_max = self.cfg.seq_len

        pn_losses = torch.zeros((T_max), device=self.device)
        n_t_pn = 0

        if self.pn_weights is not None:
            self.pn_weights = self.pn_weights.to(self.alpha_power_pn.device)

        for t in range(T_max):
            pn_logits_mask = preds[pn_masks[:, t], t, :]
            pn_target_mask = pn_target[pn_masks[:, t], t]

            outputs['pn']['prob'].append(
                self._compute_probs(pn_logits_mask, to_numpy=True))

            outputs['pn']['label'].append(self._to_numpy(pn_target_mask))

            if self.has_pn() and pn_logits_mask.shape[0] > 0 and pn_target_mask.shape[0] > 0:
                pn_pw_weights = self.pn_weights[t, :] ** self.alpha_power_pn[t] if self.pn_weights is not None else None
                pn_loss = self.crit_pn(pn_logits_mask, pn_target_mask, normalized=False, alpha=pn_pw_weights)

                pn_losses[t] = pn_loss
                n_t_pn += 1

        if n_t_pn > 0:
            prognosis_loss = pn_losses.sum() / n_t_pn
        else:
            prognosis_loss = torch.tensor(0.0, requires_grad=True)

        losses = {}

        loss = torch.tensor(0.0, requires_grad=True)
        if self.has_y0() and self.cfg.kl_coef > 0:
            loss = loss + self.cfg.kl_coef * cur_kl_loss
            losses['loss_y0'] = cur_kl_loss.item()
        else:
            losses['loss_y0'] = 0.0
        if self.cfg.prognosis_coef > 0 and n_t_pn > 0:
            loss = loss + self.cfg.prognosis_coef * prognosis_loss
            losses['loss_pn'] = prognosis_loss.item()
        else:
            losses['loss_pn'] = 0.0

        losses['loss'] = loss.item()

        if stage == "train":
            with torch.autograd.set_detect_anomaly(True):
                self.optimizer.zero_grad()
                loss.backward()
                if self.cfg.clip_norm > 0:
                    nn.utils.clip_grad_norm_(self.params_main, self.cfg.clip_norm)
                self.optimizer.step()

        return losses, outputs
