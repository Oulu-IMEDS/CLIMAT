import logging

import coloredlogs
import torch
from torch import nn

from common.losses import create_loss
from models.networks import make_network, get_output_channels

coloredlogs.install()


class FCN(nn.Module):
    def __init__(self, cfg, device, pn_weights=None):
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

            self.sz_list = [64, 64, 32, 16, 8]  # 256x256
            self.n_last_img_features = get_output_channels(self.blocks[-1], cfg.max_depth)
            self.n_last_img_ft_size = self.sz_list[cfg.max_depth - 1]

            self.n_img_features = cfg.n_img_features

        if self.n_img_features <= 0:
            self.img_ft_projection = nn.Linear(self.n_last_img_features, cfg.n_meta_features, bias=True)
            self.n_last_img_features = cfg.n_meta_features
        else:
            self.img_ft_projection = nn.Linear(self.n_last_img_features, cfg.n_img_features, bias=True)
            self.n_last_img_features = cfg.n_img_features

        self.n_all_features += self.n_last_img_features
        logging.info(f'[INFO] Num of blocks: {len(self.blocks)}')

        self.n_img_patches = self.n_last_img_ft_size * self.n_last_img_ft_size

        self.dropout = nn.Dropout(p=cfg.drop_rate)
        self.dropout_between = nn.Dropout(cfg.drop_rate_between)

        self.n_classes = cfg.n_pr_classes + cfg.n_pn_classes

        self.mlp_dim = cfg.mlp_dim = cfg.mlp_dim if isinstance(cfg.mlp_dim, int) and cfg.mlp_dim > 0 \
            else self.n_meta_features

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc_progs = nn.Sequential(
            nn.Linear(self.n_all_features, self.mlp_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.mlp_dim, self.mlp_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.mlp_dim, self.n_classes, bias=True),
        )

        self.pn_weights = torch.tensor(pn_weights) if pn_weights is not None and cfg.prognosis_coef > 0 else None
        self.configure_loss_coefs(cfg)
        self.configure_crits()
        self.configure_optimizers()
        self.batch_ind = 0
        self.to(self.device)

    def configure_loss_coefs(self, cfg):
        # alpha
        self.pn_init_power = cfg.pn_init_power

        if cfg.prognosis_coef > 0:
            self.alpha_power_pn = torch.tensor([self.pn_init_power] * cfg.seq_len, dtype=torch.float32)

        # Show class weights
        if self.pn_weights is not None and self.pn_init_power is not None:
            _pn_weights = self.pn_weights ** self.pn_init_power
        else:
            _pn_weights = None

        print(f'PN weights:\n{_pn_weights}')

    def configure_crits(self):
        self.crit_pn = create_loss(loss_name=self.cfg.loss_name,
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

    def forward(self, input, batch_i=None):
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

        preds = self.predict_prognosis(img_features, meta_features)

        return preds

    def predict_prognosis(self, img_features, meta_features):
        has_img = img_features is not None
        has_meta = meta_features != []

        if has_img or has_meta:
            if has_meta:
                meta_features = torch.cat(meta_features, -1)
                meta_features = self.dropout(meta_features)
            if has_img:
                img_features = self.img_ft_projection(img_features)
                img_features = torch.unsqueeze(img_features, 1)
                if has_meta:
                    meta_features = torch.cat((img_features, meta_features), -1)
                else:
                    meta_features = img_features
        else:
            raise ValueError(f'Not found image and features.')

        meta_features_seq = meta_features.repeat(1, self.cfg.seq_len, 1)
        preds = self.fc_progs(meta_features_seq)
        return preds

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

            img_ft = self.gap(x)
            img_ft = img_ft.squeeze(-1).squeeze(-1)
            features.append(img_ft)

        features = torch.cat(features, 1)
        return features

    def fit(self, input, target, batch_i, n_iters, epoch_i, stage="train"):
        pn_target = target['prognosis']
        pn_masks = target['prognosis_mask']

        preds = self.forward(input, batch_i)

        outputs = {'pn': {'prob': [], 'label': []}}

        T_max = self.cfg.seq_len

        pn_losses = torch.zeros((T_max), device=self.device)
        n_t_pn = 0

        if self.pn_weights is not None:
            self.pn_weights = self.pn_weights.to(self.alpha_power_pn.device)

        for t in range(T_max):
            pn_logits_mask = preds[pn_masks[:, t], t, self.cfg.n_pr_classes:]
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

        losses = {'loss_y0': None}

        loss = torch.tensor(0.0, requires_grad=True)

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
