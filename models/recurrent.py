import logging
import random
from models.networks import make_network, get_output_channels
import torch
import torch.nn as nn
from torch.optim import Adam

from common.losses import create_loss


class BiRecurrent_Model(nn.Module):
    def __init__(self, cfg, device, pn_weights=None):
        super().__init__()

        self.cfg = cfg
        self.gp_list = [cfg.gp.l0, cfg.gp.l1, cfg.gp.l2, cfg.gp.l3, cfg.gp.l4]
        self.input_data = cfg.parser.metadata
        self.pn_weights = torch.tensor(pn_weights) if pn_weights is not None else pn_weights
        # Image 0
        self.metadata_ft = {}
        self.n_meta_out_features = 0
        self.n_meta_features = cfg.n_meta_features
        self.clip_norm = cfg.clip_norm
        self.drop = nn.Dropout(cfg.drop_rate)
        self.dropout_between = nn.Dropout(cfg.drop_rate_between)

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

            self.gap = nn.AdaptiveAvgPool2d(1)

            if self.n_img_features > 0:
                self.img_ft_projection = nn.Linear(self.n_last_img_features, cfg.n_img_features, bias=True)
                self.n_last_img_features = cfg.n_img_features

            self.n_all_features += self.n_last_img_features
            logging.info(f'[INFO] Num of blocks: {len(self.blocks)}')

            # self.meta_ft_adjustment = nn.Parameter(torch.ones(1, 1, self.n_meta_out_features))
            # self.n_img_patches = self.n_last_img_ft_size * self.n_last_img_ft_size
        else:
            # self.meta_ft_adjustment = nn.Parameter(torch.ones(1, 1, self.n_all_features))
            self.n_img_patches = 0

        # Seq2Seq
        # self.encoder = self.rnn(cfg.n_all_features, cfg.rnn_dim, num_layers=2,
        #                         bias=True, batch_first=True, dropout=cfg.drop_rate, bidirectional=True)
        #
        self.dec_embedding = nn.Sequential(
            nn.Linear(1, cfg.embedding_dim, bias=True),
            nn.ReLU()
        )

        if cfg.rnn_layer == 'gru':
            self.encoder = nn.GRU(self.n_all_features, cfg.rnn_dim, num_layers=2, bias=True, batch_first=True,
                                    dropout=cfg.drop_rate, bidirectional=True)
            self.decoder = nn.GRU(self.n_all_features, cfg.rnn_dim, num_layers=2, bias=True, batch_first=True,
                                  dropout=cfg.drop_rate, bidirectional=True)
        elif cfg.rnn_layer == 'lstm':
            self.encoder = nn.LSTM(self.n_all_features, cfg.rnn_dim, num_layers=2, bias=True, batch_first=True,
                                  dropout=cfg.drop_rate, bidirectional=True)
            self.decoder = nn.LSTM(self.n_all_features, cfg.rnn_dim, num_layers=2, bias=True, batch_first=True,
                                   dropout=cfg.drop_rate, bidirectional=True)
        else:
            raise ValueError(f'Not support {cfg.rnn_layer}.')

        self.predict = nn.Linear(cfg.rnn_dim * 2, cfg.n_pn_classes, bias=True)

        # Optimizer
        self.optimizer = Adam(self.parameters(), lr=cfg['lr'], betas=(cfg['beta1'], cfg['beta2']))

        self.device = device
        self.configure_loss_coefs(cfg)
        self.configure_crits()
        self.to(device)

    def configure_crits(self):
        self.crit_pn = create_loss(loss_name=self.cfg.loss_name,
                                   normalized=False,
                                   reduction='mean').to(self.device)

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
        if self.n_img_features > 0:
            features = self.img_ft_projection(features)
        return features

    def _to_numpy(self, x):
        return x.to('cpu').detach().numpy()

    def _compute_probs(self, x, dim=-1, to_numpy=True):
        probs = torch.softmax(x, dim=dim)
        if to_numpy:
            probs = self._to_numpy(probs)
        return probs

    def forward(self, input, target):
        meta_ft = []
        img_features = None
        for input_type in self.input_data:
            if input_type.lower() == "img":
                img_features = self.forward_img(input[input_type.upper()])
            else:
                _ft = getattr(self, f"{input_type.lower()}_ft")(input[input_type.upper()])
                meta_ft.append(_ft)

        if img_features is not None:
            meta_ft.append(img_features)
        meta_ft = torch.cat(meta_ft, dim=-1)

        meta_ft = meta_ft.unsqueeze(1).repeat(1, self.cfg.seq_len, 1)

        if self.cfg.rnn_layer == 'lstm':
            enc_out, (enc_h, enc_cell) = self.encoder(meta_ft)
            dec_h = enc_h
            dec_cell = enc_cell
        elif self.cfg.rnn_layer == 'gru':
            enc_out, enc_h = self.encoder(meta_ft)
            dec_h = enc_h

        batch_size = meta_ft.shape[0]

        dec_input = -1.0 * torch.ones((batch_size, 1), dtype=torch.float32).to(self.device)  # target[:, 0]

        pn_logits_out = []

        for t in range(self.cfg.seq_len):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            dec_input = dec_input.unsqueeze(1)
            dec_input = self.dec_embedding(dec_input)

            if self.cfg.rnn_layer == 'lstm':
                dec_out, (dec_h, dec_cell) = self.decoder(dec_input, (dec_h, dec_cell))
            elif self.cfg.rnn_layer == 'gru':
                dec_out, dec_h = self.decoder(dec_input, dec_h)

            pn_logits = self.predict(dec_out).flatten(1)

            # place predictions in a tensor holding predictions for each token
            pn_logits_out.append(pn_logits)

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < self.cfg.teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = pn_logits.argmax(1, keepdim=True).float()
            # top1 = (torch.sigmoid(dec_out) > 0.5).float()

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            dec_input = target[:, t] if teacher_force else top1

        pn_logits_out = torch.stack(pn_logits_out, 1)
        return pn_logits_out

    def fit(self, input, target, stage="train", *args, **kwarg):
        pn_target = target['prognosis']
        pn_masks = target['prognosis_mask']

        pn_logits = self.forward(input, pn_target)

        outputs = {'pn': {'prob': [], 'label': []}}

        T_max = self.cfg.seq_len
        rec_losses = torch.zeros((T_max), device=pn_target.device)
        n_t_pn = 0
        for t in range(T_max):
            # Apply mask
            logits_masked = pn_logits[pn_masks[:, t], t, :]
            labels_masked = pn_target[pn_masks[:, t], t, :]
            outputs['pn']['prob'].append(self._compute_probs(logits_masked))
            outputs['pn']['label'].append(self._to_numpy(labels_masked))
            if logits_masked.shape[0] > 0 and labels_masked.shape[0] > 0:
                rec_loss = self.crit_pn(logits_masked, labels_masked)
                rec_losses[t] = rec_loss
                n_t_pn += 1

        losses = {'loss_y0': -1}
        if n_t_pn > 0:
            loss = rec_losses.sum() / n_t_pn
            losses['loss_pn'] = loss.item()
            losses['loss'] = loss.item()
            if stage == "train":
                self.optimizer.zero_grad()
                loss.backward()
                if self.clip_norm > 0:
                    nn.utils.clip_grad_norm_(self.parameters(), self.clip_norm)
                self.optimizer.step()
        else:
            losses['loss_pn'] = 0.0
            losses['loss'] = 0.0

        return losses, outputs
