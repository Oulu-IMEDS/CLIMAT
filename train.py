import random
import logging as log
import coloredlogs
import hydra
import numpy as np
import torch
import yaml
import os
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, \
    mean_squared_error, cohen_kappa_score
from tqdm import tqdm
from common.data import ItemLoader
from common.utils import proc_targets, calculate_class_weights, calculate_metric, load_metadata, init_mean_std, \
    parse_item_progs, store_model, update_max_grades, parse_img, init_transforms
from models import create_model

coloredlogs.install()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

task_names = ('grading', 'pn', 'all')
task2metrics = {'grading': ['ba', 'ka', 'mauc', 'ba.ka'],
                'pn': ['ba', 'mse', 'mauc', 'loss'],
                'all': ['loss']}

stored_models = {}
for task in task_names:
    stored_models[task] = {}
    for _name in task2metrics[task]:
        if _name == "mse" or "loss" in _name:
            stored_models[task][_name] = {'best': 1000000.0, "filename": ""}
        else:
            stored_models[task][_name] = {'best': -1, "filename": ""}


@hydra.main(config_path="configs", config_name="config_train")
def main(cfg):
    if int(cfg.seed) < 0:
        cfg.seed = random.randint(0, 1000000)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    update_max_grades(cfg)

    wdir = os.environ['PWD']
    if not os.path.isabs(cfg.meta_root):
        cfg.meta_root = os.path.join(wdir, cfg.meta_root)

    if not os.path.isdir(cfg.snapshots):
        os.makedirs(cfg.snapshots, exist_ok=True)

    print(cfg.pretty())

    with open("args.yaml", "w") as f:
        yaml.dump(OmegaConf.to_container(cfg), f, default_flow_style=False)

    # Load and split data
    oai_site_folds, oai_meta, oai_meta_test, most_meta = load_metadata(cfg, proc_targets=proc_targets, eval_only=False)

    # Compute mean and std of OAI
    oai_mean, oai_std = init_mean_std(cfg, wdir, oai_meta, parse_img)
    print(f'Mean: {oai_mean}\nStd: {oai_std}')

    y0_weights, pn_weights, pr_weights = calculate_class_weights(oai_meta, cfg)

    oai_meta.describe()

    df_train, df_val = oai_site_folds[cfg.fold_index - 1]
    df_train = df_train[df_train['visit_id'] == 0]
    df_val = df_val[df_val['visit_id'] == 0]

    loaders = dict()

    for stage, df in zip(['train', 'eval'], [df_train, df_val]):
        df['visit'] = df['visit'].astype(int)
        if stage == 'eval' and cfg.use_only_baseline:
            df = df[df['visit_id'] == 0]
        loaders[f'oai_{stage}'] = ItemLoader(
            meta_data=df, root=cfg.root, batch_size=cfg.bs, num_workers=cfg.num_workers,
            transform=init_transforms(oai_mean, oai_std)[stage], parser_kwargs=cfg.parser,
            parse_item_cb=parse_item_progs, shuffle=True if stage == "train" else False, drop_last=False)

    model = create_model(cfg, device, pn_weights=pn_weights, y0_weights=y0_weights)

    if cfg.pretrained_model and not os.path.exists(cfg.pretrained_model):
        log.fatal(f'Cannot find pretrained model {cfg.pretrained_model}')
        assert False
    elif cfg.pretrained_model:
        log.info(f'Loading pretrained model {cfg.pretrained_model}')
        try:
            model.load_state_dict(torch.load(cfg.pretrained_model), strict=True)
        except ValueError:
            log.fatal(f'Failed loading {cfg.pretrained_model}')

    for epoch_i in range(cfg.n_epochs):
        for stage in ["train", "eval"]:
        # for stage in ["eval"]:
            main_loop(loaders[f'oai_{stage}'], epoch_i, model, cfg, stage)


def whether_update_metrics(batch_i, n_iters):
    return batch_i % 10 == 0 or batch_i >= n_iters - 1


def check_y0_exists(cfg):
    return cfg.predict_current_KL and cfg.kl_coef > 0


def filter_metrics(cfg, metrics):
    global task_names, task2metrics
    filtered_metrics = metrics
    if cfg.dataset == "oai":
        # Remove missing/minor follow-ups in OAI
        followups_mask = [True, True, True, False, False, True, False, True]
        for task in task_names:
            if task == 'grading':
                continue
            for metric_name in task2metrics[task]:
                _metric = filtered_metrics[task][metric_name]
                if isinstance(_metric, list):
                    filtered_metrics[task][metric_name] = np.array(_metric)[followups_mask].tolist()
    else:
        raise ValueError(f'Not support {cfg.dataset}.')
    return filtered_metrics


def model_selection(cfg, filtered_metrics, model, epoch_i):
    global stored_models
    # y0
    if check_y0_exists(cfg):
        stored_models = store_model(
            epoch_i, 'grading', "ba.ka", filtered_metrics, stored_models, model, cfg.snapshots, cond="max",
            mode="scalar")

    # Prognosis
    if cfg.prognosis_coef > 0:
        stored_models = store_model(
            epoch_i, 'pn', "ba", filtered_metrics, stored_models, model, cfg.snapshots, cond="max",
            mode=f"{cfg.model_selection_mode}_rev" if cfg.model_selection_mode == "beta" else cfg.model_selection_mode)
        stored_models = store_model(
            epoch_i, 'pn', "loss", filtered_metrics, stored_models, model, cfg.snapshots, cond="min", mode="scalar")

    stored_models = store_model(
        epoch_i, 'all', "loss", filtered_metrics, stored_models, model, cfg.snapshots, cond="min", mode="scalar")


def prepare_display_metrics(cfg, display_metrics, metrics_by):
    if check_y0_exists(cfg):
        display_metrics[f'{cfg.grading}:ba'] = metrics_by['grading']['ba']
        display_metrics[f'{cfg.grading}:mauc'] = metrics_by['grading']['mauc']
        display_metrics[f'{cfg.grading}:ka'] = metrics_by['grading']['ka']

    if cfg.prognosis_coef:
        display_metrics[f'pn:ba'] = "-".join(
            [f'{v:.03f}' if v is not None else "" for v in metrics_by['pn']['ba'].values()])
        display_metrics[f'pn:mse'] = "-".join(
            [f'{v:.03f}' if v is not None else "" for v in metrics_by['pn']['mse'].values()])
        display_metrics[f'pn:mauc'] = "-".join(
            [f'{v:.03f}' if v is not None else "" for v in metrics_by['pn']['mauc'].values()])

    return display_metrics


def get_masked_IDs(cfg, batch, mask_name, t=None):
    IDs = batch['data']['input']['ID']
    if "classifier" in cfg.method_name and t is not None:
        t = cfg.target_time - 1

    if t is None:
        return [IDs[i] for i in range(len(IDs)) if batch[mask_name][i]]
    else:
        return [IDs[i] for i in range(len(IDs)) if batch[mask_name][i, t]]


def main_loop(loader, epoch_i, model, cfg, stage="train"):
    global best_bacc, saved_bacc_model_fullname
    global best_f1, saved_f1_model_fullname
    global best_auc, saved_auc_model_fullname
    global best_ap, saved_ap_model_fullname
    global task_names, task2metrics
    global stored_models

    n_iters = len(loader)
    progress_bar = tqdm(range(n_iters), total=n_iters, desc=f"{stage}::{epoch_i}")
    accumulated_metrics = {'ID': [], 'loss': [], 'loss_pn': [], 'loss_y0': [], 'pn': None, cfg.grading: None}
    for task in task_names:
        accumulated_metrics[task] = {}
        accumulated_metrics[task]['ID_by'] = [[] for i in range(cfg.seq_len)]
        accumulated_metrics[task]['softmax_by'] = [[] for i in range(cfg.seq_len)]
        accumulated_metrics[task]['prob_by'] = [[] for i in range(cfg.seq_len)]
        accumulated_metrics[task]['pred_by'] = [[] for i in range(cfg.seq_len)]
        accumulated_metrics[task]['label_by'] = [[] for i in range(cfg.seq_len)]

    if check_y0_exists(cfg):
        accumulated_metrics[cfg.grading] = {'ID': [], 'pred': [], 'label': [], 'softmax': [], 'prob': []}

    if stage == "eval":
        model.eval()
    else:
        model.train()

    final_metrics = {}
    task = 'pn'

    for batch_i in progress_bar:
        batch = loader.sample(1)[0]

        IDs = batch['data']['input']['ID']
        accumulated_metrics['ID'].extend(IDs)

        # Input
        input = {}
        for in_key in batch['data']['input']:
            if isinstance(batch['data']['input'][in_key], torch.Tensor):
                input[in_key] = batch['data']['input'][in_key].to(device)
            else:
                input[in_key] = batch['data']['input'][in_key]

        for inp in input.values():
            if isinstance(inp, torch.Tensor):
                batch_size = inp.shape[0]
                break
            elif (isinstance(inp, tuple) or isinstance(inp, list)) and isinstance(inp[0], torch.Tensor):
                batch_size = inp[0].shape[0]
                break

        in_seq_len = batch['prognosis'].shape[1]

        out_seq_len = cfg.seq_len

        input['label_len'] = torch.tensor([in_seq_len] * batch_size, dtype=torch.int32).to(device)

        # Target
        targets = {}
        targets[f'current_{cfg.grading}'] = batch[cfg.grading].to(device)
        targets[f'current_{cfg.grading}_mask'] = batch[f'{cfg.grading}_mask'].to(device)
        targets['prognosis'] = batch['prognosis'].to(device)
        targets['prognosis_mask'] = batch['prognosis_mask'].to(device)

        losses, outputs = model.fit(input, targets, batch_i=batch_i, n_iters=n_iters, epoch_i=epoch_i, stage=stage)

        # Metrics
        display_metrics = {}
        for loss_name in losses:
            if losses[loss_name] is not None:
                accumulated_metrics[loss_name].append(losses[loss_name])
                display_metrics[loss_name] = f'{np.array(accumulated_metrics[loss_name]).mean():.03f}'

        metrics_by = {'pn': {}, cfg.grading: {}, 'all': {}}
        for task in task_names:
            metrics_by[task] = {}
            for _name in task2metrics[task]:
                metrics_by[task][_name] = {i: None for i in range(out_seq_len)}

        accumulated_metrics['loss_pn'].append(losses['loss_pn'])
        accumulated_metrics['loss_y0'].append(losses['loss_y0'])
        accumulated_metrics['loss'].append(losses['loss'])

        for t in range(cfg.seq_len):
            task = 'pn'
            labels = outputs[task]['label'][t].flatten()
            preds = np.argmax(outputs[task]['prob'][t], axis=-1)
            probs = outputs[task]['prob'][t]

            IDs_masked = get_masked_IDs(cfg, batch, 'prognosis_mask', t)
            accumulated_metrics[task]['ID_by'][t].extend(IDs_masked)
            accumulated_metrics[task]['softmax_by'][t].append(outputs[task]['prob'][t])
            accumulated_metrics[task]['pred_by'][t].extend(list(preds))
            accumulated_metrics[task]['prob_by'][t].extend(list(probs))
            accumulated_metrics[task]['label_by'][t].extend(list(labels.astype(int)))

            if whether_update_metrics(batch_i, n_iters):
                # Prognosis
                metrics_by['pn']['ba'][t] = calculate_metric(balanced_accuracy_score,
                                                             accumulated_metrics['pn']['label_by'][t],
                                                             accumulated_metrics['pn']['pred_by'][t])
                metrics_by['pn']['mauc'][t] = calculate_metric(roc_auc_score,
                                                               accumulated_metrics['pn']['label_by'][t],
                                                               accumulated_metrics['pn']['prob_by'][t],
                                                               average='macro',
                                                               labels=[i for i in range(cfg.n_pn_classes)],
                                                               multi_class=cfg.multi_class_mode)

                metrics_by['pn']['mse'][t] = calculate_metric(mean_squared_error,
                                                              accumulated_metrics['pn']['label_by'][t],
                                                              accumulated_metrics['pn']['pred_by'][t])

        # Current KL
        if check_y0_exists(cfg) and cfg.grading in outputs:
            IDs_masked = get_masked_IDs(cfg, batch, f'{cfg.grading}_mask')
            accumulated_metrics[cfg.grading]['ID'].extend(IDs_masked)
            accumulated_metrics[cfg.grading]['pred'].extend(list(np.argmax(outputs[cfg.grading]['prob'], axis=-1)))
            accumulated_metrics[cfg.grading]['label'].extend(list(outputs[cfg.grading]['label']))
            accumulated_metrics[cfg.grading]['softmax'].append(outputs[cfg.grading]['prob'])
            accumulated_metrics[cfg.grading]['prob'].extend(list(outputs[cfg.grading]['prob']))
            if whether_update_metrics(batch_i, n_iters):
                metrics_by['grading']['ba'] = calculate_metric(balanced_accuracy_score,
                                                               accumulated_metrics[cfg.grading]['label'],
                                                               accumulated_metrics[cfg.grading]['pred'])
                metrics_by['grading']['ka'] = calculate_metric(cohen_kappa_score,
                                                               accumulated_metrics[cfg.grading]['label'],
                                                               accumulated_metrics[cfg.grading]['pred'],
                                                               weights="quadratic")
                metrics_by['grading']['mauc'] = calculate_metric(roc_auc_score,
                                                                 accumulated_metrics[cfg.grading]['label'],
                                                                 accumulated_metrics[cfg.grading]['prob'],
                                                                 average='macro',
                                                                 labels=[i for i in range(cfg.n_pn_classes)],
                                                                 multi_class=cfg.multi_class_mode)

        if whether_update_metrics(batch_i, n_iters):
            display_metrics = prepare_display_metrics(cfg, display_metrics, metrics_by)
            progress_bar.set_postfix(display_metrics)

        # Last batch
        if batch_i >= n_iters - 1:
            final_metrics = metrics_by

    metrics = {'all': {}}
    for task in task_names:
        metrics[task] = {}
        for _name in task2metrics[task]:
            if _name in final_metrics[task]:
                if task == 'grading':
                    metrics[task][_name] = final_metrics[task][_name]
                else:
                    metrics[task][_name] = list(final_metrics[task][_name].values())

    # Losses
    metrics['pn']['loss'] = np.array(accumulated_metrics['loss_pn']).mean()
    metrics['all']['loss'] = np.array(accumulated_metrics['loss']).mean()

    # Store model
    if stage == "eval" and not cfg.skip_store:
        filtered_metrics = metrics
        model_selection(cfg, filtered_metrics, model, epoch_i)

    return metrics, accumulated_metrics


if __name__ == "__main__":
    main()
