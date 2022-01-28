import json
import logging as log
import os

import coloredlogs
import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, \
    mean_squared_error, cohen_kappa_score

from common.data import ItemLoader
from common.utils import proc_targets, calculate_metric, load_metadata, init_mean_std, parse_item_progs, \
    update_max_grades, parse_img, init_transforms
from models import create_model
from . import train

# from prognosis.train import main_loop


coloredlogs.install()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

task_names = ('pr', 'pn', 'tkr')
task2metrics = {'pr': ['f1', 'ba', 'auc', 'ap'], 'pn': ['ba', 'mse'], 'tkr': ['ba', 'auc', 'ap']}


def filter_most_by_pa(ds, df_most_ex, pas=['PA05', 'PA10', 'PA15']):
    std_rows = []
    for i, row in df_most_ex.iterrows():
        std_row = dict()
        std_row['ID'] = row['ID_ex'].split('_')[0]
        std_row['visit_id'] = int(row['visit'][1:])
        std_row['PA'] = row['PA']
        std_rows.append(std_row)
    df_most_pa = pd.DataFrame(std_rows)

    ds_most_filtered = pd.merge(ds, df_most_pa, how='left', on=['ID', 'visit_id'])
    if isinstance(pas, str):
        ds_most_filtered = ds_most_filtered[ds_most_filtered['PA'] == pas]
    elif isinstance(pas, list):
        ds_most_filtered = ds_most_filtered[ds_most_filtered['PA'].isin(pas)]
    else:
        raise ValueError(f'Not support type {type(ds_most_filtered)}.')

    return ds_most_filtered


def load_data(cfg, site):
    # Compute mean and std of OAI
    wdir = os.environ['PWD']
    # Load and split data
    if cfg.dataset == "oai":
        meta_test = load_metadata(cfg, proc_targets=proc_targets, eval_only=True)
    else:
        raise ValueError(f'Not support dataset {cfg.dataset}.')

    # Loaders
    oai_mean, oai_std = init_mean_std(cfg, wdir, None, parse_img)
    print(f'Mean: {oai_mean}\nStd: {oai_std}')

    # Only choose records with baseline + first follow up
    print(f'Before filtering: {len(meta_test.index)}')

    if isinstance(cfg.use_only_grading, int) and cfg.use_only_grading >= 0:
        meta_test = meta_test[meta_test[cfg.grading] == cfg.use_only_grading]
        print(f'Only select grading {cfg.grading} = {cfg.use_only_grading}... Remaining: {len(meta_test.index)}')

    if cfg.use_only_baseline:
        print(f'Only select baseline...')
        meta_test = meta_test[meta_test['visit'] == 0]
        print(f'Only select baseline... Remaining: {len(meta_test.index)}')

    print(f'After filtering: {len(meta_test.index)}')

    # Cast visit to int
    meta_test['visit'] = meta_test['visit'].astype(int)
    loader = ItemLoader(
        meta_data=meta_test, root=cfg.root, batch_size=cfg.bs, num_workers=cfg.num_workers,
        transform=init_transforms(oai_mean, oai_std)['eval'], parser_kwargs=cfg.parser,
        parse_item_cb=parse_item_progs, shuffle=False)
    return loader


def eval(pretrained_model, loader, cfg, device, store=True):
    model = create_model(cfg, device)

    if pretrained_model and not os.path.exists(pretrained_model):
        log.fatal(f'Cannot find pretrained model {pretrained_model}')
        assert False
    elif pretrained_model:
        log.info(f'Loading pretrained model {pretrained_model}')
        try:
            model.load_state_dict(torch.load(pretrained_model), strict=False)
        except ValueError:
            log.fatal(f'Failed loading {pretrained_model}')

    metrics, accumulated_metrics = train.main_loop(loader, 0, model, cfg, "test")

    if store:
        with open("eval_metrics.json", "w") as f:
            json.dump(metrics, f)

    return metrics, accumulated_metrics


def eval_from_saved_folders(cfg, root_dir, device, patterns="bacc"):
    d = os.listdir(root_dir)
    collector = {}
    result_collector = {}
    for dir in d:
        args_fullname = os.path.join(root_dir, dir, "args.yaml")

        if os.path.isfile(args_fullname):
            config = OmegaConf.load(args_fullname)
        else:
            print(f'Not found {args_fullname}')
            config_fullname = os.path.join(root_dir, dir, ".hydra", "config.yaml")
            override_fullname = os.path.join(root_dir, dir, ".hydra", "overrides.yaml")

            config = OmegaConf.load(config_fullname)
            overriden_config = OmegaConf.load(override_fullname)

            if isinstance(overriden_config, ListConfig):
                or_config_names = ["site", "fold_index", "n_out_features", "parser"]
                for line in overriden_config:
                    k, v = line.split("=")
                    if k in or_config_names:
                        config[k] = v

        if cfg.seed != config["seed"]:
            print(f'[{cfg.seed}] Skip {dir}.')
            continue

        if "grading" not in config["parser"]:
            config["parser"]["grading"] = config["grading"]

        or_config_names = ["root", "pkl_meta_filename", "meta_root", "bs", "num_workers", "dataset", "root",
                           "most_meta_filename", "oai_meta_filename", "multi_class_mode",
                           "use_y0_class_weights", "use_pn_class_weights", "use_pr_class_weights",
                           "use_only_grading", "use_only_baseline", "model_selection_mode", "save_attn",
                           "most_followup_meta_filename"]
        eval_config_names = ['output', 'root', 'patterns', 'n_resamplings', ]
        for k in or_config_names:
            config[k] = cfg[k]

        config.eval = {}
        for k in eval_config_names:
            config.eval[k] = cfg.eval[k]

        config.pkl_meta_filename = f"cv_split_5folds_{config.grading}_oai_evalsite_{config.site}_{config.seed}.pkl"
        config.skip_store = True

        update_max_grades(config)

        print(config.pretty())

        print(f'Loading data site {config.site}...')
        loader = load_data(config, config.site)

        print(f'Finding pretrained model...')
        run_root = os.path.join(root_dir, dir)
        for r, d, f in os.walk(run_root):
            for filename in f:
                if isinstance(patterns, tuple) or isinstance(patterns, list):
                    matched = all([s in filename for s in patterns])
                else:
                    matched = patterns in filename

                if filename.endswith(".pth") and matched:
                    model_fullname = os.path.join(r, filename)

                    key = f'Site:{config.site}:{config.fold_index}'

                    print(f'{key}, model: {model_fullname}...')

                    metrics, accumulated_metrics = eval(model_fullname, loader, config, device, False)

                    result_collector = convert_metrics_to_dataframe(cfg, result_collector, accumulated_metrics)
                    collector[key] = metrics
                    continue

    output_fullname = os.path.abspath(cfg.eval.output)
    print(f'Saving output files to {output_fullname}')

    result_collector_agg, metrics_by = aggregate_dataframe(cfg, result_collector)

    # if cfg.save_predictions:
    #    with open(output_fullname[:-5] + f"_agg.pkl", "wb") as f:
    #       pickle.dump(result_collector_agg, f)

    with open(output_fullname, "w") as f:
        json.dump(metrics_by, f)


def get_sites(cfg, IDs):
    enum_sites = ['A', 'B', 'C', 'D', 'E']
    sites = IDs.str[:1]

    return sites, enum_sites


def aggregate_dataframe(cfg, result_collector):
    grading = cfg.grading
    tasks = ['pn', 'pr']
    result_collector_agg = {}

    metrics_by = {}

    metrics_by[grading] = {'ba': -1.0, 'ka': -1.0, 'mauc': -1.0, 'ece': -1.0, 'ada_ece': -1.0, 'cls_ece': -1.0}
    metrics_by['pr'] = {'ba': [-1.0] * cfg.seq_len,
                        'f1': [-1.0] * cfg.seq_len,
                        'f2': [-1.0] * cfg.seq_len,
                        'auc': [-1.0] * cfg.seq_len,
                        'ap': [-1.0] * cfg.seq_len}
    metrics_by['pn'] = {'ba': [-1.0] * cfg.seq_len,
                        'mse': [-1.0] * cfg.seq_len,
                        'mauc': [-1.0] * cfg.seq_len}

    # KL
    if grading in result_collector:
        result_collector_agg[grading] = result_collector[grading].groupby('ID', as_index=False)[
            f'{grading}_label', f'{grading}_prob'].apply(np.mean)

        result_collector_agg[grading]['Site'], list_sites = get_sites(cfg, result_collector_agg[grading].ID)

        metrics_by_site = {'ba': [], 'ka': [], 'mauc': [], 'ece': [], 'ada_ece': [], 'cls_ece': []}
        for site in list_sites:
            result_agg_site = result_collector_agg[grading][result_collector_agg[grading]['Site'] == site]
            if len(result_agg_site.index) == 0:
                print(f'Cannot find data of Side {site}.')
                continue
            labels = result_agg_site[f'{grading}_label'].tolist()
            probs = [v[0].tolist() for v in result_agg_site[f'{grading}_prob'].tolist()]
            probs = np.stack(probs, 0)
            preds = list(np.argmax(probs, -1))

            metrics_by_site['ba'].append(calculate_metric(balanced_accuracy_score, labels, preds))
            metrics_by_site['ka'].append(calculate_metric(cohen_kappa_score, labels, preds, weights="quadratic"))
            metrics_by_site['mauc'].append(calculate_metric(roc_auc_score, labels, probs, average='macro',
                                                            labels=[i for i in range(cfg.n_pn_classes)],
                                                            multi_class=cfg.multi_class_mode))
        for metric in metrics_by[grading]:
            metrics_by[grading][metric] = metrics_by_site[metric]

    # PN
    for task in tasks:
        for t in range(cfg.seq_len):
            if f'{task}_{t}' in result_collector:

                metrics_by_site = {}
                for metric in metrics_by[task]:
                    metrics_by_site[metric] = []

                result_collector_agg[f'{task}_{t}'] = result_collector[f'{task}_{t}'].groupby('ID', as_index=False)[
                    f'{task}_label_{t}', f'{task}_prob_{t}'].apply(np.mean)

                result_collector_agg[f'{task}_{t}']['Site'], list_sites = get_sites(cfg, result_collector_agg[
                    f'{task}_{t}'].ID)

                for site in list_sites:
                    result_agg_site = result_collector_agg[f'{task}_{t}'][
                        result_collector_agg[f'{task}_{t}']['Site'] == site]
                    if len(result_agg_site.index) == 0:
                        print(f'Cannot find data of Side {site} for task {task} at follow-up {t}.')
                        continue
                    labels = result_agg_site[f'{task}_label_{t}'].tolist()
                    probs = [v[0].tolist() for v in result_agg_site[f'{task}_prob_{t}'].tolist()]
                    probs = np.stack(probs, 0)
                    preds = list(np.argmax(probs, -1))

                    if task == 'pn':
                        # Prognosis
                        metrics_by_site['ba'].append(calculate_metric(balanced_accuracy_score, labels, preds))
                        metrics_by_site['mse'].append(calculate_metric(mean_squared_error, labels, preds))
                        metrics_by_site['mauc'].append(calculate_metric(roc_auc_score, labels, probs,
                                                                        average='macro',
                                                                        labels=[i for i in range(cfg.n_pn_classes)],
                                                                        multi_class=cfg.multi_class_mode))
                    else:
                        raise ValueError(f'Not support {task}.')

                for metric in metrics_by[task]:
                    metrics_by[task][metric][t] = metrics_by_site[metric]

    return result_collector_agg, metrics_by


def convert_metrics_to_dataframe(cfg, result_collector, accumulated_metrics):
    grading = cfg.grading
    tasks = []
    if cfg.progs_coef > 0.0:
        tasks.append('pr')
    if cfg.prognosis_coef > 0.0:
        tasks.append('pn')
    if cfg.kl_coef > 0.0:
        tasks.append(grading)

    for key in tasks:
        if key in accumulated_metrics:
            if key != grading and \
                    accumulated_metrics[key] is not None and \
                    'ID_by' in accumulated_metrics[key] and \
                    np.array(accumulated_metrics[key]['ID_by']).size > 0:
                seq_len = len(accumulated_metrics[key]['label_by'])
                for t in range(seq_len):
                    df = pd.DataFrame()
                    num_samples = len(accumulated_metrics[key]['label_by'][t])
                    df['ID'] = list(accumulated_metrics[key]['ID_by'][t])
                    df[f'{key}_label_{t}'] = list(accumulated_metrics[key]['label_by'][t])
                    probs = np.concatenate(accumulated_metrics[key]['softmax_by'][t], 0)
                    if num_samples > 0:
                        df[f'{key}_prob_{t}'] = np.array_split(probs, num_samples, axis=0)
                        if f'{key}_{t}' not in result_collector:
                            result_collector[f'{key}_{t}'] = df
                        else:
                            result_collector[f'{key}_{t}'] = pd.concat((result_collector[f'{key}_{t}'], df),
                                                                       ignore_index=True, sort=False)
            elif accumulated_metrics[key] is not None and \
                    'ID_by' in accumulated_metrics[key] and \
                    np.array(accumulated_metrics[key]['ID_by']).size > 0:
                df = pd.DataFrame()
                num_samples = len(accumulated_metrics[key]['ID'])
                df['ID'] = list(accumulated_metrics[key]['ID'])
                df[f'{key}_label'] = list(accumulated_metrics[key]['label'])
                probs = np.concatenate(accumulated_metrics[key]['softmax'], 0)
                df[f'{key}_prob'] = np.array_split(probs, num_samples, axis=0)
                if num_samples > 0:
                    if key not in result_collector:
                        result_collector[key] = df
                    else:
                        result_collector[key] = pd.concat((result_collector[f'{key}'], df), ignore_index=True,
                                                          sort=False)

    return result_collector


@hydra.main(config_path="configs/config_eval.yaml")
def main(cfg):
    wdir = os.environ['PWD']
    if not os.path.isabs(cfg.meta_root):
        cfg.meta_root = os.path.join(wdir, cfg.meta_root)

    if not os.path.isdir(cfg.snapshots):
        os.makedirs(cfg.snapshots)

    if cfg.eval.root:
        eval_from_saved_folders(cfg, cfg.eval.root, device, cfg.eval.patterns)
    else:
        loaders = load_data(cfg, cfg.site)
        eval(cfg.pretrained_model, loaders, cfg, device, True)


if __name__ == "__main__":
    main()
