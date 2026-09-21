"""Generate network metadata and per-node diagnostics without TensorFlow."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_synthetic_evolution import load_records, build_table, plot_evolution

TYPE_NAMES = {
    0: 'SOURCE_HEAVY', 1: 'SOURCE_MID', 2: 'SOURCE_LIGHT',
    3: 'TARGET_HEAVY', 4: 'TARGET_MID', 5: 'TARGET_LIGHT',
}


def read_json(path):
    return json.loads(path.read_text())


def network_types(data_dir):
    rows = []
    for path in data_dir.glob('*_data.json'):
        name = path.name.removesuffix('_data.json')
        if not name.isdigit():
            continue
        x = np.asarray(read_json(path)['X_train'])
        if x.ndim != 3:
            raise ValueError(f'{path}: expected 3D X_train, got {x.shape}')
        valid = np.any(x != 0, axis=2)
        types = np.unique(x[:, :, -1][valid])
        if len(types) != 1 or types[0] not in TYPE_NAMES:
            raise ValueError(f'{path}: invalid or multiple node types: {types}')
        t = int(types[0])
        rows.append({'node_id': int(name), 'node_type': t,
                     'node_type_name': TYPE_NAMES[t]})
    if not rows:
        raise ValueError(f'No numbered node datasets in {data_dir}')
    return pd.DataFrame(rows).sort_values('node_id')


def node_table(folder, records, node, node_type):
    rows, exchanges = [], []
    for s in records:
        u = s['update']
        prefix = folder / f'update_{u:03d}'
        paths = {stage: Path(f'{prefix}_{stage}.json')
                 for stage in ('before_merge', 'after_merge', 'after_training')}
        npz_path = Path(f'{prefix}_after_training.npz')
        if not all(p.exists() for p in paths.values()) or not npz_path.exists():
            raise FileNotFoundError(f'Node {node}, update {u}: incomplete diagnostics; wait for training to finish')
        metrics = {stage: read_json(p) for stage, p in paths.items()}
        b, m, a = [metrics[stage]['val'] for stage in paths]
        with np.load(npz_path) as z:
            labels = z['val_truth'][:, -1]
            losses = z['val_sample_bce']
            per_class = {c: float(losses[labels == c].mean())
                         if np.any(labels == c) else np.nan for c in (0, 1)}
        total = s['local_samples'] + s['synthetic_samples']
        overloaded = s['local_overloaded_count'] + sum(c['overloaded_count_after'] for c in s['changes'])
        row = {
            'node_id': node, 'node_type': node_type, 'update': u,
            'plot_index': s['plot_index'],
            'sender_nodes': ';'.join(str(v['node']) for v in s['senders']),
            'sender_types': ';'.join(str(v['node_type']) for v in s['senders']),
            'model_update_mode': s['model_update_mode'], 'synthetic_mode': s['synthetic_mode'],
            'val_bce_before': b['bce'], 'val_bce_after_merge': m['bce'],
            'val_bce_after': a['bce'], 'delta_val_bce': a['bce'] - b['bce'],
            'val_accuracy_before': b['accuracy'], 'val_accuracy_after': a['accuracy'],
            'val_mse_before': b['mse'], 'val_mse_after': a['mse'],
            'val_bce_class_0': per_class[0], 'val_bce_class_1': per_class[1],
            'val_count_class_0': int(np.sum(labels == 0)),
            'val_count_class_1': int(np.sum(labels == 1)),
            'changed_labels': sum(c.get('changed_labels', 0) for c in s['changes'])
                              if s['synthetic_mode'] == 'aggregate' else np.nan,
            'new_types': ';'.join(str(c['node_type']) for c in s['changes'] if c['new_type']),
            'local_samples': s['local_samples'], 'synthetic_samples': s['synthetic_samples'],
            'total_samples': total, 'overloaded_count': overloaded,
            'non_overloaded_count': total - overloaded,
            'overloaded_pct': 100 * overloaded / total if total else np.nan,
            'non_overloaded_pct': 100 * (total-overloaded) / total if total else np.nan,
        }
        rows.append(row)
        for sender in s['senders']:
            exchanges.append({'receiver_node': node, 'receiver_type': node_type,
                              'update': u, 'sender_node': sender['node'],
                              'sender_type': sender['node_type']})
    return pd.DataFrame(rows), pd.DataFrame(exchanges)


def plot_summary(table, folder, node, node_type, window, threshold):
    x = table['update']
    fig, ax = plt.subplots(4, 1, figsize=(14, 12), sharex=True, layout='constrained')
    ax[0].plot(x, table.val_bce_before, label='Prima del training', marker='.', color='steelblue')
    ax[0].plot(x, table.val_bce_after, label='Dopo il training', marker='.', color='firebrick')
    ax[0].set_ylabel('BCE validation locale'); ax[0].legend()
    for c, color in [(0, 'steelblue'), (1, 'darkorange')]:
        count = table[f'val_count_class_{c}'].iloc[0]
        label = 'non overloaded' if c == 0 else 'overloaded'
        ax[1].plot(x, table[f'val_bce_class_{c}'], color=color, marker='.', label=f'Veri {label} ({count})')
    ax[1].set_ylabel('BCE per classe\ndopo il training'); ax[1].legend()
    if table.changed_labels.notna().any():
        ax[2].bar(x, table.changed_labels, color='slateblue')
        ax[2].text(.99, .95, 'Nuovi tipi esclusi dal conteggio', ha='right', va='top', transform=ax[2].transAxes)
    else:
        ax[2].text(.5, .5, 'Confronto delle etichette disponibile solo con aggregate', ha='center', transform=ax[2].transAxes)
    ax[2].set_ylabel('Etichette sintetiche\ncambiate')
    ax[3].plot(x, table.overloaded_pct, color='darkorange', marker='.')
    ax[3].set_ylabel('Overloaded nel training\nlocale + sintetico (%)'); ax[3].set_ylim(0, 100)
    ax[3].set_xlabel('Aggiornamento gossip (parte da 1)')
    spikes = table.loc[table.delta_val_bce > threshold]
    for axis in ax:
        axis.grid(alpha=.2)
        for u in spikes['update']:
            axis.axvline(u, color='grey', linestyle=':', alpha=.5)
    for row in spikes.itertuples():
        ax[0].annotate(str(row.update), (row.update, row.val_bce_after), xytext=(0, 6), textcoords='offset points', ha='center', fontsize=8)
    first = table.iloc[0]
    fig.suptitle(f'Nodo {node} · tipo {node_type} ({TYPE_NAMES[node_type]}) · {first.model_update_mode} · {first.synthetic_mode} · finestra {window}\nTratteggi: aumento BCE > {threshold:g}; validation locale fissa')
    fig.savefig(folder / 'diagnostic_summary.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--spike-threshold', type=float, default=.05)
    args = parser.parse_args()
    root = args.run_dir / 'diagnostics'
    out = args.output_dir or root
    out.mkdir(parents=True, exist_ok=True)
    types = network_types(args.data_dir)
    types.to_csv(out / 'network_node_types.csv', index=False)
    config_path = args.run_dir / 'config.json'
    config = read_json(config_path).get('training', {}) if config_path.exists() else {}
    all_exchanges, missing = [], []
    for meta in types.itertuples(index=False):
        folder = root / f'node_{meta.node_id}'
        if not list(folder.glob('update_*_synthetic_changes.json')):
            missing.append(meta.node_id)
            continue
        records = load_records(folder)
        table, exchanges = node_table(folder, records, meta.node_id, meta.node_type)
        dest = out / folder.name
        dest.mkdir(parents=True, exist_ok=True)
        table.to_csv(dest / 'diagnostic_summary.csv', index=False)
        exchanges.to_csv(dest / 'received_exchanges.csv', index=False)
        proportions = build_table(records)
        proportions.to_csv(dest / 'class_proportions_over_updates.csv', index=False)
        plot_evolution(proportions, dest)
        plot_summary(table, dest, meta.node_id, meta.node_type,
                     config.get('synthetic_window_size', '?'), args.spike_threshold)
        all_exchanges.append(exchanges)
        print(f'Node {meta.node_id}: {len(table)} updates -> {dest}')
    if all_exchanges:
        pd.concat(all_exchanges, ignore_index=True).to_csv(out / 'received_exchanges_all_nodes.csv', index=False)
    if missing:
        print(f'WARNING: no diagnostics for nodes {missing}. No data were inferred for those nodes.')
    if not all_exchanges:
        raise SystemExit('No node diagnostics found.')


if __name__ == '__main__':
    main()
