"""One Iris example: prepare facts, run clingo, extract and display a tree.

This reference implementation demonstrates the method; it does not reproduce
the manuscript's experimental tables. Run `python demo.py all` after installing
requirements.txt. All paths are relative to this file, independent of cwd.
"""
import argparse
import csv
import html
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
FEATURES = ['sepal length', 'sepal width', 'petal length', 'petal width']
CLASSES = ['setosa', 'versicolor', 'virginica']


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n', encoding='utf-8')


def prepare(args):
    rows = list(csv.reader((ROOT / 'iris.csv').open(encoding='utf-8')))[1:]
    X = np.array([[float(v) for v in r[:4]] for r in rows])
    y = np.array([int(r[4]) for r in rows])
    train, test = train_test_split(np.arange(len(y)), test_size=0.3,
                                   stratify=y, random_state=args.seed)
    # Decimal arithmetic makes mathematical floor(alpha*x) explicit even when
    # binary floating-point would represent an exact decimal slightly below it.
    from decimal import Decimal, ROUND_FLOOR
    V = np.array([[int((Decimal(v) * args.alpha).to_integral_value(
        rounding=ROUND_FLOOR)) for v in r[:4]] for r in rows], dtype=int)
    facts = [f'sample({s+1}).' for s in range(len(train))]
    facts += [f'feature({f+1}).' for f in range(4)]
    facts += [f'class_label({c}).' for c in range(3)]
    counts = []
    for f in range(4):
        values = np.unique(V[train, f])
        thresholds = (values[:-1] + values[1:]) // 2
        if args.thresholds and len(thresholds) > args.thresholds:
            indices = np.linspace(0, len(thresholds)-1, args.thresholds, dtype=int)
            thresholds = thresholds[indices]
        facts += [f'threshold({f+1},{int(t)}).' for t in thresholds]
        counts.append(len(thresholds))
    for s, row_id in enumerate(train, 1):
        facts.append(f'true_class({s},{int(y[row_id])}).')
        facts += [f'value({s},{f+1},{int(V[row_id,f])}).' for f in range(4)]
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    (out / 'train.lp').write_text('\n'.join(facts)+'\n', encoding='utf-8')
    write_json(out / 'data.json', dict(alpha=args.alpha, seed=args.seed,
        thresholds_per_feature=counts, feature_names=FEATURES, class_names=CLASSES,
        train_indices=train.tolist(), test_indices=test.tolist(),
        values=V.tolist(), labels=y.tolist()))
    print(f'Prepared {len(train)} training / {len(test)} test samples; thresholds {counts}')


def solve(args):
    out = ROOT / args.output
    if not (out / 'train.lp').exists():
        raise RuntimeError('Run prepare first.')
    # optN proves optimality, then models=1 selects one optimal model rather than
    # enumerating every equally optimal tree. Time-limited incumbents are labelled.
    command = [sys.executable, '-m', 'clingo', str(ROOT / 'encoding.lp'),
        str(out / 'train.lp'), '--opt-mode=optN', '--models=1', '--outf=2',
        '--seed=1', f'--time-limit={args.timeout}',
        '-c', f'max_depth={args.depth}', '-c', f'size_tiebreak={int(not args.no_size)}',
        '-c', f'root_feature={args.root_feature}',
        '-c', f'banned_feature={args.banned_feature}',
        '-c', f'forbid_below={args.forbid_below}']
    proc = subprocess.run(command, capture_output=True, text=True)
    (out / 'solver.stderr.txt').write_text(proc.stderr, encoding='utf-8')
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f'clingo failed: {proc.stderr}') from exc
    write_json(out / 'solver.json', result)
    write_json(out / 'run.json', dict(command=command, max_depth_levels=args.depth,
        max_depth_edges=args.depth-1, size_tiebreak=not args.no_size,
        root_feature=args.root_feature, banned_feature=args.banned_feature,
        forbid_below=args.forbid_below, returncode=proc.returncode))
    print(f'clingo: {result.get("Result")}; raw output: {out / "solver.json"}')
    if result.get('Result') not in ('OPTIMUM FOUND', 'SATISFIABLE', 'UNKNOWN'):
        raise RuntimeError('No feasible tree: check solver.json and constraints.')


def show(args):
    out = ROOT / args.output
    result = json.loads((out / 'solver.json').read_text(encoding='utf-8'))
    data = json.loads((out / 'data.json').read_text(encoding='utf-8'))
    witnesses = [w for call in result.get('Call', []) for w in call.get('Witnesses', [])]
    if not witnesses:
        raise RuntimeError('No incumbent tree found within the solver limit.')
    best = min(witnesses, key=lambda w: tuple(w.get('Costs', [])))
    nodes = {}
    for atom in best['Value']:
        name, numbers = atom.split('(', 1)
        v = [int(n) for n in re.findall(r'-?\d+', numbers)]
        if name == 'activated':
            nodes.setdefault(v[0], {})
        elif name == 'leaf':
            nodes.setdefault(v[0], {})['leaf'] = True
        elif name == 'split':
            nodes.setdefault(v[0], {}).update(feature=v[1], threshold=v[2])
        elif name == 'predict':
            nodes.setdefault(v[0], {})['class'] = v[1]
    # Validate the exported structure rather than trusting only a pretty picture.
    def validate(n):
        node = nodes[n]
        if node.get('leaf'):
            assert node['class'] in range(3) and 'feature' not in node
        else:
            assert node['feature'] in range(1,5)
            validate(2*n)
            validate(2*n+1)
    validate(1)

    def predict(row):
        n = 1
        while not nodes[n].get('leaf'):
            f, t = nodes[n]['feature'], nodes[n]['threshold']
            n = 2*n + int(row[f-1] > t)
        return nodes[n]['class']
    train_errors = sum(predict(data['values'][i]) != data['labels'][i]
                       for i in data['train_indices'])
    test_errors = sum(predict(data['values'][i]) != data['labels'][i]
                      for i in data['test_indices'])
    if best.get('Costs'):
        assert train_errors == best['Costs'][0], 'Python and ASP predictions disagree'
        if len(best['Costs']) == 2:
            assert len(nodes) == best['Costs'][1]
    status = 'proven optimal' if result['Result'] == 'OPTIMUM FOUND' else 'incumbent only'
    header = f'{status}; training errors {train_errors}/{len(data["train_indices"])}; nodes {len(nodes)}'
    lines = [header, 'Depth parameter counts levels (root = 1).']
    def label(n):
        node = nodes[n]
        if node.get('leaf'):
            return f'node {n}: predict {CLASSES[node["class"]]}'
        return f'node {n}: floor({data["alpha"]} * {FEATURES[node["feature"]-1]}) <= {node["threshold"]}'
    def visit(n, indent=''):
        lines.append(indent + label(n))
        if not nodes[n].get('leaf'):
            lines.append(indent+'  TRUE (<=):')
            visit(2*n, indent+'    ')
            lines.append(indent+'  FALSE (>):')
            visit(2*n+1, indent+'    ')
    visit(1)
    text = '\n'.join(lines)+'\n'
    (out / 'tree.txt').write_text(text, encoding='utf-8')
    write_json(out / 'tree.json', dict(status=status, costs=best.get('Costs'),
        nodes=nodes, training_errors=train_errors, test_errors=test_errors,
        test_count=len(data['test_indices']), alpha=data['alpha']))
    def branch(n):
        content = '<li><span>'+html.escape(label(n))+'</span>'
        if not nodes[n].get('leaf'):
            content += '<ul><li><b>TRUE (left)</b><ul>'+branch(2*n)+'</ul></li>'
            content += '<li><b>FALSE (right)</b><ul>'+branch(2*n+1)+'</ul></li></ul>'
        return content+'</li>'
    page = '<!doctype html><meta charset="utf-8"><title>ASP-DT Iris tree</title>'
    page += '<style>body{font:16px system-ui;margin:40px;max-width:1000px}li{margin:14px 0}span{background:#eef3f7;padding:8px;border-radius:5px;display:inline-block}ul{border-left:2px solid #cbd5e1}p{color:#475569}</style>'
    page += '<h1>ASP-DT Iris tree</h1><p>'+html.escape(header)+'</p>'
    page += '<p>Each branch compares scaled integers. Root level is 1. This is a workflow example, not a manuscript benchmark result.</p><ul>'+branch(1)+'</ul>'
    (out / 'tree.html').write_text(page, encoding='utf-8')
    print(text)
    print(f'Open {out / "tree.html"} in a browser to view the structure.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=['prepare','solve','show','all'])
    parser.add_argument('--output', default='output')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--alpha', type=int, default=100)
    parser.add_argument('--thresholds', type=int, default=5, help='0 = all midpoints')
    parser.add_argument('--depth', type=int, default=3, help='number of levels; root = 1')
    parser.add_argument('--timeout', type=int, default=60)
    parser.add_argument('--no-size', action='store_true')
    parser.add_argument('--root-feature', type=int, default=0, help='1..4; 0 disables')
    parser.add_argument('--banned-feature', type=int, default=0, help='1..4; 0 disables')
    parser.add_argument('--forbid-below', type=int, default=1, help='level; root = 1')
    args = parser.parse_args()
    if args.alpha<1 or args.thresholds<0 or not 1<=args.depth<=8 or args.timeout<1:
        parser.error('alpha/timeout must be positive, thresholds nonnegative, depth 1..8')
    if args.root_feature not in range(5) or args.banned_feature not in range(5) or args.forbid_below<1:
        parser.error('feature indices must be 0..4; forbid-below must be positive')
    for name, function in [('prepare',prepare), ('solve',solve), ('show',show)]:
        if args.step in (name,'all'):
            function(args)


if __name__ == '__main__':
    main()
