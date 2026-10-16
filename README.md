# ASP-DT minimal Iris example

This package provides an executable reference implementation of the core
learning workflow described in the manuscript **ASP-DT: A Declarative and
Constraint-Aware Decision Tree Framework Based on Answer Set Programming**.
It was assembled for the minor revision from the manuscript's method description.
It is not the archived original experimental code and does not reproduce the
paper's means, standard deviations, timings, Adult analyses, or all ablations.

## Quick start

Use Python 3.10 or later. In this directory:

```bash
python -m pip install -r requirements.txt
python demo.py all
```

No online dataset download is needed. The bundled `iris.csv` contains the 150
rows supplied by scikit-learn's corrected Iris dataset. The seed-1 stratified
70/30 split has 105 training and 45 test rows. Thresholds are derived only
from the training rows. Default settings: alpha=100, uniform K=5 per feature,
three tree levels, training-error minimization followed by node-count tie-breaking,
solver seed 1, and a 60-second solving limit.

The limit applies to clingo solving, not Python preprocessing or a strict external
wall-clock budget. The program reports proven optimality only when clingo returns
`OPTIMUM FOUND`; otherwise any available tree is labelled an incumbent.

## Three steps

```bash
python demo.py prepare
python demo.py solve
python demo.py show
```

1. **Prepare:** scale values using mathematical floor(alpha*x), generate midpoint
   thresholds, optionally subsample them uniformly, and emit `output/train.lp`.
   `data.json` records the split row indices and scaling metadata.
2. **Solve:** pass the facts and `encoding.lp` to clingo. Raw stable models and
   objective costs are saved in `solver.json`; the actual command is in `run.json`.
3. **Show:** decode `activated`, `leaf`, `split`, and `predict` into `tree.json`,
   `tree.txt`, and `tree.html`. Open **output/tree.html** in any browser. The code
   independently evaluates routing and checks training errors against ASP costs.

An already generated example is in `example_output/`. It is illustrative and
was checked with the versions listed in `VALIDATION.md`. Equal-cost optimal
trees may have different structures across solver versions.

## Run clingo directly

After `python demo.py prepare`, the same default solve can be run from this directory:

```bash
python -m clingo encoding.lp output/train.lp --opt-mode=optN --models=1 --outf=2 --seed=1 --time-limit=60 -c max_depth=3 -c size_tiebreak=1 > output/solver.json
python demo.py show
```

Use the wrapper `python demo.py solve` when changing constraints or depth: it also
records the parameters and command in `run.json`.

## Encoding conventions

`max_depth` counts **levels**: the root is at level 1, and a value of 3 permits
at most 7 nodes and 2 root-to-leaf edges. This follows the manuscript's heap
skeleton formula Nmax=2^Dmax-1. It is not scikit-learn's edge-count depth parameter;
to match this example's structural capacity in CART, use max_depth=2.
The package does not resolve or re-evaluate the original benchmark depth settings.

The paper presents schematic rules; this runnable file explicitly supplies an
active root, depth facts, two active children for every internal node, and clingo
constants. Leaves may occur early. Each sample reaches one leaf. No test label
or test feature is used to choose a threshold or learn the tree.

Features: 1 sepal length, 2 sepal width, 3 petal length, 4 petal width (cm).
Classes: 0 setosa, 1 versicolor, 2 virginica. Routing compares integer values,
so the exported labels show the exact scaled test rather than an approximate
floating-point threshold.

## Small optional demonstrations

These demonstrate available declarative constraints; they are not new paper experiments.

```bash
python demo.py all --output root_example --root-feature 3
python demo.py all --output ban_example --banned-feature 1 --forbid-below 1
python demo.py all --output full_thresholds --thresholds 0
```

Use `--no-size` to disable the secondary node-count objective. Pass the same
`--output` directory to each step if running the steps separately. Rerun `prepare`
after changing preprocessing parameters. The `show` step reads stored metadata.

## Data and source references

- Iris source: https://archive.ics.uci.edu/dataset/53/iris
- scikit-learn Iris documentation: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html
- clingo: https://potassco.org/clingo/

The small data file was exported from scikit-learn's bundled Iris data. Use the
corrected scikit-learn version consistently; it has documented differences from
older copies of Iris. See the source dataset terms for data reuse.

## Code availability

The ASP encoding, Iris data, preprocessing, solver invocation, tree extraction,
and example outputs are publicly available in this repository:
https://github.com/shcc320/asp-dt

The manuscript links directly to this repository for code access. Run the example
with `python demo.py all` and open `output/tree.html` to inspect the learned tree.
This minimal reference implementation demonstrates the core workflow; it does
not reproduce every manuscript experiment or supply original per-trial records.
