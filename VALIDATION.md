# Validation of the minimal reference example

Validated on 11 October 2026 with Python 3.12.14, clingo 5.8.0,
numpy 2.3.5, and scikit-learn 1.8.0. The requirements allow other compatible
versions; clingo 5.6.2 was not validated here.

- Default Iris prepare → solve → show completed. clingo reported OPTIMUM FOUND.
  The selected tree has 11 training errors among 105 samples and 7 active nodes.
  These are demonstration results, not additions to the manuscript results.
- Python traversal agreed with the ASP training-error and node-count objectives.
- The root-feature-3 constraint was respected.
- A single-level tree gave one leaf and 70 training errors, as expected from
  the balanced three-class training split.
- An independent exhaustive search over all candidate stumps on a small toy
  dataset agreed with the ASP optimum: 1 error, then 3 nodes.

No paper benchmark was rerun. The saved example output is one optimal tree;
equal-cost trees and solver timings may vary. The absolute paths in the saved
run.json record the validation environment; new runs generate local paths.
