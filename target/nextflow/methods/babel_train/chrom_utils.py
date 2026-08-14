import logging
import re
from collections import OrderedDict

import numpy as np

logger = logging.getLogger(__name__)

_CHROM_RE = re.compile(r"^chr?([0-9XYM]+)[:\-]", re.IGNORECASE)


def parse_chrom_groups(var_names):
    """Group feature indices by chromosome, parsed from genomic-coordinate-style
    var_names (e.g. "chr1:1000-2000"), matching BABEL's per-chromosome grouping.

    Chromosomes are sorted lexicographically (matching BABEL's `sorted(set(...))`)
    and must be reproduced in the same order at predict time.

    Falls back to a single group spanning all features if the names don't match
    the expected chromosome-coordinate pattern.
    """
    chrom_of_feature = []
    for name in var_names:
        m = _CHROM_RE.match(str(name))
        chrom_of_feature.append(m.group(1).upper() if m else None)

    n_parsed = sum(c is not None for c in chrom_of_feature)
    unique_chroms = sorted({c for c in chrom_of_feature if c is not None})

    if n_parsed < len(var_names) or len(unique_chroms) < 2:
        logger.warning(
            "Could not parse chromosome-of-origin for all %d features from var_names "
            "(parsed %d, %d unique chromosomes); falling back to a single feature group.",
            len(var_names), n_parsed, len(unique_chroms),
        )
        chrom_groups = OrderedDict([("ALL", np.arange(len(var_names)))])
        return [len(var_names)], chrom_groups

    chrom_groups = OrderedDict()
    chrom_array = np.array(chrom_of_feature)
    for chrom in unique_chroms:
        chrom_groups[chrom] = np.where(chrom_array == chrom)[0]

    counts = [len(idxs) for idxs in chrom_groups.values()]
    return counts, chrom_groups


def reindex_to_chrom_groups(X, chrom_groups):
    """Concatenate feature columns of X in chrom_groups order (identity if already sorted).
    Used to guarantee predict-time feature order matches the order the model was trained with.
    """
    idx = np.concatenate(list(chrom_groups.values()))
    return X[:, idx]
