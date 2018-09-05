"""
This file contains a patch for scipy.signal.signaling._reverse_and_conj function.
TODO: update numpy and scipy in the future and check to see if the error is still there.

-- Full warning: --
FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated;
use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as
an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
-------------------
"""

from scipy.signal import signaltools


def patched_reverse_and_conj(x):
    reverse = [slice(None, None, -1)] * x.ndim
    reverse = tuple(reverse)  # this is the actual change
    return x[reverse].conj()


signaltools._reverse_and_conj = patched_reverse_and_conj
