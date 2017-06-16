from contextlib import contextmanager
import numpy as np
from numpy import uint64
from sklearn.model_selection._split import _BaseKFold, BaseShuffleSplit
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, _num_samples
from zlib import crc32


@contextmanager
def ignore_overflow():
    """
    Do not emit numpy warning when encountering integer overflow
    """
    current = np.seterr(over='ignore')
    yield
    np.seterr(**current)


class HashSplitMixin():
    """
    Mixin providing methods to hash strings to uint32 and re-hashing uint32s
    """
    # hashing constants
    UINT64MAX = np.iinfo(uint64).max
    POWER = uint64(32)
    BITS = uint64(64)
    BUCKETS = int(2 ** POWER)

    def _str_to_uint(self, labels):
        """
        Turn a string into an unsigned int32 using CRC32.
        As everywhere, use uint64 as type.
        """
        labels = check_array(labels, ensure_2d=False, dtype=str)

        hashes = np.zeros(len(labels), dtype=uint64)
        for i, l in enumerate(labels):
            hashes[i] = uint64(crc32(l.encode()))
        return hashes

    def _uni_hash(self, h0, rng):
        """
        Universal int to int hashing function. Uses the multiply-shift scheme described by
        Dietzfelbinger et al. to hash to m = 2**M bins. See:
        https://en.wikipedia.org/wiki/Universal_hashing#Avoiding_modular_arithmetic

        Parameters
        ----------
        h0 : numpy.array
            Array of initial uint32 hashes.

        rng : numpy.RandomState
            RandomState instance as returned by sklearn.utils.check_random_state().

        Returns
        -------
        h1 : numpy.array
            Re-hashed version of h0.
        """
        w = self.BITS
        M = self.POWER
        a = rng.randint(self.UINT64MAX, dtype=uint64)
        if a % 2 == 0:
            a += uint64(1)
        b = rng.randint(uint64(2**(w - M)), dtype=uint64)

        with ignore_overflow():
            h1 = (a * h0 + b) >> (w - M)
        return h1


class HashShuffleSplit(BaseShuffleSplit, HashSplitMixin):
    """
    Shuffle split cross-validator based on hashed labels.
    Pass labels as `groups` parameter when calling .split().
    Note that train and test size are only approximate and may diverge,
    particularly for small numbers of samples.

    Yields indices to split data into training and test sets.

    Parameters
    ----------
    n_splits : int, default 10
        Number of re-shuffling & splitting iterations.

    test_size : float, int, None, default=0.1
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. Note that setting absolute values only
        results in the approximate number of examples in the train / test set.

    train_size : float, int, or None, default=None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """
    def __init__(self, n_splits=10, test_size=0.1, train_size=None,
                 random_state=None):
        super(HashShuffleSplit, self).__init__(n_splits=n_splits,
                                               test_size=test_size,
                                               train_size=train_size,
                                               random_state=random_state)

    def _iter_indices(self, X, y, groups):
        if groups is None:
            raise ValueError("The 'groups' parameter should contain the labels to be hashed.")
        groups = check_array(groups, ensure_2d=False, dtype=str)

        n_samples = _num_samples(X)
        if len(groups) != n_samples:
            raise ValueError("'X' and 'groups' must have the same number of elements.")

        test_size, train_size = _validate_shuffle_split(n_samples,
                                                        self.test_size,
                                                        self.train_size)
        rng = check_random_state(self.random_state)

        # create initial set of int hashes
        h0 = self._str_to_uint(groups)

        n_buckets = self.BUCKETS
        train_max = n_buckets * train_size
        if train_size + test_size < 1:
            test_max = train_max + n_buckets * test_size
        else:
            test_max = n_buckets

        for i in range(self.n_splits):
            h1 = self._uni_hash(h0, rng=rng)
            is_train = h1 < train_max
            is_test = (h1 >= train_max) & (h1 < test_max)
            train = np.where(is_train)[0]
            test = np.where(is_test)[0]
            yield train, test

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.

        y : object
            Always ignored, exists for compatibility.

        groups : object
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits


class HashKFold(_BaseKFold, HashSplitMixin):
    """
    K-fold split cross-validator based on hashed labels.
    Pass labels as `groups` parameter when calling .split().
    Note that, particularly for small numbers of samples, fold sizes may differ.

    Provides train/test indices to split data in train/test sets. Split
    dataset into k consecutive folds. Each fold is then used once as a
    validation while the k - 1 remaining folds form the training set.

    Parameters
    ----------
    n_splits : int, default=3
        Number of folds. Must be at least 2.

    random_state : int, RandomState instance or None, optional, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """

    def __init__(self, n_splits=3, random_state=None):
        super(HashKFold, self).__init__(n_splits,
                                        shuffle=False,
                                        random_state=random_state)

    def _iter_test_indices(self, X, y, groups):
        if groups is None:
            raise ValueError("The 'groups' parameter should contain the labels to be hashed.")
        groups = check_array(groups, ensure_2d=False, dtype=str)

        n_samples = _num_samples(X)
        if len(groups) != n_samples:
            raise ValueError("'X' and 'groups' must have the same number of elements.")

        rng = check_random_state(self.random_state)

        # create initial set of int hashes and re-hash once for better distribution
        h0 = self._str_to_uint(groups)
        h1 = self._uni_hash(h0, rng=rng)

        n_splits = self.n_splits
        n_buckets = self.BUCKETS
        fold_sizes = (n_buckets // n_splits) * np.ones(n_splits, dtype=np.int)
        fold_sizes[:n_buckets % n_splits] += 1

        indices = np.arange(n_samples)
        current = 0
        for fold_size in fold_sizes:
            lower, upper = current, current + fold_size
            is_test = (h1 >= lower) & (h1 < upper)
            yield indices[is_test]
            current = upper


def _validate_shuffle_split(n_samples, test_size, train_size):
    """
    Validation helper to check if the test/test sizes are meaningful wrt to the
    size of the data (n_samples).
    This is mostly a copy of sklearn.model_selection._split._validate_shuffle_split() except it
    returns proportions instead of counts.
    """
    if (test_size is not None and
            np.asarray(test_size).dtype.kind == 'i' and
            test_size >= n_samples):
        raise ValueError('test_size=%d should be smaller than the number of '
                         'samples %d' % (test_size, n_samples))

    if (train_size is not None and
            np.asarray(train_size).dtype.kind == 'i' and
            train_size >= n_samples):
        raise ValueError("train_size=%d should be smaller than the number of"
                         " samples %d" % (train_size, n_samples))

    if test_size == "default":
        test_size = 0.1
    elif np.asarray(test_size).dtype.kind == 'i':
        test_size = test_size / n_samples

    if train_size is None:
        train_size = 1 - test_size
    elif np.asarray(train_size).dtype.kind == 'i':
        train_size = train_size / n_samples

    if test_size is None:
        test_size = 1 - train_size

    if train_size + test_size > 1:
        raise ValueError('The total proportion of train_size and test_size = %f, '
                         'should be smaller or equal to 1.0. Reduce test_size and/or '
                         'train_size.' % (train_size + test_size))

    return test_size, train_size
