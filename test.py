import numpy as np
from numpy.random import RandomState
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris

from hashsplit import HashShuffleSplit, HashKFold
from hashsplit.hashsplit import ignore_overflow, HashSplitMixin


def test_ignore_overflow():
    UINT64MAX = np.uint64(np.iinfo(np.uint64).max)
    one = np.uint64(1)
    # warn
    with pytest.warns(RuntimeWarning):
        UINT64MAX + one
    # don't warn
    with pytest.warns(None) as record:
        with ignore_overflow():
            UINT64MAX + one
    assert(len(record) == 0)
    # warn again
    with pytest.warns(RuntimeWarning):
        UINT64MAX + one


def test_HashSplitMixin():
    hsm = HashSplitMixin()
    labels = ['sample' + '%04d' % i for i in range(1000)]
    h0 = hsm._str_to_uint(labels)

    # make sure random seeds work as expected:
    # 1) identical, fresh seeds should yield identical results
    h1a = hsm._uni_hash(h0, RandomState(0))
    h1b = hsm._uni_hash(h0, RandomState(0))
    assert(np.array_equal(h1a, h1b))
    # 2) different seed should yield different results
    h1a = hsm._uni_hash(h0, RandomState(0))
    h1b = hsm._uni_hash(h0, RandomState(1))
    assert(not np.array_equal(h1a, h1b))
    # 2) re-using the same RandomState should yield different results
    rng = RandomState(0)
    h1a = hsm._uni_hash(h0, rng)
    h1b = hsm._uni_hash(h0, rng)
    assert(not np.array_equal(h1a, h1b))


def test_HashShuffleSplit():
    _common_checks(HashShuffleSplit)


def test_HashKFoldSplit():
    _common_checks(HashKFold)

    # in addition to the above, every label should appear exactly once in the
    # test set across all folds
    labels = np.array(['sample%04d' % i for i in range(1000)])
    X, y = _makeXy(labels)
    hkf = HashKFold(n_splits=5, random_state=0)
    test_labels = np.array([])

    for train, test in hkf.split(X, y, labels):
        test_labels = np.concatenate((test_labels, labels[test]))

    assert(np.array_equal(np.sort(labels), np.sort(test_labels)))


def _makeXy(labels):
    # helper function to create dummy X and y with dimensions matching labels
    X = np.random.rand(len(labels), 10)
    y = np.random.choice([0, 1], len(labels))
    return X, y


def _common_checks(splitter):
    """
    tests we want to run for all cross-validator
    """
    def make_instance(splitter):
        if splitter == HashShuffleSplit:
            return HashShuffleSplit(n_splits=3, test_size=0.2, random_state=0)
        elif splitter == HashKFold:
            return HashKFold(n_splits=3, random_state=0)

    def compare_runs(run1, run2):
        for r1, r2 in zip(run1, run2):
            if not np.array_equal(r1[0], r2[0]) or not np.array_equal(r1[1], r2[1]):
                return False
        return True

    labels = ['sample%04d' % i for i in range(1000)]
    X, y = _makeXy(labels)

    # two runs should yield the same result
    ss = make_instance(splitter)
    run1 = [(train, test) for train, test in ss.split(X, y, groups=labels)]
    run1b = [(train, test) for train, test in ss.split(X, y, groups=labels)]
    assert(compare_runs(run1, run1b) is True)

    # two instances  with the same random state should yield the same result
    ss2 = make_instance(splitter)
    run2 = [(train, test) for train, test in ss2.split(X, y, groups=labels)]
    assert(compare_runs(run1, run2) is True)

    # two instances with different random states shouldn't
    ss3 = make_instance(splitter)
    ss3.random_state = 1
    run3 = [(train, test) for train, test in ss3.split(X, y, groups=labels)]
    assert(compare_runs(run1, run3) is False)

    # given two data sets A and B with A ⊆ B:
    # train(A) ⊆ train(B) and test(A) ⊆ test(B)
    label_subset = labels[:800]
    X_subset, y_subset = _makeXy(label_subset)
    ss_sub = make_instance(splitter)
    for (train_sub, test_sub), (train, test) in zip(ss_sub.split(X_subset, y_subset,
                                                                 groups=label_subset),
                                                    ss.split(X, y, groups=labels)):
        assert(np.array_equal(np.intersect1d(train_sub, train), train_sub))
        assert(np.array_equal(np.intersect1d(test_sub, test), test_sub))

    # the same labels should always end up in the same set
    dup_labels = np.repeat(labels[:10], 10)
    dup_X, dup_y = _makeXy(dup_labels)
    ss_dup = make_instance(splitter)
    for train, test in ss_dup.split(dup_X, dup_y, groups=dup_labels):
        train_labels = dup_labels[train]
        test_labels = dup_labels[test]
        assert(len(np.intersect1d(train_labels, test_labels)) == 0)

    # make sure the cross-validator fits in the sklearn workflow
    iris = load_iris()
    labels = ['sample%03d' % i for i in range(iris.data.shape[0])]
    model = LogisticRegression()
    cross_val_score(model, iris.data, iris.target, cv=ss, groups=labels)
