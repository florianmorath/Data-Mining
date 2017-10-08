# MIT License
#
# Copyright (c) 2016 Olivier Bachem
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Map Reduce Framework for Python (Task1, DM2016)

https://project.las.ethz.ch/task1

1. The framework first loads the python source code specified in `sourcefile`.
  The source should contain a `mapper` and a `reducer` function.
  The `mapper(key, value)` function takes as input a (key, value) tuple where
  key is None and value is a string. It should yield (key, value) pairs.
  The `reducer(key, value)` function takes as input a key and a list of values.
  It should yield (key, value) pairs.

The source of a word count program looks as follows:

>>> def mapper(key, value):
>>>     for i in value.split():
>>>         yield i, 1
>>>
>>> def reducer(key, values):
>>>     yield key, sum(values)

2. Implementation details:
  - Keys produced by the mapper *must* only be strings, ints or floats.
  - The (key, value) pairs produced by the mapper must be pickable by cPickle
    (https://docs.python.org/2/library/pickle.html).
  - The (key, value) pairs produced by the reducer must be convertable to JSON
    (https://docs.python.org/2/library/json.html?).

3. The training files are then used to run the example.

4. For debugging purposes, logging to STDERR can be enabled using the `--log` or `-l` flag.

(c) 2016 Olivier Bachem
"""
from collections import defaultdict
import argparse
import glob
import imp
import multiprocessing
import os
import random
import resource
import sys
from sets import Set

from itertools import chain, islice
try:
    import simplejson as json
except:
    import json
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def chunks(iterable, size=10):
    iterator = iter(iterable)
    for first in iterator:
        yield list(chain([first], islice(iterator, size - 1)))


def isolated_batch_call(f, arguments):
    """Calls the function f in a separate process and returns list of results"""

    def lf(q):
        r = []
        for args in arguments:
            r.extend(list(f(*args)))
        q.put(r)

    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=lf, args=(q, ))
    p.start()
    # Set timeout for individual mappers and reducers.
    res = q.get()
    p.join()
    return res


def mapreduce(input, mapper, reducer, batch_size=50, log=False):
    """Python function that runs a worst-case map reduce framework on the provided data

    Args:
      input -- list or generator of (key, value) tuple
      mapper -- function that takes (key, value) pair as input and returns iterable key-value pairs
      reducer -- function that takes key + list of values and outputs (key, value) pair
      log -- whether log messages should be generated (default: False)

    Returns list of (key, value pairs)
    """
    # Set initial random seed
    random.seed(0)
    # Run mappers
    if log: logger.info("Starting mapping phase!")
    d = defaultdict(list)
    for pairs_generator in chunks(input, batch_size):
        pairs = list(pairs_generator)
        if log:
            for k, v in pairs:
                logger.debug("  Running mapper for '%s' key with value '%s'...", k, v)
        for k2, v2 in isolated_batch_call(mapper, pairs):
            if log: logger.debug("    Mapper produced (%s, %s) pair...", k2, v2)
            if not isinstance(k2, (basestring, int, float)):
                raise Exception("Keys must be strings, ints or floats (provided '%s')!"% k2)
            d[k2].append(v2)
    if log: logger.info("Finished mapping phase!")
    # Random permutations of both keys and values.
    keys = d.keys()
    random.shuffle(keys)
    for k in keys:
        random.shuffle(d[k])
    # Run reducers
    if log: logger.info("Starting reducing phase!")
    res = []
    for key_chunk in chunks(keys, batch_size):
        if log:
            for k in key_chunk:
                logger.debug("  Running reducer for '%s' key with values '%s'...", k, d[k])
        r = isolated_batch_call(reducer, ([k, d[k]] for k in key_chunk))
        for k, v in r:
            if log: logger.debug("    Reducer produced (%s, %s) pair...", k, v)
        res.extend(r)
    if log: logger.info("Finished reducing phase!")
    return res


def yield_pattern(path):
    """Yield lines from each file in specified folder"""
    for i in glob.iglob(path):
        if os.path.isfile(i):
            with open(i, "r") as fin:
                for line in fin:
                    yield None, line


def import_from_file(f):
    """Import code from the specified file"""
    mod = imp.new_module("mod")
    exec f in mod.__dict__
    return mod


def evaluate(reported_duplicates, true_duplicates):
    tp = fp = fn = 0
    seen = Set()
    # Count true positives and false positives.
    for pair in reported_duplicates:
        # Skip already seen pairs.
        if pair in seen: continue
        seen.add(pair)
        if pair in true_duplicates:
            tp += 1
        else:
            fp += 1

    # Count false negatives.
    for pair in true_duplicates:
        if pair not in reported_duplicates:
            fn += 1

    logging.info("TP=%d, FP=%d, FN=%d", tp, fp, fn)
    
    # If either precision or recall are zero, return zero.
    if (tp + fp == 0) or (tp + fn == 0):
        return 0
    
    precision = 1.0 * tp / (tp + fp)
    recall = 1.0 * tp / (tp + fn)
    if precision + recall == 0:
        return 0

    f1 = 2.0 * precision * recall / (precision + recall)
    logging.info("Precision: %.3f, recall: %.3f, F1 score: %.3f",
                 precision, recall, f1)
    return f1


def run(sourcestring, input_pattern, duplicates_file, batch, log):
    mod = import_from_file(sourcestring)
    input = yield_pattern(input_pattern)

    reported_duplicates = Set()
    for output in mapreduce(input, mod.mapper, mod.reducer, batch, log):
        reported_duplicates.add(output)

    true_duplicates = Set()
    with open(duplicates_file, "r") as inf:
        for line in inf:
            a, b = line.strip().split(",")
            true_duplicates.add((int(a), int(b)))

    return evaluate(reported_duplicates, true_duplicates)



def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        'source_file', help='.py file with mapper and reducer function')
    parser.add_argument(
        '--log', '-l', help='Enable logging for debugging', action='store_true')
    args = parser.parse_args()
    INPUT_PATTERN = "data/handout_shingles.txt"
    DUPLICATES = "data/handout_duplicates.txt"
    
    
    BATCH = 50
    
    with open(args.source_file, "r") as fin:
        source = fin.read()

    print run(source, INPUT_PATTERN, DUPLICATES, BATCH, args.log)

if __name__ == "__main__":
    main()
