import sys
import time
from os import listdir
from os.path import isfile, join
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import EnglishStemmer
from collections import OrderedDict
import pandas


vocab = OrderedDict()  # the features used in the classifier
stop_words_dict = {}


def get_file_names(path):
    return [path + "/" + f for f in listdir(path) if isfile(join(path, f))]


def remove_punct(in_str):
    import re
    return re.sub("[^\w]", " ", in_str)


def remove_stop_and_short_words(token_words, min_word_size=2):
    global stop_words_dict
    clean_words = []
    for word in token_words:
        if len(word) >= min_word_size and word not in stop_words_dict:
            clean_words.append(word)
    return clean_words


def stem_tokens(tokens, stemmer):
    return [stemmer.stem(token) for token in tokens]


def file_to_tokens(file_name, min_word_size, stemmer=None):
    with open(file_name) as f:
        file_string = f.read()
    tokens = remove_punct(file_string).split()
    tokens = remove_stop_and_short_words(tokens, min_word_size)
    if stemmer:
        return stem_tokens(tokens, stemmer)
    else:
        return tokens


def sort_vocab_by_freq():
    global vocab
    vocab = sorted(vocab.items(), key=lambda t: t[1], reverse=True)


def build_stop_words():
    global stop_words_dict
    with open('stopwords.txt') as f:
            stopwords = f.read().lower().split()
    stop_words_dict = dict(zip(stopwords, stopwords))


def buildvocab(skip_first_n, num_words, min_word_size, stemmer, debug=False):
    print "\nBuilding the dictionary..."
    global vocab
    global stop_words_dict
    build_stop_words()
    all_file_names = get_file_names('pos') + get_file_names('neg')
    # Build the vocab
    for file_name in all_file_names:
        # the below call to file_to_tokens does not need to stem, just leave
        # off the stemmer arg (Porter, Lancaster, English)
        tokens = file_to_tokens(file_name, min_word_size, stemmer)
        for token in tokens:
            if token in vocab:
                vocab[token] += 1
            else:
                vocab[token] = 1
    sort_vocab_by_freq()
    print "{0} words found, keeping {1} of them.".format(
        len(vocab), num_words)
    vocab = OrderedDict(vocab[skip_first_n:skip_first_n + num_words])
    # look at dictionary in debug mode
    if debug:
        words_per_page = 20
        vocab_debug_printer(words_per_page)
    print "Done."
    return vocab


def vocab_debug_printer(words_per_page):
    count = 0
    for word in vocab:
        print "\nVocabulary list is {} words long.".format(len(vocab))
        if count % words_per_page == 0:
            resp = raw_input(
                '''please press enter to see next {}, or press 's' to skip printing vocabulary
                '''.format(words_per_page))
            if resp == 's':
                break
        else:
            print word + " {}".format(vocab[word])
        count += 1
    print "\nWARNING: debug mode produces inaccurate dictionary build times!"


def vectorize(file_name, min_word_size, stemmer):
    # Create vector representation of review
    global vocab
    vector = np.zeros(len(vocab))
    tokens = file_to_tokens(file_name, min_word_size, stemmer)
    for token in tokens:
        if token in vocab:
            vector[vocab.keys().index(token)] += 1
    # print "The vector is:\n {}".format(vector)
    return vector


def make_classifier(min_word_size, stemmer):
    print "Training the classifier..."
    # Build X matrix of vector representations of review files,
    # and y vector of labels
    pos_file_names = get_file_names('pos')
    neg_file_names = get_file_names('neg')
    # m is the number of training examples
    m_pos = len(pos_file_names)
    m_neg = len(neg_file_names)
    m = m_pos + m_neg
    pos_labels = np.ones(m_pos)
    neg_labels = -np.ones(m_neg)
    y = np.concatenate((pos_labels, neg_labels), axis=0)
    # get dimensions of data
    dimensions = len(vocab)

    # initialize X
    X = np.zeros((m, dimensions))
    message = "{:.2%} percent done\r"
    # build X
    for i in xrange(m_pos):
        X[i, :] = vectorize(pos_file_names[i], min_word_size, stemmer)
        sys.stdout.write(message.format(i / float(m)))
        sys.stdout.flush()
    for j in xrange(m_neg):
        X[j + m_pos, :] = vectorize(neg_file_names[j], min_word_size, stemmer)
        sys.stdout.write(message.format((m_pos + j) / float(m)))
        sys.stdout.flush()
    # make the logistic regression function
    lr = LR()
    lr.fit(X, y)

    return lr


def test_classifier(lr, min_word_size, stemmer):
    print "Running test data through classifier..."
    global vocab
    test = np.zeros((len(listdir('test')), len(vocab)))
    test_file_names = []
    i = 0
    y = []
    for file_name in listdir('test'):
        test_file_names.append(file_name)
        test[i] = vectorize(join('test', file_name), min_word_size, stemmer)
        ind = int(file_name.split('_')[0][-1])
        y.append(1 if ind == 3 else -1)
        i += 1

    assert(sum(y) == 0)
    p = lr.predict(test)

    r, w = 0, 0
    for i, x in enumerate(p):
        if x == y[i]:
            r += 1
        else:
            w += 1
            print(test_file_names[i])
    print "Done\n"
    print "Correct on {0} reviews, wrong on {1} reviews.".format(r, w)
    print "That's {:.2%}.\n".format(float(r)/(r+w))
    return r


def classify(params, debug=False):
    skip_first_n, n, min_word_size, stemmer = params
    # buildvocab(skip_first_n, n, min_word_size, stemmer, debug=debug)
    lr = make_classifier(min_word_size, stemmer)
    return test_classifier(lr, min_word_size, stemmer)


def repeat_classify(repeat, params, debug=False):
    skip_first_n, n, min_word_size, stemmer = params
    vocab_start_time = time.clock()
    buildvocab(skip_first_n, n, min_word_size, stemmer, debug=debug)
    vocab_finish_time = time.clock()
    vocab_time = vocab_finish_time - vocab_start_time
    print "Time to build vocabulary: {}\n".format(vocab_time)
    idx = ['run{0}'.format(run) for run in xrange(1, repeat+1)]
    cols = [
        'correct', 'words', 'skip_first',
        'min_word_size', 'stemmer', 'elapsed_time']
    results = pandas.DataFrame(index=idx, columns=cols)
    for run in xrange(1, repeat+1):
        # classify_start_time includes vocab time as well
        classify_start_time = time.clock() - vocab_time
        find_best_runs(run, classify_start_time, results, params, repeat == 1)
    finish_time = time.clock()
    if repeat > 1:
        print "\nAll results:\n"
        print results
        print """
        * Note that for multi-runs the vocabulary is not rebuilt for each run,
          but the time to build the vocabulary *is* included in elapsed_time
          for each run.
        """
    print "Actual time for all runs: {}".format(
        finish_time-vocab_start_time)


def find_best_runs(run, start_time, results, params, single=False):
    skip, n, word_size, stemmer = params
    run_msg = (
        "Run {0} with parameters words={1}, skip_first={2}, "
        "min_word_size={3}, stemmer={4}"
    )
    run_msg = run_msg.format(run, n, skip, word_size, stemmer)
    run_id = 'run{}'.format(run)
    print "\n\n" + run_id + "\n" + run_msg
    correct = classify(params)
    finish_time = time.clock()
    time_elapsed = finish_time - start_time
    result = [correct, n, skip, word_size, stemmer, time_elapsed]
    print "Result of run {0}: {1}".format(run, result)
    prev_best = results.iloc[0, 0]
    if not single and run > 1 and correct > prev_best:
        print "New best found on {0} with {1} right.\n".format(
            run_id, correct)
    elif correct == prev_best:  # and not np.isnan(prev_best):
        print "Tie found on {0} with {1} right.\n".format(
            run_id, correct)
    run += 1
    results.loc[run_id] = result
    if not single:
        results.sort(
            columns='correct', ascending=False, inplace=True)
        print "Top 10 results so far:"
        print results[0:10]


def optimize(param_ranges):
    start_time = time.clock()
    min_n, max_n, n_gap = param_ranges[0:3]
    min_skip_first_n, max_skip_first_n, skips_gap = param_ranges[3:6]
    min_min_word_size, max_min_word_size, stemmer_types = param_ranges[6:9]
    print "\n\n"
    ns = max_n - min_n + 1
    skips = max_skip_first_n - min_skip_first_n + 1
    stemmers = len(stemmer_types)
    word_sizes = max_min_word_size - min_min_word_size + 1
    runs = (ns/n_gap) * (skips/skips_gap) * stemmers * word_sizes
    print "{} runs to do\n".format(runs)

    idx = ['run{0}'.format(run) for run in xrange(1, runs+1)]
    cols = ['correct', 'words', 'skip_first',
            'min_word_size', 'stemmer', 'elapsed_time']
    results = pandas.DataFrame(index=idx, columns=cols)
    run = 1
    for n in xrange(min_n, max_n+1, n_gap):
        for skip in xrange(min_skip_first_n, max_skip_first_n+1, skips_gap):
            for word_size in xrange(min_min_word_size, max_min_word_size+1):
                for stemmer in stemmer_types:
                    params = (skip, n, word_size, stemmer)
                    vocab_start_time = time.clock()
                    buildvocab(skip, n, word_size, stemmer)
                    vocab_finish_time = time.clock()
                    vocab_time = vocab_finish_time - vocab_start_time
                    print "Time to build vocabulary: {}".format(vocab_time)
                    find_best_runs(run, time.clock(), results, params)
                    run += 1
    finish_time = time.clock()
    print "Top 50 results:\n"
    print results[0:50]
    print "Time for all runs: {}".format(finish_time-start_time)


def get_single_run_params_from_user():
        n = raw_input("""
        How many words would you like to use? \n(Higher numbers will
        take longer, we suggest <= 10000)
        """)
        n = int(n)

        skip_first_n = raw_input("""
        How many of the most frequent words would you like to skip? (>=0)
        """)
        skip_first_n = int(skip_first_n)

        min_word_size = raw_input("""
        What is the minimum word size you want to use? (>=1)
        """)
        min_word_size = int(min_word_size)

        stem_response = raw_input("""
        Which stemmer would you like to use for normalization? \n
        Choose 0 for none, choose 1 for Porter, 2 for Lancaster, and 3 for
        Snowball English.
        """)
        if stem_response == '0':
            stemmer = None
        elif stem_response == '1':
            stemmer = PorterStemmer()
        elif stem_response == '2':
            stemmer = LancasterStemmer()
        elif stem_response == '3':
            stemmer = EnglishStemmer()
        return (skip_first_n, n, min_word_size, stemmer)


def get_multi_run_params_from_user():
        min_n = int(raw_input("\n\nMin number of words in vocab: "))
        max_n = int(raw_input("Max number of words in vocab: "))
        n_gap = int(raw_input("For each run increase words in vocab by: "))
        min_skip_first_n = int(raw_input(
            "Minimum top words in vocab to skip: "))
        max_skip_first_n = int(raw_input(
            "Maximum top words in vocab to skip: "))
        skips_gap = int(raw_input("For each run increase words skipped by: "))
        min_min_word_size = int(raw_input(
            "Min-min word size to include in vocab: "))
        max_min_word_size = int(raw_input(
            "Max-min word size to include in vocab: "))
        stemmer_types = []
        print("\nFor stemmer options, please respond y/n "
              "for all desired in optimization\n")
        if raw_input("Run without stemmer? (y/n): ") == 'y':
            stemmer_types.append(None)
        if raw_input("Run with Porter stemmer? (y/n): ") == 'y':
            stemmer_types.append(PorterStemmer())
        if raw_input("Run with Lancaster stemmer? (y/n): ") == 'y':
            stemmer_types.append(LancasterStemmer())
        if raw_input("Run with snowball English stemmer? (y/n): ") == 'y':
            stemmer_types.append(EnglishStemmer())
        return (min_n, max_n, n_gap, min_skip_first_n, max_skip_first_n,
                skips_gap, min_min_word_size, max_min_word_size, stemmer_types)


if __name__ == '__main__':
    has_args = len(sys.argv) > 1
    if has_args:
        mode = sys.argv[1]
        if mode == '--optimize':
            param_ranges = get_multi_run_params_from_user()
            optimize(param_ranges)
        elif mode == '--single-run':
            debug = raw_input(
                "\nWould you like to build dictionary in debug mode (y/n)) ")
            if debug == 'y':
                debug = True
            else:
                debug = False
            params = get_single_run_params_from_user()
            repeat_classify(1, params, debug)
        elif mode == '--multi-run':
            runs = raw_input(
                "\nHow many runs would you like with the same parameters? ")
            runs = int(runs)
            params = get_single_run_params_from_user()
            repeat_classify(runs, params)
        else:
            print """
            Sorry that is not an option, for interactive modes try:\n
            '--optimize' to run optimizer\n
            '--single-run' to run a single\n
            '--multi-run' to do multiple runs with the same parameters\n
            or with no command line args for a demo.
            """
    else:
        print "\nYou can also run in optimizer, single-run, and multi-run modes."
        print 'give command line argument "--options" for all options.'
        print "\n\nRunning demo..."
        skip_first_n = 2
        n = 9979
        min_word_size = 1
        stemmer = None
        params = (skip_first_n, n, min_word_size, stemmer)
        repeat_classify(1, params)