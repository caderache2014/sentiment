from IPython import parallel
from sklearn.datasets import fetch_20newsgroups_vectorized


def get_results():
    # get data
    data = fetch_20newsgroups_vectorized(remove=('headers',
                                                 'footers',
                                                 'quotes'))
    alphas = [1E-4, 1E-3, 1E-2, 1E-1]
    # set up dview for imports
    clients = parallel.Client()
    dview = clients[:]
    with dview.sync_imports():
        # doesn't seem to like import numpy as np, using numpy instead
        import numpy
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.cross_validation import cross_val_score
    dview.block = True
    # send data to clients
    dview['data'] = data
    # set up load balanced view for parallel processing
    lview = clients.load_balanced_view()
    # set blocking to True to get all results once processing is done
    lview.block = True
    results = lview.map(get_single_result, alphas)
    return results


def get_single_result(alpha):
    clf = MultinomialNB(alpha)
    result = (alpha, numpy.mean(cross_val_score(clf, data.data, data.target)))
    return result


if __name__ == '__main__':
    results = get_results()
    best_result = (0, 0)
    for result in results:
        if result[1] > best_result[1]:
            best_result = result
    print "\nThe best result is:"
    print "alpha = {}".format(best_result[0])
    print "score = {}%".format(round(best_result[1] * 100, 2))