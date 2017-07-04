import pickle

class A(object):
    pass

fname = 'a.pickle'
with open(fname, mode='wb') as fp:
    pickle.dump(A, fp)

with open(fname, mode='rb') as fp:
    a = pickle.load(fp)
    print(A.__name__)