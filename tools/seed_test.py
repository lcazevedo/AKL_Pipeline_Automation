"""EVoC seed sensitivity test (run from inside a run workspace, runs/<name>/).

Usage:  cd runs/<name> && python ../../tools/seed_test.py [seeds...]   (default: 42 7 123)

1. Refits EVoC (noise_level=0.12) on data/checkpoints/step6_embeddings.pkl for each seed,
   printing topics per layer and % noise.
2. For seed 42, applies the Step 6 noise reassignment (nearest topic centroid, cosine) and
   compares each saved layer (data/checkpoints/step6_layer_assignments.pkl) with the refit:
   ARI and exact identity. Only layers saved by Step 6 (3-5) can be compared.
"""
import pickle, sys, numpy as np, evoc
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics.pairwise import cosine_similarity

seeds = [int(s) for s in sys.argv[1:]] or [42, 7, 123]
E = pickle.load(open("data/checkpoints/step6_embeddings.pkl", "rb"))
A = pickle.load(open("data/checkpoints/step6_layer_assignments.pkl", "rb"))

def fit(seed):
    m = evoc.EVoC(noise_level=0.12, random_state=seed); m.fit(E)
    return [np.asarray(l).copy() for l in m.cluster_layers_]

def reassign(c):
    c = c.copy()
    te = np.array([E[c == t].mean(axis=0) for t in range(len(set(c)) - 1) if (c == t).any()])
    new = cosine_similarity(E, te).argmax(axis=1); m = c == -1; c[m] = new[m]
    return c

for seed in seeds:
    L = fit(seed)
    print("seed", seed, "topics", [len(set(l)) - (1 if -1 in l else 0) for l in L],
          "noise%", [round(float(100 * (l == -1).mean()), 1) for l in L])
    if seed == 42:
        for i, l in enumerate(L):
            if i in A:
                r = reassign(l); nn = l != -1
                print(" layer", i, "ARI reassigned", round(ari(A[i], r), 4),
                      "identical", bool((A[i] == r).all()), "ARI non-noise", round(ari(A[i][nn], l[nn]), 4))
            else:
                print(" layer", i, "no saved assignment")
