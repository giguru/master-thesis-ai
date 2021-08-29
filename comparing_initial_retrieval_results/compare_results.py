import json
from collections import defaultdict

import matplotlib.pyplot as plt

sets = {
    "Anserini, Direchlet, mu=2500": [
        './results_quretec_anserini_direchlet_mu-2500.json',
        './results_castorini-t5-base-canard_anserini_direchlet_mu-2500.json'
    ],
    "Anserini, BM25, default settings": [
        './results_qurectec_anserini_bm25_default.json',
        './results_castorini-t5-base-canard_anserini_bm25_default.json'
    ]
}


percentages = defaultdict(list)
for key in sets.keys():
    file_a, file_b = sets[key]
    with open(file_a, "r", encoding='utf-8') as f1:
        list_a = json.load(f1)

    with open(file_b, "r", encoding='utf-8') as f2:
        list_b = json.load(f2)

    for qid in list_a.keys():
        set_a = set(list_a[qid])
        set_b = set(list_b[qid])
        overlap = list(set_a & set_b)
        percentages[key].append(len(overlap) / len(set_a) * 100)

bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 101]
plt.title("Overlap initial retrieval")
plt.xlabel("% Percentage")
plt.ylabel("# Queries")
plt.hist(percentages.values(),
         bins=bins,
         histtype='bar',
         label=list(percentages.keys()))
plt.legend(loc="upper left")
plt.show()