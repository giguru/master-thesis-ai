import datasets
from compare_qrels import compute_qrels_df

sets = {
    "sample": [
        './compare_qrels_sample_data/results_quretec_anserini_bm25_default.json',
        './compare_qrels_sample_data/results_castorini-t5-base-canard_anserini_bm25_default.json'
    ],
    "sample2": [
        './compare_qrels_sample_data/results_quretec_anserini_bm25_default.json',
        './compare_qrels_sample_data/results_perfect_quretec_anserini_bm25_default.json'
    ],
}

topics = datasets.load_dataset('uva-irlab/trec-cast-2019-multi-turn', 'topics', split="test")
qrels = datasets.load_dataset('uva-irlab/trec-cast-2019-multi-turn', 'qrels', split="test")
# Convert into the right data format
qrels = {d['qid']: {d['qrels']['docno'][i]: int(d['qrels']['relevance'][i]) for i in range(len(d['qrels']['docno']))} for d in qrels}
topics = {topic['qid']: topic['query'] for k, topic in enumerate(topics)}

used_set = sets['sample2']
res = compute_qrels_df(qrels, topics, used_set[0], used_set[1], {'recall.1000', 'recip_rank', 'map'})
res.print_analysis('recall_1000')
res.print_metric_graph('recall_1000')
res.print_delta_metric_graph('recall_1000')
res.print_overlap_hist()
res.print_table()
