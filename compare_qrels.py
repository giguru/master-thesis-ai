from typing import List
from dataclasses import dataclass, field
import json
import pandas
import pytrec_eval
import matplotlib.pyplot as plt
import logging

__all__ = ['CompareData', '']

logger = logging.getLogger(__name__)
DELTA = u'Δ'


@dataclass
class CompareData:
    run_a: dict
    run_b: dict
    data: pandas.DataFrame
    name_run_a: str = field(default_factory=lambda:"Run A")
    name_run_b: str = field(default_factory=lambda:"Run B")
    bins: List = field(default_factory=lambda:[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    def print_overlap_hist(self):
        percentages = []
        n_queries = len(self.run_a.keys())
        for qid in self.run_a.keys():
            set_a = set(self.run_a[qid].keys())
            set_b = set(self.run_b[qid].keys())
            overlap = list(set_a & set_b)
            percentages.append(len(overlap) / len(set_a) * 100)

        plt.title(f"Results overlap (n_queries={n_queries})")
        plt.xlabel("% Percentage of overlap")
        plt.ylabel("# Queries")
        plt.hist(percentages, bins=self.bins, histtype='bar', rwidth=0.8)
        plt.xticks(self.bins)
        plt.savefig('overlap.png')
        plt.show()

    def analyse(self, metric: str, limit_hard: float = 0.2, limit_outperforms: float = 0.4, limit_easy: float = 0.8):
        hard_queries = self.data[(self.data[f"{metric}_1"] < limit_hard) & (self.data[f"{metric}_2"] < limit_hard)]
        easy_queries = self.data[(self.data[f"{metric}_1"] > limit_easy) & (self.data[f"{metric}_2"] > limit_easy)]
        run_b_outperforms = self.data[(self.data[f"{DELTA}{metric}"] > limit_outperforms)]
        run_a_outperforms = self.data[(self.data[f"{DELTA}{metric}"] < -1*limit_outperforms)]

        print(f"{len(hard_queries)} hard queries ({metric} < {limit_hard}) for both methods: {hard_queries[['qid']].tolist()}")
        print(f"{len(easy_queries)} easy queries ({metric} > {limit_easy}) for both methods: {easy_queries[['qid']].tolist()}")
        print(f"{len(run_b_outperforms)} queries where {self.name_run_b} outperforms {self.name_run_a} ({DELTA}{metric} > {limit_outperforms}): {run_b_outperforms['qid'].tolist()}")
        print(f"{len(run_a_outperforms)} queries where {self.name_run_a} outperforms {self.name_run_b} ({DELTA}{metric} > {limit_outperforms}): {run_a_outperforms['qid'].tolist()}")

    def print_table(self, save_file="table.html"):
        html_file = open(save_file, "w")
        html_file.write(self.data.style.to_html(encoding="UTF-8"))
        html_file.close()

    def print_metric_graph(self, metric: str, save_file="metrics.png"):
        from pylab import plot, show, savefig, xlim, figure, \
            ylim, legend, boxplot, setp, axes, xlabel, title, ylabel

        # function for setting the colors of the box plots pairs
        def setBoxColors(bp):
            setp(bp['boxes'][0], color='blue')
            setp(bp['caps'][0], color='blue')
            setp(bp['caps'][1], color='blue')
            setp(bp['whiskers'][0], color='blue')
            setp(bp['whiskers'][1], color='blue')
            setp(bp['medians'][0], color='blue')

            setp(bp['boxes'][1], color='red')
            setp(bp['caps'][2], color='red')
            setp(bp['caps'][3], color='red')
            setp(bp['whiskers'][2], color='red')
            setp(bp['whiskers'][3], color='red')
            setp(bp['medians'][1], color='red')

        bin_values = []
        for b in range(len(self.bins) - 1):
            is_last_bound = self.bins[b + 1] == self.bins[-1]
            right_bound = self.bins[b + 1] + 1 if is_last_bound else self.bins[b + 1]
            bin_df = self.data[(self.data['percentage'] >= self.bins[b]) & (self.data['percentage'] < right_bound)]
            bin_values.append([
                bin_df[f"{metric}_1"].tolist(),
                bin_df[f"{metric}_2"].tolist()
            ])

        fig = figure()
        ax = axes()
        n_bins = len(bin_values)
        xticks, xticklabels = [], []
        for i in range(n_bins):
            l, r = (i*3)+1, (i*3)+2
            middle = (r+l) / 2
            xticks.append(middle)
            xticklabels.append(f"{self.bins[i]}-{self.bins[i+1]}")
            bp = boxplot(bin_values[i],
                         positions=[l, r],
                         widths=0.6,
                         showfliers=False)
            setBoxColors(bp)

        # set axes limits and labels
        title(f"{metric} ")
        ylabel(f"{metric} score")
        xlabel("% Percentage of overlap")
        xlim(0, n_bins * 3)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)

        # draw temporary red and blue lines and use them to create a legend
        hB, = plot([1, 1], 'b-')
        hR, = plot([1, 1], 'r-')
        legend((hB, hR), (self.name_run_a, self.name_run_b))
        hB.set_visible(False)
        hR.set_visible(False)

        savefig(save_file)
        show()

    def print_delta_metric_graph(self, metric: str, save_file='delta.png'):
        metric = DELTA + metric
        from pylab import plot, show, savefig, xlim, figure, \
            ylim, legend, boxplot, setp, axes, xlabel, title, ylabel

        # function for setting the colors of the box plots pairs
        def setBoxColors(bp):
            setp(bp['boxes'][0], color='blue')
            setp(bp['caps'][0], color='blue')
            setp(bp['caps'][1], color='blue')
            setp(bp['whiskers'][0], color='blue')
            setp(bp['whiskers'][1], color='blue')
            # setp(bp['fliers'][0], color='blue')
            # setp(bp['fliers'][1], color='blue')
            setp(bp['medians'][0], color='blue')

        bin_values = []
        for b in range(len(self.bins) - 1):
            is_last_bound = self.bins[b+1] == self.bins[-1]
            # Include the edge for the last bin as well
            right_bound = self.bins[b+1] + 1 if is_last_bound else self.bins[b+1]
            bin_df = self.data[(self.data['percentage'] >= self.bins[b]) & (self.data['percentage'] < right_bound)]
            bin_values.append([
                bin_df[f"{metric}"].tolist(),
            ])

        fig = figure()
        ax = axes()
        n_bins = len(bin_values)
        xticks, xticklabels = [], []
        for i in range(n_bins):
            l, r = (i*3)+1, (i*3)+2
            middle = (r + l) / 2
            xticks.append(middle)
            xticklabels.append(f"{self.bins[i]}-{self.bins[i+1]}")
            bp = boxplot(bin_values[i],
                         positions=[middle],
                         widths=1.0,
                         showfliers=False, showcaps=True
                         )
            setBoxColors(bp)

        # set axes limits and labels
        title(f"{metric} ")
        ylabel(f"{metric} score")
        xlabel("% Percentage of overlap")
        xlim(0, n_bins * 3)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        if save_file:
            savefig(save_file)
        show()

def compute_qrels_df(qrels: dict, topics: dict, file_a: str, file_b: str, metrics):
    with open(file_a, "r", encoding='utf-8') as f1:
        run_a = json.load(f1)
    with open(file_b, "r", encoding='utf-8') as f2:
        run_b = json.load(f2)

    if len(run_a.keys()) != len(run_b.keys()):
        raise ValueError('Both lists should have the same length')

    qrels_keys = list(qrels.keys())
    missing_keys_in_qrels_a = [k for k in run_a.keys() if k not in qrels_keys]
    if len(missing_keys_in_qrels_a) > 0:
        logger.warning(f"Keys in run_a that are missing qrels: {missing_keys_in_qrels_a}")

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)

    # The run should have the format {query_id: {doc_id: rank_score}}
    res_a = evaluator.evaluate(run_a)
    res_b = evaluator.evaluate(run_b)

    labels = ['Query ID', 'Percentage']
    qids = list(res_a.keys())[0]
    metric_labels = res_a[qids].keys()
    for metric_label in metric_labels:
        labels += [f"{metric_label}_1", f"{metric_label}_2", f"{DELTA}{metric_label}"]

    cols_per_qid = {}
    for qid in run_a.keys():
        if qid in missing_keys_in_qrels_a:
            continue
        set_a = set(run_a[qid].keys())
        set_b = set(run_b[qid].keys())
        overlap = list(set_a & set_b)
        percentage = len(overlap) / len(set_a) * 100

        cols_per_qid[qid] = {
            'qid': qid,
            'topic': topics[qid],
            'percentage': percentage,
        }
        for metric_label in metric_labels:
            cols_per_qid[qid][f"{metric_label}_1"] = res_a[qid][metric_label]
            cols_per_qid[qid][f"{metric_label}_2"] = res_b[qid][metric_label]
            cols_per_qid[qid][f"{DELTA}{metric_label}"] = res_b[qid][metric_label] - res_a[qid][metric_label]
    pandas.options.display.float_format = "{:,.3f}".format
    df = pandas.DataFrame.from_dict(cols_per_qid.values())
    return CompareData(data=df, run_a=run_a, run_b=run_b, name_run_a="QuReTec", name_run_b="Rewriting")
