#!/usr/bin/env mdl
import glob
import argparse
import itertools
from pathlib import Path
from typing import List, Tuple
from collections import Counter

# import ujson as json
import json
from tabulate import tabulate

import lovelive2
from lovelive2.client.dataset_client import DatasetClient
from nori2.utils import smart_open
DC = DatasetClient(lovelive2.BASE_URL)


parser = argparse.ArgumentParser()
parser.add_argument("eval_jsons", nargs='+', help="evaluation jsons")
parser.add_argument("--threshold", type=float, default=40.0)
parser.add_argument("--limit", type=int, default=10)
parser.add_argument("--sort", type=str, default="TP")
parser.add_argument("--ignore", help="ignore some atom benchmarks, splited by ','", default="")
parser.add_argument('--datasets', nargs='+', default=None,
    help='lovelive2 dataset_id(s) to be evaluated')
parser.add_argument('--datasets_attent', nargs='+', default=None,
    help='lovelive2 dataset_id(s) to be show')
args = parser.parse_args()


IGNORE_ATOMS = set(args.ignore.split(','))
MODE = "score" if args.threshold < 1 else "FN"
SORT = args.sort if args.sort in ["AUC", "TN"] else "TN"


def get_atoms(datasets):
    import lovelive2
    lc = lovelive2.Client(lovelive2.BASE_URL)
    atoms_list = []
    for ds_id in datasets:
        atoms_list.extend(lc.get_dataset_atoms_expanded(ds_id))
    atom_ids = set([i['atom_id'] for i in atoms_list])

    return atom_ids


def get_score_threshold(res: List[Tuple[str, float]], cnt: int):
    if len(res) <= cnt:
        return 0.0

    res = sorted(res)
    return res[-cnt]


# def process_json(pf: Path, atom_ids, atoms_ll_all_dict, atom_ids_attent):
def process_json(pf: Path, val_label):
    val_score = dict()
    with open(str(pf)) as jf:
        val_result = json.load(jf)['scores']
        for re in val_result:
            val_score[re['atom_id']] = re['score']

    attack_list, genuine_list = [], []

    for atom in val_label.keys():
        label = float(val_label[atom])
        score = float(val_score[atom])
        # score = 1-score
        if label == 1:  # genuine
            genuine_list.append(score)
        elif label == 0:  # ATTACK
            attack_list.append(score)

    if MODE == "FN":
        thres = get_score_threshold(
            attack_list, int(args.threshold)
        )
    else:
        thres = args.threshold

    cnter = Counter()
    for k, score_list in [
        ('tn', attack_list),
        ('fn', genuine_list),
    ]:
        for score in score_list:
            if score > thres:
                cnter[k] += 1

    tn, fn = cnter['tn'], cnter['fn']

    # auc  死人漏检一定的情况下统计真人误杀
    recall_list = []  # zhen ren tong guo
    thre_list = []
    p_scores = genuine_list.copy()
    p_scores.sort()
    n_scores = attack_list.copy()
    n_scores.sort()
    # for i in range(len(p_scores)):
    j = len(p_scores)
    for i in range(len(n_scores)):
        # print(n_scores)
        # while j < len(p_scores) and p_scores[j] > n_scores[i]:
        while j > 0 and p_scores[-j] < n_scores[i]:
            j -= 1
        recall_list.append(j)
        thre_list.append(n_scores[i])
    # auc=sum(recall_list)/float(len(n_scores)*len(recall_list))
    start1 = max(0, len(recall_list)//6000)
    end1 = max(0, len(recall_list)//1000)
    if start1 == end1:
        start1 = 0
        end1 = len(recall_list)
    # auc = sum(recall_list[start1:end1])/float(max(1, len(p_scores)*len(recall_list[start1:end1])))
    auc = sum(recall_list[-end1:-start1])/float(max(1, len(p_scores)*len(recall_list[-end1:-start1])))

    # get acc of attent benchmark
    num_right = 0
    # num_total = len(res_attent['scores'])
    num_total = len(val_score)
    # for atom in res_attent['scores']:
    #     label = atoms_ll_all_dict[atom['atom_id']]
    #     score = 1-atom['score']
    for atom in val_label.keys():
        label = float(val_label[atom])
        score = float(val_score[atom])

        if score == -1:
            continue
        if label == 0 and score < thres:  # GENUINE
            num_right = num_right + 1
            #print('score_0: ', score)

        elif label == 1 and score > thres:  # ATTACK
            num_right = num_right + 1
    if num_total == 0:
        acc = 1
    else:
        acc = float(num_right) / float(num_total)
    return {
        "threshold": thres,
        "TN": tn,
        "FN": fn,
        "AUC": auc,
        "ACC": 1-acc,
    }


def main():
    eval_jsons = itertools.chain(*[glob.glob(pf) for pf in args.eval_jsons])
    eval_jsons = [Path(pf) for pf in eval_jsons]

    val_lable_path = '../../../dataset/val_private_list.txt'
    # val_score_path = 'train_log/eval/val_224_192_epoch_0_1.json'
    # eval_jsons = glob.glob('train_log/eval/*.json')

    atoms_all = []
    val_score = dict()
    val_label = dict()

    with open(val_lable_path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            line = line.split(' ')
            val_label[line[1]] = line[-1]
            atoms_all.append(line[1])


    num = 0
    for atom in val_label.keys():
        if atom not in val_score.keys():
            print(atom)
        num += 1

    all_results = [
        (pf.name, process_json(pf, val_label), )
        for pf in eval_jsons
    ]


    '''
    atom_ids = None if args.datasets is None else get_atoms(args.datasets)
    atom_ids_attent = None if args.datasets is None else get_atoms(args.datasets_attent)

    atoms_ll_all = []
    for dataset in args.datasets:
        atoms_ll_all.extend(DC.dataset_get_atoms(dataset, compact=False))
    for dataset in args.datasets_attent:
        atoms_ll_all.extend(DC.dataset_get_atoms(dataset, compact=False))
    atoms_ll_all_dict = dict()
    for atoms_ll in atoms_ll_all:
        atoms_ll_all_dict[atoms_ll['atom_id']] = atoms_ll['label']
    '''

    '''
    all_results = [
        (pf.name, process_json(pf, atom_ids, atoms_ll_all_dict, atom_ids_attent), )
        for pf in eval_jsons
    ]
    '''

    all_results = sorted(
        all_results, key=lambda x: -x[1]["ACC"]
    )
    table = [
        [name, r["TN"], r["FN"], 1-r["threshold"], r["AUC"], r["ACC"]]
        for name, r in all_results[:args.limit]
    ]

    print(tabulate(table, headers=("name", "FN", "TN",
                                   "threshold attack", "AUC[0.1%~1%]", "ACC")))

if __name__ == "__main__":
    main()

# vim: ts=4 sw=4 sts=4 expandtab

