#!/usr/bin/env mdl
import json
import numpy as np
import argparse


def get_nid2scores(scoreFilePath):
    nid_scores = dict()
    with open(scoreFilePath, 'r') as js:
        dataInfo = json.load(js)
        scoresInfo = dataInfo['scores']
        for scoreInfo in scoresInfo:
            nid = scoreInfo['atom_id']
            score = scoreInfo['score']
            nid_scores[nid] = score
    return nid_scores


def normalScore(score, thre):
    return 1/(1+np.exp(np.log(1/score-1)-np.log(1/thre-1)))


parser = argparse.ArgumentParser()
parser.add_argument("eval_jsons", type=str, help="evaluation jsons")
parser.add_argument("threshold", type=float, default=0.5)
args = parser.parse_args()


scoreFilePath = args.eval_jsons
thre = args.threshold
nid_scores = get_nid2scores(scoreFilePath)


val_list_f = open('../../../dataset/test_public_list.txt', 'r')
val_list = val_list_f.readlines()
val_list_f.close()
# file_nids = get_file2nid()
with open('result.txt', 'w') as fw:
    for val_name in val_list:
        # file_depth = val_name.split(' ')[1]
        nid = val_name.split(' ')[1]
        # nid = file_nids[file_rgb]
        # print(nid_scores)
        if(nid in nid_scores.keys()):
            score = nid_scores[nid]
            if score > 0 and score < 1:
                score = normalScore(float(score), thre)
                # score = np.abs(1-score)
                score = score
            elif score == 0:
                score = 1
            elif score == 1:
                score = 0
            else:
                print('error!!!')
        else:
            # score = 0.0
            print(nid)
            print('error')
            xxx
        fw.write(val_name.strip() + ' ' + str(score) + '\n')


# vim: ts=4 sw=4 sts=4 expandtab
