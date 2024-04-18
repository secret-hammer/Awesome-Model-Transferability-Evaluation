import argparse
import os
import json
import numpy as np
from scipy.stats import pearsonr, spearmanr
import evaluate_results as er


def evaluate_predicted_measure(metric):
    gt = er.GT
    tt = getattr(er,metric)
    
    result = {}
    for factor_name in tt.keys():
        if factor_name not in ['network_width', 'network_depth', 'network_layer', 'early_stop', 'diversity_of_train', 'density_of_train']:
            continue

        gtf, ttf = np.array(gt[factor_name]), np.array(tt[factor_name])
        result[factor_name] = {}
        result[factor_name]['accracy'] = {}
        result[factor_name]['accracy']['pearson'],  result[factor_name]['accracy']['spearman'] = evaluate_accuracy(gtf.T, ttf.T)
        result[factor_name]['effectiveness'] = {}
        result[factor_name]['effectiveness']['pearson'],  result[factor_name]['effectiveness']['spearman'] = evaluate_accuracy(gtf, ttf)

    
    # compute the mean of all factors
    acc_sum_p, acc_sum_s, sta_sum_p, sta_sum_s, eff_sum_p, eff_sum_s = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for fn, score in result.items():
        acc_sum_p += score['accracy']['pearson'][1]
        acc_sum_s += score['accracy']['spearman'][1]
        sta_sum_p += score['accracy']['pearson'][2]
        sta_sum_s += score['accracy']['spearman'][2]
        eff_sum_p += score['effectiveness']['pearson'][1]
        eff_sum_s += score['effectiveness']['spearman'][1]
    
    result['acc_mean'], result['sta_mean'], result['eff_mean'] = {}, {}, {}
    factor_num = len(result.keys()) - 3
    result['acc_mean']['pearson'] = acc_sum_p / factor_num
    result['acc_mean']['spearman'] = acc_sum_s / factor_num
    
    result['sta_mean']['pearson'] = sta_sum_p / factor_num
    result['sta_mean']['spearman'] = sta_sum_s / factor_num

    result['eff_mean']['pearson'] = eff_sum_p / factor_num
    result['eff_mean']['spearman'] = eff_sum_s / factor_num

    return result

'''
    input : gt: [3, 3], tt: [3, 3]
    return: list
            [([p1, p2, p3], p_mean, p_var), ([s1, s2, s3], s_mean, s_var)]
    Pearson系数, Spearman系数均按行计算
'''
def evaluate_accuracy(gt: np.ndarray, tt: np.ndarray):
    result = []
    pearsons = []
    spearmans = []

    for i in range(gt.shape[0]):
        pearsons.append(pearsonr(gt[i], tt[i])[0])
        spearmans.append(spearmanr(gt[i], tt[i])[0])

    print(pearsons)
    print(np.var(np.array(pearsons), ddof=1))
    result.append((pearsons, np.mean(np.array(pearsons)), 1.0 - np.var(np.array(pearsons), ddof=1)))
    result.append((spearmans, np.mean(np.array(spearmans)), 1.0 - np.var(np.array(spearmans), ddof=1)))
    return result


parser = argparse.ArgumentParser()
parser.add_argument('--metric', type=str)

if __name__ == '__main__':  

    seed=10
    np.random.seed(seed)
    
    args = parser.parse_args()
    log_path = f'/nfs4/wjx/transferbility/experiment/log/QMAB/measure_score_acc/{args.metric}/result.json'
    path_list, _ = os.path.split(log_path)
    
    if not os.path.exists(path_list):
        os.makedirs(path_list)
    
    score = evaluate_predicted_measure(args.metric)
    # 保存为json格式
    with open(log_path, 'a') as f:
        json.dump(score, f, indent=4)
    
    print(f'Success! The result is saved in {log_path}')
