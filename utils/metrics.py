from functools import partial
import numpy as np
from numpy import random

from skimage import exposure
from skimage import img_as_float
from skimage.transform import resize
import math
import cv2

EPSILON = np.finfo('float').eps


def inner_worker(saliency_map, gt_s):
    # mground_truth = mground_truth.astype(np.float32)
    # saliency_map = saliency_map.astype(np.float32)
    gt_sal, gt_fix = gt_s[0], gt_s[1]
    # Calculate metrics
    aucj = AUC_Judd(saliency_map, gt_fix)
    kl = KLD(saliency_map, gt_sal)
    cc = CC(saliency_map, gt_sal)
    sim = SIM(saliency_map, gt_sal)
    nss = NSS(saliency_map, gt_fix)
    scores = {
        "AUC-J": aucj,
        "KL": kl,
        "CC": cc,
        "SIM": sim,
        "NSS": nss
    }
    return scores


def norm(s):
    return (s - s.min()) / (s.max() - s.min())


def offline_inner_worker(pred_path, gt_path_s, gt_path_f, sqr=False):
    pow = 2 if sqr else 1
    #     mground_truth = 255 * norm(cv2.imread(gt_path_s, cv2.IMREAD_GRAYSCALE))**pow
    mground_truth = cv2.imread(gt_path_s, cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    target_h, target_w = pred.shape
    if gt_path_f:
        fground_truth = cv2.imread(gt_path_f, cv2.IMREAD_GRAYSCALE)
        fground_truth = fground_truth.astype(np.float32)
        target_h, target_w = fground_truth.shape

    mground_truth = cv2.resize(mground_truth, (target_w, target_h))
    saliency_map = cv2.resize(pred, (target_w, target_h))

    mground_truth = mground_truth.astype(np.float32)
    saliency_map = saliency_map.astype(np.float32)

    # Calculate metrics
    aucj = 0
    nss = 0
#     if gt_path_f:
    fground_truth = mground_truth.copy()
#     fground_truth[fground_truth < 0.5] = 0.
    fground_truth[fground_truth < 200] = 0.
    fground_truth[fground_truth >= 200] = 1.
#     fground_truth[fground_truth >= 0] = 1.
    aucj = AUC_Judd(saliency_map, fground_truth)
    nss = NSS(saliency_map, fground_truth)
    kl = KLD(saliency_map, mground_truth)

    cc = CC(saliency_map, mground_truth)

    sim = SIM(saliency_map, mground_truth)

    scores = {
        "AUC-J": aucj,
        "KL": kl,
        "CC": cc,
        "SIM": sim,
        "NSS": nss
    }
    return scores


def normalize(x, method='standard', axis=None):
    x = np.array(x, copy=False)
    if axis is not None:
        y = np.rollaxis(x, axis).reshape([x.shape[axis], -1])
        shape = np.ones(len(x.shape))
        shape[axis] = x.shape[axis]
        if method == 'standard':
            res = (x - np.mean(y, axis=1).reshape(shape)) / np.std(y, axis=1).reshape(shape)
        elif method == 'range':
            res = (x - np.min(y, axis=1).reshape(shape)) / (np.max(y, axis=1) - np.min(y, axis=1)).reshape(shape)
        elif method == 'sum':
            res = x / np.float_(np.sum(y, axis=1).reshape(shape))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    else:
        if method == 'standard':
            res = (x - np.mean(x)) / np.std(x)
        elif method == 'range':
            res = (x - np.min(x)) / (np.max(x) - np.min(x))
        elif method == 'sum':
            res = x / float(np.sum(x))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    return res


def match_hist(image, cdf, bin_centers, nbins=256):
    image = img_as_float(image)
    old_cdf, old_bin = exposure.cumulative_distribution(image, nbins)
    new_bin = np.interp(old_cdf, cdf, bin_centers)
    out = np.interp(image.ravel(), old_bin, new_bin)
    return out.reshape(image.shape)


def KLD(p, q):
    p = normalize(p, method='sum')
    q = normalize(q, method='sum')
    return np.sum(np.where(p != 0, p * np.log((p + EPSILON) / (q + EPSILON)), 0))


def AUC_Judd(saliency_map, fixation_map, jitter=False):
    saliency_map = np.array(saliency_map, copy=False)
    fixation_map = np.array(fixation_map, copy=False) > 0.5
    # If there are no fixation to predict, return NaN
    if not np.any(fixation_map):
        # print('no fixation to predict')
        return np.nan
    # Make the saliency_map the size of the fixation_map
    if saliency_map.shape != fixation_map.shape:
        saliency_map = resize(saliency_map, fixation_map.shape, order=3, mode='constant')
    # Jitter the saliency map slightly to disrupt ties of the same saliency value
    if jitter:
        saliency_map += random.rand(*saliency_map.shape) * 1e-7
    # Normalize saliency map to have values between [0,1]
    saliency_map = normalize(saliency_map, method='range')

    S = saliency_map.ravel()
    F = fixation_map.ravel()
    S_fix = S[F]  # Saliency map values at fixation locations
    n_fix = len(S_fix)
    n_pixels = len(S)
    # Calculate AUC
    thresholds = sorted(S_fix, reverse=True)
    tp = np.zeros(len(thresholds) + 2)
    fp = np.zeros(len(thresholds) + 2)
    tp[0] = 0
    tp[-1] = 1
    fp[0] = 0
    fp[-1] = 1
    for k, thresh in enumerate(thresholds):
        above_th = np.sum(S >= thresh)  # Total number of saliency map values above threshold
        tp[k + 1] = (k + 1) / float(n_fix)  # Ratio saliency map values at fixation locations above threshold
        fp[k + 1] = (above_th - k - 1) / float(n_pixels - n_fix)  # Ratio other saliency map values above threshold
    return np.trapz(tp, fp)  # y, x


def AUC_Borji(saliency_map, fixation_map, n_rep=100, step_size=0.1, rand_sampler=None):
    saliency_map = np.array(saliency_map, copy=False)
    fixation_map = np.array(fixation_map, copy=False) > 0.5
    # If there are no fixation to predict, return NaN
    if not np.any(fixation_map):
        # print('no fixation to predict')
        return np.nan
    # Make the saliency_map the size of the fixation_map
    if saliency_map.shape != fixation_map.shape:
        saliency_map = resize(saliency_map, fixation_map.shape, order=3, mode='constant')
    # Normalize saliency map to have values between [0,1]
    saliency_map = normalize(saliency_map, method='range')

    S = saliency_map.ravel()
    F = fixation_map.ravel()
    S_fix = S[F]  # Saliency map values at fixation locations
    n_fix = len(S_fix)
    n_pixels = len(S)
    # For each fixation, sample n_rep values from anywhere on the saliency map
    if rand_sampler is None:
        r = random.randint(0, n_pixels, [n_fix, n_rep])
        S_rand = S[r]  # Saliency map values at random locations (including fixated locations!? underestimated)
    else:
        S_rand = rand_sampler(S, F, n_rep, n_fix)
    # Calculate AUC per random split (set of random locations)
    auc = np.zeros(n_rep) * np.nan
    print(n_rep)
    for rep in range(n_rep):
        thresholds = np.r_[0:np.max(np.r_[S_fix, S_rand[:, rep]]):step_size][::-1]
        tp = np.zeros(len(thresholds) + 2)
        fp = np.zeros(len(thresholds) + 2)
        tp[0] = 0
        tp[-1] = 1
        fp[0] = 0
        fp[-1] = 1
        for k, thresh in enumerate(thresholds):
            tp[k + 1] = np.sum(S_fix >= thresh) / float(n_fix)
            fp[k + 1] = np.sum(S_rand[:, rep] >= thresh) / float(n_fix)
        auc[rep] = np.trapz(tp, fp)
    return np.mean(auc)  # Average across random splits


def AUC_shuffled(saliency_map, fixation_map, other_map, n_rep=100, step_size=0.1):
    other_map = np.array(other_map, copy=False) > 0.5
    if other_map.shape != fixation_map.shape:
        raise ValueError('other_map.shape != fixation_map.shape')

    # For each fixation, sample n_rep values (from fixated locations on other_map) on the saliency map
    def sample_other(other, S, F, n_rep, n_fix):
        fixated = np.nonzero(other)[0]
        indexer = list(map(lambda x: np.random.permutation(x)[:n_fix], np.tile(range(len(fixated)), [n_rep, 1])))
        r = fixated[np.transpose(indexer)]
        S_rand = S[r]  # Saliency map values at random locations (including fixated locations!? underestimated)
        return S_rand

    return AUC_Borji(saliency_map, fixation_map, n_rep, step_size, partial(sample_other, other_map.ravel()))


def discretize_gt(gt):
    import warnings
    warnings.warn('can improve the way GT is discretized')
    return gt / 255.0


def auc_shuff(s_map, gt, other_map, splits=100, stepsize=0.1):
    gt = discretize_gt(gt)
    other_map = discretize_gt(other_map)

    num_fixations = np.sum(gt)

    x, y = np.where(other_map == 1)
    other_map_fixs = []
    for j in zip(x, y):
        other_map_fixs.append(j[0] * other_map.shape[0] + j[1])
    ind = len(other_map_fixs)
    assert ind == np.sum(other_map), 'something is wrong in auc shuffle'

    num_fixations_other = min(ind, num_fixations)

    num_pixels = s_map.shape[0] * s_map.shape[1]
    random_numbers = []
    for i in range(0, splits):
        temp_list = []
        t1 = np.random.permutation(ind)
        for k in t1:
            temp_list.append(other_map_fixs[k])
        random_numbers.append(temp_list)

    aucs = []
    # for each split, calculate auc
    for i in random_numbers:
        r_sal_map = []
        for k in i:
            r_sal_map.append(s_map[k % s_map.shape[0] - 1, k / s_map.shape[0]])
        # in these values, we need to find thresholds and calculate auc
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        r_sal_map = np.array(r_sal_map)

        # once threshs are got
        thresholds = sorted(set(thresholds))
        area = []
        area.append((0.0, 0.0))
        for thresh in thresholds:
            # in the salience map, keep only those pixels with values above threshold
            temp = np.zeros(s_map.shape)
            temp[s_map >= thresh] = 1.0
            num_overlap = np.where(np.add(temp, gt) == 2)[0].shape[0]
            tp = num_overlap / (num_fixations * 1.0)

            # fp = (np.sum(temp) - num_overlap)/((np.shape(gt)[0] * np.shape(gt)[1]) - num_fixations)
            # number of values in r_sal_map, above the threshold, divided by num of random locations = num of fixations
            fp = len(np.where(r_sal_map > thresh)[0]) / (num_fixations * 1.0)

            area.append((round(tp, 4), round(fp, 4)))

        area.append((1.0, 1.0))
        area.sort(key=lambda x: x[0])
        tp_list = [x[0] for x in area]
        fp_list = [x[1] for x in area]

        aucs.append(np.trapz(np.array(tp_list), np.array(fp_list)))

    return np.mean(aucs)


def NSS(saliency_map, fixation_map):
    s_map = np.array(saliency_map, copy=False)
    f_map = np.array(fixation_map, copy=False) > 0.5
    if not np.any(f_map):
        # print('no fixation to predict')
        return np.nan
    if s_map.shape != f_map.shape:
        s_map = resize(s_map, f_map.shape)
    # Normalize saliency map to have zero mean and unit std
    s_map = normalize(s_map, method='standard')
    # Mean saliency value at fixation locations
    return np.mean(s_map[f_map])


def CC(saliency_map1, saliency_map2):
    map1 = np.array(saliency_map1, copy=False)
    map2 = np.array(saliency_map2, copy=False)
    if map1.shape != map2.shape:
        map1 = resize(map1, map2.shape, order=3,
                      mode='constant')  # bi-cubic/nearest is what Matlab imresize() does by default
    # Normalize the two maps to have zero mean and unit std
    map1 = normalize(map1, method='standard')
    map2 = normalize(map2, method='standard')
    # Compute correlation coefficient
    return np.corrcoef(map1.ravel(), map2.ravel())[0, 1]


# saliencyMeasures.py
def SIM(saliency_map1, saliency_map2):
    map1 = np.array(saliency_map1, copy=False)
    map2 = np.array(saliency_map2, copy=False)
    if map1.shape != map2.shape:
        map1 = resize(map1, map2.shape, order=3,
                      mode='constant')  # bi-cubic/nearest is what Matlab imresize() does by default
    # Normalize the two maps to have values between [0,1] and sum up to 1
    map1 = normalize(map1, method='range')
    map2 = normalize(map2, method='range')
    map1 = normalize(map1, method='sum')
    map2 = normalize(map2, method='sum')
    # Compute histogram intersection
    intersection = np.minimum(map1, map2)
    return np.sum(intersection)


def genERP(i, j, N):
    val = math.pi / N
    # w_map[i+j*w] = cos ((j - (h/2) + 0.5) * PI/h)
    w = math.cos((j - (N / 2) + 0.5) * val)
    return w


def compute_map_ws(h, w):
    """calculate weights for the sphere, the function provide weighting map for a given video
        :img    the input original video
    """
    equ = np.zeros((h, w))

    for j in range(0, equ.shape[0]):
        for i in range(0, equ.shape[1]):
            equ[j, i] = genERP(i, j, equ.shape[0])
    equ = equ / equ.max()
    return equ
