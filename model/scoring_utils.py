import os
import re
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve,auc
from tqdm import tqdm
import pickle
from model.dataset import shanghaitech_hr_skip
from model.dataset import SCENE_SH, SCENE_UB

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def score_dataset(score, metadata, args=None):
    gt_arr, scores_arr , scores_arr_cil= get_dataset_scores(score, metadata, args=args)
    scores_arr = smooth_scores(scores_arr)
    scores_arr_cil = smooth_scores(scores_arr_cil)
    gt_np = np.concatenate(gt_arr)
    scores_np = np.concatenate(scores_arr)
    scores_cil_np = np.concatenate(scores_arr_cil)

    min_val = np.min(scores_np)
    max_val = np.max(scores_np)
    if max_val > min_val:
        scores_np = (scores_np - min_val) / (max_val - min_val)
    else:
        scores_np = np.zeros_like(scores_np)

    min_val = np.min(scores_cil_np)
    max_val = np.max(scores_cil_np)
    if max_val > min_val:
        scores_cil_np = (scores_cil_np - min_val) / (max_val - min_val)
    else:
         scores_cil_np= np.zeros_like(scores_cil_np)
    
    auc_roc, pr_auc, eer, eer_th = score_auc(scores_np, gt_np)
    auc_roc_cil, pr_auc_cil, eer_cil, eer_th_cil = score_auc(scores_cil_np, gt_np)
    return {
        'auc_roc': auc_roc,
        'auc_roc_cil': auc_roc_cil,
        'scores_np': scores_np,
        'auc_pr': pr_auc,
        'auc_pr_cil': pr_auc_cil,
        'eer': eer,
        'eer_cil': eer_cil,
        'eer_th': eer_th,
        'eer_th_cil': eer_th_cil,
        'gt_np': gt_np
    }


def get_dataset_scores(scores, metadata, args=None):
    dataset_gt_arr = []
    dataset_scores_arr = []
    dataset_scores_arr_cil = []
    metadata_np = np.array(metadata)

    if args.dataset == 'UBnormal':
        pose_segs_root = 'data/UBnormal/pose/test'
        clip_list = os.listdir(pose_segs_root)
        clip_list = sorted(
            fn.replace("alphapose_tracked_person.json", "tracks.txt") for fn in clip_list if fn.endswith('.json'))
        per_frame_scores_root = 'data/UBnormal/gt/'

    else: 
        per_frame_scores_root = 'data/ShanghaiTech/gt/test_frame_mask/'
        clip_list = os.listdir(per_frame_scores_root)
        clip_list = sorted(fn for fn in clip_list if fn.endswith('.npy'))


    print("Scoring {} clips".format(len(clip_list)))
    for clip in tqdm(clip_list):
        clip_gt, clip_score ,clip_score_cil= get_clip_score(scores, clip, metadata_np, metadata, per_frame_scores_root, args)
        if clip_score is not None:
            dataset_gt_arr.append(clip_gt)
            dataset_scores_arr.append(clip_score)
            dataset_scores_arr_cil.append(clip_score_cil)

    for scores_arr in [dataset_scores_arr, dataset_scores_arr_cil]:
        scores_np = np.concatenate(scores_arr, axis=0)
        scores_np[scores_np == np.inf] = scores_np[scores_np != np.inf].max()
        scores_np[scores_np == -np.inf] = scores_np[scores_np != -np.inf].min()
        index = 0
        for i in range(len(scores_arr)):
            for t in range(scores_arr[i].shape[0]):
                scores_arr[i][t] = scores_np[index]
                index += 1

    return dataset_gt_arr, dataset_scores_arr ,dataset_scores_arr_cil



def score_auc(scores_np, gt):
    scores_np[scores_np == np.inf] = scores_np[scores_np != np.inf].max()
    scores_np[scores_np == -1 * np.inf] = scores_np[scores_np != -1 * np.inf].min()
    auc_roc = roc_auc_score(gt, scores_np)
    precision, recall, thresholds = precision_recall_curve(gt, scores_np)
    auc_precision_recall = auc(recall, precision)
    fpr, tpr, threshold = roc_curve(gt, scores_np, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return auc_roc, auc_precision_recall, eer, eer_threshold


def smooth_scores(scores_arr, sigma=7):
    for s in range(len(scores_arr)):
        for sig in range(1, sigma):
            scores_arr[s] = gaussian_filter1d(scores_arr[s], sigma=sig)
    return scores_arr

   
def get_clip_score(scores, clip, metadata_np, metadata, per_frame_scores_root, args):
    if args.dataset == 'UBnormal':
        type, scene_id, clip_id = re.findall('(abnormal|normal)_scene_(\d+)_scenario(.*)_tracks.*', clip)[0]
        clip_id = type + "_" + clip_id
        
    else:
        scene_id, clip_id = [int(i) for i in clip.replace("label", "001").split('.')[0].split('_')]
        if shanghaitech_hr_skip((args.dataset == 'ShanghaiTech-HR'), scene_id, clip_id):
            return None, None,None
    clip_metadata_inds = np.where((metadata_np[:, 1] == clip_id) &
                                  (metadata_np[:, 0] == scene_id))[0]
    clip_metadata = metadata[clip_metadata_inds]
    clip_fig_idxs = set([arr[2] for arr in clip_metadata])
    clip_res_fn = os.path.join(per_frame_scores_root, clip)
    clip_gt = np.load(clip_res_fn)
    if args.dataset != "UBnormal":
        clip_gt = np.ones(clip_gt.shape) - clip_gt  # 1 is normal, 0 is abnormal
    scores_zeros = np.ones(clip_gt.shape[0]) * np.inf
    if len(clip_fig_idxs) == 0:
        clip_person_scores_dict = {0: np.copy(scores_zeros)}
    else:
        clip_person_scores_dict = {i: np.copy(scores_zeros) for i in clip_fig_idxs}

    for person_id in clip_fig_idxs:
        person_metadata_inds = \
            np.where(
                (metadata_np[:, 1] == clip_id) & (metadata_np[:, 0] == scene_id) & (metadata_np[:, 2] == person_id))[0]
        pid_scores = scores[person_metadata_inds]
        
        pid_frame_inds = np.array([metadata[i][3] for i in person_metadata_inds]).astype(int)
        clip_person_scores_dict[person_id][pid_frame_inds + int(args.seg_len / 2)] = pid_scores

    clip_ppl_score_arr = np.stack(list(clip_person_scores_dict.values()))
    clip_score = np.amin(clip_ppl_score_arr, axis=0)
    lam = 0.7 if 'shanghai' in args.dataset.lower() else 4.0
    scene_idx = int(scene_id) - 1
    scene_score = SCENE_SH[scene_idx] if 'shanghai' in args.dataset.lower() else SCENE_UB[scene_idx]
    clip_score_cil = np.copy(clip_score)
    if 'shanghai' in args.dataset.lower():
            clip_score_cil += scene_score * lam
    else:
            clip_score_cil -= scene_score * lam
    if args.eval_ubnormal_hr:
        hr_ubnormal_masknpy_filename=generate_hr_ubnormal_masknpy_filename(scene_id,clip_id)

        root_mask = "data/UBnormal/pose/testmask/test_frame_mask"
        mask_clip_path = os.path.join(root_mask, hr_ubnormal_masknpy_filename)
        processing_mask = np.load(mask_clip_path)
        clip_score = clip_score[processing_mask == 1]
        clip_score_cil = clip_score_cil[processing_mask == 1]
        clip_gt    = clip_gt[processing_mask == 1]

    return clip_gt, clip_score, clip_score_cil


def generate_hr_ubnormal_masknpy_filename(scene_id, clip_id):
    keyword_mapping = {
        "fog": "51",
        "fire": "52",
        "smoke": "53"
    }

    import re
   
    match = re.match(r'(abnormal|normal)__?(\d+)_?(.*)', clip_id)
    if not match:
        raise ValueError(f"Invalid clip_id format: {clip_id}")


    clip_type, clip_number, clip_part = match.groups()

    
    prefix = "0" if clip_type == "abnormal" else "1"


    scene_id_formatted = f"{int(scene_id):02d}"
    clip_number_formatted = f"{int(clip_number):02d}"


    if not clip_part: 
        clip_part_formatted = "00"
    elif clip_part.isdigit(): 
        clip_part_formatted = f"{int(clip_part):02d}"
    elif clip_part in keyword_mapping:  
        clip_part_formatted = keyword_mapping[clip_part]
    else: 
        raise ValueError(f"Unrecognized clip_part: {clip_part}")

  
    return f"{prefix}{scene_id_formatted}_{clip_number_formatted}{clip_part_formatted}.npy"
