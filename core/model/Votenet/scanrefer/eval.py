import os
import sys
import json
import pickle
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
# from lib.config import CONF
PATH_OUTPUT = os.getcwd()
from models.ap_helper import  APCalculator, parse_predictions, parse_groundtruths
from .loss_helper import get_loss
from .eval_helper import get_eval
# from models.refnet import RefNet
# from data.scannet.model_util_scannet import ScannetDatasetConfig


def eval_ref(args, model, dataloader, DC): # DC: config
    print("evaluate localization...")
    # # constant
    # DC = ScannetDatasetConfig()

    # # init training dataset
    # print("preparing data...")
    # scanrefer, scene_list = get_scanrefer(args)
    scanrefer = dataloader.dataset.scanrefer

    # # dataloader
    # _, dataloader = get_dataloader(args, scanrefer, scene_list, "val", DC)

    # # model
    # model = get_model(args, DC)

    # config
    POST_DICT = {
        "remove_empty_box": True, 
        "use_3d_nms": True, 
        "nms_iou": 0.25,
        "use_old_type_nms": False, 
        "cls_nms": True, 
        "per_class_proposal": True,
        "conf_thresh": 0.05,
        "dataset_config": DC
    } if not args.no_nms else None

    # random seeds
    seeds = [args.seed] + [2 * i for i in range(args.repeat - 1)]

    # evaluate
    print("evaluating...")
    score_path = os.path.join(PATH_OUTPUT, args.folder, "scores.p")
    pred_path = os.path.join(PATH_OUTPUT, args.folder, "predictions.p")
    gen_flag = (not os.path.exists(score_path)) or args.force or args.repeat > 1
    if gen_flag:
        ref_acc_all = []
        ious_all = []
        masks_all = []
        others_all = []
        lang_acc_all = []
        for seed in seeds:
            # reproducibility
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            np.random.seed(seed)

            print("generating the scores for seed {}...".format(seed))
            ref_acc = []
            ious = []
            masks = []
            others = []
            lang_acc = []
            predictions = {}
            for data in tqdm(dataloader):
                for key in data:
                    data[key] = data[key].cuda()

                # feed
                data = model._forward(data)
                _, data = get_loss(
                    data_dict=data, 
                    config=DC, 
                    detection=True,
                    reference=True, 
                    use_lang_classifier=not args.no_lang_cls
                )
                data = get_eval(
                    data_dict=data, 
                    config=DC,
                    reference=True, 
                    use_lang_classifier=not args.no_lang_cls,
                    use_oracle=args.use_oracle,
                    use_cat_rand=args.use_cat_rand,
                    use_best=args.use_best,
                    post_processing=POST_DICT
                )

                ref_acc += data["ref_acc"]
                ious += data["ref_iou"]
                masks += data["ref_multiple_mask"]
                others += data["ref_others_mask"]
                lang_acc.append(data["lang_acc"].item())

                # store predictions
                ids = data["scan_idx"].detach().cpu().numpy()
                for i in range(ids.shape[0]):
                    idx = ids[i]
                    scene_id = scanrefer[idx]["scene_id"]
                    object_id = scanrefer[idx]["object_id"]
                    ann_id = scanrefer[idx]["ann_id"]

                    if scene_id not in predictions:
                        predictions[scene_id] = {}

                    if object_id not in predictions[scene_id]:
                        predictions[scene_id][object_id] = {}

                    if ann_id not in predictions[scene_id][object_id]:
                        predictions[scene_id][object_id][ann_id] = {}

                    predictions[scene_id][object_id][ann_id]["pred_bbox"] = data["pred_bboxes"][i]
                    predictions[scene_id][object_id][ann_id]["gt_bbox"] = data["gt_bboxes"][i]
                    predictions[scene_id][object_id][ann_id]["iou"] = data["ref_iou"][i]

            with open(pred_path, "wb") as f:
                pickle.dump(predictions, f)

            # save to global
            ref_acc_all.append(ref_acc)
            ious_all.append(ious)
            masks_all.append(masks)
            others_all.append(others)
            lang_acc_all.append(lang_acc)

        # convert to numpy array
        ref_acc = np.array(ref_acc_all)
        ious = np.array(ious_all)
        masks = np.array(masks_all)
        others = np.array(others_all)
        lang_acc = np.array(lang_acc_all)

        # save the global scores
        with open(score_path, "wb") as f:
            scores = {
                "ref_acc": ref_acc_all,
                "ious": ious_all,
                "masks": masks_all,
                "others": others_all,
                "lang_acc": lang_acc_all
            }
            pickle.dump(scores, f)

    else:
        print("loading the scores...")
        with open(score_path, "rb") as f:
            scores = pickle.load(f)

            # unpack
            ref_acc = np.array(scores["ref_acc"])
            ious = np.array(scores["ious"])
            masks = np.array(scores["masks"])
            others = np.array(scores["others"])
            lang_acc = np.array(scores["lang_acc"])

    multiple_dict = {
        "unique": 0,
        "multiple": 1
    }
    others_dict = {
        "not_in_others": 0,
        "in_others": 1
    }

    # evaluation stats
    stats = {k: np.sum(masks[0] == v) for k, v in multiple_dict.items()}
    stats["overall"] = masks[0].shape[0]
    stats = {}
    for k, v in multiple_dict.items():
        stats[k] = {}
        for k_o, v_o in others_dict.items():
            stats[k][k_o] = np.sum(np.logical_and(masks[0] == v, others[0] == v_o))

        stats[k]["overall"] = np.sum(masks[0] == v)

    stats["overall"] = {}
    for k_o, v_o in others_dict.items():
        stats["overall"][k_o] = np.sum(others[0] == v_o)
    
    stats["overall"]["overall"] = masks[0].shape[0]

    # aggregate scores
    scores = {}
    for k, v in multiple_dict.items():
        for k_o in others_dict.keys():
            ref_accs, acc_025ious, acc_05ious = [], [], []
            for i in range(masks.shape[0]):
                running_ref_acc = np.mean(ref_acc[i][np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o])]) \
                    if np.sum(np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o])) > 0 else 0
                running_acc_025iou = ious[i][np.logical_and(np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o]), ious[i] >= 0.25)].shape[0] \
                    / ious[i][np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o])].shape[0] \
                    if np.sum(np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o])) > 0 else 0
                running_acc_05iou = ious[i][np.logical_and(np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o]), ious[i] >= 0.5)].shape[0] \
                    / ious[i][np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o])].shape[0] \
                    if np.sum(np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o])) > 0 else 0

                # store
                ref_accs.append(running_ref_acc)
                acc_025ious.append(running_acc_025iou)
                acc_05ious.append(running_acc_05iou)

            if k not in scores:
                scores[k] = {k_o: {} for k_o in others_dict.keys()}

            scores[k][k_o]["ref_acc"] = np.mean(ref_accs)
            scores[k][k_o]["acc@0.25iou"] = np.mean(acc_025ious)
            scores[k][k_o]["acc@0.5iou"] = np.mean(acc_05ious)

        ref_accs, acc_025ious, acc_05ious = [], [], []
        for i in range(masks.shape[0]):
            running_ref_acc = np.mean(ref_acc[i][masks[i] == multiple_dict[k]]) if np.sum(masks[i] == multiple_dict[k]) > 0 else 0
            running_acc_025iou = ious[i][np.logical_and(masks[i] == multiple_dict[k], ious[i] >= 0.25)].shape[0] \
                / ious[i][masks[i] == multiple_dict[k]].shape[0] if np.sum(masks[i] == multiple_dict[k]) > 0 else 0
            running_acc_05iou = ious[i][np.logical_and(masks[i] == multiple_dict[k], ious[i] >= 0.5)].shape[0] \
                / ious[i][masks[i] == multiple_dict[k]].shape[0] if np.sum(masks[i] == multiple_dict[k]) > 0 else 0

            # store
            ref_accs.append(running_ref_acc)
            acc_025ious.append(running_acc_025iou)
            acc_05ious.append(running_acc_05iou)

        scores[k]["overall"] = {}
        scores[k]["overall"]["ref_acc"] = np.mean(ref_accs)
        scores[k]["overall"]["acc@0.25iou"] = np.mean(acc_025ious)
        scores[k]["overall"]["acc@0.5iou"] = np.mean(acc_05ious)

    scores["overall"] = {}
    for k_o in others_dict.keys():
        ref_accs, acc_025ious, acc_05ious = [], [], []
        for i in range(masks.shape[0]):
            running_ref_acc = np.mean(ref_acc[i][others[i] == others_dict[k_o]]) if np.sum(others[i] == others_dict[k_o]) > 0 else 0
            running_acc_025iou = ious[i][np.logical_and(others[i] == others_dict[k_o], ious[i] >= 0.25)].shape[0] \
                / ious[i][others[i] == others_dict[k_o]].shape[0] if np.sum(others[i] == others_dict[k_o]) > 0 else 0
            running_acc_05iou = ious[i][np.logical_and(others[i] == others_dict[k_o], ious[i] >= 0.5)].shape[0] \
                / ious[i][others[i] == others_dict[k_o]].shape[0] if np.sum(others[i] == others_dict[k_o]) > 0 else 0

            # store
            ref_accs.append(running_ref_acc)
            acc_025ious.append(running_acc_025iou)
            acc_05ious.append(running_acc_05iou)

        # aggregate
        scores["overall"][k_o] = {}
        scores["overall"][k_o]["ref_acc"] = np.mean(ref_accs)
        scores["overall"][k_o]["acc@0.25iou"] = np.mean(acc_025ious)
        scores["overall"][k_o]["acc@0.5iou"] = np.mean(acc_05ious)
   
    ref_accs, acc_025ious, acc_05ious = [], [], []
    for i in range(masks.shape[0]):
        running_ref_acc = np.mean(ref_acc[i])
        running_acc_025iou = ious[i][ious[i] >= 0.25].shape[0] / ious[i].shape[0]
        running_acc_05iou = ious[i][ious[i] >= 0.5].shape[0] / ious[i].shape[0]

        # store
        ref_accs.append(running_ref_acc)
        acc_025ious.append(running_acc_025iou)
        acc_05ious.append(running_acc_05iou)

    # aggregate
    scores["overall"]["overall"] = {}
    scores["overall"]["overall"]["ref_acc"] = np.mean(ref_accs)
    scores["overall"]["overall"]["acc@0.25iou"] = np.mean(acc_025ious)
    scores["overall"]["overall"]["acc@0.5iou"] = np.mean(acc_05ious)

    # report
    print("\nstats:")
    for k_s in stats.keys():
        for k_o in stats[k_s].keys():
            print("{} | {}: {}".format(k_s, k_o, stats[k_s][k_o]))

    for k_s in scores.keys():
        print("\n{}:".format(k_s))
        for k_m in scores[k_s].keys():
            for metric in scores[k_s][k_m].keys():
                print("{} | {} | {}: {}".format(k_s, k_m, metric, scores[k_s][k_m][metric]))

    print("\nlanguage classification accuracy: {}".format(np.mean(lang_acc)))
    return np.mean(lang_acc)

def eval_det(args, model, dataloader, DC):  # DC: dataset config
    print("evaluate detection...")
    # # constant
    # DC = ScannetDatasetConfig()
    # 
    # # init training dataset
    # print("preparing data...")
    # scanrefer, scene_list = get_scanrefer(args)

    # # dataloader
    # _, dataloader = get_dataloader(args, scanrefer, scene_list, "val", DC)

    # # model
    # model = get_model(args, DC)

    # config
    POST_DICT = {
        "remove_empty_box": True, 
        "use_3d_nms": True, 
        "nms_iou": 0.25,
        "use_old_type_nms": False, 
        "cls_nms": True, 
        "per_class_proposal": True,
        "conf_thresh": 0.05,
        "dataset_config": DC
    }
    AP_IOU_THRESHOLDS = [0.25, 0.5]
    AP_CALCULATOR_LIST = [APCalculator(iou_thresh, DC.class2type) for iou_thresh in AP_IOU_THRESHOLDS]

    sem_acc = []
    for data in tqdm(dataloader):
        for key in data:
            data[key] = data[key].cuda()

        # feed
        with torch.no_grad():
            data = model._forward(data)
            _, data = get_loss(
                data_dict=data, 
                config=DC, 
                detection=True,
                reference=False
            )
            data = get_eval(
                data_dict=data, 
                config=DC, 
                reference=False,
                post_processing=POST_DICT
            )

        sem_acc.append(data["sem_acc"].item())

        batch_pred_map_cls = parse_predictions(data, POST_DICT) 
        batch_gt_map_cls = parse_groundtruths(data, POST_DICT) 
        for ap_calculator in AP_CALCULATOR_LIST:
            ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)

    # aggregate object detection results and report
    print("\nobject detection sem_acc: {}".format(np.mean(sem_acc)))
    for i, ap_calculator in enumerate(AP_CALCULATOR_LIST):
        print()
        print("-"*10, "iou_thresh: %f"%(AP_IOU_THRESHOLDS[i]), "-"*10)
        metrics_dict = ap_calculator.compute_metrics()
        for key in metrics_dict:
            print("eval %s: %f"%(key, metrics_dict[key]))
    return np.mean(sem_acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, help="Folder containing the model")
    parser.add_argument("--gpu", type=str, help="gpu", default="0")
    parser.add_argument("--batch_size", type=int, help="batch size", default=8)
    parser.add_argument("--num_points", type=int, default=40000, help="Point Number [default: 40000]")
    parser.add_argument("--num_proposals", type=int, default=256, help="Proposal number [default: 256]")
    parser.add_argument("--num_scenes", type=int, default=-1, help="Number of scenes [default: -1]")
    parser.add_argument("--force", action="store_true", help="enforce the generation of results")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--repeat", type=int, default=1, help="Number of times for evaluation")
    parser.add_argument("--no_height", action="store_true", help="Do NOT use height signal in input.")
    parser.add_argument("--no_lang_cls", action="store_true", help="Do NOT use language classifier.")
    parser.add_argument("--no_nms", action="store_true", help="do NOT use non-maximum suppression for post-processing.")
    parser.add_argument("--use_color", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_normal", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_multiview", action="store_true", help="Use multiview images.")
    parser.add_argument("--use_bidir", action="store_true", help="Use bi-directional GRU.")
    parser.add_argument("--use_train", action="store_true", help="Use train split in evaluation.")
    parser.add_argument("--use_oracle", action="store_true", help="Use ground truth bounding boxes.")
    parser.add_argument("--use_cat_rand", action="store_true", help="Use randomly selected bounding boxes from correct categories as outputs.")
    parser.add_argument("--use_best", action="store_true", help="Use best bounding boxes as outputs.")
    parser.add_argument("--reference", action="store_true", help="evaluate the reference localization results")
    parser.add_argument("--detection", action="store_true", help="evaluate the object detection results")
    args = parser.parse_args()

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # evaluate
    if args.reference: eval_ref(args)
    if args.detection: eval_det(args)

