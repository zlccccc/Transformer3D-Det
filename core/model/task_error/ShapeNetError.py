import numpy as np

seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
               'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
               'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
               'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat
num_classes = 16
num_part = 50
class_to_label = dict(zip(seg_classes.keys(), range(len(seg_classes))))


def ShapeNetError(input, output):
    total_seen_class = [0 for _ in range(num_part)]
    total_correct_class = [0 for _ in range(num_part)]
    shape_ious = {cat: [] for cat in seg_classes.keys()}
    target = input['seg'][:, :, 0]
    out = output['value']
    cur_batch_size, NUM_POINT = out.shape[0], out.shape[1]

    cur_pred_val = out.cpu().data.numpy()
    cur_pred_val_logits = cur_pred_val
    cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
    target = target.cpu().data.numpy()
    for i in range(cur_batch_size):
        cat = seg_label_to_cat[target[i, 0]]
        logits = cur_pred_val_logits[i, :, :]
        cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]
    # correct = np.sum(cur_pred_val == target)
    # total_correct += correct
    # total_seen += (cur_batch_size * NUM_POINT)

    for l in range(num_part):  # for one point
        total_seen_class[l] += np.sum(target == l)
        total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

    for i in range(cur_batch_size):  # for one object
        segp = cur_pred_val[i, :]
        segl = target[i, :]
        cat = seg_label_to_cat[segl[0]]
        part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
        for l in seg_classes[cat]:
            if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):  # part is not present, no prediction as well
                part_ious[l - seg_classes[cat][0]] = 1.0
            else:
                part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(np.sum((segl == l) | (segp == l)))
        shape_ious[cat].append(np.mean(part_ious))
    # all calculate

    shape_ious_arr = [0 for _ in range(num_classes)]
    shape_ious_arr_count = [0 for _ in range(num_classes)]
    all_shape_ious, all_shape_ious_count = 0, 0
    for cat in shape_ious.keys():
        for iou in shape_ious[cat]:
            all_shape_ious += iou
            all_shape_ious_count += 1
            shape_ious_arr[class_to_label[cat]] += iou
            shape_ious_arr_count[class_to_label[cat]] += 1
        # shape_ious[cat] = np.mean(shape_ious[cat])
    # mean_shape_ious = np.mean(list(shape_ious.values()))

    # test_metrics['accuracy'] = total_correct / float(total_seen)
    # test_metrics['class_avg_accuracy'] = np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
    # for cat in sorted(shape_ious.keys()):
    #     log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
    # test_metrics['class_avg_iou'] = mean_shape_ious
    # test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)
    output['class_avg_accruancy(error)'] = np.array(total_correct_class)
    output['class_avg_accruancy(error_count)'] = np.array(total_seen_class)
    output['class_avg_iou(error)'] = np.array(shape_ious_arr)
    output['class_avg_iou(error_count)'] = np.array(shape_ious_arr_count)
    output['instance_avg_iou(error)'] = all_shape_ious
    output['instance_avg_iou(error_count)'] = all_shape_ious_count
    # for key, value in sorted(output.items()):
    #     if '(error)' in key:
    #         print(key, value, type(value))
    return output
