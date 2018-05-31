import torch

def IoU(pred, target, n_classes=21):
    ious = []

    pred = pred.view(-1)
    target = target.view(-1)

    for cls in xrange(0, n_classes):
        pred_indexes = pred == cls
        target_indexes = target == cls
        intersection = (pred_indexes[target_indexes]).long().sum().data.cpu()[0]  # Cast to long to prevent overflows
        union = pred_indexes.long().sum().data.cpu()[0] + target_indexes.long().sum().data.cpu()[0] - intersection
        if not union:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / float(max(union, 1)))
    return np.array(ious)

def mIoU(pred, target, n_classes=21):
    iou = IoU(pred, target, n_classes=21)
    return iou.mean()
