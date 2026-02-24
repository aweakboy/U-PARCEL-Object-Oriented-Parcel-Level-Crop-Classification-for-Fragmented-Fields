import numpy as np
from sklearn.metrics import f1_score, accuracy_score, jaccard_score

# def compute_metrics_detailed(preds_list, gts_list, num_classes):
#     preds = np.concatenate([p.flatten() for p in preds_list])
#     gts = np.concatenate([g.flatten() for g in gts_list])
#
#     f1_per_class = f1_score(gts, preds, average=None, labels=range(num_classes), zero_division=0)
#     mean_f1 = f1_per_class.mean()
#     acc = accuracy_score(gts, preds)
#     iou = jaccard_score(gts, preds, average=None, labels=range(num_classes), zero_division=0)
#     mean_iou = iou.mean()
#
#     return {
#         "overall_accuracy": float(acc),
#         "mean_f1": float(mean_f1),
#         "mean_iou": float(mean_iou),
#         "f1_per_class": f1_per_class.tolist(),
#         "iou_per_class": iou.tolist()
#     }
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, jaccard_score

def compute_metrics_detailed(preds_list, gts_list, num_classes):
    """
    参数
    -------
    preds_list : List[ array-like ]  # 每个元素可以是 int / list / ndarray / Tensor
    gts_list   : List[ array-like ]
    """
    # 把任何格式都转成 1-D numpy，再 concatenate
    preds = np.concatenate([np.asarray(p).ravel() for p in preds_list])
    gts   = np.concatenate([np.asarray(g).ravel() for g in gts_list])

    f1_per_class = f1_score(gts, preds,
                            average=None,
                            labels=range(num_classes),
                            zero_division=0)
    mean_f1 = f1_per_class.mean()

    acc = accuracy_score(gts, preds)

    iou_per_class = jaccard_score(gts, preds,
                                  average=None,
                                  labels=range(num_classes),
                                  zero_division=0)
    mean_iou = iou_per_class.mean()

    return {
        "overall_accuracy": float(acc),
        "mean_f1": float(mean_f1),
        "mean_iou": float(mean_iou),
        "f1_per_class": f1_per_class.tolist(),
        "iou_per_class": iou_per_class.tolist()
    }
