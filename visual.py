import random

import numpy

from eval.eval_detection import ANETdetection
import numpy as np
import matplotlib.pyplot as plt


def choose(gt,video_name):

    choose_gt = gt.loc[gt['video-id'] == video_name]
    if len(gt.values[0]) > 4:
        choose_gt = choose_gt.loc[choose_gt['score']>0.5]
    x_gt = numpy.arange(0, 250, 0.1)[None, :].repeat(len(choose_gt), 0)  # (choose_gt,300)

    ts_gt = np.array(choose_gt['t-start'][:])[:, None]
    ys_gt = np.array(x_gt >= ts_gt, dtype=np.int)

    te_gt = np.array(choose_gt['t-end'][:])[:, None]
    ye_gt = np.array(x_gt > te_gt, dtype=np.int)

    y_gt = (ys_gt - ye_gt).sum(axis=0)
    # y_gt = np.array(y_gt>=1,dtype=np.int)
    return y_gt[:,None]


def plot(ygt,ytopk,ymean,ymax,videoname):
    pixel_per_bar = 4  # 线宽像素
    dpi = 800  # 分辨率

    _,fig = plt.subplots(4,1,figsize=(len(y_gt)*pixel_per_bar/dpi, 2), dpi=dpi)

    fig[0].imshow((ygt).reshape(1,-1),
          cmap='binary',  # 设置为二值图
          aspect='auto')
    fig[0].set_xticks([])
    fig[0].set_yticks([])
    fig[0].set_title(videoname,fontsize=14)
    # fig[0].set_ylabel('gt')

    fig[1].imshow((ytopk).reshape(1,-1),
          cmap='Oranges',  # 设置为二值图
          aspect='auto')
    fig[1].set_xticks([])
    fig[1].set_yticks([])
    # fig[1].set_ylabel('topkmean')

    fig[2].imshow((ymean).reshape(1,-1),
          cmap='Blues',  # 设置为二值图
          aspect='auto')
    fig[2].set_xticks([])
    fig[2].set_yticks([])
    # fig[2].set_ylabel('mean')

    fig[3].imshow((ymax).reshape(1, -1),
                  cmap='Greens',  # 设置为二值图
                  aspect='auto')
    fig[3].set_xticks([])
    fig[3].set_yticks([])
    # fig[2].set_ylabel('max')

    plt.show()


if __name__ == "__main__":
    gt_path = './data/gt.json'
    res_path_topk = './experiments/result_topk.json'
    res_path_mean = './experiments/result_mean.json'
    res_path_max = './experiments/result_max.json'
    tIOU_thresh = tIoU_thresh = np.linspace(0.1, 0.7, 7)

    anet_detection = ANETdetection(gt_path, res_path_topk,
                                   subset='test', tiou_thresholds=tIoU_thresh,
                                   verbose=False, check_status=False)
    anet_detection1 = ANETdetection(gt_path, res_path_mean,
                                   subset='test', tiou_thresholds=tIoU_thresh,
                                   verbose=False, check_status=False)
    anet_detection2 = ANETdetection(gt_path, res_path_max,
                                    subset='test', tiou_thresholds=tIoU_thresh,
                                    verbose=False, check_status=False)

    gt = anet_detection.ground_truth

    pred_topk = anet_detection.prediction
    pred_mean = anet_detection1.prediction
    pred_max = anet_detection2.prediction

    video_name = list(set(gt.values[:, 0]))

    video_id = random.randint(0, 209)
    video_name = video_name[video_id]
    # video_name = 'video_test_0001468'

    y_gt = choose(gt,video_name)
    y_predtopk = choose(pred_topk,video_name)
    y_predmean = choose(pred_mean,video_name)
    y_predmax = choose(pred_max,video_name)

    plot(y_gt,y_predtopk,y_predmean,y_predmax,video_name)



