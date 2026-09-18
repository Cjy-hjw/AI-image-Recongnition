===== 检测诊断摘要 =====
[static ] n_gt= 1117 n_pred= 1389 TP= 919 匹配率=0.8227 | AP current=0.7955 oracle=0.8227 iou_ranked=0.8227
[dynamic ] n_gt= 71 n_pred= 780 TP= 28 匹配率=0.3944 | AP current=0.2140 oracle=0.3944 iou_ranked=0.3944
[combined] n_gt= 1188 n_pred= 2169 TP= 947 匹配率=0.7971 | AP current=0.7608 oracle=0.7971 iou_ranked=0.7971
[mech_static] 漏检=198 主桶={'mech5': 48, 'mech2': 35, 'mech3': 2, 'mech4': 0, 'mech1': 17, 'unclassified': 96, 'unobservable': 0} | 并集占比={'mech5': 0.242, 'mech2': 0.177, 'mech3': 0.01, 'mech4': 0.035, 'mech1': 0.086} | 近失带=57 远失=83 零IoU=58
[mech_dynamic] 漏检=43 主桶={'mech5': 2, 'mech2': 12, 'mech3': 0, 'mech4': 0, 'mech1': 1, 'unclassified': 28, 'unobservable': 0} | 并集占比={'mech5': 0.047, 'mech2': 0.279, 'mech3': 0.0, 'mech4': 0.023, 'mech1': 0.023} | 近失带=14 远失=25 零IoU=3
[stats_static] 临界带[0.35,0.5)占比=0.051 | 主病灶匹配率=0.8727272727272727 次病灶=0.5659340659340659 | 
GT-iou桶={'miss': 58, '(0,0.1)': 23, '(0.1,0.35)': 60, '[0.35,0.5)': 57, '>=0.5': 919, 'invalid': 0} | 
每例实例数={'0': 19, '1': 693, '2': 194, '3-5': 79, '6+': 6}
[stats_dynamic] 临界带[0.35,0.5)占比=0.197 | 主病灶匹配率=0.5 次病灶=0.17391304347826086 | 
GT-iou桶={'miss': 3, '(0,0.1)': 1, '(0.1,0.35)': 24, '[0.35,0.5)': 14, '>=0.5': 29, 'invalid': 0} | 每例实例数={'0': 0, '1': 0, '2': 0, '3-5': 2, '6+': 46}
[diag] 报告: /2026aicompetition/workspace/arch_push/detection_diag_20260918_045543.json
[diag] sidecar: /2026aicompetition/workspace/arch_push/detection_diag_20260918_045543.cases.jsonl

对拍
[log] | tail -40
===== 恒定人口分类分 =====
frozen 人口: 947
[S_det_seg] baseline=None candidate=None
[S_cls_frozen]
    malignancy_auc_pr       base=0.9436335839617068 cand=0.9436335839617068  Δ=+0.0000
    tirads_weighted_kappa   base=0.9001589671553512 cand=0.9001589671553512  Δ=+0.0000
    ete_weighted_kappa      base=0.566039721162699  cand=0.566039721162699   Δ=+0.0000
    location_weighted_auc   base=0.6135969390789756 cand=0.6135969390789756  Δ=+0.0000
    clinical_t_auc          base=0.9061411839189617 cand=0.9061411839189617  Δ=+0.0000
    score                   base=55.59768337265008  cand=55.59768337265008   Δ=+0.0000
    n_records               base=947 cand=947 Δ=+0.0000
[S_clin_sim]
    malignancy_auc_pr       base=0.9436335839617068 cand=0.9436335839617068  Δ=+0.0000
    tirads_weighted_kappa   base=0.9001589671553512 cand=0.9001589671553512  Δ=+0.0000
    ete_weighted_kappa      base=0.566039721162699  cand=0.566039721162699   Δ=+0.0000
    location_weighted_auc   base=0.6135969390789756 cand=0.6135969390789756  Δ=+0.0000
    clinical_t_auc          base=0.9061411839189617 cand=0.9061411839189617  Δ=+0.0000
    score                   base=55.59768337265008  cand=55.59768337265008   Δ=+0.0000
    n_records               base=947 cand=947 Δ=+0.0000
[population] 找回=0 丢失=0
[bootstrap]
    malignancy_auc_pr       Δmean=0.0 ci95=[0.0, 0.0] P(Δ>0)=0.0
    tirads_weighted_kappa   Δmean=0.0 ci95=[0.0, 0.0] P(Δ>0)=0.0
    ete_weighted_kappa      Δmean=0.0 ci95=[0.0, 0.0] P(Δ>0)=0.0
    location_weighted_auc   Δmean=0.0 ci95=[0.0, 0.0] P(Δ>0)=0.0
    clinical_t_auc          Δmean=0.0 ci95=[0.0, 0.0] P(Δ>0)=0.0
[out] /2026aicompetition/workspace/arch_push/cmp_selfpair.json

===== 检测诊断摘要 =====
[static  ] n_gt= 1117 n_pred= 1519 TP=  946 匹配率=0.8469 | AP current=0.8149 oracle=0.8469 iou_ranked=0.8469
[dynamic] n_gt=    71 n_pred=  850 TP=    29 匹配率=0.4085 | AP current=0.2354 oracle=0.4085 iou_ranked=0.4085
[combined] n_gt= 1188 n_pred= 1569 TP=  975 匹配率=0.8207 | AP current=0.7791 oracle=0.8207 iou_ranked=0.8207
[mech_static] 漏检=171 主桶={'mech5': 31, 'mech2': 28, 'mech3': 4, 'mech4': 1, 'mech1': 12, 'unclassified': 95, 'unobservable': 0} | 并集占比={'mech5': 0.181, 'mech2': 0.164, 'mech3': 0.023, 'mech4': 0.029, 'mech1': 0.07} | 近失焦=51 远失焦=76 零IoU=44
[mech_dynamic] 漏检=42 主桶={'mech5': 1, 'mech2': 13, 'mech3': 0, 'mech4': 0, 'mech1': 3, 'unclassified': 25, 'unobservable': 0} | 并集占比={'mech5': 0.024, 'mech2': 0.31, 'mech3': 0.0, 'mech4': 0.048, 'mech1': 0.071} | 近失焦=20 远失焦=20 零IoU=1
[stats_static] 临界带[0.35,0.5]占比=0.046 ; 主病灶匹配率=0.8919786096256684 次病灶=0.6153846153846154 | GT-iou桶={'miss': 44, '(0,0.1)': 18, '(0.1,0.35)': 58, '[0.35,0.5)': 51, '>=0.5': 946, 'invalid': 0} | 每例实例数={'0': 10, '1': 628, '2': 233, '3-5': 116, '6+': 4}
[stats_dynamic] 临界带[0.35,0.5]占比=0.282 ; 主病灶匹配率=0.5208333333333334 次病灶=0.17391304347826086 | GT-iou桶={'miss': 1, '(0,0.1)': 3, '(0.1,0.35)': 17, '[0.35,0.5)': 20, '>=0.5': 30, 'invalid': 0} | 每例实例数={'0': 0, '1': 0, '2': 0, '3-5': 2, '6+': 46}
[diag] 报告: /2026aicompetition/workspace/arch_push_segonly/detection_diag_20260918_072457.json
[diag] sidecar: /2026aicompetition/workspace/arch_push_segonly/detection_diag_20260918_072457.cases.jsonl

2>&1 | tail -50
===== 恒定人口分类分 =====
frozen 人口: 947
[S_det_seg] baseline=None candidate=None
[S_cls_frozen]
  malignancy_auc_pr      base=0.9436335839617068 cand=0.9436005869045635 Δ=-0.0000
  tirads_weighted_kappa    base=0.9001589671553512 cand=0.9001589671553512 Δ=+0.0000
  ete_weighted_kappa       base=0.566039721162699 cand=0.566039721162699 Δ=+0.0000
  location_weighted_auc    base=0.6135969390789756 cand=0.6135969390789756 Δ=+0.0000
  clinical_t_auc           base=0.9061411839189617 cand=0.9062598923710036 Δ=+0.0001
  score                    base=55.59768337265008 cand=55.598672204982066 Δ=+0.0010
  n_records                base=947 cand=947 Δ=+0.0000
[S_clin_sim]
  malignancy_auc_pr      base=0.9436335839617068 cand=0.9416930839250336 Δ=-0.0019
  tirads_weighted_kappa    base=0.9001589671553512 cand=0.8984731512158529 Δ=-0.0017
  ete_weighted_kappa       base=0.566039721162699 cand=0.5675564681724846 Δ=+0.0015
  location_weighted_auc    base=0.6135969390789756 cand=0.6115844713087716 Δ=-0.0020
  clinical_t_auc           base=0.9061411839189617 cand=0.90185828213836 Δ=-0.0043
  score                    base=55.59768337265008 cand=55.478809613502385 Δ=-0.1189
  n_records                base=947 cand=975 Δ=+28.0000
[population] 找回=44 丢失=16
[bootstrap]
  malignancy_auc_pr       Δmean=-4.371801890497629e-05 ci95=[-0.000461888152298498, 0.000420782990426319] P(Δ>0)=0.42
  tirads_weighted_kappa     Δmean=0.0 ci95=[0.0, 0.0] P(Δ>0)=0.0
  ete_weighted_kappa       Δmean=-0.00038698649840389175 ci95=[-0.023354554554559082, 0.020854009707403736] P(Δ>0)=0.47
  location_weighted_auc    Δmean=0.0 ci95=[0.0, 0.0] P(Δ>0)=0.0
  clinical_t_auc           Δmean=0.0001252586643161685 ci95=[-0.00012035775566556493, 0.0004951910274431426] P(Δ>0)=0.79
[out] /2026aicompetition/workspace/arch_push_segonly/cmp_segonly_vs_r5.json