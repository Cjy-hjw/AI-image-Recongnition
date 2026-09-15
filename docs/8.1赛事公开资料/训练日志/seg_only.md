--> Successfully loaded offline local backbone pretrained weights.
--> Found offline local backbone checkpoint: checkpoints/efficientnet_b2_pretrained.safetensors. Loading weights locally.
--> Successfully loaded offline local backbone pretrained weights.
--> [warm-start] 加载 /root/cloud_train_pkg/checkpoints/official_heads_seg_best.pth; 四头权重全部命中(missing=0,全量热启动)
 [Val Diag] pred_prob: mean=0.0125, std=0.1035, min=0.0000, max=0.9991
--> [护栏] warm-start val Dice 基线 = 0.6754
--> [SEG-only 护栏基线] {'malignancy_auc_pr': 0.9425965464578624, 'tirads_weighted_kappa': 0.8780235057493917, 'location_weighted_auc': 0.9482540717762212, 'clinical_t_auc': 0.908511080812516, 'ete_weighted_kappa': 0.3769557971194748, 'composite': 0.7882874148650073}
--> [护栏] baseline 指标已写 JSONL(epoch=0/phase=val): {'malignancy_auc_pr': 0.9426, 'location_weighted_auc': 0.9483, 'ete_weighted_kappa': 0.377, 'clinical_t_auc': 0.9085, 'tirads_weighted_kappa': 0.878, 'seg_dice': 0.6754, 'composite': 0.7883, 'n_lesions': 1213, 'ete_extreme_error_rate': 0.0198, 'ete_kappa_gap': 0.0407, 'seg_prob_mean': 0.0339, 'seg_prob_max': 1.0, 'seg_over_half_frac': 0.0344}
 [optimizer rebuilt] SEG-only finetune (r5 frozen, seg chain only) head_lr=1.20e-04
 [Val Diag] pred_prob: mean=0.0123, std=0.1005, min=0.0000, max=0.9888
Epoch 01/1 [SEG-only finetune (r5 frozen, seg chain only)] train_loss=3.6595 val_dice=0.6832 seg_p=0.032771425105416384 seg_pmax=0.995564341545105 seg_over_half=0.033321539001679605 composite=0.7917341866354687 mal_auc_pr=0.9427038051730272 loc_auc=0.9482540717762212 ete_kappa=0.3897112967524059 t_auc=0.9084879717144639 tirads_kappa=0.8780235057493917

| [SEG-only 门] 未过] val_dice=0.6832 < 0.72,不满足上线门槛(不 abort,留权重供分析)
--> 保存 best(composite=0.7917)→ checkpoints/tmp_seg_only_probe.pth