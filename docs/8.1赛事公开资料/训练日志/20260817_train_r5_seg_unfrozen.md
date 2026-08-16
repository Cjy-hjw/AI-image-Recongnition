nohup: ignoring input [train_official] device=cuda bs=16 img=512 lr_base=0.0024 epochs=35 mock=False freeze_legacy=False --> Found offline local backbone checkpoint: checkpoints/convnext_nano_pretrained.safetensors. Loading weights locally. --> Successfully loaded offline local backbone pretrained weights. --> Found offline local backbone checkpoint: checkpoints/efficientnet_b2_pretrained.safetensors. Loading weights locally. --> Successfully loaded offline local backbone pretrained weights. [Val Diag] pred_prob: mean=0.4454, std=0.0003, min=0.4425, max=0.4463 --> [护栏] warm-start val Dice 基线 = 0.0007 [optimizer rebuilt] Official Stage A: heads+seg warmup (cls_backbone/CBM frozen) head_lr=2.40e-04 [Val Diag] pred_prob: mean=0.4345, std=0.3703, min=0.1101, max=1.0000 Epoch 01/35 [Official Stage A: heads+seg warmup (cls_backbone/CBM frozen)] train_loss=5.0111 val_dice=0.1566 seg_p=0.4138582204220563 seg_pmax=0.9999992847442627 seg_over_half=0.3305257340154704 composite=0.32847180953050664 mal_auc_pr=0.4999421425153562 loc_auc=0.5677581807166784 ete_kappa=0.0 t_auc=0.6288265306122449 tirads_kappa=0.11768877889780893 --> 保存 best(composite=0.3285)-> checkpoints/official_heads_seg_best.pth [Val Diag] pred_prob: mean=0.2694, std=0.3740, min=0.0168, max=1.0000 Epoch 02/35 [Official Stage A: heads+seg warmup (cls_backbone/CBM frozen)] train_loss=4.7869 val_dice=0.1757 seg_p=0.2708261821573303 seg_pmax=1.0 seg_over_half=0.25760382872361404 composite=0.357658901660632 mal_auc_pr=0.508571752740719 loc_auc=0.5695538035564243 ete_kappa=0.0 t_auc=0.7062925170068027 tirads_kappa=0.19359282569352532 --> 保存 best(composite=0.3577)-> checkpoints/official_heads_seg_best.pth [Val Diag] pred_prob: mean=0.0778, std=0.2220, min=0.0030, max=1.0000 Epoch 03/35 [Official Stage A: heads+seg warmup (cls_backbone/CBM frozen)] train_loss=4.9220 val_dice=0.3024 seg_p=0.08892169533043924 seg_pmax=0.9999983310699463 seg_over_half=0.08836529805110051 composite=0.3930035658583659 mal_auc_pr=0.501694489952109 loc_auc=0.5863671844038492 ete_kappa=0.0 t_auc=0.6960459183673469 tirads_kappa=0.2715313859768299 --> 保存 best(composite=0.3930)-> checkpoints/official_heads_seg_best.pth 

memory.used [MiB], memory.total [MiB]
1072 MiB, 32607 MiB
1170 MiB, 32607 MiB
4018 MiB, 32607 MiB
8708 MiB, 32607 MiB
2988 MiB, 32607 MiB
2766 MiB, 32607 MiB
8716 MiB, 32607 MiB
8716 MiB, 32607 MiB
8716 MiB, 32607 MiB
8716 MiB, 32607 MiB
8716 MiB, 32607 MiB
8716 MiB, 32607 MiB
8716 MiB, 32607 MiB

13:05.204Z", "epoch": 1, "step": 0, "phase": "train", "mode": "training", "data_source": "official/
13:07.758Z", "epoch": 1, "step": 20, "phase": "train", "mode": "training", "data_source": "official
13:11.755Z", "epoch": 1, "step": 40, "phase": "train", "mode": "training", "data_source": "official
13:14.627Z", "epoch": 1, "step": 60, "phase": "train", "mode": "training", "data_source": "official
13:18.192Z", "epoch": 1, "step": 80, "phase": "train", "mode": "training", "data_source": "official
13:21.461Z", "epoch": 1, "step": 100, "phase": "train", "mode": "training", "data_source": "official
13:25.655Z", "epoch": 1, "step": 120, "phase": "train", "mode": "training", "data_source": "official
13:29.089Z", "epoch": 1, "step": 140, "phase": "train", "mode": "training", "data_source": "official
13:31.972Z", "epoch": 1, "step": 160, "phase": "train", "mode": "training", "data_source": "official
13:35.082Z", "epoch": 1, "step": 180, "phase": "train", "mode": "training", "data_source": "official
13:38.142Z", "epoch": 1, "step": 200, "phase": "train", "mode": "training", "data_source": "official
13:41.665Z", "epoch": 1, "step": 220, "phase": "train", "mode": "training", "data_source": "official
13:44.519Z", "epoch": 1, "step": 240, "phase": "train", "mode": "training", "data_source": "official
13:49.073Z", "epoch": 1, "step": 260, "phase": "train", "mode": "training", "data_source": "official
13:51.974Z", "epoch": 1, "step": 280, "phase": "train", "mode": "training", "data_source": "official
13:54.890Z", "epoch": 1, "step": 300, "phase": "train", "mode": "training", "data_source": "official
13:58.012Z", "epoch": 1, "step": 320, "phase": "train", "mode": "training", "data_source": "official
14:39.502Z", "epoch": 1, "step": 0, "phase": "val", "mode": "training", "data_source": "official/tr
14:39.798Z", "epoch": 1, "step": 0, "phase": "val", "mode": "training", "data_source": "official/tr
14:42.122Z", "epoch": 2, "step": 0, "phase": "train", "mode": "training", "data_source": "official/
14:45.054Z", "epoch": 2, "step": 20, "phase": "train", "mode": "training", "data_source": "official
14:47.893Z", "epoch": 2, "step": 40, "phase": "train", "mode": "training", "data_source": "official
14:50.794Z", "epoch": 2, "step": 60, "phase": "train", "mode": "training", "data_source": "official
14:53.617Z", "epoch": 2, "step": 80, "phase": "train", "mode": "training", "data_source": "official
14:56.567Z", "epoch": 2, "step": 100, "phase": "train", "mode": "training", "data_source": "official
14:59.588Z", "epoch": 2, "step": 120, "phase": "train", "mode": "training", "data_source": "official
15:02.545Z", "epoch": 2, "step": 140, "phase": "train", "mode": "training", "data_source": "official
15:05.535Z", "epoch": 2, "step": 160, "phase": "train", "mode": "training", "data_source": "official
15:08.491Z", "epoch": 2, "step": 180, "phase": "train", "mode": "training", "data_source": "official
15:11.458Z", "epoch": 2, "step": 200, "phase": "train", "mode": "training", "data_source": "official
15:14.381Z", "epoch": 2, "step": 220, "phase": "train", "mode": "training", "data_source": "official
15:17.324Z", "epoch": 2, "step": 240, "phase": "train", "mode": "training", "data_source": "official
15:20.320Z", "epoch": 2, "step": 260, "phase": "train", "mode": "training", "data_source": "official
15:23.300Z", "epoch": 2, "step": 280, "phase": "train", "mode": "training", "data_source": "official
15:26.278Z", "epoch": 2, "step": 300, "phase": "train", "mode": "training", "data_source": "official
15:29.070Z", "epoch": 2, "step": 320, "phase": "train", "mode": "training", "data_source": "official

Epoch 08/35 [Official Stage A: heads+seg warmup (cls_backbone/CBM frozen)] train_loss=7.2589 val_dice=0.4994 seg_p=0.0256982427673 seg_over_half=0.020484574447722125 composite=0.43929445214254864 mal_auc_pr=0.5012373162115837 loc_auc=0.623664146798309 ete_kappa=0.0 t_auc=0.7121598639455782 tirads_kappa=0.2992891023861537
--> 保存 best(composite=0.4393)-> checkpoints/official_heads_seg_best.pth
[Val Diag] pred_prob: mean=0.0322, std=0.1636, min=0.0000, max=0.9970
Epoch 09/35 [Official Stage A: heads+seg warmup (cls_backbone/CBM frozen)] train_loss=7.7866 val_dice=0.5309 seg_p=0.044341468811 seg_over_half=0.04157674665281758 composite=0.4521475198099752 mal_auc_pr=0.5024064749007768 loc_auc=0.6226686952194066 ete_kappa=0.0 t_auc=0.6918792517006804 tirads_kappa=0.36505876965348794
--> 保存 best(composite=0.4521)-> checkpoints/official_heads_seg_best.pth
[Val Diag] pred_prob: mean=0.0182, std=0.1241, min=0.0000, max=0.9992
Epoch 10/35 [Official Stage A: heads+seg warmup (cls_backbone/CBM frozen)] train_loss=8.4612 val_dice=0.4904 seg_p=0.0511241912842 seg_over_half=0.028802224176169854 composite=0.43375918760518856 mal_auc_pr=0.5028667981957584 loc_auc=0.6219729925031319 ete_kappa=0.0 t_auc=0.6907738095238095 tirads_kappa=0.2965507271881639
[optimizer rebuilt] Official Stage B: heads+seg + LLRD unfreeze cls Top-2 head_lr=2.40e-03
[Val Diag] pred_prob: mean=0.0153, std=0.1120, min=0.0000, max=0.9990
Epoch 11/35 [Official Stage B: heads+seg + LLRD unfreeze cls Top-2] train_loss=3.8066 val_dice=0.3944 seg_p=0.018230851558 seg_over_half=0.01814650360649154 composite=0.5022749702004219 mal_auc_pr=0.8659990199792391 loc_auc=0.6134266511890548 ete_kappa=0.0 t_auc=0.6343537414965985 tirads_kappa=0.5054619178745423
--> 保存 best(composite=0.5023)-> checkpoints/official_heads_seg_best.pth
[Val Diag] pred_prob: mean=0.0242, std=0.1447, min=0.0000, max=0.9996
Epoch 12/35 [Official Stage B: heads+seg + LLRD unfreeze cls Top-2] train_loss=4.2581 val_dice=0.5457 seg_p=0.028275463047 seg_over_half=0.028669602772187903 composite=0.5753337135842619 mal_auc_pr=0.8778830698190692 loc_auc=0.7038254178576754 ete_kappa=0.0 t_auc=0.6844812925170068 tirads_kappa=0.6400867587474206
--> 保存 best(composite=0.5753)-> checkpoints/official_heads_seg_best.pth
[Val Diag] pred_prob: mean=0.0331, std=0.1713, min=0.0000, max=0.9997
Epoch 13/35 [Official Stage B: heads+seg + LLRD unfreeze cls Top-2] train_loss=3.8921 val_dice=0.5927 seg_p=0.03386113673 seg_over_half=0.03413325653979059 composite=0.587899185018737 mal_auc_pr=0.873362992867946 loc_auc=0.7239576834279996 ete_kappa=0.0 t_auc=0.6785714285714286 tirads_kappa=0.6587601782570423
--> 保存 best(composite=0.5879)-> checkpoints/official_heads_seg_best.pth
[Val Diag] pred_prob: mean=0.0256, std=0.1470, min=0.0000, max=0.9970
Epoch 14/35 [Official Stage B: heads+seg + LLRD unfreeze cls Top-2] train_loss=3.5837 val_dice=0.5785 seg_p=0.028579658357 seg_over_half=0.028787691917645155 composite=0.5900730941229928 mal_auc_pr=0.8640304439203997 loc_auc=0.7429696437789332 ete_kappa=0.0 t_auc=0.7112244897959185 tirads_kappa=0.6436924096784885
--> 保存 best(composite=0.5901)-> checkpoints/official_heads_seg_best.pth
[Val Diag] pred_prob: mean=0.0358, std=0.1771, min=0.0000, max=0.9997
Epoch 15/35 [Official Stage B: heads+seg + LLRD unfreeze cls Top-2] train_loss=3.2285 val_dice=0.6193 seg_p=0.03306819932 seg_over_half=0.03349745485204211 composite=0.6010430305333317 mal_auc_pr=0.8908897291920069 loc_auc=0.7393387017790443 ete_kappa=0.0 t_auc=0.6983843537414965 tirads_kappa=0.65833059249166
--> 保存 best(composite=0.6010)-> checkpoints/official_heads_seg_best.pth
[Val Diag] pred_prob: mean=0.0329, std=0.1698, min=0.0000, max=0.9991
Epoch 16/35 [Official Stage B: heads+seg + LLRD unfreeze cls Top-2] train_loss=2.9167 val_dice=0.6253 seg_p=0.0346711594 seg_over_half=0.033500582621647762 composite=0.610497676367928 mal_auc_pr=0.8776630961036542 loc_auc=0.7518710748219214 ete_kappa=0.0 t_auc=0.7275085034013605 tirads_kappa=0.6806051599690303
--> 保存 best(composite=0.6105)-> checkpoints/official_heads_seg_best.pth
[Val Diag] pred_prob: mean=0.0305, std=0.1651, min=0.0000, max=0.9997
Epoch 17/35 [Official Stage B: heads+seg + LLRD unfreeze cls Top-2] train_loss=2.5596 val_dice=0.6318 seg_p=0.031711274961 seg_over_half=0.03234490276088376 composite=0.6132179261596186 mal_auc_pr=0.8988221122303334 loc_auc=0.7432201005693725 ete_kappa=0.0 t_auc=0.7167091836734695 tirads_kappa=0.6887378398603143
--> 保存 best(composite=0.6132)-> checkpoints/official_heads_seg_best.pth
[Val Diag] pred_prob: mean=0.0266, std=0.1528, min=0.0000, max=0.9996
Epoch 18/35 [Official Stage B: heads+seg + LLRD unfreeze cls Top-2] train_loss=2.3200 val_dice=0.6207 seg_p=0.02650026675 seg_over_half=0.02708739500779372 composite=0.611583138922532 mal_auc_pr=0.8870937278000426 loc_auc=0.7349924595431861 ete_kappa=0.0 t_auc=0.7420493197278911 tirads_kappa=0.6846441502153862
[Val Diag] pred_prob: mean=0.0330, std=0.1709, min=0.0000, max=0.9997
Epoch 19/35 [Official Stage B: heads+seg + LLRD unfreeze cls Top-2] train_loss=2.0321 val_dice=0.6405 seg_p=0.03211591466 seg_over_half=0.03257669076411682 composite=0.6073423459304893 mal_auc_pr=0.8773609560194493 loc_auc=0.7343328221086897 ete_kappa=-0.0038569424964935983 t_auc=0.7441326530612244 tirads_kappa=0.6516234714479816
[Val Diag] pred_prob: mean=0.0306, std=0.1643, min=0.0000, max=0.9997
Epoch 20/35 [Official Stage B: heads+seg + LLRD unfreeze cls Top-2] train_loss=1.7495 val_dice=0.6364 seg_p=0.03046998503 seg_over_half=0.030962520802514795 composite=0.6125179484556167 mal_auc_pr=0.8755776132592992 loc_auc=0.7360865729193337 ete_kappa=0.04828388598022093 t_auc=0.7212585034013607 tirads_kappa=0.6575036481400885
[Val Diag] pred_prob: mean=0.0256, std=0.1528, min=0.0000, max=0.9997

Epoch 19/35 [Official Stage B: heads+seg + LLRD unfreeze cls Top-2] train_loss=2.0321 val_dice=0.6405 seg_p=0.03211591224676582 seg_pmax=0.9999743700027466 seg_over_half=0.03257669076411682 composite=0.6073423459304893 mal_auc_pr=0.8773609560194493 loc_auc=0.7343328221086897 ete_kappa=-0.0038569424964935983 t_auc=0.7441326530612244 tirads_kappa=0.6848441502153862
[Val Diag] pred_prob: mean=0.0306, std=0.1643, min=0.0000, max=0.9997

Epoch 20/35 [Official Stage B: heads+seg + LLRD unfreeze cls Top-2] train_loss=1.7495 val_dice=0.6364 seg_p=0.03046998549142533 seg_pmax=0.9999548196792603 seg_over_half=0.030962520802514795 composite=0.6125179484556167 mal_auc_pr=0.8755776132592992 loc_auc=0.7360865729193337 ete_kappa=0.04828388598022093 t_auc=0.7212585034013607 tirads_kappa=0.6575036481400885
[Val Diag] pred_prob: mean=0.0346, std=0.1737, min=0.0000, max=0.9990

Epoch 21/35 [Official Stage B: heads+seg + LLRD unfreeze cls Top-2] train_loss=1.5674 val_dice=0.6350 seg_p=0.03441246478272789 seg_pmax=0.9999219179153442 seg_over_half=0.034945225574561124 composite=0.6368161518742967 mal_auc_pr=0.8844387558079075 loc_auc=0.7294812293964478 ete_kappa=0.20861633233296473 t_auc=0.7237244897959184 tirads_kappa=0.6396766104849549
--> 保存 best(composite=0.6368)-> checkpoints/official_heads_seg_best.pth
[Val Diag] pred_prob: mean=0.0303, std=0.1637, min=0.0000, max=0.9998

Epoch 22/35 [Official Stage B: heads+seg + LLRD unfreeze cls Top-2] train_loss=1.4896 val_dice=0.6460 seg_p=0.03038426847941071 seg_pmax=0.9999963045120239 seg_over_half=0.030970731430505154 composite=0.6221679099305302 mal_auc_pr=0.8770155577986696 loc_auc=0.7276837245527441 ete_kappa=0.13995327102803734 t_auc=0.7155612244897958 tirads_kappa=0.6268065328929331
[Val Diag] pred_prob: mean=0.0339, std=0.1720, min=0.0000, max=0.9992

Epoch 23/35 [Official Stage B: heads+seg + LLRD unfreeze cls Top-2] train_loss=1.2918 val_dice=0.6417 seg_p=0.03233577256482205 seg_pmax=0.9999274015426636 seg_over_half=0.03275898927767601 composite=0.6315095467320768 mal_auc_pr=0.8606087849271227 loc_auc=0.7312345204740047 ete_kappa=0.17892295398288727 t_auc=0.7187712585034013 tirads_kappa=0.6578281276878086
[Val Diag] pred_prob: mean=0.0326, std=0.1698, min=0.0000, max=0.9994

Epoch 24/35 [Official Stage B: heads+seg + LLRD unfreeze cls Top-2] train_loss=1.1588 val_dice=0.6479 seg_p=0.03300861543121599 seg_pmax=0.9999780654907227 seg_over_half=0.03359713074723644 composite=0.6271082737964871 mal_auc_pr=0.8514227688471007 loc_auc=0.7318652784236105 ete_kappa=0.13465446090034983 t_auc=0.7181972789115646 tirads_kappa=0.6786365843734824
[Val Diag] pred_prob: mean=0.0354, std=0.1759, min=0.0000, max=0.9988

Epoch 25/35 [Official Stage B: heads+seg + LLRD unfreeze cls Top-2] train_loss=0.9582 val_dice=0.6433 seg_p=0.035384145411029015 seg_pmax=0.9998534917831421 seg_over_half=0.03597448281282504 composite=0.6120329350700845 mal_auc_pr=0.8569169991563105 loc_auc=0.7312267904441083 ete_kappa=0.0966411068511418 t_auc=0.6935799319727891 tirads_kappa=0.6505741680485767
[Val Diag] pred_prob: mean=0.0314, std=0.1668, min=0.0000, max=0.9997

Epoch 26/35 [Official Stage B: heads+seg + LLRD unfreeze cls Top-2] train_loss=0.9834 val_dice=0.6521 seg_p=0.031022669152307086 seg_pmax=0.9999872446060181 seg_over_half=0.03148115316086267 composite=0.6298586117993524 mal_auc_pr=0.8672389610156304 loc_auc=0.7320447292142557 ete_kappa=0.15477622528608004 t_auc=0.7066113945578232 tirads_kappa=0.6663345272399213
[Val Diag] pred_prob: mean=0.0376, std=0.1808, min=0.0000, max=0.9995

Epoch 27/35 [Official Stage B: heads+seg + LLRD unfreeze cls Top-2] train_loss=0.9808 val_dice=0.6452 seg_p=0.035548193341661136 seg_pmax=0.9999469518661499 seg_over_half=0.035868134018937514 composite=0.6293973166449626 mal_auc_pr=0.8697129022346481 loc_auc=0.7373360024566511 ete_kappa=0.1276587418060855 t_auc=0.7119047619047618 tirads_kappa=0.684560582223757
[Val Diag] pred_prob: mean=0.0291, std=0.1616, min=0.0000, max=0.9994

Epoch 28/35 [Official Stage B: heads+seg + LLRD unfreeze cls Top-2] train_loss=0.9277 val_dice=0.6474 seg_p=0.03016152476462034 seg_pmax=0.9999681711196899 seg_over_half=0.030702300325653256 composite=0.6278545657030276 mal_auc_pr=0.8648984508766988 loc_auc=0.7339438649241113 ete_kappa=0.15670103092783505 t_auc=0.6926870748299321 tirads_kappa=0.6715222141296431
[Val Diag] pred_prob: mean=0.0330, std=0.1712, min=0.0000, max=0.9996

Epoch 29/35 [Official Stage B: heads+seg + LLRD unfreeze cls Top-2] train_loss=0.8971 val_dice=0.6508 seg_p=0.032623300725627226 seg_pmax=0.9999866485595703 seg_over_half=0.03307825291650535 composite=0.6281239413159448 mal_auc_pr=0.8628716447201725 loc_auc=0.733084542053051 ete_kappa=0.151364873813553 t_auc=0.7094600340136055 tirads_kappa=0.6611243124023579
[Val Diag] pred_prob: mean=0.0316, std=0.1674, min=0.0000, max=0.9991

Epoch 30/35 [Official Stage B: heads+seg + LLRD unfreeze cls Top-2] train_loss=0.9652 val_dice=0.6447 seg_p=0.03185432466000495 seg_over_half=0.03224136138103417 composite=0.6266473874912365 mal_auc_pr=0.8509918210491778 loc_auc=0.7339271976198115 ete_kappa=0.1542915147185986 t_auc=0.7031462585034014 tirads_kappa=0.6728058028354764
[optimizer rebuilt] Official Stage C: heads+seg low-lr calibration head_lr=1.20e-04
[Val Diag] pred_prob: mean=0.0323, std=0.1700, min=0.0000, max=0.9994

Epoch 31/35 [Official Stage C: heads+seg low-lr calibration] train_loss=0.7187 val_dice=0.6528 seg_p=0.03220077870226294 seg_pmax=0.999969482421875 seg_over_half=0.03244596402320636 composite=0.6253039990801639 mal_auc_pr=0.8512768652057016 loc_auc=0.7357893767957516 ete_kappa=0.13598047459851914 t_auc=0.7006802721088435 tirads_kappa=0.6753254437869822
[Val Diag] pred_prob: mean=0.0331, std=0.1724, min=0.0000, max=0.9994

Epoch 32/35 [Official Stage C: heads+seg low-lr calibration] train_loss=0.9607 val_dice=0.6525 seg_p=0.032437904159257396 seg_pmax=0.9999589920043945 seg_over_half=0.03265962092834111 composite=0.6254114003918819 mal_auc_pr=0.8515138584985706 loc_auc=0.7358339674374997 ete_kappa=0.14492459455619378 t_auc=0.7023809523809523 tirads_kappa=0.6653345365476344
[Val Diag] pred_prob: mean=0.0334, std=0.1729, min=0.0000, max=0.9993

Epoch 33/35 [Official Stage C: heads+seg low-lr calibration] train_loss=0.86
Epoch 33/35 [Official Stage C: heads+seg low-lr calibration] train_loss=0.8662 val_dice=0.6517 seg_p=0.03274277593237312 seg_pmax=0.9999727010726929 seg_over_half=0.03313767733915195 composite=0.6256694487527133 mal_auc_pr=0.8522461872397647 loc_auc=0.7356626330954699 ete_kappa=0.1496424473070127 t_auc=0.7022534013605441 tirads_kappa=0.6664032746569426
[Val Diag] pred_prob: mean=0.0332, std=0.1724, min=0.0000, max=0.9993

Epoch 34/35 [Official Stage C: heads+seg low-lr calibration] train_loss=0.7214 val_dice=0.6535 seg_p=0.03257562648404395 seg_pmax=0.9999743700027466 seg_over_half=0.03291856974876957 composite=0.6257693127168467 mal_auc_pr=0.8524081709933412 loc_auc=0.7357673797478187 ete_kappa=0.15070553935860058 t_auc=0.7017389455782312 tirads_kappa=0.6653577522959462
[Val Diag] pred_prob: mean=0.0331, std=0.1721, min=0.0000, max=0.9993

Epoch 35/35 [Official Stage C: heads+seg low-lr calibration] train_loss=0.8506 val_dice=0.6528 seg_p=0.03265160228538556 seg_pmax=0.9999738931655884 seg_over_half=0.03303829272834756 composite=0.6257779000571999 mal_auc_pr=0.8524335845322251 loc_auc=0.7357840470921185 ete_kappa=0.151364873813553 t_auc=0.7017389455782312 tirads_kappa=0.6653577522959462
[Val Diag] pred_prob: mean=0.0331, std=0.1721, min=0.0000, max=0.9993