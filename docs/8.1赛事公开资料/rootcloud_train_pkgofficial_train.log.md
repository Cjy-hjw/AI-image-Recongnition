```text
--> [护栏] warm-start val Dice 基线 = 0.0007
[Val Diag] pred_prob: mean=0.4454, std=0.0003, min=0.4425, max=0.4463
Epoch 01/35 [Official Stage A: heads warmup (backbone/seg/CBM frozen)] train_loss=4.7887 val_dice=0.0007 composite=0.3633560739265397 loc_auc=0.5685671070990249 ete_kappa=0.0 t_auc=0.6207057823129252 tirads_kappa=0.12601021232826393
--> 保存 best(composite=0.3634)-> checkpoints/official_heads_best.pth
[Val Diag] pred_prob: mean=0.4454, std=0.0003, min=0.4425, max=0.4463
Epoch 02/35 [Official Stage A: heads warmup (backbone/seg/CBM frozen)] train_loss=4.6374 val_dice=0.0007 composite=0.3911204221952326 loc_auc=0.5598014144282364 ete_kappa=0.0 t_auc=0.6870748299319729 tirads_kappa=0.207228598723469
--> 保存 best(composite=0.3911)-> checkpoints/official_heads_best.pth
[Val Diag] pred_prob: mean=0.4454, std=0.0003, min=0.4425, max=0.4463
Epoch 03/35 [Official Stage A: heads warmup (backbone/seg/CBM frozen)] train_loss=4.7718 val_dice=0.0007 composite=0.4077422720324185 loc_auc=0.5760652071848764 ete_kappa=0.0 t_auc=0.7055272108843538 tirads_kappa=0.2556216742003776
--> 保存 best(composite=0.4077)-> checkpoints/official_heads_best.pth
[Val Diag] pred_prob: mean=0.4454, std=0.0003, min=0.4425, max=0.4463
Epoch 04/35 [Official Stage A: heads warmup (backbone/seg/CBM frozen)] train_loss=5.0846 val_dice=0.0007 composite=0.40234706536241643 loc_auc=0.5792201987749852 ete_kappa=0.0 t_auc=0.6612670068027212 tirads_kappa=0.269750853341891
[Val Diag] pred_prob: mean=0.4454, std=0.0003, min=0.4425, max=0.4463
Epoch 05/35 [Official Stage A: heads warmup (backbone/seg/CBM frozen)] train_loss=5.3974 val_dice=0.0007 composite=0.4045401918270626 loc_auc=0.587075348034747 ete_kappa=0.0 t_auc=0.7009778911564625 tirads_kappa=0.23315045205161888
[Val Diag] pred_prob: mean=0.4454, std=0.0003, min=0.4425, max=0.4463
Epoch 06/35 [Official Stage A: heads warmup (backbone/seg/CBM frozen)] train_loss=5.7965 val_dice=0.0007 composite=0.4221947946133839 loc_auc=0.5931620686619514 ete_kappa=0.0 t_auc=0.726062925170068 tirads_kappa=0.2902517113424158
--> 保存 best(composite=0.4222)-> checkpoints/official_heads_best.pth
[Val Diag] pred_prob: mean=0.4454, std=0.0003, min=0.4425, max=0.4463
Epoch 07/35 [Official Stage A: heads warmup (backbone/seg/CBM frozen)] train_loss=6.5020 val_dice=0.0007 composite=0.4245384548060465 loc_auc=0.6179153497549782 ete_kappa=0.0 t_auc=0.6923469387755101 tirads_kappa=0.3109327176072598
--> 保存 best(composite=0.4245)-> checkpoints/official_heads_best.pth
[Val Diag] pred_prob: mean=0.4454, std=0.0003, min=0.4425, max=0.4463
Epoch 08/35 [Official Stage A: heads warmup (backbone/seg/CBM frozen)] train_loss=7.0275 val_dice=0.0007 composite=0.4182445647452256 loc_auc=0.6161292688644698 ete_kappa=0.0 t_auc=0.7064625850340137 tirads_kappa=0.2671337019351595
[Val Diag] pred_prob: mean=0.4454, std=0.0003, min=0.4425, max=0.4463
Epoch 09/35 [Official Stage A: heads warmup (backbone/seg/CBM frozen)] train_loss=7.6579 val_dice=0.0007 composite=0.4276885982494121 loc_auc=0.6062518874010692 ete_kappa=0.0 t_auc=0.7107993197278912 tirads_kappa=0.3198945162256155
--> 保存 best(composite=0.4277)-> checkpoints/official_heads_best.pth
[Val Diag] pred_prob: mean=0.4454, std=0.0003, min=0.4425, max=0.4463
Epoch 10/35 [Official Stage A: heads warmup (backbone/seg/CBM frozen)] train_loss=8.2464 val_dice=0.0007 composite=0.43894448611202774 loc_auc=0.6144783026994772 ete_kappa=0.0 t_auc=0.6972789115646258 tirads_kappa=0.3814679484035509
--> 保存 best(composite=0.4389)-> checkpoints/official_heads_best.pth
[optimizer rebuilt] Official Stage B: LLRD unfreeze cls Top-2 head_lr=2.40e-03
[Val Diag] pred_prob: mean=0.4454, std=0.0003, min=0.4425, max=0.4463
Epoch 11/35 [Official Stage B: LLRD unfreeze cls Top-2] train_loss=2.7759 val_dice=0.0007 composite=0.5120394684268793 loc_auc=0.5840753404145805 ete_kappa=0.0 t_auc=0.656547619047619 tirads_kappa=0.45969931830396293
--> 保存 best(composite=0.5120)-> checkpoints/official_heads_best.pth
[Val Diag] pred_prob: mean=0.4454, std=0.0003, min=0.4425, max=0.4463
Epoch 12/35 [Official Stage B: LLRD unfreeze cls Top-2] train_loss=4.4849 val_dice=0.0007 composite=0.5712902724574928 loc_auc=0.6622384918191357 ete_kappa=0.0 t_auc=0.6940051020408162 tirads_kappa=0.6175181115833934
--> 保存 best(composite=0.5713)-> checkpoints/official_heads_best.pth
[Val Diag] pred_prob: mean=0.4454, std=0.0003, min=0.4425, max=0.4463
Epoch 13/35 [Official Stage B: LLRD unfreeze cls Top-2] train_loss=4.0890 val_dice=0.0007 composite=0.5837940428515924 loc_auc=0.692288187493401 ete_kappa=0.0 t_auc=0.6602891156462585 tirads_kappa=0.6718323375956313
--> 保存 best(composite=0.5838)-> checkpoints/official_heads_best.pth
[Val Diag] pred_prob: mean=0.4454, std=0.0003, min=0.4425, max=0.4463
Epoch 14/35 [Official Stage B: LLRD unfreeze cls Top-2] train_loss=3.7701 val_dice=0.0007 composite=0.5965183577402722 loc_auc=0.7249691464434078 ete_kappa=0.0 t_auc=0.7008928571428572 tirads_kappa=0.66214565155685
--> 保存 best(composite=0.5965)-> checkpoints/official_heads_best.pth
[Val Diag] pred_prob: mean=0.4454, std=0.0003, min=0.4425, max=0.4463
Epoch 15/35 [Official Stage B: LLRD unfreeze cls Top-2] train_loss=3.4097 val_dice=0.0007 composite=0.603690132292725 loc_auc=0.7337534535829793 ete_kappa=0.0 t_auc=0.7162414965986396 tirads_kappa=0.6800525059098846
--> 保存 best(composite=0.6037)-> checkpoints/official_heads_best.pth
```

Epoch 16/35 [Official Stage B: LLRD unfreeze cls Top-2] train_loss=3.1577 val_dice=0.0007 composite=0.6029334285519473 loc_auc=0.740436383677468 ete_kappa=0.0 t_auc=0.7133928571428572 tirads_kappa=0.6726669794691658 [Val Diag] pred_prob: mean=0.4454, std=0.0003, min=0.4425, max=0.4463 Epoch 17/35 [Official Stage B: LLRD unfreeze cls Top-2] train_loss=2.8403 val_dice=0.0007 composite=0.596690224279693 loc_auc=0.7342857987200057 ete_kappa=0.0 t_auc=0.6933673469387756 tirads_kappa=0.6732360959984351 [Val Diag] pred_prob: mean=0.4454, std=0.0003, min=0.4425, max=0.4463 Epoch 18/35 [Official Stage B: LLRD unfreeze cls Top-2] train_loss=2.5914 val_dice=0.0007 composite=0.5970827625524981 loc_auc=0.7219495227793867 ete_kappa=0.0 t_auc=0.7105867346938776 tirads_kappa=0.6673645659678923 [Val Diag] pred_prob: mean=0.4454, std=0.0003, min=0.4425, max=0.4463 Epoch 19/35 [Official Stage B: LLRD unfreeze cls Top-2] train_loss=2.3885 val_dice=0.0007 composite=0.5984740717276236 loc_auc=0.7203611481908208 ete_kappa=0.0 t_auc=0.7026360544217687 tirads_kappa=0.6862898307668852 [Val Diag] pred_prob: mean=0.4454, std=0.0003, min=0.4425, max=0.4463 Epoch 20/35 [Official Stage B: LLRD unfreeze cls Top-2] train_loss=2.1363 val_dice=0.0007 composite=0.59727616418442 loc_auc=0.7309068545196999 ete_kappa=0.0 t_auc=0.6925595238095239 tirads_kappa=0.6820811977340168 [Val Diag] pred_prob: mean=0.4454, std=0.0003, min=0.4425, max=0.4463 Epoch 21/35 [Official Stage B: LLRD unfreeze cls Top-2] train_loss=1.9111 val_dice=0.0007 composite=0.593500164774127 loc_auc=0.720286706025912 ete_kappa=0.0 t_auc=0.6911139455782314 tirads_kappa=0.6883302088912695 [Val Diag] pred_prob: mean=0.4454, std=0.0003, min=0.4425, max=0.4463 Epoch 22/35 [Official Stage B: LLRD unfreeze cls Top-2] train_loss=1.7783 val_dice=0.0007 composite=0.5936681005750263 loc_auc=0.7193082309352122 ete_kappa=0.0 t_auc=0.7104804421768708 tirads_kappa=0.6665842149326433 [Val Diag] pred_prob: mean=0.4454, std=0.0003, min=0.4425, max=0.4463 Epoch 23/35 [Official Stage B: LLRD unfreeze cls Top-2] train_loss=1.7132 val_dice=0.0007 composite=0.5962170148873663 loc_auc=0.72441658149336 ete_kappa=0.0 t_auc=0.7175595238095239 tirads_kappa=0.6804524369162657 [Val Diag] pred_prob: mean=0.4454, std=0.0003, min=0.4425, max=0.4463

Epoch 23/35 [Official Stage B: LLRD unfreeze cls Top-2] train_loss=1.7132 val_dice=0.0007 composite=0.5962170148873663 loc_auc=0.72441658149336 ete_kappa=0.0 t_auc=0.7175595238095239 tirads_kappa=0.6804524369162657
  [Val Diag] pred_prob: mean=0.4454, std=0.0003, min=0.4425, max=0.4463
Epoch 24/35 [Official Stage B: LLRD unfreeze cls Top-2] train_loss=1.6624 val_dice=0.0007 composite=0.6004482659866808 loc_auc=0.7200812602024568 ete_kappa=0.0 t_auc=0.7167942176870749 tirads_kappa=0.7050187408677975
  [Val Diag] pred_prob: mean=0.4454, std=0.0003, min=0.4425, max=0.4463
Epoch 25/35 [Official Stage B: LLRD unfreeze cls Top-2] train_loss=1.4294 val_dice=0.0007 composite=0.593290665926008 loc_auc=0.717266724537485 ete_kappa=0.0 t_auc=0.7155187074829932 tirads_kappa=0.6826596885049498
  [Val Diag] pred_prob: mean=0.4454, std=0.0003, min=0.4425, max=0.4463
Epoch 26/35 [Official Stage B: LLRD unfreeze cls Top-2] train_loss=1.5239 val_dice=0.0007 composite=0.5828341478784631 loc_auc=0.7227954164172217 ete_kappa=-0.027542570019540413 t_auc=0.702593537414966 tirads_kappa=0.6688171496786975
  [Val Diag] pred_prob: mean=0.4454, std=0.0003, min=0.4425, max=0.4463
Epoch 27/35 [Official Stage B: LLRD unfreeze cls Top-2] train_loss=1.4117 val_dice=0.0007 composite=0.5826837398690679 loc_auc=0.7243383298788674 ete_kappa=-0.05205973816562359 t_auc=0.7178571428571429 tirads_kappa=0.6834217818676147
  [Val Diag] pred_prob: mean=0.4454, std=0.0003, min=0.4425, max=0.4463
Epoch 28/35 [Official Stage B: LLRD unfreeze cls Top-2] train_loss=1.3904 val_dice=0.0007 composite=0.5948305842239303 loc_auc=0.7188155262227575 ete_kappa=-0.0038569424964935983 t_auc=0.7279761904761906 tirads_kappa=0.688372443974949
  [Val Diag] pred_prob: mean=0.4454, std=0.0003, min=0.4425, max=0.4463
Epoch 29/35 [Official Stage B: LLRD unfreeze cls Top-2] train_loss=1.3948 val_dice=0.0007 composite=0.5829223134626693 loc_auc=0.7218046502147223 ete_kappa=-0.05414271037975604 t_auc=0.7263180272108845 tirads_kappa=0.6763108836813592
  [Val Diag] pred_prob: mean=0.4454, std=0.0003, min=0.4425, max=0.4463
Epoch 30/35 [Official Stage B: LLRD unfreeze cls Top-2] train_loss=1.3258 val_dice=0.0007 composite=0.59085860190603 loc_auc=0.7200489855339143 ete_kappa=-0.011424897373757181 t_auc=0.715093537414966 tirads_kappa=0.6829410378593497
  [optimizer rebuilt] Official Stage C: heads low-lr calibration head_lr=1.20e-04
  [Val Diag] pred_prob: mean=0.4454, std=0.0003, min=0.4425, max=0.4463
Epoch 31/35 [Official Stage C: heads low-lr calibration] train_loss=1.1919 val_dice=0.0007 composite=0.5947775513505004 loc_auc=0.721346220978238 ete_kappa=-0.0038569424964935983 t_auc=0.7163052721088435 tirads_kappa=0.6924588600662362
  [Val Diag] pred_prob: mean=0.4454, std=0.0003, min=0.4425, max=0.4463
Epoch 32/35 [Official Stage C: heads low-lr calibration] train_loss=1.1307 val_dice=0.0007 composite=0.5936320319729538 loc_auc=0.721353188330723 ete_kappa=-0.0038569424964935983 t_auc=0.7163477891156462 tirads_kappa=0.6866817788192145
  [Val Diag] pred_prob: mean=0.4454, std=0.0003, min=0.4425, max=0.4463
Epoch 33/35 [Official Stage C: heads low-lr calibration] train_loss=1.1083 val_dice=0.0007 composite=0.5951522577860315 loc_auc=0.722331838413552 ete_kappa=-0.0038569424964935983 t_auc=0.7180697278911564 tirads_kappa=0.6915823190262652
  [Val Diag] pred_prob: mean=0.4454, std=0.0003