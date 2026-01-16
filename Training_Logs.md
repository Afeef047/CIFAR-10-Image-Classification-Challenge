CIFAR-10 Challenge Training Log


Dataset:
- Train: 40,000 images
- Validation: 5,000 images
- Test: 5,000 images
- Split: 80/10/10

Hardware:
- Google Colab (NVIDIA T4 GPU)
- Automatic Mixed Precision (AMP): Enabled

--------------------------------
Level 1 — Baseline (ResNet-18)
--------------------------------
Model: ResNet-18 (ImageNet Pretrained)
Optimizer: Adam (lr=1e-3)
Epochs: 8
Batch Size: 128

Epoch | Train Loss | Val Acc
--------------------------------
1     | 0.5801     | 0.8380
2     | 0.3739     | 0.8655
3     | 0.3037     | 0.8782
4     | 0.2575     | 0.8871
5     | 0.2339     | 0.8963
6     | 0.1974     | 0.9012
7     | 0.1732     | 0.9075
8     | 0.1618     | 0.8940

Level 1 Final Test Accuracy: 0.8978
Checkpoint saved: models/level1.pth

--------------------------------
Level 2 — Intermediate (ConvNeXt-Tiny)
--------------------------------
Model: ConvNeXt-Tiny (Pretrained on ImageNet)
Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)
Scheduler: CosineAnnealingLR
AMP: Enabled
Epochs: 12

Epoch | Train Loss | Val Acc
--------------------------------
1     | 0.5979     | 0.8417
2     | 0.3715     | 0.8702
3     | 0.2862     | 0.8752
4     | 0.2316     | 0.8763
5     | 0.1786     | 0.8881
6     | 0.1299     | 0.8869
7     | 0.0908     | 0.9003
8     | 0.0617     | 0.9031
9     | 0.0405     | 0.9120
10    | 0.0226     | 0.9123
11    | 0.0166     | 0.9197
12    | 0.0126     | 0.9159

Level 2 Final Test Accuracy: 0.9264
Checkpoint saved: models/level2.pth

--------------------------------
Level 3 — Custom Head + Partial Fine-Tuning
--------------------------------
Model: ConvNeXt-Tiny + Custom MLP Head (512 → 10)
Fine-Tuning: Last Stage + Head
Optimizer: AdamW
Epochs: 18
AMP: Enabled

Epoch | Train Loss | Val Acc
--------------------------------
1     | 0.5571     | 0.8563
2     | 0.3380     | 0.8670
3     | 0.2568     | 0.8743
4     | 0.1999     | 0.8855
5     | 0.1578     | 0.8784
6     | 0.1221     | 0.8893
7     | 0.0957     | 0.8884
8     | 0.0757     | 0.8944
9     | 0.0607     | 0.8903
10    | 0.0460     | 0.8959
11    | 0.0378     | 0.8964
12    | 0.0292     | 0.8935
13    | 0.0255     | 0.9010
14    | 0.0175     | 0.8996
15    | 0.0136     | 0.9052
16    | 0.0107     | 0.9041
17    | 0.0104     | 0.9003
18    | 0.0088     | 0.9056

Level 3 Final Test Accuracy: 0.9093
Confusion Matrix Generated
Checkpoint saved: models/level3.pth

--------------------------------
Level 4 — Ensemble (Weighted Soft Voting)
--------------------------------

Ensemble Components:
- Level 2: ConvNeXt-Tiny
- Level 3: ConvNeXt-Tiny + MLP Head

Weight Search (w2, w3):

w2 | w3 | Test Acc
--------------------
0.3 | 0.7 | 0.9280
0.4 | 0.6 | 0.9301
0.5 | 0.5 | 0.9330  <-- best
0.6 | 0.4 | 0.9314
0.7 | 0.3 | 0.9275

Best Ensemble Weights: (0.5, 0.5)

Final Ensemble Test Accuracy: 0.9330
Plot saved: results/level4_ensemble.png


