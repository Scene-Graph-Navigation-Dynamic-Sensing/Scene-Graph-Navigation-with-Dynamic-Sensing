"""Evaluation of the scene graph pipeline against dataset ground truth.

replica_gt     -- Replica ground truth: labelled mesh vertices, classes, visibility.
open_vocab     -- dataset-independent node matching and open-vocabulary metrics
                  (Acc@k / AUC, object R@k, node precision / recall / N_dup,
                  mIoU / mAcc / F-mIoU).
artifacts      -- per-scene files for checking results by eye (video, voxel
                  maps, gallery, tables, results.md).

Drivers: scripts/eval_replica_scene.py (one scene), scripts/combine_replica_results.py.
See docs/evaluation_metrics.md for which metrics are reported and why.
"""
