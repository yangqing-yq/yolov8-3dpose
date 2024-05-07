import os
import json
import pandas as pd
import numpy as np

cur_dir = os.path.abspath(os.path.dirname(__file__))

def saveMetric(metrics, filename):
    print("Saving to json...")
    data = {
        "metrics.names": metrics.names,
        "metrics.speed": metrics.speed,
        
        "metrics.box.mean_precision": metrics.box.mp,
        "metrics.box.precision": metrics.box.p.tolist(),
        "metrics.box.mean_recall": metrics.box.mr,
        "metrics.box.recall": metrics.box.r.tolist(),
        "metrics.box.f1": metrics.box.f1.tolist(),
        "metrics.box.map50": metrics.box.map50,
        "metrics.box.ap50": metrics.box.ap50.tolist(),
        "metrics.box.map50-95": metrics.box.map,
        "metrics.box.maps": metrics.box.maps.tolist(),

        "metrics.pose.mean_precision": metrics.pose.mp,
        "metrics.pose.precision": metrics.pose.p.tolist(),
        "metrics.pose.mean_recall": metrics.pose.mr,
        "metrics.pose.recall": metrics.pose.r.tolist(),
        "metrics.pose.f1": metrics.pose.f1.tolist(),
        "metrics.pose.map50": metrics.pose.map50,
        "metrics.pose.ap50": metrics.pose.ap50.tolist(),
        "metrics.pose.map50-95": metrics.pose.map,
        "metrics.pose.maps": metrics.pose.maps.tolist()
        }

    json_path = os.path.join(cur_dir, '..', filename + '.json')
    with open(json_path, 'w') as f:
        json.dump(data, f, indent = 4)

    print("Saving to pandas...")
    table = {
        "Class": ['all'] + list(metrics.names.values()),
        
        "Box.Precision" : [round(metrics.box.mp, 4)] + metrics.box.p.round(4).tolist(),
        "Box.Recall" : [round(metrics.box.mr, 4)] + metrics.box.r.round(4).tolist(),
        "Box.mAP50" : [round(metrics.box.map50, 4)] + metrics.box.ap50.round(4).tolist(),
        "Box.mAP50-90" : [round(metrics.box.map, 4)] + metrics.box.maps.round(4).tolist(),

        "Pose.Precision" : [round(metrics.pose.mp, 4)] + metrics.pose.p.round(4).tolist(),
        "Pose.Recall" : [round(metrics.pose.mr, 4)] + metrics.pose.r.round(4).tolist(),
        "Pose.mAP50" : [round(metrics.pose.map50, 4)] + metrics.pose.ap50.round(4).tolist(),
        "Pose.mAP50-90" : [round(metrics.pose.map, 4)] + metrics.pose.maps.round(4).tolist()
        }

    df = pd.DataFrame.from_dict(table)
    df.to_csv(os.path.join(cur_dir, '..', filename + '.csv'), index = False)