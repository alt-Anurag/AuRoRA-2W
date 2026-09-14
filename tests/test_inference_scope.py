import numpy as np
import torch
from aurora2w.config import CLASS_NAMES
from aurora2w.inference import render_overlay, scoped_detection_output


def test_untrained_masks_never_paint_partial_model_video():
    frame = np.zeros((64,96,3),dtype=np.uint8)
    out = {"road_logits":torch.full((1,1,16,24),20.),"lane_logits":torch.full((1,1,16,24),20.)}
    prediction={"boxes":torch.empty((0,4)),"labels":torch.empty(0,dtype=torch.long),"scores":torch.empty(0)}
    result=render_overlay(frame,out,prediction,np.eye(3),(64,96),supervised_tasks=[])
    np.testing.assert_array_equal(result,frame)
    road=render_overlay(frame,out,prediction,np.eye(3),(64,96),supervised_tasks=["road"])
    assert np.any(road!=frame)
    assert road[32,32,1]>road[32,32,0] # road green, not the stronger untrained lane overlay


def test_classes_are_suppressed_before_decode_without_modifying_model_outputs():
    original={"detection":[{"class_logits":torch.ones(1,len(CLASS_NAMES),2,3)}]}
    scoped=scoped_detection_output(original,["pothole"])
    probability=scoped["detection"][0]["class_logits"].sigmoid()
    assert probability[:, :8].count_nonzero()==0
    assert (probability[:,8]>.5).all()
    assert (original["detection"][0]["class_logits"]==1).all()
