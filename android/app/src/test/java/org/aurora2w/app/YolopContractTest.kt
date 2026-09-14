package org.aurora2w.app

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

class YolopContractTest {
    private fun fixture()=JSONObject(javaClass.getResourceAsStream("/yolop-contract.json")!!.bufferedReader().use {it.readText()})
    private fun rejects(change: (JSONObject)->Unit) {
        val j=fixture();change(j)
        try {ModelMetadata.parse(j.toString());fail("Invalid YOLOP contract accepted")}
        catch(_: IllegalArgumentException) {}
    }
    @Test fun pretrainedModelHasTwoFullResolutionMasksAndNoImuOrDetections() {
        val m=ModelMetadata.parse(fixture().toString())
        assertTrue(m.isYolop);assertFalse(m.usesImu);assertEquals(1,m.maskStride)
        assertTrue(m.classes.isEmpty());assertTrue(m.supervisedDetectionClasses.isEmpty())
        assertEquals(setOf("road","lane"),m.supervisedTasks)
    }
    @Test fun syntheticOrUnknownWeightsCannotUseThePretrainedRoute() {
        rejects {it.put("synthetic_smoke",true)}
        rejects {it.put("training",JSONObject().put("synthetic_smoke",true))}
        rejects {it.put("trained_on_real_data",false)}
        rejects {it.put("training_status","unknown")}
        rejects {it.put("live_inference_allowed",false)}
    }
    @Test fun fakeDetectionsAndSensorInputsAreRejected() {
        rejects {it.put("classes",JSONArray(listOf("car")))}
        rejects {it.put("supervised_detection_classes",JSONArray(listOf("pothole")))}
        rejects {it.put("alignment","idfa")}
        rejects {it.put("input_names",JSONArray(listOf("image","imu")))}
    }
    @Test fun wrongOutputSemanticsOrShapesAreRejected() {
        rejects {it.put("mask_stride",4)}
        rejects {it.put("mask_scores","probability")}
        rejects {it.getJSONObject("output_shapes").put("road_logits",JSONArray(listOf(1,1,80,80)))}
        rejects {it.put("outputs",JSONArray(listOf("road_logits","lane_logits","class_logits")))}
    }
    @Test fun preprocessingDriftIsRejected() {
        rejects {it.getJSONObject("letterbox").put("resize","area")}
        rejects {it.put("mean",JSONArray(listOf(0,0,0)))}
        rejects {it.put("dtype","float16")}
    }
    @Test fun otherFamiliesAndUnsubstantiatedFinetuningAreRejected() {
        rejects {it.put("model_family","arbitrary_onnx")}
        rejects {it.put("idd_finetuned",true)}
        rejects {it.getJSONObject("source").put("repository","https://example.com/unknown")}
    }
}
