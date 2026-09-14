package org.aurora2w.app

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

class ModelMetadataTest {
    private fun fixture(): JSONObject = JSONObject()
        .put("format_version",2).put("layout","NCHW").put("color","RGB")
        .put("image_shape",JSONArray(listOf(1,3,384,640)))
        .put("classes",JSONArray(listOf("person","rider","bicycle","motorcycle","autorickshaw","car","bus","truck","pothole")))
        .put("mean",JSONArray(listOf(.485,.456,.406))).put("std",JSONArray(listOf(.229,.224,.225)))
        .put("pixel_scale",255).put("letterbox",JSONObject().put("pad_value",JSONArray(listOf(114,114,114)))
            .put("dimension_rounding","ties_to_even").put("resize","bilinear_half_pixel")
            .put("placement","center_floor_left_top").put("independent_xy_scale_after_rounding",true))
        .put("alignment","idfa").put("outputs",JSONArray(ModelMetadata.REQUIRED_OUTPUTS))
        .put("dtype","float32").put("input_names",JSONArray(listOf("image","imu")))
        .put("input_shapes",JSONObject().put("image",JSONArray(listOf(1,3,384,640))).put("imu",JSONArray(listOf(1,2))))
        .put("output_shapes",JSONObject().put("road_logits",JSONArray(listOf(1,1,96,160)))
            .put("lane_logits",JSONArray(listOf(1,1,96,160))).put("class_logits",JSONArray(listOf(1,5040,9)))
            .put("boxes_xyxy",JSONArray(listOf(1,5040,4))).put("centerness_logits",JSONArray(listOf(1,5040))))
        .put("model_sha256","a".repeat(64)).put("training_status","trained")
        .put("live_inference_allowed",true).put("trained_on_real_data",true).put("synthetic_smoke",false)
        .put("supervised_tasks",JSONArray(listOf("road","lane")))
        .put("supervised_detection_classes",JSONArray(listOf("pothole")))
    @Test fun trainedIdfaContractHasExplicitImuAndLocationCount() {
        val m = ModelMetadata.parse(fixture().toString())
        assertTrue(m.usesImu); assertEquals(5040,m.expectedLocations)
        assertArrayEquals(longArrayOf(1,3,384,640),m.inputShape)
    }
    @Test(expected = IllegalArgumentException::class) fun syntheticModelCannotActivateLiveCamera() {
        ModelMetadata.parse(fixture().put("synthetic_smoke",true).toString())
    }
    @Test(expected = IllegalArgumentException::class) fun unknownTrainingCannotActivateLiveCamera() {
        ModelMetadata.parse(fixture().put("training_status","unknown").toString())
    }
    @Test(expected = IllegalArgumentException::class) fun reorderedClassesAreRejected() {
        val f = fixture(); val a = f.getJSONArray("classes"); a.put(0,"pothole"); a.put(8,"person")
        ModelMetadata.parse(f.toString())
    }
    @Test(expected = IllegalArgumentException::class) fun incompatibleSpatialShapeIsRejected() {
        ModelMetadata.parse(fixture().put("image_shape",JSONArray(listOf(1,3,383,640))).toString())
    }
    @Test(expected = IllegalArgumentException::class) fun unversionedExportIsRejected() {
        ModelMetadata.parse(fixture().put("format_version",1).toString())
    }
    @Test(expected = IllegalArgumentException::class) fun metadataShapeDriftIsRejected() {
        val f = fixture(); f.getJSONObject("input_shapes").put("imu",JSONArray(listOf(1,3)))
        ModelMetadata.parse(f.toString())
    }
    @Test(expected = IllegalArgumentException::class) fun metadataDtypeDriftIsRejected() {
        ModelMetadata.parse(fixture().put("dtype","float16").toString())
    }
    @Test fun potholeOnlyModelKeepsSegmentationDisabled() {
        val f = fixture().put("supervised_tasks",JSONArray())
        val metadata = ModelMetadata.parse(f.toString())
        assertTrue(metadata.supervisedTasks.isEmpty())
        assertEquals(setOf("pothole"),metadata.supervisedDetectionClasses)
    }
    @Test(expected = IllegalArgumentException::class) fun emptyTrainedScopeCannotEnableLivePredictions() {
        ModelMetadata.parse(fixture().put("supervised_tasks",JSONArray())
            .put("supervised_detection_classes",JSONArray()).toString())
    }
}
