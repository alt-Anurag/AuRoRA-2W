package org.aurora2w.app

import org.json.JSONObject

/** Version 3 is a narrow segmentation-only adapter, independent of Aurora v2. */
internal object YolopContract {
    fun parse(json: JSONObject): ModelMetadata {
        require(json.getString("model_family") == "yolop_segmentation")
        require(json.getString("layout") == "NCHW" && json.getString("color") == "RGB")
        require(json.getString("dtype") == "float32" && json.getString("alignment") == "none")
        val shape=json.getJSONArray("image_shape")
        require(shape.length()==4 && shape.getInt(0)==1 && shape.getInt(1)==3)
        val h=shape.getInt(2);val w=shape.getInt(3)
        require(h==w && h in setOf(320,640)) { "Supported YOLOP exports are 320 or 640 square" }
        fun strings(key: String)=json.getJSONArray(key).let { a->List(a.length()) {a.getString(it)} }
        require(strings("classes").isEmpty() && strings("supervised_detection_classes").isEmpty())
        require(strings("input_names")==listOf("image"))
        val outputs=listOf("road_logits","lane_logits")
        require(strings("outputs")==outputs && strings("supervised_tasks")==listOf("road","lane"))
        fun shapes(key: String,expected: Map<String,List<Int>>) {
            val table=json.getJSONObject(key)
            require(table.keys().asSequence().toSet()==expected.keys)
            for((name,dims) in expected) {
                val a=table.getJSONArray(name)
                require(a.length()==dims.size && dims.indices.all {a.getInt(it)==dims[it]})
            }
        }
        shapes("input_shapes",mapOf("image" to listOf(1,3,h,w)))
        shapes("output_shapes",outputs.associateWith {listOf(1,1,h,w)})
        fun vector(key: String,expected: FloatArray): FloatArray {
            val a=json.getJSONArray(key)
            require(a.length()==3)
            return FloatArray(3) {a.getDouble(it).toFloat()}.also {require(it.contentEquals(expected))}
        }
        val mean=vector("mean",floatArrayOf(.485f,.456f,.406f))
        val std=vector("std",floatArrayOf(.229f,.224f,.225f))
        require(json.getDouble("pixel_scale")==255.0)
        val box=json.getJSONObject("letterbox")
        val pad=box.getJSONArray("pad_value")
        require(pad.length()==3 && (0..2).all {pad.getInt(it)==114})
        require(box.getString("dimension_rounding")=="ties_to_even" && box.getString("resize")=="bilinear_half_pixel")
        require(box.getString("placement")=="center_floor_left_top" && box.getBoolean("independent_xy_scale_after_rounding"))
        require(json.getInt("mask_stride")==1 && json.getString("mask_scores")=="foreground_minus_background_presigmoid")
        require(json.getString("mask_projection")=="sigmoid_then_crop_bilinear_threshold")
        require(json.getString("training_status")=="pretrained" && json.getBoolean("trained_on_real_data")
            && json.getBoolean("live_inference_allowed") && !json.getBoolean("synthetic_smoke"))
        require(json.optJSONObject("training")?.optBoolean("synthetic_smoke",false) != true)
        require(!json.getBoolean("idd_finetuned")) { "This adapter declares the original pretrained baseline" }
        val source=json.getJSONObject("source")
        require(source.getString("repository")=="https://github.com/hustvl/YOLOP")
        require(source.getString("commit").matches(Regex("[0-9a-f]{40}")))
        require(source.getString("onnx_sha256").matches(Regex("[0-9a-f]{64}")))
        require(source.getString("training_dataset")=="BDD100K")
        require(source.getString("license")=="MIT" && source.getString("license_text").isNotBlank())
        val hash=json.getString("model_sha256")
        require(hash.matches(Regex("[0-9a-f]{64}")))
        return ModelMetadata(w,h,emptyList(),mean,std,"none",outputs,hash,false,
            "YOLOP authors' baseline · device and night accuracy unvalidated",
            setOf("road","lane"),emptySet(),json,"yolop_segmentation")
    }
}
