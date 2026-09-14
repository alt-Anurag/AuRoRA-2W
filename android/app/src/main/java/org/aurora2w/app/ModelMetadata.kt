package org.aurora2w.app

import org.json.JSONObject

data class ModelMetadata(
    val width: Int, val height: Int, val classes: List<String>, val mean: FloatArray,
    val std: FloatArray, val alignment: String, val outputs: List<String>,
    val sha256: String, val synthetic: Boolean, val provenance: String,
    val supervisedTasks: Set<String>, val supervisedDetectionClasses: Set<String>,
    val raw: JSONObject, val family: String = "aurora"
) {
    val isYolop get() = family == "yolop_segmentation"
    val maskStride get() = if(isYolop) 1 else 4
    val usesImu get() = alignment == "idfa"
    val inputShape get() = longArrayOf(1, 3, height.toLong(), width.toLong())
    val expectedLocations get() = listOf(8,16,32).sumOf { (height/it)*(width/it) }
    companion object {
        val REQUIRED_OUTPUTS = listOf("road_logits", "lane_logits", "class_logits", "boxes_xyxy", "centerness_logits")
        fun parse(text: String): ModelMetadata {
            require(text.length <= 1_048_576) { "Metadata exceeds 1 MiB" }
            val json = JSONObject(text)
            if(json.optInt("format_version") == 3) return YolopContract.parse(json)
            require(json.getInt("format_version") == 2) { "Export a version 2 Aurora bundle" }
            require(json.getString("layout") == "NCHW" && json.getString("color") == "RGB") { "Expected normalized NCHW RGB" }
            val shape = json.optJSONArray("image_shape") ?: json.getJSONArray("input_shape")
            require(shape.length() == 4 && shape.getInt(0) == 1 && shape.getInt(1) == 3)
            val h = shape.getInt(2); val w = shape.getInt(3)
            require(h in 64..1024 && w in 64..1536 && h%32 == 0 && w%32 == 0) { "Unsupported static image size" }
            val names = json.getJSONArray("classes").let { a -> List(a.length()) { a.getString(it) } }
            require(names == listOf("person", "rider", "bicycle", "motorcycle", "autorickshaw", "car", "bus", "truck", "pothole")) { "Class order differs from Aurora" }
            fun vector(name: String): FloatArray {
                val a = json.optJSONArray(name) ?: json.getJSONObject("normalization").getJSONArray(name)
                require(a.length() == 3)
                return FloatArray(3) { a.getDouble(it).toFloat() }.also { values -> require(values.all { it.isFinite() }) }
            }
            val mean = vector("mean"); val std = vector("std")
            require(std.all { it > 0f })
            require(json.getDouble("pixel_scale") == 255.0) { "Unsupported pixel scale" }
            val letterbox = json.getJSONObject("letterbox")
            val pad = letterbox.getJSONArray("pad_value")
            require(pad.length() == 3 && (0..2).all { pad.getInt(it) == 114 }) { "Unsupported letterbox padding" }
            require(letterbox.getString("dimension_rounding") == "ties_to_even")
            require(letterbox.getString("resize") == "bilinear_half_pixel")
            val alignment = json.getString("alignment")
            require(alignment in listOf("none", "idfa")) { "Unsupported alignment" }
            require(json.getString("dtype") == "float32") { "Only float32 bundles are supported" }
            require(letterbox.getString("placement") == "center_floor_left_top" &&
                letterbox.getBoolean("independent_xy_scale_after_rounding")) { "Unsupported letterbox placement" }
            val expectedInputs = if (alignment == "idfa") setOf("image","imu") else setOf("image")
            val declaredInputs = json.getJSONArray("input_names").let { a -> List(a.length()) { a.getString(it) } }
            require(declaredInputs.toSet() == expectedInputs && declaredInputs.distinct().size == declaredInputs.size)
            fun declaredShape(table: JSONObject, name: String, expected: List<Int>) {
                val a = table.getJSONArray(name)
                require(a.length() == expected.size && expected.indices.all { a.getInt(it) == expected[it] }) {
                    "Declared $name shape differs from the Aurora contract"
                }
            }
            val inputShapes = json.getJSONObject("input_shapes")
            require(inputShapes.keys().asSequence().toSet() == expectedInputs)
            declaredShape(inputShapes,"image",listOf(1,3,h,w))
            if (alignment == "idfa") declaredShape(inputShapes,"imu",listOf(1,2))
            val outputs = json.getJSONArray("outputs").let { a -> List(a.length()) { a.getString(it) } }
            require(outputs.toSet().containsAll(REQUIRED_OUTPUTS) && outputs.distinct().size == outputs.size)
            require(outputs.all { it in REQUIRED_OUTPUTS || it == "pothole_logits" })
            val outputShapes = json.getJSONObject("output_shapes")
            require(outputShapes.keys().asSequence().toSet() == outputs.toSet())
            val locations = listOf(8,16,32).sumOf { (h/it)*(w/it) }
            for (name in outputs) declaredShape(outputShapes,name,when (name) {
                "class_logits" -> listOf(1,locations,names.size)
                "boxes_xyxy" -> listOf(1,locations,4)
                "centerness_logits" -> listOf(1,locations)
                else -> listOf(1,1,h/4,w/4)
            })
            val hash = json.optString("model_sha256", json.optString("sha256", "")).lowercase()
            require(hash.matches(Regex("[0-9a-f]{64}"))) { "Bundle must include model_sha256" }
            val synthetic = json.optBoolean("synthetic_smoke", false) ||
                json.optJSONObject("training")?.optBoolean("synthetic_smoke", false) == true
            require(!synthetic && json.optBoolean("live_inference_allowed", false) &&
                json.optBoolean("trained_on_real_data", false) && json.optString("training_status") == "trained") {
                "Live inference needs a real-data trained bundle. Synthetic or unverified weights are not enabled."
            }
            fun stringSet(name: String): Set<String> {
                val a = json.getJSONArray(name)
                val values = List(a.length()) { a.getString(it) }
                require(values.distinct().size == values.size)
                return values.toSet()
            }
            val supervisedTasks = stringSet("supervised_tasks")
            val supervisedClasses = stringSet("supervised_detection_classes")
            require(supervisedTasks.all { it in setOf("road","lane") } && supervisedClasses.all { it in names })
            require(supervisedTasks.isNotEmpty() || supervisedClasses.isNotEmpty()) { "No trained task scope declared" }
            val provenance = if (json.optBoolean("requires_resolution_reevaluation",false))
                "RESIZED EXPORT · accuracy needs reevaluation"
                else "RESEARCH MODEL · accuracy not verified on this device"
            return ModelMetadata(w,h,names,mean,std,alignment,outputs,hash,synthetic,provenance,
                supervisedTasks,supervisedClasses,json)
        }
    }
}
