package org.aurora2w.app

import ai.onnxruntime.*
import android.graphics.Bitmap
import android.graphics.Color
import java.io.Closeable
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.*

data class InferenceResult(val mask: Bitmap, val boxes: List<Detection>, val networkMs: Double,
                           val preprocessingMs: Double, val postprocessingMs: Double)

/** Owns one ORT CPU session (XNNPACK preferred for YOLOP); all calls, replacement and close happen on analyzer executor. */
class InferenceEngine(modelFile: File, val metadata: ModelMetadata,
    useXnnpack: Boolean = metadata.isYolop, useNnapi: Boolean = false, autoSelect: Boolean = false) : Closeable {
    private val environment = OrtEnvironment.getEnvironment()
    private lateinit var session: OrtSession
    var runtimeLabel: String = "ONNX CPU"
        private set
    private val workers=min(2,max(1,Runtime.getRuntime().availableProcessors()/2))
    private var workspace: Workspace? = null
    private class Workspace(val t: Letterbox,val stride: Int) {
        val tensor=FloatArray(3*t.width*t.height)
        val rgb=IntArray(t.sourceWidth*t.sourceHeight)
        val pixels=IntArray(t.sourceWidth*t.sourceHeight)
        val compactPixels=IntArray(t.resizedWidth*t.resizedHeight)
        val crop=FloatArray(t.resizedWidth*t.resizedHeight)
        val resize=SamplingGrid(t.sourceWidth,t.sourceHeight,t.resizedWidth,t.resizedHeight)
        val projection=SamplingGrid(t.resizedWidth,t.resizedHeight,t.sourceWidth,t.sourceHeight)
        val upsample=if(stride==1) null else SamplingGrid(t.width/stride,t.height/stride,t.width,t.height)
    }
    private fun workspace(t: Letterbox): Workspace {
        val old=workspace
        if(old!=null && old.t==t) return old
        return Workspace(t,metadata.maskStride).also {workspace=it}
    }
    init {
        var accelerated=false
        if(useNnapi) {
            try {
                session=createSession(modelFile,false,true)
                try {validateContract();warmup()} catch(failure: Exception) {session.close();throw failure}
                accelerated=true
                runtimeLabel="ONNX NNAPI · hardware + CPU fallback"
            } catch(failure: Exception) {
                android.util.Log.w("AuroraRuntime","NNAPI unavailable; using CPU",failure)
            }
        }
        if(!accelerated && useXnnpack) {
            try {
                session=createSession(modelFile,true)
                try {validateContract();warmup()} catch(failure: Exception) {session.close();throw failure}
                accelerated=true
                runtimeLabel="ONNX XNNPACK CPU · $workers workers"
            } catch(failure: Exception) {
                android.util.Log.w("AuroraRuntime","XNNPACK unavailable; using CPU",failure)
            }
        }
        if(!accelerated) {
            session=createSession(modelFile,false)
            try {validateContract();warmup()} catch(failure: Exception) {session.close();throw failure}
            runtimeLabel="ONNX CPU · $workers workers"
        }
        if(autoSelect && metadata.isYolop && android.os.Build.VERSION.SDK_INT>=29) selectMeasuredHardware(modelFile)
    }
    /** CPU remains available until an NNAPI candidate passes a numerical and timing check.
     * Three fixed-pattern runs compare this device and model. This is not an accuracy benchmark.
     */
    private fun selectMeasuredHardware(modelFile: File) {
        val cpu=session
        val cpuLabel=runtimeLabel
        var candidate: OrtSession?=null
        val input=FloatArray(3*metadata.width*metadata.height) {((it*37)%251)/125f-1f}
        fun measure(): Pair<Double,Map<String,FloatArray>> {
            val times=DoubleArray(3)
            var output=emptyMap<String,FloatArray>()
            for(i in times.indices) {
                val start=System.nanoTime()
                output=runTensor(input,0f,0f)
                times[i]=(System.nanoTime()-start)/1e6
            }
            times.sort()
            return times[1] to output
        }
        try {
            val baseline=measure()
            candidate=createSession(modelFile,false,true)
            session=candidate
            validateContract();warmup()
            val hardware=measure()
            val parity=metadata.outputs.all {name->
                val wanted=baseline.second.getValue(name)
                val actual=hardware.second.getValue(name)
                actual.size==wanted.size && actual.indices.all {
                    abs(actual[it]-wanted[it])<=5e-3f+2e-3f*abs(wanted[it])
                }
            }
            val select=parity && hardware.first<baseline.first*.90
            android.util.Log.i("AuroraRuntime","Auto backend: cpu_ms=${baseline.first}, nnapi_ms=${hardware.first}, parity=$parity, hardware_selected=$select")
            if(select) {
                runtimeLabel="ONNX NNAPI · hardware + CPU fallback"
                cpu.close()
            } else {
                candidate.close();session=cpu;runtimeLabel=cpuLabel
            }
        } catch(failure: Exception) {
            candidate?.close();session=cpu;runtimeLabel=cpuLabel
            android.util.Log.w("AuroraRuntime","Auto backend kept CPU",failure)
        }
    }
    private fun createSession(modelFile: File,xnnpack: Boolean,nnapi: Boolean=false): OrtSession {
        val options = OrtSession.SessionOptions()
        val threads=workers
        options.setIntraOpNumThreads(if(xnnpack)1 else threads)
        options.setInterOpNumThreads(1)
        // Keep CPU available for CameraX, sensors and UI between inference work.
        options.addConfigEntry("session.intra_op.allow_spinning","0")
        options.addConfigEntry("session.inter_op.allow_spinning","0")
        options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        try {
            if(nnapi) options.addNnapi(java.util.EnumSet.of(ai.onnxruntime.providers.NNAPIFlags.CPU_DISABLED))
            if(xnnpack) options.addXnnpack(mapOf("intra_op_num_threads" to threads.toString()))
            return environment.createSession(modelFile.absolutePath, options)
        } finally { options.close() }
    }
    private fun checkShape(info: NodeInfo?, expected: LongArray, name: String) {
        val tensor = info?.info as? TensorInfo ?: error("Missing tensor $name")
        require(tensor.type == OnnxJavaType.FLOAT && tensor.shape.contentEquals(expected)) {
            "$name must be float32 ${expected.contentToString()}, got ${tensor.shape.contentToString()}"
        }
    }
    private fun validateContract() {
        require(session.inputNames == (if (metadata.usesImu) setOf("image", "imu") else setOf("image"))) { "Graph inputs do not match metadata" }
        checkShape(session.inputInfo["image"], metadata.inputShape, "image")
        if (metadata.usesImu) checkShape(session.inputInfo["imu"], longArrayOf(1,2), "imu")
        require(session.outputNames == metadata.outputs.toSet()) { "Graph outputs differ from metadata" }
        val n = metadata.expectedLocations.toLong()
        for (name in metadata.outputs) {
            val shape = when (name) {
                "class_logits" -> longArrayOf(1,n,metadata.classes.size.toLong())
                "boxes_xyxy" -> longArrayOf(1,n,4)
                "centerness_logits" -> longArrayOf(1,n)
                else -> longArrayOf(1,1,(metadata.height/metadata.maskStride).toLong(),(metadata.width/metadata.maskStride).toLong())
            }
            checkShape(session.outputInfo[name], shape, name)
        }
    }
    private fun runTensor(input: FloatArray, roll: Float, confidence: Float): Map<String, FloatArray> {
        val tensors = HashMap<String, OnnxTensor>()
        try {
            tensors["image"] = OnnxTensor.createTensor(environment, FloatBuffer.wrap(input), metadata.inputShape)
            if (metadata.usesImu) tensors["imu"] = OnnxTensor.createTensor(environment,
                FloatBuffer.wrap(floatArrayOf(roll,confidence)), longArrayOf(1,2))
            session.run(tensors).use { result ->
                return metadata.outputs.associateWith { name ->
                    val tensor = result.get(name).orElseThrow { IllegalArgumentException("Missing output $name") } as OnnxTensor
                    val buffer = tensor.floatBuffer
                    FloatArray(buffer.remaining()).also { buffer.get(it) }.also { values ->
                        require(values.all { it.isFinite() }) { "Non-finite $name output; model rejected" }
                    }
                }
            }
        } finally { tensors.values.forEach { it.close() } }
    }
    private fun warmup() {
        // Import-time execution verifies actual runtime operator support, not just ONNX syntax.
        runTensor(FloatArray(3*metadata.height*metadata.width), 0f, 0f)
    }
    /** For recorded tensor parity tests; the app UI never calls this with fixture weights. */
    fun rawOutputsForValidation(input: FloatArray, roll: Float, confidence: Float): Map<String,FloatArray> =
        runTensor(input,roll,confidence)
    fun infer(bitmap: Bitmap, roll: Float, confidence: Float, threshold: Float, maskThreshold: Float, compactMask: Boolean = false): InferenceResult {
        val start = System.nanoTime()
        val transform = Letterbox(bitmap.width,bitmap.height,metadata.width,metadata.height)
        val input = preprocess(bitmap, transform)
        val networkStart = System.nanoTime()
        val output = runTensor(input, roll, confidence)
        val postStart = System.nanoTime()
        val boxes = if(metadata.isYolop) emptyList() else Postprocess.decode(output.getValue("class_logits"), output.getValue("boxes_xyxy"),
            output.getValue("centerness_logits"), metadata.classes.size, transform, threshold,.5f,
            metadata.classes.indices.filter { metadata.classes[it] in metadata.supervisedDetectionClasses }.toSet())
        val mask = renderMasks(output, transform, maskThreshold,compactMask)
        return InferenceResult(mask,boxes,(postStart-networkStart)/1e6,(networkStart-start)/1e6,
            (System.nanoTime()-postStart)/1e6)
    }
    fun preprocessForValidation(bitmap: Bitmap): FloatArray = preprocess(bitmap,Letterbox(bitmap.width,bitmap.height,metadata.width,metadata.height)).copyOf()
    private fun preprocess(bitmap: Bitmap, t: Letterbox): FloatArray {
        val count = t.width*t.height
        val buffers=workspace(t)
        val tensor=buffers.tensor
        val rgb=buffers.rgb
        bitmap.getPixels(rgb,0,bitmap.width,0,0,bitmap.width,bitmap.height)
        val sampling=buffers.resize
        for (c in 0..2) {
            val pad = (114f/255f-metadata.mean[c])/metadata.std[c]
            java.util.Arrays.fill(tensor,c*count,(c+1)*count,pad)
        }
        for(y in 0 until t.resizedHeight) for(x in 0 until t.resizedWidth) {
            sampling.writeNormalizedRgb(rgb,x,y,tensor,(y+t.top)*t.width+x+t.left,count,metadata.mean,metadata.std)
        }
        return tensor
    }
    private fun renderMasks(output: Map<String,FloatArray>, t: Letterbox, threshold: Float,compact: Boolean): Bitmap {
        val buffers=workspace(t)
        // Display-only fast path. Network input/output and evaluation retain their exact contract.
        val compactYolop=compact && metadata.isYolop && t.sourceWidth>t.resizedWidth
        val pixels=if(compactYolop)buffers.compactPixels else buffers.pixels
        java.util.Arrays.fill(pixels,0)
        val colors = if(metadata.isYolop) mapOf("road_logits" to Color.argb(120,255,217,72),
            "lane_logits" to Color.argb(220,255,77,145)) else mapOf("road_logits" to Color.argb(76,54,224,160),
            "lane_logits" to Color.argb(205,255,221,81), "pothole_logits" to Color.argb(185,255,95,140))
        val projection=buffers.projection
        val upsample=buffers.upsample
        for ((task,color) in colors) {
            if (task.removeSuffix("_logits") !in metadata.supervisedTasks) continue
            val logits = output[task] ?: continue
            // Padding is discarded by the contract, so do not sigmoid or allocate it.
            val crop=buffers.crop
            for(y in 0 until t.resizedHeight) for(x in 0 until t.resizedWidth) {
                crop[y*t.resizedWidth+x]=Postprocess.sigmoid(
                    if(metadata.maskStride==1) logits[(y+t.top)*t.width+x+t.left]
                    else upsample!!.sample(logits,x+t.left,y+t.top))
            }
            if(compactYolop) {
                for(i in crop.indices) if(crop[i]>=threshold) pixels[i]=color
            } else {
                for (y in 0 until t.sourceHeight) for (x in 0 until t.sourceWidth) {
                    val value = projection.sample(crop,x,y)
                    if (value >= threshold) pixels[y*t.sourceWidth+x] = color
                }
            }
        }
        return Bitmap.createBitmap(pixels,if(compactYolop)t.resizedWidth else t.sourceWidth,
            if(compactYolop)t.resizedHeight else t.sourceHeight,Bitmap.Config.ARGB_8888)
    }
    override fun close() = session.close()
}
