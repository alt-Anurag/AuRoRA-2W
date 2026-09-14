package org.aurora2w.app

import kotlin.math.*

/** Pixel edges, matching Python's rounded letterbox (separate sx and sy). */
data class Letterbox(val sourceWidth: Int, val sourceHeight: Int, val width: Int, val height: Int) {
    init { require(minOf(sourceWidth, sourceHeight, width, height) > 0) }
    private val scale = min(width.toDouble() / sourceWidth, height.toDouble() / sourceHeight)
    val resizedWidth = max(1, round(sourceWidth * scale).toInt())
    val resizedHeight = max(1, round(sourceHeight * scale).toInt())
    val left = (width - resizedWidth) / 2
    val top = (height - resizedHeight) / 2
    val sx = resizedWidth.toFloat() / sourceWidth
    val sy = resizedHeight.toFloat() / sourceHeight
    fun inverse(box: Detection): Detection = box.copy(
        x1 = ((box.x1 - left) / sx).coerceIn(0f, sourceWidth.toFloat()),
        y1 = ((box.y1 - top) / sy).coerceIn(0f, sourceHeight.toFloat()),
        x2 = ((box.x2 - left) / sx).coerceIn(0f, sourceWidth.toFloat()),
        y2 = ((box.y2 - top) / sy).coerceIn(0f, sourceHeight.toFloat()))
}

data class Detection(val x1: Float, val y1: Float, val x2: Float, val y2: Float,
                     val label: Int, val score: Float)

object Postprocess {
    fun sigmoid(value: Float): Float = (1.0 / (1.0 + exp(-value.toDouble()))).toFloat()
    fun iou(a: Detection, b: Detection): Float {
        val intersection = max(0f, min(a.x2, b.x2) - max(a.x1, b.x1)) *
            max(0f, min(a.y2, b.y2) - max(a.y1, b.y1))
        val union = max(0f, a.x2-a.x1)*max(0f, a.y2-a.y1) +
            max(0f, b.x2-b.x1)*max(0f, b.y2-b.y1) - intersection
        return if (union > 0f) intersection / union else 0f
    }
    fun nms(candidates: List<Detection>, threshold: Float = .5f, maximum: Int = 100): List<Detection> {
        require(threshold in 0f..1f && maximum > 0)
        val selected = ArrayList<Detection>()
        for (box in candidates.sortedByDescending { it.score }) {
            if (listOf(box.x1, box.y1, box.x2, box.y2, box.score).any { !it.isFinite() } ||
                box.x2 <= box.x1 || box.y2 <= box.y1) continue
            if (selected.none { it.label == box.label && iou(it, box) > threshold }) selected.add(box)
            if (selected.size >= maximum) break
        }
        return selected
    }
    fun decode(classes: FloatArray, boxes: FloatArray, centers: FloatArray,
               classCount: Int, transform: Letterbox, threshold: Float, nmsThreshold: Float,
               enabledClasses: Set<Int> = (0 until classCount).toSet()): List<Detection> {
        require(boxes.size == centers.size * 4 && classes.size == centers.size * classCount)
        // A bounded heap prevents quadratic suppression on an untrained graph.
        val heap = java.util.PriorityQueue<Detection>(compareBy { it.score })
        for (i in centers.indices) {
            val center = sigmoid(centers[i])
            for (c in 0 until classCount) {
                if (c !in enabledClasses) continue
                val score = sqrt(sigmoid(classes[i * classCount + c]) * center)
                if (!score.isFinite() || score <= threshold) continue
                val box = Detection(boxes[4*i], boxes[4*i+1], boxes[4*i+2], boxes[4*i+3], c, score)
                if (heap.size < 1000) heap.add(box)
                else if (score > heap.peek()!!.score) { heap.poll(); heap.add(box) }
            }
        }
        val clipped = heap.map { box -> box.copy(
            x1 = box.x1.coerceIn(0f,transform.width.toFloat()), x2 = box.x2.coerceIn(0f,transform.width.toFloat()),
            y1 = box.y1.coerceIn(0f,transform.height.toFloat()), y2 = box.y2.coerceIn(0f,transform.height.toFloat())) }
        return nms(clipped, nmsThreshold).map(transform::inverse).filter {
            it.x2 - it.x1 >= 1f && it.y2 - it.y1 >= 1f
        }
    }
    /** PyTorch/OpenCV half-pixel sampling; edge replication, no align_corners. */
    fun bilinear(values: FloatArray, width: Int, height: Int, x: Double, y: Double): Float {
        val xx = x.coerceIn(0.0, (width-1).toDouble())
        val yy = y.coerceIn(0.0, (height-1).toDouble())
        val x0 = floor(xx).toInt(); val y0 = floor(yy).toInt()
        val x1 = min(x0+1, width-1); val y1 = min(y0+1, height-1)
        val dx = (xx-x0).toFloat(); val dy = (yy-y0).toFloat()
        return (values[y0*width+x0]*(1-dx) + values[y0*width+x1]*dx)*(1-dy) +
            (values[y1*width+x0]*(1-dx) + values[y1*width+x1]*dx)*dy
    }
}

/** Unit wxyz quaternion mapping Android device coordinates to world (world Z up). */
data class Quaternion(val w: Double, val x: Double, val y: Double, val z: Double) {
    fun normalized(): Quaternion {
        val norm = sqrt(w*w+x*x+y*y+z*z)
        require(norm.isFinite() && norm > 1e-8) { "Invalid attitude quaternion" }
        return Quaternion(w/norm, x/norm, y/norm, z/norm)
    }
    fun gravityDevice(): DoubleArray {
        val q = normalized()
        return doubleArrayOf(-2*(q.x*q.z-q.y*q.w), -2*(q.y*q.z+q.x*q.w),
            -(1-2*(q.x*q.x+q.y*q.y)))
    }
    companion object {
        fun slerp(a: Quaternion, b: Quaternion, t: Double): Quaternion {
            require(t in 0.0..1.0)
            val p = a.normalized(); var q = b.normalized()
            var dot = p.w*q.w+p.x*q.x+p.y*q.y+p.z*q.z
            if (dot < 0) { q = Quaternion(-q.w,-q.x,-q.y,-q.z); dot = -dot }
            dot = dot.coerceIn(-1.0, 1.0)
            val u: Double; val v: Double
            if (dot > .9995) { u = 1-t; v = t }
            else { val angle = acos(dot); u = sin((1-t)*angle)/sin(angle); v = sin(t*angle)/sin(angle) }
            return Quaternion(u*p.w+v*q.w, u*p.x+v*q.x, u*p.y+v*q.y, u*p.z+v*q.z).normalized()
        }
    }
}

data class AttitudeSample(val timeNs: Long, val q: Quaternion, val accurate: Boolean)

class AttitudeBuffer(private val maxGapNs: Long = 50_000_000L) {
    private val samples = ArrayDeque<AttitudeSample>()
    @Synchronized fun add(sample: AttitudeSample) {
        if (samples.isNotEmpty() && sample.timeNs <= samples.last().timeNs) return
        samples.addLast(sample)
        while (samples.size > 1000) samples.removeFirst()
    }
    @Synchronized fun clear() = samples.clear()
    @Synchronized fun at(timeNs: Long): Quaternion? {
        if (samples.size < 2 || timeNs < samples.first().timeNs || timeNs > samples.last().timeNs) return null
        var before = samples.first()
        for (after in samples) {
            if (after.timeNs == timeNs) return if (after.accurate) after.q else null
            if (after.timeNs > timeNs) {
                val gap = after.timeNs-before.timeNs
                if (!before.accurate || !after.accurate || gap > maxGapNs) return null
                return Quaternion.slerp(before.q, after.q, (timeNs-before.timeNs).toDouble()/gap)
            }
            before = after
        }
        return null
    }
}

object CameraGeometry {
    fun wrap(angle: Double): Double = atan2(sin(angle), cos(angle))
    /** Rear optical x right/y down/z forward; positive clockwise SCENE roll.
     * Natural upright image -> device is diag(1,-1,-1).
     * E = diag(1,-1,-1) Rz(SENSOR_ORIENTATION - ImageProxy.rotationDegrees).
     */
    fun sceneRoll(q: Quaternion, sensorOrientation: Int, imageRotation: Int): Double? {
        require(sensorOrientation in listOf(0,90,180,270) && imageRotation in listOf(0,90,180,270))
        val gravity = q.gravityDevice()
        val a = Math.toRadians((sensorOrientation-imageRotation).toDouble())
        val naturalX = gravity[0]; val naturalY = -gravity[1]
        val x = cos(a)*naturalX + sin(a)*naturalY
        val y = -sin(a)*naturalX + cos(a)*naturalY
        if (hypot(x,y) < .1) return null
        return atan2(-x, y)
    }
}
