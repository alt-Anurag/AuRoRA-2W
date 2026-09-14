@file:androidx.annotation.OptIn(androidx.camera.view.TransformExperimental::class)

package org.aurora2w.app

import android.content.Context
import android.graphics.*
import android.view.View
import android.widget.FrameLayout
import android.os.SystemClock
import androidx.camera.view.PreviewView
import androidx.camera.view.transform.OutputTransform
import androidx.camera.view.transform.CoordinateTransform
import kotlin.math.min

data class DashboardFrame(val image: Bitmap, val prediction: InferenceResult?, val classes: List<String>,
                          val roll: Float, val imuTrusted: Boolean, val banner: String,
                          val sourceTransform: OutputTransform? = null,
                          val capturedNs: Long? = null, val orientation: Quaternion? = null)

/** Hardware camera preview is independent of inference. Matched mode retains the exact source.
 * Live masks use CameraX viewport transforms plus an explicit age and angular-drift gate.
 */
class DashboardView(context: Context) : FrameLayout(context) {
    val preview = PreviewView(context).apply {
        implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        scaleType = PreviewView.ScaleType.FIT_CENTER
    }
    private val overlay = object : View(context) {
        override fun onDraw(canvas: Canvas) { drawFrame(canvas) }
    }
    private var currentOrientation: Quaternion? = null
    var matchedFrames = false
        set(value) { field=value;overlay.invalidate() }
    var paused = false
        set(value) { field=value;overlay.invalidate() }
    init {
        setBackgroundColor(Color.rgb(5,12,15))
        addView(preview,LayoutParams(-1,-1))
        addView(overlay,LayoutParams(-1,-1))
    }
    fun updateMotion(value: MotionReading) {
        currentOrientation=value.orientation
        overlay.invalidate()
    }
    fun maskStatus(): String {
        if(paused) return "Paused on a matched frame"
        if(matchedFrames) return "Matched view · camera updates with model"
        val data=frame ?: return "Live camera · waiting for perception"
        if(!roadAndLaneVisible) return "Live camera · overlays off"
        if(data.prediction==null) return "Live camera · no prediction"
        val now=SystemClock.elapsedRealtimeNanos()
        val reason=OverlayFreshness.reason(data.capturedNs,now,data.orientation,currentOrientation)
        if(reason!=null) return reason
        if(data.sourceTransform==null || preview.outputTransform==null) return "Aligning camera preview"
        val age=((now-data.capturedNs!!)/50_000_000L)*50
        return "Live view · mask ${age} ms old"
    }
    private var frame: DashboardFrame? = null
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val palette = intArrayOf(Color.rgb(125,193,255),Color.rgb(161,146,255),Color.rgb(120,230,255),
        Color.rgb(255,171,90),Color.rgb(204,236,114),Color.rgb(94,210,255),Color.rgb(224,139,243),
        Color.rgb(255,195,100),Color.rgb(255,96,142))
    var roadAndLaneVisible = true
        set(value) {field=value;overlay.invalidate()}
    private fun sp(value: Float) = android.util.TypedValue.applyDimension(android.util.TypedValue.COMPLEX_UNIT_SP,value,resources.displayMetrics)
    fun submit(next: DashboardFrame) {
        // Bitmaps are immutable once handed to the UI; recycle only replaced frame buffers.
        val old = frame; frame = next
        old?.image?.takeIf { it !== next.image }?.recycle()
        old?.prediction?.mask?.takeIf { it !== next.prediction?.mask }?.recycle()
        overlay.invalidate()
    }
    fun clear() { frame?.image?.recycle(); frame?.prediction?.mask?.recycle(); frame = null; overlay.invalidate() }
    private fun drawFrame(canvas: Canvas) {
        val matched=matchedFrames || paused
        if(matched) canvas.drawColor(Color.rgb(5,12,15))
        val data = frame
        if (data == null) {
            if(!matched) return
            paint.color = Color.rgb(146,173,177); paint.textSize = sp(18f)
            paint.textAlign = Paint.Align.CENTER
            canvas.drawText("Camera view will appear here",width/2f,height/2f,paint)
            paint.textAlign = Paint.Align.LEFT; return
        }
        val scale = min(width.toFloat()/data.image.width,height.toFloat()/data.image.height)
        val left = (width-data.image.width*scale)/2; val top = (height-data.image.height*scale)/2
        val transform=Matrix()
        if(matched) {
            transform.setScale(scale,scale)
            transform.postTranslate(left,top)
        } else {
            if(!roadAndLaneVisible || OverlayFreshness.reason(data.capturedNs,SystemClock.elapsedRealtimeNanos(),
                data.orientation,currentOrientation)!=null) return
            val source=data.sourceTransform ?: return
            val target=preview.outputTransform ?: return
            transform.set(CameraFrameTransform.toPreview(source,target))
        }
        canvas.save(); canvas.concat(transform)
        canvas.clipRect(0,0,data.image.width,data.image.height)
        paint.isFilterBitmap = true; paint.style = Paint.Style.FILL
        if(matched) canvas.drawBitmap(data.image,0f,0f,paint)
        if (roadAndLaneVisible) data.prediction?.mask?.let { canvas.drawBitmap(it,null,RectF(0f,0f,data.image.width.toFloat(),data.image.height.toFloat()),paint) }
        for (box in data.prediction?.boxes.orEmpty()) {
            paint.style = Paint.Style.STROKE; paint.strokeWidth = 2.2f/scale; paint.color = palette[box.label%palette.size]
            canvas.drawRect(box.x1,box.y1,box.x2,box.y2,paint)
            val label = "${data.classes.getOrElse(box.label) { "class ${box.label}" }} ${(box.score*100).toInt()}%"
            paint.style = Paint.Style.FILL; paint.textSize = sp(12f)/scale
            val textY = (box.y1-5/scale).coerceAtLeast(paint.textSize)
            val textWidth = paint.measureText(label)
            val color = paint.color; paint.color = Color.argb(200,4,16,20)
            canvas.drawRect(box.x1,textY-paint.textSize,box.x1+textWidth+6/scale,textY+3/scale,paint)
            paint.color = color; canvas.drawText(label,box.x1+3/scale,textY,paint)
        }
        canvas.restore()
        // A small gravity marker is a sensor diagnostic, never a suggested road trajectory.
        if (matched && data.imuTrusted) {
            canvas.save(); canvas.translate(width/2f,28*resources.displayMetrics.density)
            canvas.rotate(Math.toDegrees(data.roll.toDouble()).toFloat())
            paint.color = Color.rgb(115,245,194); paint.strokeWidth = 2f*resources.displayMetrics.density
            canvas.drawLine(-22f,0f,22f,0f,paint); canvas.restore()
        }
        if (data.banner.isNotEmpty()) {
            paint.color = Color.argb(210,7,19,21); paint.style = Paint.Style.FILL
            val dh = 30*resources.displayMetrics.density
            canvas.drawRect(0f,height-dh,width.toFloat(),height.toFloat(),paint)
            paint.color = Color.rgb(255,221,142); paint.textSize = sp(11f)
            canvas.drawText(data.banner,12*resources.displayMetrics.density,height-10*resources.displayMetrics.density,paint)
        }
    }
}
