package org.aurora2w.app

import android.content.Context
import android.graphics.*
import android.view.View
import java.util.ArrayDeque
import java.util.Locale
import kotlin.math.*

/** A schematic of relative PHONE attitude, not a reconstruction of bike position.
 * Draws six seconds of measured samples. Gaps are disconnected; angles are never extrapolated.
 */
internal class MotionView(context: Context) : View(context) {
    private data class Point(val time: Long,val roll: Float?,val pitch: Float?)
    private val samples = ArrayDeque<Point>()
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val path = Path()
    private var reading = MotionReading()
    private var now = 0L
    private var shownRoll = 0f
    private var shownPitch = 0f
    private val cyan = AuroraColors.mint
    private val gold = AuroraColors.amber
    private val fontScale = resources.configuration.fontScale.coerceIn(1f,1.3f)

    fun clearHistory() { samples.clear();shownRoll=0f;shownPitch=0f;invalidate() }
    fun submit(value: MotionReading,timeNs: Long) {
        if (reading.referenced && !value.referenced) clearHistory()
        val fresh = value.rollDegrees != null && value.pitchDegrees != null
        val dt = if(now == 0L) .05f else ((timeNs-now)/1e9f).coerceIn(0f,.25f)
        val alpha = 1f-exp(-dt/ .10f)
        shownRoll = if(fresh) shownRoll + (value.rollDegrees!!.coerceIn(-35f,35f)-shownRoll)*alpha else 0f
        shownPitch = if(fresh) shownPitch + (value.pitchDegrees!!.coerceIn(-50f,50f)-shownPitch)*alpha else 0f
        reading=value;now=timeNs
        samples.addLast(Point(timeNs,value.rollDegrees,value.pitchDegrees))
        while(samples.isNotEmpty() && (timeNs-samples.first.time > 6_000_000_000L || samples.size > 140)) samples.removeFirst()
        // No live-region announcements: TalkBack can inspect the current values on demand.
        contentDescription = if(fresh) String.format(Locale.US,"Relative phone roll %.1f degrees, pitch %.1f degrees. Bike is schematic.",value.rollDegrees,value.pitchDegrees)
            else value.status
        invalidate()
    }
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val scale=min(width/320f,height/146f)
        canvas.save()
        canvas.translate((width-320f*scale)/2,(height-146f*scale)/2)
        canvas.scale(scale,scale)
        graph(canvas,12f,88f,true,cyan)
        graph(canvas,220f,88f,false,gold)
        bike(canvas,160f,70f,reading.rollDegrees != null)
        text(canvas,"SCHEMATIC",160f,137f,8f,AuroraColors.muted,Paint.Align.CENTER)
        canvas.restore()
    }
    private fun text(c: Canvas,value: String,x: Float,y: Float,size: Float,color: Int,align: Paint.Align = Paint.Align.LEFT) {
        paint.reset();paint.isAntiAlias=true;paint.color=color
        paint.textSize=size*fontScale;paint.textAlign=align
        paint.typeface=Typeface.create("sans-serif-medium",Typeface.NORMAL)
        c.drawText(value,x,y,paint)
    }
    private fun graph(c: Canvas,x: Float,w: Float,roll: Boolean,color: Int) {
        val angle=if(roll)reading.rollDegrees else reading.pitchDegrees
        text(c,if(roll)"ROLL" else "PITCH",x,17f,9f,AuroraColors.muted)
        text(c,angle?.let { String.format(Locale.US,"%+.1f°",it) } ?: "—",x,44f,22f,if(angle == null)AuroraColors.muted else color)
        val top=59f;val bottom=104f;val middle=(top+bottom)/2
        paint.style=Paint.Style.STROKE;paint.strokeWidth=.6f;paint.color=AuroraColors.edge
        for(y in listOf(top,middle,bottom)) c.drawLine(x,y,x+w,y,paint)
        c.save();c.clipRect(x,top,x+w,bottom)
        path.reset();var connected=false
        for(s in samples) {
            val v=if(roll)s.roll else s.pitch
            if(v == null) { connected=false;continue }
            val px=x+w*(1f-(now-s.time)/6_000_000_000f)
            val py=middle-v.coerceIn(-45f,45f)/45f*(bottom-top)/2
            if(connected)path.lineTo(px,py) else path.moveTo(px,py)
            connected=true
        }
        paint.color=color;paint.strokeWidth=1.5f;paint.strokeJoin=Paint.Join.ROUND
        c.drawPath(path,paint);c.restore()
        text(c,"±45°  /  6 s",x,121f,8f,AuroraColors.muted)
    }
    private fun bike(c: Canvas,x: Float,y: Float,valid: Boolean) {
        c.save();c.translate(x,y)
        // Decorative halo describes no lane boundary, obstacle, distance or predicted path.
        paint.style=Paint.Style.FILL
        paint.shader=RadialGradient(0f,0f,57f,intArrayOf(Color.argb(if(valid)44 else 12,92,231,222),Color.TRANSPARENT),null,Shader.TileMode.CLAMP)
        c.drawCircle(0f,0f,57f,paint);paint.shader=null
        paint.style=Paint.Style.STROKE;paint.color=AuroraColors.edge;paint.strokeWidth=.7f
        c.drawOval(-46f,-58f,46f,58f,paint)
        for(side in listOf(-1f,1f)) for(tick in -2..2) c.drawLine(side*48f,tick*14f,side*(if(tick==0)54f else 51f),tick*14f,paint)
        c.rotate(shownRoll)
        // Signed pitch translates the schematic, so forward/back tilt is clearly visible.
        // This is an attitude cue, not measured vehicle displacement.
        c.translate(0f,shownPitch*.15f)
        c.scale(1f,cos(Math.toRadians(shownPitch.toDouble())).toFloat())
        val accent=if(valid)cyan else Color.rgb(100,129,138)
        fun rect(l: Float,t: Float,r: Float,b: Float,radius: Float,color: Int,outline: Boolean=false) {
            paint.style=if(outline)Paint.Style.STROKE else Paint.Style.FILL;paint.color=color;paint.strokeWidth=1.2f
            c.drawRoundRect(l,t,r,b,radius,radius,paint)
        }
        rect(-5f,-54f,5f,-33f,4f,Color.rgb(4,10,14))
        rect(-6f,33f,6f,55f,4f,Color.rgb(4,10,14))
        rect(-5f,-53f,5f,-34f,4f,accent,true)
        rect(-6f,35f,6f,54f,4f,accent,true)
        path.reset();path.moveTo(0f,-47f)
        path.cubicTo(-11f,-44f,-17f,-27f,-14f,-13f)
        path.lineTo(-8f,9f);path.lineTo(-9f,30f);path.quadTo(0f,49f,9f,30f)
        path.lineTo(8f,9f);path.lineTo(14f,-13f)
        path.cubicTo(17f,-27f,11f,-44f,0f,-47f);path.close()
        paint.style=Paint.Style.FILL
        paint.shader=LinearGradient(-16f,0f,16f,0f,intArrayOf(Color.rgb(24,55,61),if(valid)Color.rgb(57,124,124) else Color.rgb(55,74,79),Color.rgb(17,41,49)),null,Shader.TileMode.CLAMP)
        c.drawPath(path,paint);paint.shader=null
        paint.color=accent;paint.style=Paint.Style.STROKE;paint.strokeWidth=1.1f;c.drawPath(path,paint)
        rect(-5f,8f,5f,33f,4f,AuroraColors.background)
        // Handlebar and mirrors.
        paint.strokeWidth=2f;c.drawLine(-23f,-22f,23f,-22f,paint)
        c.drawLine(-23f,-22f,-26f,-29f,paint);c.drawLine(23f,-22f,26f,-29f,paint)
        rect(-31f,-33f,-23f,-28f,2f,accent)
        rect(23f,-33f,31f,-28f,2f,accent)
        // Rider shoulders, arms and helmet.
        path.reset();path.moveTo(-7f,-24f);path.quadTo(-14f,-20f,-17f,-8f)
        path.lineTo(-11f,10f);path.quadTo(0f,17f,11f,10f);path.lineTo(17f,-8f)
        path.quadTo(14f,-20f,7f,-24f);path.close()
        paint.style=Paint.Style.FILL;paint.color=Color.rgb(17,31,39);c.drawPath(path,paint)
        paint.style=Paint.Style.STROKE;paint.color=accent;paint.strokeWidth=1f;c.drawPath(path,paint)
        paint.strokeWidth=3f;c.drawLine(-15f,-12f,-21f,-22f,paint);c.drawLine(15f,-12f,21f,-22f,paint)
        paint.style=Paint.Style.FILL;paint.color=Color.rgb(181,207,207)
        c.drawOval(-7f,-33f,7f,-15f,paint)
        paint.color=Color.rgb(25,64,72);c.drawOval(-6f,-29f,6f,-22f,paint)
        paint.color=gold;paint.strokeWidth=2f;c.drawLine(-3f,41f,3f,41f,paint)
        c.restore()
    }
}
