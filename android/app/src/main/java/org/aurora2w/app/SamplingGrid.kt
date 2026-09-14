package org.aurora2w.app

import kotlin.math.*

/** Precomputed half-pixel bilinear coordinates; same Float operation order as Postprocess.bilinear. */
internal class SamplingGrid(sourceWidth: Int,sourceHeight: Int,width: Int,height: Int) {
    private class Axis(source: Int,target: Int) {
        val low=IntArray(target)
        val high=IntArray(target)
        val weight=FloatArray(target)
        init {
            require(source>0 && target>0)
            for(i in 0 until target) {
                val coordinate=((i+.5)*source/target-.5).coerceIn(0.0,(source-1).toDouble())
                low[i]=floor(coordinate).toInt();high[i]=min(low[i]+1,source-1)
                weight[i]=(coordinate-low[i]).toFloat()
            }
        }
    }
    private val xs=Axis(sourceWidth,width)
    private val ys=Axis(sourceHeight,height)
    private val stride=sourceWidth
    fun sample(values: FloatArray,x: Int,y: Int): Float {
        val dx=xs.weight[x];val dy=ys.weight[y]
        val a=ys.low[y]*stride;val b=ys.high[y]*stride
        return (values[a+xs.low[x]]*(1-dx)+values[a+xs.high[x]]*dx)*(1-dy)+
            (values[b+xs.low[x]]*(1-dx)+values[b+xs.high[x]]*dx)*dy
    }
    fun channel(values: IntArray,shift: Int,x: Int,y: Int): Float {
        val dx=xs.weight[x];val dy=ys.weight[y]
        val a=ys.low[y]*stride;val b=ys.high[y]*stride
        val aa=((values[a+xs.low[x]] shr shift) and 255).toFloat()
        val ab=((values[a+xs.high[x]] shr shift) and 255).toFloat()
        val ba=((values[b+xs.low[x]] shr shift) and 255).toFloat()
        val bb=((values[b+xs.high[x]] shr shift) and 255).toFloat()
        return (aa*(1-dx)+ab*dx)*(1-dy)+(ba*(1-dx)+bb*dx)*dy
    }
    /** Fetch four source pixels once for all channels. Preserve Float order and uint8 rounding. */
    fun writeNormalizedRgb(values: IntArray,x: Int,y: Int,tensor: FloatArray,index: Int,
        plane: Int,mean: FloatArray,std: FloatArray) {
        val dx=xs.weight[x];val dy=ys.weight[y]
        val a=ys.low[y]*stride;val b=ys.high[y]*stride
        val aa=values[a+xs.low[x]];val ab=values[a+xs.high[x]]
        val ba=values[b+xs.low[x]];val bb=values[b+xs.high[x]]
        for(c in 0..2) {
            val shift=16-8*c
            val value=(((aa shr shift) and 255).toFloat()*(1-dx)+((ab shr shift) and 255).toFloat()*dx)*(1-dy)+
                (((ba shr shift) and 255).toFloat()*(1-dx)+((bb shr shift) and 255).toFloat()*dx)*dy
            tensor[c*plane+index]=(round(value)/255f-mean[c])/std[c]
        }
    }
}
