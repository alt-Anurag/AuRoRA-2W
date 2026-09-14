@file:androidx.annotation.OptIn(androidx.camera.view.TransformExperimental::class)

package org.aurora2w.app

import android.graphics.*
import android.util.Size
import androidx.camera.core.ImageProxy
import androidx.camera.core.ImageInfo
import androidx.camera.view.transform.OutputTransform
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import java.lang.reflect.Proxy
import kotlin.math.min

@RunWith(AndroidJUnit4::class)
class CameraMappingTest {
    @Test fun croppedBitmapAndLiveOverlayAgreeForAllRotationsAndViewShapes() {
        val crop=Rect(40,30,600,450)
        val full=Bitmap.createBitmap(640,480,Bitmap.Config.ARGB_8888)
        val anchors=arrayOf(floatArrayOf(80f,70f),floatArrayOf(530f,350f),floatArrayOf(140f,230f))
        val colors=intArrayOf(Color.RED,Color.GREEN,Color.BLUE)
        val canvas=Canvas(full);val paint=Paint()
        anchors.forEachIndexed {i,p->paint.color=colors[i];canvas.drawRect(p[0]-4,p[1]-4,p[0]+5,p[1]+5,paint)}
        try {
            for(rotation in listOf(0,90,180,270)) {
                val info=Proxy.newProxyInstance(ImageInfo::class.java.classLoader,arrayOf(ImageInfo::class.java)) {_,m,_->
                    when(m.name) {"getRotationDegrees"->rotation;else->error(m.name)}
                } as ImageInfo
                val proxy=Proxy.newProxyInstance(ImageProxy::class.java.classLoader,arrayOf(ImageProxy::class.java)) {_,m,_->
                    when(m.name) {"getWidth"->640;"getHeight"->480;"getCropRect"->crop;"getImageInfo"->info;else->error(m.name)}
                } as ImageProxy
                val source=CameraFrameTransform.source(proxy)
                val image=CameraFrameTransform.upright(full,crop,rotation)
                fun rotated(x: Float,y: Float)=when(rotation) {
                    90->floatArrayOf(crop.height()-y,x)
                    180->floatArrayOf(crop.width()-x,crop.height()-y)
                    270->floatArrayOf(y,crop.width()-x)
                    else->floatArrayOf(x,y)
                }
                for(i in anchors.indices) {
                    val p=rotated(anchors[i][0]-crop.left+.5f,anchors[i][1]-crop.top+.5f)
                    assertEquals("rotated source color $rotation",colors[i],image.getPixel(p[0].toInt(),p[1].toInt()))
                }
                for(view in listOf(Size(300,700),Size(700,300))) {
                    val scale=min(view.width.toFloat()/image.width,view.height.toFloat()/image.height)
                    val left=(view.width-image.width*scale)/2
                    val top=(view.height-image.height*scale)/2
                    // Independently map three canonical viewport corners to displayed pixels.
                    val n=floatArrayOf(-1f,-1f,1f,-1f,-1f,1f)
                    val corners=arrayOf(rotated(0f,0f),rotated(crop.width().toFloat(),0f),rotated(0f,crop.height().toFloat()))
                    val dest=corners.flatMap {listOf(left+it[0]*scale,top+it[1]*scale)}.toFloatArray()
                    val targetMatrix=Matrix().apply {assertTrue(setPolyToPoly(n,0,dest,0,3))}
                    val target=OutputTransform(targetMatrix,Size(crop.width(),crop.height()))
                    val mapping=CameraFrameTransform.toPreview(source,target)
                    for(anchor in anchors) {
                        val p=rotated(anchor[0]-crop.left,anchor[1]-crop.top)
                        val mapped=p.copyOf();mapping.mapPoints(mapped)
                        assertEquals(left+p[0]*scale,mapped[0],.001f)
                        assertEquals(top+p[1]*scale,mapped[1],.001f)
                    }
                }
                image.recycle()
            }
        } finally {full.recycle()}
    }
}
