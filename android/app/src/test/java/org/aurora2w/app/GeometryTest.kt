package org.aurora2w.app

import org.junit.Assert.*
import org.junit.Test
import kotlin.math.*

class GeometryTest {
    private fun pose(screenRotationDegrees: Double): Quaternion {
        // Rx(90) Rz(screenRotation): upright rear camera with gravity in image plane.
        val a = Math.toRadians(screenRotationDegrees)/2
        val s = sqrt(.5)
        return Quaternion(s*cos(a),s*cos(a),-s*sin(a),s*sin(a))
    }
    @Test fun rearCameraSceneSignAndDisplayRotations() {
        for (imageRotation in listOf(0,90,180,270)) for (lean in listOf(-42.0,0.0,31.0)) {
            val q = pose(90.0-imageRotation+lean)
            assertEquals(Math.toRadians(lean),CameraGeometry.sceneRoll(q,90,imageRotation)!!,1e-9)
        }
    }
    @Test fun lookingStraightDownHasNoHorizon() {
        assertNull(CameraGeometry.sceneRoll(Quaternion(1.0,0.0,0.0,0.0),90,90))
    }
    @Test fun everySensorOrientationMatchesIndependentWorldHorizonProjection() {
        val poses=listOf(pose(-28.0),pose(37.0),Quaternion(.62,.51,-.2,.4).normalized(),Quaternion(.7,-.4,.2,.3).normalized())
        for (q in poses) {
            val w=q.w;val x=q.x;val y=q.y;val z=q.z
            val r=arrayOf(doubleArrayOf(1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)),
                doubleArrayOf(2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)),
                doubleArrayOf(2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)))
            // Rear optical forward is minus the device Z axis. The intersection
            // of the world horizontal plane and image plane is forward × world up.
            val worldHorizon=doubleArrayOf(-r[1][2],r[0][2],0.0)
            val deviceX=(0..2).sumOf {r[it][0]*worldHorizon[it]}
            val deviceY=(0..2).sumOf {r[it][1]*worldHorizon[it]}
            for (sensor in listOf(0,90,180,270)) for (image in listOf(0,90,180,270)) {
                var px=deviceX;var py=-deviceY
                repeat(((image-sensor+360)%360)/90) {val old=px;px=-py;py=old}
                val expected=atan2(py,px)
                val actual=CameraGeometry.sceneRoll(q,sensor,image)!!
                assertEquals("sensor=$sensor image=$image",0.0,CameraGeometry.wrap(actual-expected),1e-8)
            }
        }
    }
    @Test fun portraitLetterboxRetainsFullImageWithoutAnUntrackedCrop() {
        val t=Letterbox(1080,1920,640,384)
        assertEquals(216,t.resizedWidth);assertEquals(384,t.resizedHeight);assertEquals(212,t.left)
        val original=t.inverse(Detection(212f,0f,428f,384f,8,.8f))
        assertEquals(0f,original.x1,1e-4f);assertEquals(1080f,original.x2,1e-4f)
        assertEquals(1920f,original.y2,1e-3f)
    }
    @Test fun antipodalQuaternionsAreSameAttitude() {
        val a = pose(25.0); val b = Quaternion(-a.w,-a.x,-a.y,-a.z)
        val halfway = Quaternion.slerp(a,b,.5)
        assertArrayEquals(a.gravityDevice(),halfway.gravityDevice(),1e-10)
    }
    @Test fun slerpCrossesWrapOnShortPath() {
        val middle = Quaternion.slerp(pose(179.0),pose(-179.0),.5)
        assertEquals(Math.PI,abs(CameraGeometry.sceneRoll(middle,90,90)!!),1e-8)
    }
    @Test fun timelineRejectsGapsExtrapolationAndUnreliableSamples() {
        val b = AttitudeBuffer()
        b.add(AttitudeSample(0,pose(0.0),true)); b.add(AttitudeSample(100_000_000,pose(30.0),true))
        assertNull(b.at(33_333_333)); assertNull(b.at(-1)); assertNull(b.at(100_000_001))
        assertNotNull(b.at(100_000_000))
        val c = AttitudeBuffer()
        c.add(AttitudeSample(0,pose(0.0),true)); c.add(AttitudeSample(10_000_000,pose(30.0),false))
        assertNull(c.at(5_000_000)); assertNull(c.at(10_000_000))
    }
    @Test fun timelineInterpolatesAtExposureTime() {
        val b = AttitudeBuffer()
        b.add(AttitudeSample(0,pose(0.0),true)); b.add(AttitudeSample(20_000_000,pose(40.0),true))
        assertEquals(Math.toRadians(10.0),CameraGeometry.sceneRoll(b.at(5_000_000)!!,90,90)!!,1e-8)
    }
    @Test fun roundedLetterboxPreservesIndependentScalesAndPixelEdges() {
        val t = Letterbox(1001,563,640,384)
        assertEquals(640,t.resizedWidth); assertEquals(360,t.resizedHeight); assertEquals(12,t.top)
        val source = t.inverse(Detection(t.left.toFloat(),t.top.toFloat(),
            (t.left+t.resizedWidth).toFloat(),(t.top+t.resizedHeight).toFloat(),0,1f))
        assertEquals(0f,source.x1,1e-4f); assertEquals(1001f,source.x2,1e-4f); assertEquals(563f,source.y2,1e-4f)
    }
    @Test fun nmsDoesNotSuppressDifferentClasses() {
        val a = Detection(0f,0f,10f,10f,0,.9f)
        val result = Postprocess.nms(listOf(a,a.copy(label=1,score=.8f),a.copy(score=.7f)))
        assertEquals(listOf(0,1),result.map { it.label })
    }
    @Test fun clippingBeforeNmsMatchesBorderBoxSemantics() {
        val t = Letterbox(64,64,64,64)
        val result = Postprocess.decode(floatArrayOf(8f,7f),floatArrayOf(-100f,0f,20f,20f,0f,0f,20f,20f),
            floatArrayOf(8f,8f),1,t,.3f,.5f)
        assertEquals(1,result.size)
        assertEquals(0f,result.single().x1,0f)
    }
    @Test fun scoreThresholdIsStrictAndNanNeverBecomesDetection() {
        val t = Letterbox(64,64,64,64)
        val result = Postprocess.decode(floatArrayOf(0f,Float.NaN),floatArrayOf(0f,0f,10f,10f,0f,0f,10f,10f),
            floatArrayOf(0f,0f),1,t,.5f,.5f)
        assertTrue(result.isEmpty())
    }
    @Test fun halfPixelResizeClampsAtEdges() {
        val a = floatArrayOf(0f,10f,20f,30f)
        assertEquals(15f,Postprocess.bilinear(a,2,2,.5,.5),1e-6f)
        assertEquals(0f,Postprocess.bilinear(a,2,2,-.25,-.25),1e-6f)
        assertEquals(30f,Postprocess.bilinear(a,2,2,2.0,2.0),1e-6f)
    }
    @Test fun untrainedClassIsExcludedBeforeTopKAndNms() {
        val t = Letterbox(64,64,64,64)
        val result = Postprocess.decode(floatArrayOf(100f,2f),floatArrayOf(0f,0f,20f,20f),
            floatArrayOf(5f),2,t,.3f,.5f,setOf(1))
        assertEquals(1,result.size); assertEquals(1,result.single().label)
    }
    @Test(expected = IllegalArgumentException::class) fun invalidQuaternionFailsClosed() {
        Quaternion(0.0,0.0,0.0,0.0).normalized()
    }
}
