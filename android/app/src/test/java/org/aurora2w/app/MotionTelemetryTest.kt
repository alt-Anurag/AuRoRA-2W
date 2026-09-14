package org.aurora2w.app

import org.junit.Assert.*
import org.junit.Test
import kotlin.math.*

class MotionTelemetryTest {
    private val identity=Quaternion(1.0,0.0,0.0,0.0)
    private fun axis(x: Double,y: Double,z: Double,deg: Double): Quaternion {
        val a=Math.toRadians(deg)/2
        return Quaternion(cos(a),x*sin(a),y*sin(a),z*sin(a))
    }
    private fun multiply(a: Quaternion,b: Quaternion)=Quaternion(
        a.w*b.w-a.x*b.x-a.y*b.y-a.z*b.z,
        a.w*b.x+a.x*b.w+a.y*b.z-a.z*b.y,
        a.w*b.y-a.x*b.z+a.y*b.w+a.z*b.x,
        a.w*b.z+a.x*b.y-a.y*b.x+a.z*b.w)
    private fun add(m: MotionTelemetry,t: Long,q: Quaternion=identity,accurate: Boolean=true) =
        m.add(AttitudeSample(t,q,accurate))

    @Test fun noAnglesWithoutExplicitReference() {
        val m=MotionTelemetry()
        assertNull(m.read(1).rollDegrees)
        add(m,10)
        assertNull(m.read(10).rollDegrees)
        m.setReference(10)
        assertEquals(0f,m.read(10).rollDegrees!!,1e-5f)
        assertEquals(0f,m.read(10).pitchDegrees!!,1e-5f)
    }
    @Test fun arbitraryMountAndIndependentAxisSigns() {
        for(base in listOf(identity,axis(1.0,0.0,0.0,72.0),Quaternion(.62,.51,-.2,.4).normalized())) {
            val m=MotionTelemetry();add(m,1,base);m.setReference(1)
            add(m,2,multiply(base,axis(0.0,0.0,1.0,27.0)))
            assertEquals(-27f,m.read(2).rollDegrees!!,1e-4f)
            assertEquals(0f,m.read(2).pitchDegrees!!,1e-4f)
            add(m,3,multiply(base,axis(1.0,0.0,0.0,-18.0)))
            assertEquals(0f,m.read(3).rollDegrees!!,1e-4f)
            assertEquals(-18f,m.read(3).pitchDegrees!!,1e-4f)
        }
    }
    @Test fun equivalentQuaternionSignsAndWrapBounds() {
        val m=MotionTelemetry();add(m,1);m.setReference(1)
        for((i,d) in listOf(-179.0,-90.0,0.0,90.0,179.0).withIndex()) {
            val q=axis(0.0,0.0,1.0,d)
            add(m,i+2L,Quaternion(-q.w,-q.x,-q.y,-q.z))
            assertEquals(-d.toFloat(),m.read(i+2L).rollDegrees!!,1e-4f)
        }
    }
    @Test fun staleUnreliableInvalidAndFutureSamplesWithholdAngles() {
        val m=MotionTelemetry(100);add(m,10);m.setReference(10)
        assertNull(m.read(111).rollDegrees)
        assertNull(m.read(9).rollDegrees)
        add(m,20,accurate=false);assertNull(m.read(20).rollDegrees)
        add(m,30,Quaternion(Double.NaN,0.0,0.0,0.0));assertNull(m.read(30).rollDegrees)
        add(m,40);assertNotNull(m.read(40).rollDegrees)
        add(m,39,axis(0.0,0.0,1.0,30.0))
        assertEquals(0f,m.read(40).rollDegrees!!,1e-4f)
    }
    @Test fun referenceRejectsMissingStaleAndUnreliableSamples() {
        val m=MotionTelemetry(100)
        fun rejects(t: Long) {
            try {m.setReference(t);fail("Invalid sample must not create a reference")}
            catch(_: IllegalArgumentException) {}
        }
        rejects(1);add(m,10,accurate=false);rejects(10)
        add(m,20);rejects(121)
        m.setReference(20)
        m.reset()
        assertFalse(m.read(20).referenced);rejects(20)
    }
    @Test fun coupledEulerAnglesAreWithheldAtSingularity() {
        val m=MotionTelemetry();add(m,1);m.setReference(1)
        for((i,d) in listOf(-90.0,90.0).withIndex()) {
            add(m,i+2L,axis(0.0,1.0,0.0,d))
            assertNull(m.read(i+2L).rollDegrees);assertNull(m.read(i+2L).pitchDegrees)
        }
    }
    @Test fun accelerationDeviationIsFreshMagnitudeDifferenceNotVehicleAcceleration() {
        val m=MotionTelemetry(100)
        m.acceleration(10,9.80665)
        assertEquals(0f,m.read(10).accelerationDeviation!!,1e-5f)
        m.acceleration(20,8.80665)
        assertEquals(1f,m.read(20).accelerationDeviation!!,1e-5f)
        m.acceleration(19,30.0)
        assertEquals(1f,m.read(20).accelerationDeviation!!,1e-5f)
        assertNull(m.read(121).accelerationDeviation)
        m.acceleration(130,Double.NaN)
        assertNull(m.read(130).accelerationDeviation)
    }
    @Test fun automaticReferenceNeedsSteadyFreshDataAndDoesNotFollowMovement() {
        val m=MotionTelemetry(100)
        assertFalse(m.autoReference(1,true))
        add(m,10);assertFalse(m.autoReference(10,false))
        assertFalse(m.autoReference(111,true))
        assertTrue(m.autoReference(10,true))
        add(m,20,axis(0.0,0.0,1.0,20.0))
        assertFalse(m.autoReference(20,true))
        assertEquals(-20f,m.read(20).rollDegrees!!,1e-4f)
        m.reset();add(m,30,accurate=false)
        assertFalse(m.autoReference(30,true))
    }
    @Test fun uncalibratedGameUiCanMoveWithoutRelaxingDefaultTrust() {
        val m=MotionTelemetry(100)
        m.add(AttitudeSample(10,identity,false),allowUncalibrated=true)
        assertTrue(m.autoReference(10,true))
        m.add(AttitudeSample(20,axis(1.0,0.0,0.0,25.0),false),allowUncalibrated=true)
        val reading=m.read(20)
        assertEquals(25f,reading.pitchDegrees!!,1e-4f)
        assertTrue(reading.status.contains("unreported"))
        assertNull(m.read(121).pitchDegrees)
        m.add(AttitudeSample(130,Quaternion(Double.NaN,0.0,0.0,0.0),false),allowUncalibrated=true)
        assertNull(m.read(130).pitchDegrees)
        val strict=MotionTelemetry()
        strict.add(AttitudeSample(10,identity,false))
        assertFalse(strict.autoReference(10,true))
    }
}
