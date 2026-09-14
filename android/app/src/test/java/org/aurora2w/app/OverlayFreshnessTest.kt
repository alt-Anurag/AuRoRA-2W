package org.aurora2w.app

import org.junit.Assert.*
import org.junit.Test
import kotlin.math.*

class OverlayFreshnessTest {
    private val identity=Quaternion(1.0,0.0,0.0,0.0)
    @Test fun timestampBoundsMissingClocksAndAbsentSensors() {
        val t=1_000_000_000L
        assertNull(OverlayFreshness.reason(t,t+500_000_000,identity,identity))
        assertNotNull(OverlayFreshness.reason(t,t+500_000_001,identity,identity))
        assertNotNull(OverlayFreshness.reason(t,t-1,identity,identity))
        assertNotNull(OverlayFreshness.reason(null,t,identity,identity))
        assertNull(OverlayFreshness.reason(t,t+200_000_000,null,null))
        assertNotNull(OverlayFreshness.reason(t,t+200_000_001,null,identity))
    }
    @Test fun motionIsCheckedOnEveryAxisAndQuaternionSignsAreEquivalent() {
        for(axis in 0..2) {
            val a=Math.toRadians(7.0)/2
            val xyz=DoubleArray(3);xyz[axis]=sin(a)
            assertNotNull(OverlayFreshness.reason(10,20,identity,Quaternion(cos(a),xyz[0],xyz[1],xyz[2])))
        }
        assertNull(OverlayFreshness.reason(10,20,identity,Quaternion(-1.0,0.0,0.0,0.0)))
        assertNotNull(OverlayFreshness.reason(10,20,identity,Quaternion(Double.NaN,0.0,0.0,0.0)))
    }
}
