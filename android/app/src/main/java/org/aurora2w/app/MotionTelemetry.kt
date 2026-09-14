package org.aurora2w.app

import kotlin.math.*

/** UI telemetry only. Never substitutes for an exposure-aligned FusionReading.
 * Reference axes are the physical phone axes at capture, independent of display rotation.
 * Roll is clockwise about the reference phone Z axis; pitch is rotation about reference X.
 * These are relative ZYX Euler components, not measured vehicle attitude.
 */
data class MotionReading(
    val rollDegrees: Float? = null, val pitchDegrees: Float? = null,
    val accelerationDeviation: Float? = null, val status: String = "Waiting for sensors",
    val referenced: Boolean = false, val timestampNs: Long = 0,
    val orientation: Quaternion? = null
)

class MotionTelemetry(private val freshnessNs: Long = 250_000_000L) {
    private var latest: AttitudeSample? = null
    private var reference: Quaternion? = null
    private var uncalibrated = false
    private var acceleration: Pair<Long, Double>? = null
    @Synchronized fun reset() { latest = null; reference = null; acceleration = null;uncalibrated=false }
    @Synchronized fun add(sample: AttitudeSample, allowUncalibrated: Boolean = false) {
        if (latest != null && sample.timeNs <= latest!!.timeNs) return
        val q = try { sample.q.normalized() } catch (_: IllegalArgumentException) {
            latest = sample.copy(accurate = false); return
        }
        uncalibrated=!sample.accurate
        // A relative UI animation can display uncalibrated game rotation with an explicit status.
        // The exposure-aligned model timeline retains the original, stricter accuracy gate.
        latest = sample.copy(q = q,accurate=sample.accurate || allowUncalibrated)
    }
    @Synchronized fun acceleration(timeNs: Long, magnitude: Double) {
        if (acceleration != null && timeNs <= acceleration!!.first) return
        acceleration = timeNs to if (magnitude.isFinite()) abs(magnitude - 9.80665) else Double.NaN
    }
    @Synchronized fun setReference(nowNs: Long) {
        val s = latest
        require(s != null && s.accurate && nowNs - s.timeNs in 0..freshnessNs) { "Wait for a fresh, reliable orientation signal" }
        reference = s.q
    }
    /** Only captures the first reference; moving later never recenters the display. */
    @Synchronized fun autoReference(nowNs: Long, stationary: Boolean): Boolean {
        if (reference != null || !stationary) return false
        val s = latest ?: return false
        if (!s.accurate || nowNs-s.timeNs !in 0..freshnessNs) return false
        reference = s.q
        return true
    }
    @Synchronized fun read(nowNs: Long): MotionReading {
        val a = acceleration?.takeIf { nowNs - it.first in 0..freshnessNs && it.second.isFinite() }?.second?.toFloat()
        val s = latest
        if (s == null || nowNs - s.timeNs !in 0..freshnessNs)
            return MotionReading(accelerationDeviation = a, status = "Waiting for fresh orientation", referenced = reference != null)
        if (!s.accurate) return MotionReading(accelerationDeviation = a, status = "Orientation signal unreliable", referenced = reference != null)
        val r = reference ?: return MotionReading(accelerationDeviation = a, status = "Hold still briefly to start the bike", timestampNs = s.timeNs, orientation = s.q)
        val q = relative(r, s.q)
        // ZYX is singular at +/-90 degrees about Y: withhold the coupled angles.
        val sinY = 2 * (q.w * q.y - q.z * q.x)
        if (abs(sinY) > .999) return MotionReading(accelerationDeviation = a, status = "Orientation near axis limit", referenced = true, timestampNs = s.timeNs, orientation = s.q)
        val roll = -atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        val pitch = atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y))
        return MotionReading(Math.toDegrees(roll).toFloat(), Math.toDegrees(pitch).toFloat(), a,
            if(uncalibrated) "Relative phone motion · accuracy unreported" else "Relative phone orientation",
            true, s.timeNs, s.q)
    }
    private fun relative(a: Quaternion, b: Quaternion) = Quaternion(
        a.w*b.w+a.x*b.x+a.y*b.y+a.z*b.z,
        a.w*b.x-a.x*b.w-a.y*b.z+a.z*b.y,
        a.w*b.y+a.x*b.z-a.y*b.w-a.z*b.x,
        a.w*b.z-a.x*b.y+a.y*b.x-a.z*b.w).normalized()
}
