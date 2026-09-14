package org.aurora2w.app

import kotlin.math.*

/** Live overlays are delayed estimates. This bounds age and angular drift, not translation.
 * Times are elapsedRealtime nanoseconds; quaternions are device-to-world wxyz.
 */
object OverlayFreshness {
    const val MAX_AGE_NS = 500_000_000L
    fun reason(capturedNs: Long?, nowNs: Long, source: Quaternion?, current: Quaternion?): String? {
        if (capturedNs == null) return "Use matched view: camera clock unavailable"
        val age=nowNs-capturedNs
        if (age < 0 || age > MAX_AGE_NS) return "Waiting for a newer mask"
        if (source == null || current == null) {
            return if(age > 200_000_000L) "Waiting for fresh motion or mask" else null
        }
        val a=try {source.normalized()} catch(_: IllegalArgumentException) {return "Motion unavailable"}
        val b=try {current.normalized()} catch(_: IllegalArgumentException) {return "Motion unavailable"}
        val dot=abs(a.w*b.w+a.x*b.x+a.y*b.y+a.z*b.z).coerceIn(0.0,1.0)
        return if(Math.toDegrees(2*acos(dot)) > 6.0) "Phone moved: waiting for a new mask" else null
    }
}