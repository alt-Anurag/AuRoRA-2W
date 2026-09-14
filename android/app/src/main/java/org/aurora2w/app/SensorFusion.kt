package org.aurora2w.app

import android.content.Context
import android.hardware.*
import android.os.*
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.*

data class CaptureTiming(val timestampNs: Long, val exposureNs: Long?, val skewNs: Long?,
                         val frameNumber: Long, val eisOff: Boolean, val oisOff: Boolean,
                         val crop: String, val focalLength: Float?) {
    // First-row exposure start + half exposure + half scan duration approximates image center.
    val midpointNs get() = timestampNs+(exposureNs ?: 0L)/2+(skewNs ?: 0L)/2
}
data class FusionReading(val roll: Float, val confidence: Float, val status: String,
                         val timestampNs: Long, val rawRoll: Double? = null,
                         val calibrationKey: String = "",val generation: Int = 0)

class SensorFusion(context: Context, private val recorder: SessionRecorder) : SensorEventListener {
    private val manager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val thread = HandlerThread("aurora-imu").apply { start() }
    private val handler = Handler(thread.looper)
    private val attitude = manager.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR)
        ?: manager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
    private val gyro = manager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
    private val accel = manager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val preferences = context.getSharedPreferences("mount-calibration",Context.MODE_PRIVATE)
    private val timeline = AttitudeBuffer()
    private val motionTimeline = AttitudeBuffer()
    private val motion = MotionTelemetry()
    @Volatile private var gyroNorm = Double.POSITIVE_INFINITY
    @Volatile private var accelNorm = 0.0
    @Volatile private var gyroTime = 0L
    @Volatile private var accelTime = 0L
    @Volatile private var steadySinceNs = 0L
    @Volatile private var lastReading: FusionReading? = null
    private val readingGeneration=AtomicInteger()
    val available get() = attitude != null
    fun start() {
        timeline.clear();motionTimeline.clear();motion.reset();invalidateReading();gyroTime=0;accelTime=0;steadySinceNs=0
        gyroNorm=Double.POSITIVE_INFINITY;accelNorm=0.0
        for (sensor in listOfNotNull(attitude,gyro,accel)) manager.registerListener(this,sensor,10_000,0,handler)
    }
    fun stop() { manager.unregisterListener(this);timeline.clear();motionTimeline.clear();motion.reset();invalidateReading();steadySinceNs=0;gyroTime=0;accelTime=0 }
    fun invalidateReading() { readingGeneration.incrementAndGet();lastReading=null }
    fun close() { stop(); thread.quitSafely() }
    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit
    override fun onSensorChanged(event: SensorEvent) {
        val values = event.values.copyOf()
        when (event.sensor.type) {
            Sensor.TYPE_GAME_ROTATION_VECTOR, Sensor.TYPE_ROTATION_VECTOR -> {
                val q = FloatArray(4)
                SensorManager.getQuaternionFromVector(q, values)
                val sample = AttitudeSample(event.timestamp,
                    Quaternion(q[0].toDouble(),q[1].toDouble(),q[2].toDouble(),q[3].toDouble()),
                    event.accuracy != SensorManager.SENSOR_STATUS_UNRELIABLE)
                motion.add(sample,allowUncalibrated=event.sensor.type == Sensor.TYPE_GAME_ROTATION_VECTOR)
                try {
                    val normalized=sample.copy(q=sample.q.normalized())
                    timeline.add(normalized)
                    motionTimeline.add(normalized.copy(accurate=normalized.accurate ||
                        event.sensor.type==Sensor.TYPE_GAME_ROTATION_VECTOR))
                } catch (_: IllegalArgumentException) { }
            }
            Sensor.TYPE_GYROSCOPE -> {
                gyroNorm = sqrt(values.take(3).sumOf { it.toDouble()*it }); gyroTime = event.timestamp
            }
            Sensor.TYPE_ACCELEROMETER -> {
                accelNorm = sqrt(values.take(3).sumOf { it.toDouble()*it }); accelTime = event.timestamp
                motion.acceleration(event.timestamp,accelNorm)
            }
        }
        if (gyroNorm < .08 && abs(accelNorm-9.80665) < .7 &&
            abs(event.timestamp-gyroTime) < 100_000_000 && abs(event.timestamp-accelTime) < 100_000_000) {
            if (steadySinceNs == 0L) steadySinceNs = event.timestamp
        } else steadySinceNs = 0
        if (recorder.active) recorder.event(JSONObject().put("type","sensor").put("sensor_type",event.sensor.type)
            .put("timestamp_ns",event.timestamp).put("accuracy",event.accuracy)
            .put("values",JSONArray(values.map { if (it.isFinite()) it else JSONObject.NULL })))
    }
    fun read(timing: CaptureTiming?, timestampRealtime: Boolean, sensorOrientation: Int?,
             imageRotation: Int, cameraId: String): FusionReading {
        val key = "$cameraId:$sensorOrientation:$imageRotation"
        val generation=readingGeneration.get()
        fun publish(reading: FusionReading): FusionReading {
            if (generation != readingGeneration.get()) return FusionReading(0f,0f,"Camera orientation changed",0)
            return reading.copy(calibrationKey=key,generation=generation).also {lastReading=it}
        }
        fun missing(reason: String) = publish(FusionReading(0f,0f,reason,timing?.midpointNs ?: 0))
        if (!available) return missing("No rotation-vector sensor")
        if (!timestampRealtime) return missing("Camera clock is not comparable to IMU")
        if (timing == null || timing.exposureNs == null) return missing("Capture timing unavailable")
        if (!timing.eisOff || !timing.oisOff) return missing("Stabilization state prevents trusted roll")
        if (sensorOrientation == null || sensorOrientation !in listOf(0,90,180,270)) return missing("Camera axes unavailable")
        // Wait at most 12 ms for the upper sensor bracket; never extrapolate from latest callback.
        var q = timeline.at(timing.midpointNs)
        for (attempt in 0 until 6) {
            if (q != null) break
            Thread.sleep(2)
            q = timeline.at(timing.midpointNs)
        }
        val rotation = q ?: return missing("IMU bracket missing or gap > 50 ms")
        val raw = CameraGeometry.sceneRoll(rotation,sensorOrientation,imageRotation) ?: return missing("Horizon roll undefined")
        val offset = preferences.getString(key,null)?.toDoubleOrNull()?.takeIf {it.isFinite()}
        return if (offset == null) publish(FusionReading(0f,0f,"Level-mount calibration needed",timing.midpointNs,raw))
        else publish(FusionReading(CameraGeometry.wrap(raw-offset).toFloat(),1f,"Roll input · level calibrated",timing.midpointNs,raw))
    }
    private fun stationary(now: Long) = steadySinceNs != 0L && now-steadySinceNs >= 750_000_000L &&
        now-gyroTime in 0..100_000_000L && now-accelTime in 0..100_000_000L
    fun motionReading(): MotionReading {
        val now=SystemClock.elapsedRealtimeNanos()
        if (!available) return motion.read(now).copy(status="No rotation-vector sensor on this phone")
        if (motion.autoReference(now,stationary(now))) {
            recorder.event(JSONObject().put("type","motion_reference").put("automatic",true)
                .put("timestamp_ns",now).put("axes","relative physical phone axes; UI only"))
        }
        return motion.read(now)
    }
    /** Read-only, no waiting and no model calibration. Used to hide an old preview overlay. */
    fun orientationAt(timestampNs: Long): Quaternion? = motionTimeline.at(timestampNs)
    fun referenceMotion(): String {
        val now = SystemClock.elapsedRealtimeNanos()
        require(stationary(now)) {
            "Hold the phone still for one second"
        }
        motion.setReference(now)
        recorder.event(JSONObject().put("type","motion_reference").put("timestamp_ns",now)
            .put("axes","relative physical phone axes; UI only"))
        return "Motion reference set for this session"
    }
    fun calibrate(): String {
        val reading = lastReading ?: error("Wait for the camera and IMU")
        val raw = reading.rawRoll ?: error(reading.status)
        require(reading.generation==readingGeneration.get()) {"Wait for the current camera orientation"}
        val now = SystemClock.elapsedRealtimeNanos()
        require(now-reading.timestampNs in 0..500_000_000L) { "Camera frame is stale" }
        require(steadySinceNs != 0L && now-steadySinceNs >= 750_000_000L &&
            now-gyroTime < 100_000_000L && now-accelTime < 100_000_000L) { "Hold the phone still for one second" }
        preferences.edit().putString(reading.calibrationKey,raw.toString()).apply()
        recorder.event(JSONObject().put("type","calibration").put("key",reading.calibrationKey).put("neutral_roll_rad",raw)
            .put("timestamp_ns",now))
        return "Level reference saved for this rear camera and orientation"
    }
    fun clearCalibration() { preferences.edit().clear().apply();invalidateReading() }
    fun metadata() = JSONObject().put("attitude_sensor",attitude?.name ?: "unavailable")
        .put("gyro_sensor",gyro?.name ?: "unavailable").put("accelerometer",accel?.name ?: "unavailable")
        .put("attitude_quaternion","wxyz device-to-world; world z up")
        .put("roll_convention","positive clockwise scene, radians")
        .put("maximum_sensor_gap_ns",50_000_000).put("calibration",JSONObject(preferences.all))
}
