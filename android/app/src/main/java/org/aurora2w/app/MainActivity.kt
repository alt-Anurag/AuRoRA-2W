@file:androidx.annotation.OptIn(androidx.camera.view.TransformExperimental::class)

package org.aurora2w.app

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.hardware.display.DisplayManager
import android.graphics.*
import android.hardware.camera2.*
import android.net.Uri
import android.os.*
import android.view.*
import android.widget.*
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.*
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.transform.ImageProxyTransformFactory
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.TreeMap
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.*

@androidx.annotation.OptIn(ExperimentalCamera2Interop::class,ExperimentalCameraInfo::class)
class MainActivity : ComponentActivity() {
    private val analyzerExecutor = Executors.newSingleThreadExecutor { task ->
        Thread({ Process.setThreadPriority(2);task.run() },"aurora-inference")
    }
    private val fileExecutor = Executors.newSingleThreadExecutor()
    private lateinit var recorder: SessionRecorder
    private lateinit var fusion: SensorFusion
    private lateinit var store: ModelStore
    private lateinit var dashboard: DashboardView
    private lateinit var ui: DashboardUi
    private var uiState = DashboardState()
    private var uiPreferences = DashboardPreferences()
    private var controlsDialog: DashboardControls? = null
    private val motionHandler = Handler(Looper.getMainLooper())
    private val motionTick = object : Runnable {
        override fun run() {
            if (!foreground) return
            val motion=fusion.motionReading()
            dashboard.updateMotion(motion)
            ui.renderMotion(motion,SystemClock.elapsedRealtimeNanos())
            ui.renderMaskStatus(dashboard.maskStatus())
            motionHandler.postDelayed(this,50)
        }
    }
    private var lastStatsUpdateNs = 0L
    private lateinit var displayManager: DisplayManager
    private var imageAnalysis: ImageAnalysis? = null
    private var cameraPreview: Preview? = null
    private var boundCameraInfo: CameraInfo? = null
    private var boundCamera: androidx.camera.core.Camera? = null
    private var torchGeneration=0
    private var displayRotation = -1
    @Volatile private var expectedImageRotation: Int? = null
    @Volatile private var frameGeneration = 0
    private val displayListener = object: DisplayManager.DisplayListener {
        override fun onDisplayAdded(id: Int) = Unit
        override fun onDisplayRemoved(id: Int) = Unit
        override fun onDisplayChanged(id: Int) { if (::dashboard.isInitialized && dashboard.display?.displayId == id) updateTargetRotation() }
    }
    @Volatile private var engine: InferenceEngine? = null // Analyzer executor owns session lifetime.
    private var cameraProvider: ProcessCameraProvider? = null
    private val captureTimings = TreeMap<Long,CaptureTiming>()
    private val captured = AtomicLong()
    private val captureFailures = AtomicLong()
    @Volatile private var foreground = false
    @Volatile private var paused = false
    @Volatile private var importing = false
    @Volatile private var realtime = false
    @Volatile private var sensorOrientation: Int? = null
    @Volatile private var cameraId = "unknown"
    @Volatile private var eisUnsupported = false
    @Volatile private var oisUnsupported = false
    @Volatile private var detectionThreshold = .35f
    @Volatile private var maskThreshold = .5f
    @Volatile private var matchedMasks=false
    @Volatile private var maximumInferenceFps = 15
    @Volatile private var recordMetadata = JSONObject()
    private var previousFrameNumber = -1L
    private var skippedFrames = 0L
    private var analysisCount = 0L
    private var lastProcessedNs = 0L
    private var rateStartNs = 0L
    private var rateCount = 0
    private var fps = 0.0
    private var consecutiveErrors = 0
    private var lastSharedFile: File? = null
    private val cameraPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) startCamera() else { uiState=uiState.copy(cameraReady=false,cameraStatus="Camera access needed"); renderUi() }
    }
    private val modelPicker = registerForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { uris ->
        if (uris.isNotEmpty()) importModel(uris)
    }
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        WindowCompat.setDecorFitsSystemWindows(window,false)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        recorder = SessionRecorder(this); fusion = SensorFusion(this,recorder); store = ModelStore(this)
        displayManager = getSystemService(DISPLAY_SERVICE) as DisplayManager
        val settings = getSharedPreferences("dashboard-controls",MODE_PRIVATE)
        uiPreferences = DashboardPreferences(settings.getFloat("detection",.35f),settings.getFloat("mask",.5f),
            settings.getInt("fps",15),settings.getBoolean("overlays",true),settings.getBoolean("frames",false),settings.getBoolean("matched",false))
        detectionThreshold=uiPreferences.detection;maskThreshold=uiPreferences.mask;maximumInferenceFps=uiPreferences.fps
        matchedMasks=uiPreferences.matchedFrames
        paused=savedInstanceState?.getBoolean("paused") ?: false
        importing=true
        createDashboard()
        analyzerExecutor.execute {
            try {
                engine = (store.loadActive() ?: store.loadStarter())?.engine
                runOnUiThread { showModelStatus() }
            } catch (failure: Exception) { runOnUiThread { uiState=uiState.copy(modelDetail="Model unavailable: ${failure.message}");renderUi();toast("Model unavailable: ${failure.message}") } }
            finally { importing=false;runOnUiThread { renderUi() } }
        }
    }
    override fun onStart() {
        super.onStart(); foreground = true; fusion.start()
        motionHandler.removeCallbacks(motionTick);motionHandler.post(motionTick)
        displayManager.registerDisplayListener(displayListener,Handler(Looper.getMainLooper()))
        renderUi()
        dashboard.post { ensureCameraPermission();updateTargetRotation() }
    }
    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        controlsDialog?.dismiss();controlsDialog=null
        frameGeneration++;fusion.invalidateReading();dashboard.clear()
        createDashboard()
        dashboard.post { startCamera() }
    }
    override fun onSaveInstanceState(outState: Bundle) { outState.putBoolean("paused",paused);super.onSaveInstanceState(outState) }
    override fun onStop() {
        foreground = false;frameGeneration++;dashboard.clear()
        motionHandler.removeCallbacks(motionTick);fusion.stop()
        torchGeneration++;boundCamera?.cameraControl?.enableTorch(false)
        uiState=uiState.copy(torchOn=false,torchPending=false)
        displayManager.unregisterDisplayListener(displayListener)
        if (recorder.active) fileExecutor.execute { recorder.stop() }
        super.onStop()
    }
    override fun onDestroy() {
        cameraProvider?.unbindAll(); fusion.close()
        controlsDialog?.dismiss()
        analyzerExecutor.execute { engine?.close(); engine = null }
        analyzerExecutor.shutdown(); fileExecutor.shutdown()
        dashboard.clear()
        super.onDestroy()
    }
    private fun createDashboard() {
        val landscape = resources.configuration.screenWidthDp > resources.configuration.screenHeightDp
        ui = DashboardUi(this,landscape,DashboardActions(::togglePause,::toggleRecording,
            { modelPicker.launch(arrayOf("*/*")) },::calibrationDialog,::showControls,::ensureCameraPermission,
            ::motionReferenceDialog,::toggleOverlays,::toggleTorch))
        dashboard = ui.cameraView
        dashboard.roadAndLaneVisible = uiPreferences.overlays
        dashboard.matchedFrames=uiPreferences.matchedFrames
        dashboard.paused=paused
        setContentView(ui.root)
        renderUi()
    }
    private fun renderUi() {
        uiState=uiState.copy(paused=paused,recording=recorder.active,importing=importing,overlays=uiPreferences.overlays)
        dashboard.paused=paused
        ui.render(uiState)
    }
    private fun toggleTorch() {
        val camera=boundCamera ?: return
        if(!camera.cameraInfo.hasFlashUnit()) {toast("This camera has no torch");return}
        if(uiState.torchPending || !foreground) return
        val generation=++torchGeneration
        val future=camera.cameraControl.enableTorch(!uiState.torchOn)
        uiState=uiState.copy(torchPending=true);renderUi()
        future.addListener({
            if(generation==torchGeneration && camera===boundCamera) {
                try {future.get()} catch(failure: Exception) {toast("Torch unavailable: ${failure.cause?.message ?: failure.message}")}
                uiState=uiState.copy(torchPending=false);renderUi()
            }
        },ContextCompat.getMainExecutor(this))
    }
    private fun toggleOverlays() {
        uiPreferences=uiPreferences.copy(overlays=!uiPreferences.overlays)
        dashboard.roadAndLaneVisible=uiPreferences.overlays;dashboard.invalidate()
        getSharedPreferences("dashboard-controls",MODE_PRIVATE).edit().putBoolean("overlays",uiPreferences.overlays).apply()
        renderUi()
    }
    private fun motionReferenceDialog() {
        android.app.AlertDialog.Builder(this).setTitle("Set motion reference")
            .setMessage("Keep the phone stationary in its mounted position for one second, then tap Set reference. The bike and graphs show changes in phone orientation from this position. The first reference is automatic after a steady hold. Reset it here if you change the mount. It starts again when you reopen the app and does not calibrate the perception model.")
            .setNegativeButton("Cancel",null)
            .setPositiveButton("Set reference") { _,_ ->
                try { toast(fusion.referenceMotion());ui.clearMotionHistory() }
                catch (failure: Exception) { toast(failure.message ?: "Reference unavailable") }
            }.show()
    }
    private fun togglePause() {
        paused=!paused;frameGeneration++
        if (!paused) analyzerExecutor.execute { previousFrameNumber=-1L;rateStartNs=0;rateCount=0 }
        renderUi()
    }
    private fun showControls() {
        controlsDialog?.dismiss()
        controlsDialog=DashboardControls(this,uiState,uiPreferences,{ preferences ->
            if(matchedMasks!=preferences.matchedFrames) {frameGeneration++;dashboard.clear()}
            matchedMasks=preferences.matchedFrames
            uiPreferences=preferences
            detectionThreshold=preferences.detection;maskThreshold=preferences.mask;maximumInferenceFps=preferences.fps
            dashboard.roadAndLaneVisible=preferences.overlays
            dashboard.matchedFrames=preferences.matchedFrames
            getSharedPreferences("dashboard-controls",MODE_PRIVATE).edit()
                .putFloat("detection",preferences.detection).putFloat("mask",preferences.mask)
                .putInt("fps",preferences.fps).putBoolean("overlays",preferences.overlays)
                .putBoolean("frames",preferences.recordFrames).putBoolean("matched",preferences.matchedFrames).apply()
            renderUi()
        },{ modelPicker.launch(arrayOf("*/*")) },::calibrationDialog,
            { fusion.clearCalibration();toast("Level references cleared") },::shareTrace,::ensureCameraPermission)
        controlsDialog?.show()
    }
    private fun updateTargetRotation() {
        val rotation=dashboard.display?.rotation ?: return
        if (rotation == displayRotation) return
        displayRotation=rotation
        frameGeneration++
        fusion.invalidateReading()
        dashboard.clear()
        uiState=uiState.copy(roll=null,rawRoll=null,cameraAxes="Updating camera axes",sensorStatus="Waiting for current camera orientation");renderUi()
        expectedImageRotation=boundCameraInfo?.getSensorRotationDegrees(rotation)
        imageAnalysis?.targetRotation=rotation
        cameraPreview?.targetRotation=rotation
        recorder.event(JSONObject().put("type","display_rotation").put("rotation",rotation)
            .put("expected_image_rotation_deg",expectedImageRotation).put("timestamp_ns",SystemClock.elapsedRealtimeNanos()))
    }
    private fun ensureCameraPermission() {
        if (ContextCompat.checkSelfPermission(this,Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) startCamera()
        else cameraPermission.launch(Manifest.permission.CAMERA)
    }
    private fun startCamera() {
        if(!foreground || isDestroyed) return
        frameGeneration++;dashboard.clear()
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            try {
                if(!foreground || isDestroyed) return@addListener
                val provider = future.get(); cameraProvider = provider
                val builder = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .setTargetRotation(dashboard.display?.rotation ?: Surface.ROTATION_0)
                    .setResolutionSelector(ResolutionSelector.Builder().setResolutionStrategy(
                        ResolutionStrategy(android.util.Size(640,480),ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER)).build())
                Camera2Interop.Extender(builder)
                    .setCaptureRequestOption(CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE,CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_OFF)
                    .setCaptureRequestOption(CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE,CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE_OFF)
                    .setSessionCaptureCallback(object : CameraCaptureSession.CaptureCallback() {
                        override fun onCaptureCompleted(session: CameraCaptureSession, request: CaptureRequest, result: TotalCaptureResult) {
                            val ts = result.get(CaptureResult.SENSOR_TIMESTAMP) ?: return
                            captured.incrementAndGet()
                            val timing = CaptureTiming(ts,result.get(CaptureResult.SENSOR_EXPOSURE_TIME),
                                result.get(CaptureResult.SENSOR_ROLLING_SHUTTER_SKEW),result.frameNumber,
                                eisUnsupported || result.get(CaptureResult.CONTROL_VIDEO_STABILIZATION_MODE) == CaptureResult.CONTROL_VIDEO_STABILIZATION_MODE_OFF,
                                oisUnsupported || result.get(CaptureResult.LENS_OPTICAL_STABILIZATION_MODE) == CaptureResult.LENS_OPTICAL_STABILIZATION_MODE_OFF,
                                result.get(CaptureResult.SCALER_CROP_REGION)?.toShortString() ?: "unknown",result.get(CaptureResult.LENS_FOCAL_LENGTH))
                            synchronized(captureTimings) {
                                captureTimings[ts] = timing
                                while (captureTimings.size > 150) captureTimings.pollFirstEntry()
                            }
                            if (recorder.active) recorder.event(JSONObject().put("type","capture").put("timestamp_ns",ts)
                                .put("frame_number",result.frameNumber).put("exposure_ns",timing.exposureNs)
                                .put("rolling_shutter_skew_ns",timing.skewNs).put("eis_off",timing.eisOff)
                                .put("ois_off",timing.oisOff).put("crop",timing.crop).put("focal_length_mm",timing.focalLength))
                        }
                        override fun onCaptureFailed(session: CameraCaptureSession,request: CaptureRequest,failure: CaptureFailure) { captureFailures.incrementAndGet() }
                    })
                val analysis = builder.build()
                imageAnalysis = analysis
                analysis.setAnalyzer(analyzerExecutor,::analyze)
                torchGeneration++
                boundCameraInfo?.torchState?.removeObservers(this)
                boundCamera?.cameraControl?.enableTorch(false)
                provider.unbindAll()
                val rotation=dashboard.display?.rotation ?: Surface.ROTATION_0
                val preview=Preview.Builder().setTargetRotation(rotation)
                    .setTargetAspectRatio(AspectRatio.RATIO_4_3).build()
                preview.setSurfaceProvider(dashboard.preview.surfaceProvider)
                cameraPreview=preview
                // A shared viewport makes analysis and preview depict the same sensor region.
                val sensorRotation=provider.getCameraInfo(CameraSelector.DEFAULT_BACK_CAMERA).getSensorRotationDegrees(rotation)
                val ratio=if(sensorRotation%180==0) android.util.Rational(4,3) else android.util.Rational(3,4)
                val group=UseCaseGroup.Builder().addUseCase(preview).addUseCase(analysis)
                    .setViewPort(ViewPort.Builder(ratio,rotation).setScaleType(ViewPort.FIT).build()).build()
                val camera = provider.bindToLifecycle(this,CameraSelector.DEFAULT_BACK_CAMERA,group)
                boundCameraInfo=camera.cameraInfo;boundCamera=camera
                uiState=uiState.copy(torchAvailable=camera.cameraInfo.hasFlashUnit(),torchOn=false,torchPending=false)
                camera.cameraInfo.torchState.observe(this) { state ->
                    if(boundCamera===camera) {
                        uiState=uiState.copy(torchOn=state==TorchState.ON);renderUi()
                        recorder.event(JSONObject().put("type","torch").put("enabled",state==TorchState.ON)
                            .put("timestamp_ns",SystemClock.elapsedRealtimeNanos()))
                    }
                }
                val initialRotation=dashboard.display?.rotation ?: Surface.ROTATION_0
                displayRotation=initialRotation
                expectedImageRotation=camera.cameraInfo.getSensorRotationDegrees(initialRotation)
                val info = Camera2CameraInfo.from(camera.cameraInfo)
                cameraId = info.cameraId
                sensorOrientation = info.getCameraCharacteristic(CameraCharacteristics.SENSOR_ORIENTATION)
                realtime = info.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE) == CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME
                eisUnsupported = info.getCameraCharacteristic(CameraCharacteristics.CONTROL_AVAILABLE_VIDEO_STABILIZATION_MODES)?.all { it == 0 } == true
                oisUnsupported = info.getCameraCharacteristic(CameraCharacteristics.LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION)?.all { it == 0 } == true
                recordMetadata = JSONObject().put("camera_id",cameraId).put("sensor_orientation_deg",sensorOrientation)
                    .put("timestamp_source",if (realtime) "REALTIME" else "UNKNOWN").put("lens_facing","rear")
                    .put("intrinsic_calibration",info.getCameraCharacteristic(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION)?.let { JSONArray(it.toList()) })
                    .put("sensor_physical_size",info.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)?.toString())
                    .put("active_array",info.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)?.toShortString())
                uiState=uiState.copy(cameraReady=true,cameraStatus="Camera active");renderUi()
            } catch (failure: Exception) { uiState=uiState.copy(cameraReady=false,cameraStatus="Camera unavailable",torchAvailable=false,torchOn=false,torchPending=false);renderUi();toast(failure.message ?: "Camera initialization failed") }
        },ContextCompat.getMainExecutor(this))
    }
    private fun analyze(proxy: ImageProxy) {
        var image: Bitmap? = null
        try {
            if (!foreground || paused || importing) return
            val activeModel=engine
            if(activeModel==null && !recorder.active) return
            val generation=frameGeneration
            val startNs = SystemClock.elapsedRealtimeNanos()
            if (lastProcessedNs != 0L && startNs-lastProcessedNs < 1_000_000_000L/maximumInferenceFps) return
            lastProcessedNs = startNs
            val timestamp = proxy.imageInfo.timestamp
            var timing = synchronized(captureTimings) { captureTimings[timestamp] }
            for (attempt in 0 until 3) {
                if (timing != null || activeModel?.metadata?.usesImu != true && !recorder.active) break
                Thread.sleep(2); timing = synchronized(captureTimings) { captureTimings[timestamp] }
            }
            val rotation = proxy.imageInfo.rotationDegrees
            if (expectedImageRotation != null && rotation != expectedImageRotation) return
            val sourceTransform=CameraFrameTransform.source(proxy)
            val full = proxy.toBitmap()
            val crop = proxy.cropRect
            image = CameraFrameTransform.upright(full,crop,rotation)
            if (image !== full) full.recycle()
            val sensor = if(activeModel?.metadata?.usesImu == true || recorder.active)
                fusion.read(timing,realtime,sensorOrientation,rotation,cameraId)
                else FusionReading(0f,0f,"Phone motion active independently; model has no IMU input",timestamp)
            val exposureNs=timing?.midpointNs ?: timestamp
            val orientation=if(realtime) fusion.orientationAt(exposureNs) else null
            recorder.frame(image,timestamp)
            val inference = activeModel?.infer(image,sensor.roll,sensor.confidence,detectionThreshold,maskThreshold,compactMask=!matchedMasks)
            if (timing != null) {
                if (previousFrameNumber >= 0) skippedFrames += max(0,timing.frameNumber-previousFrameNumber-1)
                previousFrameNumber = timing.frameNumber
            }
            analysisCount++; rateCount++
            if (rateStartNs == 0L) rateStartNs = startNs
            val now = SystemClock.elapsedRealtimeNanos()
            if (now-rateStartNs >= 1_000_000_000) { fps = rateCount*1e9/(now-rateStartNs); rateCount = 0; rateStartNs = now }
            val totalMs = (now-startNs)/1e6
            recorder.event(JSONObject().put("type","analysis").put("frame_timestamp_ns",timestamp)
                .put("exposure_midpoint_ns",sensor.timestampNs).put("rotation_degrees",rotation)
                .put("width",image.width).put("height",image.height).put("roll_rad",sensor.roll)
                .put("raw_camera_roll_rad",sensor.rawRoll).put("sensor_orientation_degrees",sensorOrientation)
                .put("imu_confidence",sensor.confidence).put("imu_status",sensor.status)
                .put("network_ms",inference?.networkMs).put("pipeline_ms",totalMs)
                .put("preprocessing_ms",inference?.preprocessingMs).put("postprocessing_ms",inference?.postprocessingMs)
                .put("runtime",activeModel?.runtimeLabel)
                .put("detection_count",inference?.boxes?.size).put("score_threshold",detectionThreshold)
                .put("mask_threshold",maskThreshold).put("unprocessed_capture_frames",skippedFrames))
            val frame = DashboardFrame(image,inference,activeModel?.metadata?.classes.orEmpty(),sensor.roll,sensor.confidence > 0f,"",
                sourceTransform,if(realtime) exposureNs else null,orientation)
            image = null // Ownership transfers to UI below.
            val rate = fps; val dropped = skippedFrames; val analyzed = analysisCount
            runOnUiThread {
                if (!foreground || isDestroyed || generation != frameGeneration) { frame.image.recycle(); frame.prediction?.mask?.recycle(); return@runOnUiThread }
                dashboard.submit(frame)
                val ageMs = if (realtime) (SystemClock.elapsedRealtimeNanos()-timestamp)/1e6 else null
                uiState=uiState.copy(fps=if(inference!=null)rate else null,networkMs=inference?.networkMs,pipelineMs=totalMs,ageMs=ageMs,
                    preprocessingMs=inference?.preprocessingMs,postprocessingMs=inference?.postprocessingMs,
                    runtime=activeModel?.runtimeLabel ?: "No model",
                    frameCount=analyzed,skipped=dropped,objects=inference?.boxes?.size,
                    potholes=inference?.boxes?.count { activeModel?.metadata?.classes?.get(it.label) == "pothole" },
                    sensorStatus=sensor.status,roll=if(sensor.confidence>0f)sensor.roll else null,
                    rawRoll=sensor.rawRoll,cameraAxes="Rear sensor ${sensorOrientation ?: "?"}° · image rotation $rotation°",
                    imuUsed=activeModel?.metadata?.usesImu == true)
                if (now-lastStatsUpdateNs >= 500_000_000L) {
                    lastStatsUpdateNs=now;renderUi()
                }
            }
            consecutiveErrors = 0
        } catch (failure: Exception) {
            consecutiveErrors++
            recorder.event(JSONObject().put("type","pipeline_error").put("message",failure.message))
            if (consecutiveErrors >= 3) {
                engine?.close(); engine = null
                runOnUiThread { showModelStatus();uiState=uiState.copy(modelDetail="Inference stopped: ${failure.message}");renderUi() }
            }
        } finally { image?.recycle(); proxy.close() }
    }
    private fun importModel(uris: List<Uri>) {
        importing = true;frameGeneration++;dashboard.clear();renderUi()
        analyzerExecutor.execute {
            try {
                val loaded = store.import(uris)
                val previous = engine; engine = loaded.engine; previous?.close()
                runOnUiThread { showModelStatus(); toast("Model imported and runtime checked") }
            } catch (failure: Exception) {
                runOnUiThread { showModelStatus(); toast("Import failed: ${failure.message}") }
            } finally { importing = false;runOnUiThread {renderUi()} }
        }
    }
    private fun showModelStatus() {
        val metadata = engine?.metadata
        uiState=uiState.copy(modelLoaded=metadata != null,
            modelTitle=if(metadata==null) "Your camera is ready" else if(metadata.isYolop) "YOLOP ${metadata.width} · road + lanes" else "Aurora · ${metadata.width} × ${metadata.height} · ${metadata.alignment.uppercase()}",
            modelDetail=if(metadata==null) "Import a trained Aurora model to enable road, lane and pothole perception."
                else if(metadata.isYolop) "Pretrained on BDD100K. No IDD fine-tuning yet.\nRoad and lane masks only; no potholes, vehicles or IMU fusion.\n${metadata.provenance}"
                else "Trained: ${(metadata.supervisedTasks+metadata.supervisedDetectionClasses).joinToString(", ")}\n${metadata.provenance}")
        renderUi()
    }
    private fun calibrationDialog() {
        android.app.AlertDialog.Builder(this).setTitle("Set a level camera reference")
            .setMessage("Hold level and still for one second, with the visible horizon level. Do not save a reference to zero a tilted mount. This saves a roll reference for the current rear-camera orientation. Recalibrate after moving the mount.")
            .setNegativeButton("Cancel",null).setPositiveButton("Save level reference") { _,_ ->
                try { toast(fusion.calibrate()) } catch (failure: Exception) { toast(failure.message ?: "Calibration unavailable") }
            }.show()
    }
    private fun toggleRecording() {
        if (recorder.active) {
            fileExecutor.execute { recorder.stop(); runOnUiThread {renderUi();toast("Trace saved in app storage")} }
        } else {
            val meta = JSONObject().put("schema_version",1).put("device",Build.MODEL).put("manufacturer",Build.MANUFACTURER)
                .put("android_api",Build.VERSION.SDK_INT).put("camera",recordMetadata).put("sensors",fusion.metadata())
                .put("model",engine?.metadata?.raw).put("note","Optional 2 FPS source JPEG frames; sensor and camera metadata. No audio or location.")
            recorder.start(meta,uiPreferences.recordFrames);renderUi();toast("Recording timestamp and sensor trace")
        }
    }
    private fun shareTrace() {
        fileExecutor.execute {
            try {
                val file = recorder.archive() ?: error("Record a trace first")
                lastSharedFile = file
                runOnUiThread {
                    renderUi()
                    val uri = FileProvider.getUriForFile(this,"$packageName.files",file)
                    val intent = Intent(Intent.ACTION_SEND).setType("application/zip").putExtra(Intent.EXTRA_STREAM,uri)
                        .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                    startActivity(Intent.createChooser(intent,"Share Aurora trace"))
                }
            } catch (failure: Exception) { runOnUiThread { toast(failure.message ?: "Could not export trace") } }
        }
    }
    private fun toast(message: String) = Toast.makeText(this,message,Toast.LENGTH_LONG).show()
}
