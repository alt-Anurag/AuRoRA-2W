package org.aurora2w.app

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import org.json.JSONObject
import java.io.File
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicBoolean
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

/** Bounded asynchronous trace: disk I/O never blocks camera or sensor callbacks. */
class SessionRecorder(private val context: Context) {
    private sealed interface Item {
        data class Line(val value: String): Item
        data class Frame(val bitmap: Bitmap,val timestampNs: Long): Item
    }
    private val queue = ArrayBlockingQueue<Item>(2048)
    private val rejected = AtomicLong()
    private val frameBusy = AtomicBoolean()
    private val rejectedFrames = AtomicLong()
    @Volatile private var includeFrames = false
    @Volatile private var startedNs = 0L
    private var lastFrameNs = 0L
    private var frameCount = 0
    @Volatile var lastError: String? = null
        private set
    @Volatile private var running = false
    private var worker: Thread? = null
    private var directory: File? = null
    val active get() = running
    @Synchronized fun start(metadata: JSONObject,recordFrames: Boolean = false): File {
        check(!running && worker?.isAlive != true)
        queue.clear(); rejected.set(0); rejectedFrames.set(0); frameBusy.set(false)
        includeFrames = recordFrames; startedNs = SystemClock.elapsedRealtimeNanos(); lastFrameNs = 0; frameCount = 0; lastError = null
        val dir = File(context.filesDir,"sessions/session-${System.currentTimeMillis()}").apply { mkdirs() }
        metadata.put("frame_recording",recordFrames).put("maximum_recorded_frames",120).put("maximum_frame_recording_seconds",60)
        File(dir,"metadata.json").writeText(metadata.toString(2))
        if (recordFrames) File(dir,"frames").mkdirs()
        directory = dir; running = true
        worker = Thread({
            try {
                File(dir,"trace.jsonl").bufferedWriter().use { writer ->
                    while (running || queue.isNotEmpty()) {
                        val item = queue.poll(100,java.util.concurrent.TimeUnit.MILLISECONDS) ?: continue
                        when (item) {
                            is Item.Line -> { writer.write(item.value); writer.newLine() }
                            is Item.Frame -> try {
                                val filename = "frames/${item.timestampNs}.jpg"
                                File(dir,filename).outputStream().use { stream ->
                                    check(item.bitmap.compress(Bitmap.CompressFormat.JPEG,88,stream)) { "JPEG encoder failed" }
                                }
                                writer.write(JSONObject().put("type","recorded_frame").put("frame_timestamp_ns",item.timestampNs)
                                    .put("file",filename).put("width",item.bitmap.width).put("height",item.bitmap.height).toString())
                                writer.newLine()
                            } finally { item.bitmap.recycle(); frameBusy.set(false) }
                        }
                    }
                    writer.write(JSONObject().put("type","recorder_summary").put("dropped_log_events",rejected.get())
                        .put("dropped_frame_recordings",rejectedFrames.get()).put("scheduled_frames",frameCount).toString())
                    writer.newLine()
                }
            } catch (failure: Exception) { lastError = failure.message; running = false }
            finally {
                while (true) { val item = queue.poll() ?: break; if (item is Item.Frame) item.bitmap.recycle() }
                frameBusy.set(false)
            }
        },"aurora-log").apply { start() }
        return dir
    }
    fun event(value: JSONObject) { if (running && !queue.offer(Item.Line(value.toString()))) rejected.incrementAndGet() }
    @Synchronized fun frame(bitmap: Bitmap,timestampNs: Long) {
        if (!running || !includeFrames) return
        val now = SystemClock.elapsedRealtimeNanos()
        if (frameCount >= 120 || now-startedNs > 60_000_000_000L || now-lastFrameNs < 500_000_000L) return
        lastFrameNs = now
        if (!frameBusy.compareAndSet(false,true)) { rejectedFrames.incrementAndGet(); return }
        val copy = bitmap.copy(Bitmap.Config.ARGB_8888,false)
        if (copy == null || !queue.offer(Item.Frame(copy,timestampNs))) {
            copy?.recycle(); frameBusy.set(false); rejectedFrames.incrementAndGet()
        } else frameCount++
    }
    @Synchronized fun stop(): File? {
        if (!running) return directory
        running = false
        worker?.join(3000)
        if (worker?.isAlive == true) error("Trace writer is still flushing; retry export")
        worker = null
        return directory
    }
    fun archive(): File? {
        val dir = stop() ?: File(context.filesDir,"sessions").listFiles()?.filter { it.isDirectory }?.maxByOrNull { it.name } ?: return null
        val zipFile = File(dir.parentFile,"${dir.name}.zip")
        ZipOutputStream(zipFile.outputStream()).use { zip ->
            dir.walkTopDown().filter { it.isFile }.forEach { file ->
                zip.putNextEntry(ZipEntry(file.relativeTo(dir).invariantSeparatorsPath)); file.inputStream().use { it.copyTo(zip) }; zip.closeEntry()
            }
        }
        return zipFile
    }
}
