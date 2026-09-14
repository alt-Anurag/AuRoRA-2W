package org.aurora2w.app

import android.graphics.BitmapFactory
import android.graphics.Color
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.*

/** Pretrained graph parity and mask projection. Pattern images are tests, never road accuracy. */
@RunWith(AndroidJUnit4::class)
class YolopParityTest {
    @Test fun pretrainedGraphPreprocessingAndMasksMatchDesktopInBothAspectRatios() {
        val instrumentation=InstrumentationRegistry.getInstrumentation()
        val source=instrumentation.targetContext.assets
        val references=instrumentation.context.assets
        fun floats(name: String): FloatArray {
            val bytes=references.open("yolop_reference/$name").use {it.readBytes()}
            val buffer=ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            return FloatArray(buffer.remaining()).also {buffer.get(it)}
        }
        val metadata=ModelMetadata.parse(source.open("yolop_starter/model.json").bufferedReader().use {it.readText()})
        assertTrue(metadata.isYolop);assertFalse(metadata.usesImu)
        val expected=JSONObject(references.open("yolop_reference/expected.json").bufferedReader().use {it.readText()})
        assertEquals(expected.getString("model_sha256"),metadata.sha256)
        val model=File(instrumentation.targetContext.cacheDir,"yolop-parity.onnx")
        source.open("yolop_starter/model.onnx").use {input->model.outputStream().use {input.copyTo(it)}}
        try {
            for(backend in listOf("CPU","XNNPACK","NNAPI","AUTO")) InferenceEngine(model,metadata,backend=="XNNPACK" || backend=="AUTO",backend=="NNAPI",autoSelect=backend=="AUTO").use {engine->
                android.util.Log.i("AuroraParity","Validating ${engine.runtimeLabel}")
                val cases=expected.getJSONArray("cases")
                for(index in 0 until cases.length()) {
                    val case=cases.getJSONObject(index);val name=case.getString("name")
                    val bitmap=references.open("yolop_reference/$name.png").use {BitmapFactory.decodeStream(it)}!!
                    val input=engine.preprocessForValidation(bitmap)
                    val referenceInput=floats("$name-input.bin")
                    assertEquals(referenceInput.size,input.size)
                    for(i in input.indices) assertEquals("resize case $index at $i",referenceInput[i],input[i],2e-6f)
                    val output=engine.rawOutputsForValidation(input,0f,0f)
                    for(task in listOf("road","lane")) {
                        val actual=output.getValue(task+"_logits");val wanted=floats("$name-$task.bin")
                        assertEquals(wanted.size,actual.size)
                        for(i in actual.indices) assertTrue("$task case $index at $i: ${actual[i]} vs ${wanted[i]}",
                            actual[i].isFinite() && abs(actual[i]-wanted[i]) <= 5e-3f+2e-3f*abs(wanted[i]))
                    }
                    val result=engine.infer(bitmap,0f,0f,.35f,.5f)
                    assertTrue(result.boxes.isEmpty())
                    assertEquals(bitmap.width,result.mask.width);assertEquals(bitmap.height,result.mask.height)
                    val pixels=IntArray(bitmap.width*bitmap.height)
                    result.mask.getPixels(pixels,0,bitmap.width,0,0,bitmap.width,bitmap.height)
                    val flags=references.open("yolop_reference/$name-mask.bin").use {it.readBytes()}
                    var mismatches=0
                    for(i in pixels.indices) {
                        val actual=when {Color.alpha(pixels[i])==0->0;Color.green(pixels[i])==77->2;else->1}
                        val wanted=when {flags[i].toInt() and 2 != 0->2;flags[i].toInt() and 1 != 0->1;else->0}
                        if(actual!=wanted)mismatches++
                    }
                    assertTrue("Projected mask mismatch: $mismatches of ${pixels.size}",mismatches <= max(2,(pixels.size*.0002).toInt()))
                    if(index==0) {
                        val large=android.graphics.Bitmap.createScaledBitmap(bitmap,480,640,true)
                        val t=Letterbox(large.width,large.height,metadata.width,metadata.height)
                        val raw=engine.rawOutputsForValidation(engine.preprocessForValidation(large),0f,0f)
                        val compact=engine.infer(large,0f,0f,.35f,.5f,compactMask=true)
                        assertEquals(t.resizedWidth,compact.mask.width)
                        assertEquals(t.resizedHeight,compact.mask.height)
                        val painted=IntArray(compact.mask.width*compact.mask.height)
                        compact.mask.getPixels(painted,0,compact.mask.width,0,0,compact.mask.width,compact.mask.height)
                        for(y in 0 until compact.mask.height) for(x in 0 until compact.mask.width) {
                            val src=(y+t.top)*t.width+x+t.left
                            val lane=Postprocess.sigmoid(raw.getValue("lane_logits")[src])>=.5f
                            val road=Postprocess.sigmoid(raw.getValue("road_logits")[src])>=.5f
                            val wanted=if(lane) Color.argb(220,255,77,145) else if(road) Color.argb(120,255,217,72) else 0
                            assertEquals(wanted,painted[y*compact.mask.width+x])
                        }
                        // A later inference must not mutate the bitmap already owned by the UI.
                        engine.infer(large,0f,0f,.35f,.95f,compactMask=true).mask.recycle()
                        val after=IntArray(painted.size)
                        compact.mask.getPixels(after,0,compact.mask.width,0,0,compact.mask.width,compact.mask.height)
                        assertArrayEquals(painted,after)
                        compact.mask.recycle();large.recycle()
                    }
                    result.mask.recycle();bitmap.recycle()
                }
            }
        } finally {model.delete()}
    }
}
