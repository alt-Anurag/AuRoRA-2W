package org.aurora2w.app

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Test
import java.io.File

/** Measures a fixed pattern, not road accuracy. Results identify the tested device explicitly. */
class RuntimeTimingTest {
    @Test fun recordSteadyInferenceAndHostProcessingCosts() {
        val i=InstrumentationRegistry.getInstrumentation()
        val assets=i.targetContext.assets
        val metadata=ModelMetadata.parse(assets.open("yolop_starter/model.json").bufferedReader().use {it.readText()})
        val file=File(i.targetContext.cacheDir,"timing.onnx")
        assets.open("yolop_starter/model.onnx").use {input->file.outputStream().use {input.copyTo(it)}}
        val pattern=i.context.assets.open("yolop_reference/0.png").use {BitmapFactory.decodeStream(it)}!!
        val image=Bitmap.createScaledBitmap(pattern,480,640,true)
        try {
            for(backend in listOf("CPU","XNNPACK","NNAPI")) InferenceEngine(file,metadata,backend=="XNNPACK",backend=="NNAPI").use {engine->
                repeat(3) {engine.infer(image,0f,0f,.35f,.5f,compactMask=true).mask.recycle()}
                val samples=List(8) {
                    engine.infer(image,0f,0f,.35f,.5f,compactMask=true).also {it.mask.recycle()}
                }
                fun median(values: List<Double>)=values.sorted().let {(it[3]+it[4])/2}
                Log.i("AuroraTiming","device=${android.os.Build.MODEL}; runtime=${engine.runtimeLabel}; " +
                    "mode=compact_live; network_ms=${median(samples.map {it.networkMs})}; pre_ms=${median(samples.map {it.preprocessingMs})}; " +
                    "post_ms=${median(samples.map {it.postprocessingMs})}")
            }
        } finally {image.recycle();pattern.recycle();file.delete()}
    }
}
