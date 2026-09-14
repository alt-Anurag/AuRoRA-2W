package org.aurora2w.app

import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import kotlin.math.abs

/** Fixture weights live exclusively in the test APK, never the production app. */
@RunWith(AndroidJUnit4::class)
class PortableIdfaTest {
    @Test fun portableIdfaExecutesAndMatchesPythonReferenceOnAndroidCpu() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val assets = instrumentation.context.assets
        val metadataText = assets.open("untrained_idfa_fixture/model.json").bufferedReader().use { it.readText() }
        val json = JSONObject(metadataText)
        // Assert the production import guard stays closed for this fixture.
        assertTrue(json.getBoolean("synthetic_smoke"))
        assertFalse(json.getBoolean("live_inference_allowed"))
        try { ModelMetadata.parse(metadataText); fail("Synthetic fixture unexpectedly passed live import") }
        catch (_: IllegalArgumentException) { }
        val shape = json.getJSONArray("image_shape")
        fun strings(name: String): List<String> = json.getJSONArray(name).let { a -> List(a.length()) { a.getString(it) } }
        fun floats(name: String): FloatArray = json.getJSONArray(name).let { a -> FloatArray(a.length()) { a.getDouble(it).toFloat() } }
        // Test-only direct metadata construction is not exposed by ModelStore or the camera UI.
        val metadata = ModelMetadata(shape.getInt(3),shape.getInt(2),strings("classes"),floats("mean"),floats("std"),
            json.getString("alignment"),strings("outputs"),json.getString("model_sha256"),true,"TEST FIXTURE",
            emptySet(),emptySet(),json)
        val model = File(instrumentation.targetContext.cacheDir,"instrumentation-idfa.onnx")
        assets.open("untrained_idfa_fixture/model.onnx").use { input -> model.outputStream().use { input.copyTo(it) } }
        try {
            val expected = JSONObject(assets.open("untrained_idfa_fixture/expected.json").bufferedReader().use { it.readText() })
            val cases = expected.getJSONArray("cases")
            assertTrue(cases.length() >= 3)
            InferenceEngine(model,metadata).use { engine ->
                for (index in 0 until cases.length()) {
                    val case = cases.getJSONObject(index)
                    val values = case.getJSONArray("input")
                    val input = FloatArray(values.length()) { values.getDouble(it).toFloat() }
                    val imu = case.getJSONArray("imu")
                    val output = engine.rawOutputsForValidation(input,imu.getDouble(0).toFloat(),imu.getDouble(1).toFloat())
                    val references = case.getJSONObject("outputs")
                    for ((name,actual) in output) {
                        val reference = references.getJSONArray(name)
                        assertEquals("$name case $index length",reference.length(),actual.size)
                        for (i in actual.indices) {
                            val wanted = reference.getDouble(i)
                            val tolerance = 2e-4 + 2e-3*abs(wanted)
                            assertTrue("$name case $index element $i: ${actual[i]} vs $wanted",actual[i].isFinite() && abs(actual[i]-wanted) <= tolerance)
                        }
                    }
                }
            }
        } finally { model.delete() }
    }
}
