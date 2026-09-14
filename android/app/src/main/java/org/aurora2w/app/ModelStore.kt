package org.aurora2w.app

import android.content.Context
import android.net.Uri
import java.io.File
import java.io.InputStream
import java.security.MessageDigest
import java.util.UUID
import java.util.zip.ZipInputStream

data class LoadedModel(val engine: InferenceEngine, val directory: File)

/** SAF grant only; no broad storage permission. Imports are atomic after validation. */
class ModelStore(private val context: Context) {
    private val models = File(context.filesDir,"models").apply { mkdirs() }
    private val prefs = context.getSharedPreferences("model", Context.MODE_PRIVATE)
    fun loadActive(): LoadedModel? {
        val name = prefs.getString("active",null) ?: return null
        require(name.matches(Regex("[0-9a-f-]{36}")))
        return open(File(models,name))
    }
    /** Built-in, attributed real pretrained model. Never overwrites an active imported model. */
    fun loadStarter(): LoadedModel? {
        if(context.assets.list("yolop_starter")?.contains("model.onnx") != true) return null
        val directory=File(models,UUID.randomUUID().toString()).apply {mkdirs()}
        try {
            for(name in listOf("model.onnx","model.json")) context.assets.open("yolop_starter/$name").use {
                copyBounded(it,File(directory,name),if(name.endsWith(".json"))1_048_576L else 268_435_456L)
            }
            val loaded=open(directory)
            prefs.edit().putString("active",directory.name).apply()
            return loaded
        } catch(failure: Throwable) {directory.deleteRecursively();throw failure}
    }
    private fun open(directory: File): LoadedModel {
        val model = File(directory,"model.onnx")
        val metadata = ModelMetadata.parse(File(directory,"model.json").readText())
        val digest = MessageDigest.getInstance("SHA-256")
        model.inputStream().use { input ->
            val block = ByteArray(65536)
            while (true) { val n = input.read(block); if (n < 0) break; digest.update(block,0,n) }
        }
        val actual = digest.digest().joinToString("") { "%02x".format(it) }
        require(actual == metadata.sha256) { "Model checksum differs from metadata" }
        return LoadedModel(InferenceEngine(model,metadata,autoSelect=metadata.isYolop),directory)
    }
    private fun copyBounded(input: InputStream, destination: File, limit: Long) {
        destination.outputStream().use { output ->
            val block = ByteArray(65536); var total = 0L
            while (true) {
                val count = input.read(block); if (count < 0) break
                total += count; require(total <= limit) { "Imported file exceeds the size limit" }
                output.write(block,0,count)
            }
            require(total > 0) { "Empty model file" }
        }
    }
    fun import(uris: List<Uri>): LoadedModel {
        require(uris.size in 1..2) { "Choose one ZIP, or the ONNX and JSON files together" }
        val directory = File(models, UUID.randomUUID().toString()).apply { mkdirs() }
        try {
            if (uris.size == 1) {
                context.contentResolver.openInputStream(uris.single())!!.use { raw ->
                    ZipInputStream(raw).use { zip ->
                        val found = HashSet<String>()
                        while (true) {
                            val entry = zip.nextEntry ?: break
                            require(!entry.isDirectory && entry.name in setOf("model.onnx","model.json")) {
                                "Bundle may contain only model.onnx and model.json at its root"
                            }
                            require(found.add(entry.name)) { "Duplicate ZIP member" }
                            copyBounded(zip,File(directory,entry.name),if (entry.name.endsWith(".json")) 1_048_576L else 268_435_456L)
                        }
                        require(found.size == 2) { "Bundle needs model.onnx and model.json" }
                    }
                }
            } else {
                val found = HashSet<String>()
                for (uri in uris) {
                    val name = context.contentResolver.query(uri,arrayOf(android.provider.OpenableColumns.DISPLAY_NAME),null,null,null)?.use {
                        if (it.moveToFirst()) it.getString(0) else ""
                    } ?: ""
                    val suffix = name.substringAfterLast('.',"").lowercase()
                    require(suffix in setOf("onnx","json") && found.add(suffix)) { "Choose one ONNX and one JSON file" }
                    context.contentResolver.openInputStream(uri)!!.use { input ->
                        copyBounded(input,File(directory,"model.$suffix"),if (suffix == "json") 1_048_576L else 268_435_456L)
                    }
                }
            }
            val loaded = open(directory)
            prefs.edit().putString("active",directory.name).apply()
            return loaded
        } catch (failure: Throwable) { directory.deleteRecursively(); throw failure }
    }
}
