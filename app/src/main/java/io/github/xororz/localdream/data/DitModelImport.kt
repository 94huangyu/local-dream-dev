package io.github.xororz.localdream.data

import android.content.Context
import android.net.Uri
import android.system.ErrnoException
import android.system.Os
import android.text.format.Formatter
import android.util.Log
import androidx.annotation.StringRes
import io.github.xororz.localdream.R
import java.io.File
import java.io.IOException
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import org.json.JSONException
import org.json.JSONObject

/**
 * Imports user-supplied DiT weights (a Z-Image Turbo, FLUX.2 Klein or Qwen
 * Image 2.1 fine-tune or merge) as a custom model directory in the layout
 * PipelineDit expects:
 *
 *   dit.safetensors | dit.gguf, llm.gguf, [llm_vision.gguf], vae.safetensors,
 *   tokenizer.json, plus the kind's marker file.
 *
 * The marker is written last, so an interrupted import is never listed.
 *
 * Fine-tunes normally keep the base model's text encoder, VAE and tokenizer,
 * so each of those can come from the installed built-in package of the same
 * kind. Those files are hard-linked (falling back to a symlink) rather than
 * copied, which saves several GB per imported model.
 */
object DitModelImport {
    private const val TAG = "DitModelImport"
    private const val COPY_BUFFER_BYTES = 1 shl 20

    // A safetensors header is a JSON index; anything larger is not one.
    private const val MAX_SAFETENSORS_HEADER_BYTES = 64L shl 20

    // Left free after the copy so the import cannot fill the device.
    private const val FREE_SPACE_MARGIN_BYTES = 512L shl 20

    // Prefixes of the non-DiT parts of an all-in-one checkpoint. The engine is
    // handed this file as the diffusion model only, so those parts would be
    // ignored at best.
    private val BUNDLED_PREFIXES = listOf(
        "first_stage_model.",
        "cond_stage_model.",
        "conditioner.",
        "text_encoders.",
        "text_encoder.",
        "vae.",
    )

    enum class Component(val fileName: String, @param:StringRes val labelRes: Int) {
        TEXT_ENCODER("llm.gguf", R.string.dit_component_text_encoder),
        VISION_ENCODER("llm_vision.gguf", R.string.dit_component_vision_encoder),
        VAE("vae.safetensors", R.string.dit_component_vae),
        TOKENIZER("tokenizer.json", R.string.dit_component_tokenizer),
    }

    enum class Kind(
        val ditKind: String,
        val displayName: String,
        val builtinModelId: String,
        val components: List<Component>,
        // Tensor-name fragment unique to this architecture, matched against
        // both ComfyUI and diffusers key layouts.
        val signature: String,
    ) {
        Z_IMAGE(
            "zimage",
            "Z-Image Turbo",
            "z_image_turbo",
            listOf(Component.TEXT_ENCODER, Component.VAE, Component.TOKENIZER),
            "cap_embedder.",
        ),
        FLUX2_KLEIN(
            "klein",
            "FLUX.2 Klein",
            "flux2_klein_4b",
            listOf(Component.TEXT_ENCODER, Component.VAE, Component.TOKENIZER),
            "double_stream_modulation_img",
        ),
        QWEN_IMAGE_2_1(
            "qwen21",
            "Qwen Image 2.1",
            "qwen_image_2_1",
            listOf(
                Component.TEXT_ENCODER,
                Component.VISION_ENCODER,
                Component.VAE,
                Component.TOKENIZER,
            ),
            "txt_in.text_norm",
        ),
    }

    private enum class WeightFormat(val fileName: String) {
        SAFETENSORS("dit.safetensors"),
        GGUF("dit.gguf"),
    }

    class ImportException(message: String) : Exception(message)

    /**
     * Sampling defaults for an imported DiT of [ditKind]: the settings of the
     * built-in model it derives from, not SD-style CFG and negative prompts,
     * which these distilled models do not use.
     */
    fun defaultsFor(ditKind: String): ModelConfig = when (ditKind) {
        "zimage" -> ModelConfig(
            prompt = "a lovely cat wearing black sunglasses, studio photo,",
            negativePrompt = "",
            steps = 8f,
            cfg = 1f,
            scheduler = "euler",
        )

        "klein" -> ModelConfig(
            prompt = "a lovely cat wearing black sunglasses, studio photo,",
            negativePrompt = "",
            steps = 4f,
            cfg = 1f,
            scheduler = "euler",
            denoiseStrength = 1f,
        )

        "qwen21" -> ModelConfig(
            prompt = "a lovely cat wearing black sunglasses, studio photo,",
            negativePrompt = "",
            steps = 20f,
            cfg = 1f,
            scheduler = "euler",
            denoiseStrength = 1f,
        )

        else -> ModelConfig()
    }

    /**
     * [component] of the installed built-in package of [kind], or null when
     * that package is not fully downloaded.
     */
    fun builtinFile(context: Context, kind: Kind, component: Component): File? {
        val dir = File(Model.getModelsDir(context), kind.builtinModelId)
        if (!File(dir, Model.markerFileName(kind.ditKind)).isFile) return null
        return File(dir, component.fileName).takeIf { it.isFile && it.length() > 0L }
    }

    /**
     * Creates custom model [modelName] from [ditUri], taking each component
     * from [pickedComponents] when the user chose a file for it and from the
     * built-in package otherwise. Throws [ImportException] with a
     * user-facing message; on any failure the partial directory is removed.
     */
    suspend fun import(
        context: Context,
        modelName: String,
        kind: Kind,
        ditUri: Uri,
        pickedComponents: Map<Component, Uri>,
        onStage: (String) -> Unit,
        onBytes: (copiedBytes: Long, totalBytes: Long) -> Unit,
    ) = withContext(Dispatchers.IO) {
        val modelId = modelName.replace(" ", "")
        if (modelId.isEmpty()) throw ImportException(context.getString(R.string.dit_import_error_name))
        if (ModelRepository.isReservedModelId(modelId)) {
            throw ImportException(context.getString(R.string.custom_model_id_reserved))
        }
        val modelsDir = Model.getModelsDir(context)
        val modelDir = File(modelsDir, modelId)
        if (modelDir.exists()) throw ImportException(context.getString(R.string.rename_name_exists))

        onStage(context.getString(R.string.dit_import_checking))
        val format = inspectDit(context, ditUri, kind)

        // Resolve every source before writing anything.
        val copies = mutableListOf(ditUri to File(modelDir, format.fileName))
        val links = mutableListOf<Pair<File, File>>()
        for (component in kind.components) {
            val target = File(modelDir, component.fileName)
            val picked = pickedComponents[component]
            if (picked != null) {
                copies += picked to target
            } else {
                val builtin = builtinFile(context, kind, component)
                    ?: throw ImportException(
                        context.getString(
                            R.string.dit_import_error_missing,
                            context.getString(component.labelRes),
                        ),
                    )
                links += builtin to target
            }
        }

        val sizes = copies.map { (uri, _) -> sizeOf(context, uri) }
        val totalBytes = sizes.filter { it > 0 }.sum()
        if (sizes.all { it >= 0 }) {
            val free = modelsDir.usableSpace
            if (free < totalBytes + FREE_SPACE_MARGIN_BYTES) {
                throw ImportException(
                    context.getString(
                        R.string.dit_import_error_space,
                        Formatter.formatFileSize(context, totalBytes + FREE_SPACE_MARGIN_BYTES),
                        Formatter.formatFileSize(context, free),
                    ),
                )
            }
        }

        var completed = false
        if (!modelDir.mkdirs()) throw ImportException(context.getString(R.string.cannot_open_file))
        try {
            var copiedBytes = 0L
            for ((uri, target) in copies) {
                onStage(context.getString(R.string.dit_import_copying, target.name))
                copyUri(context, uri, target) { delta ->
                    copiedBytes += delta
                    onBytes(copiedBytes, totalBytes)
                }
            }
            if (links.isNotEmpty()) {
                onStage(context.getString(R.string.dit_import_linking))
                for ((source, target) in links) linkShared(source, target)
            }
            File(modelDir, Model.markerFileName(kind.ditKind)).createNewFile()
            completed = true
        } finally {
            if (!completed) modelDir.deleteRecursively()
        }
    }

    /**
     * Checks that [uri] holds DiT weights the engine can load for [kind] and
     * returns their container format. Only safetensors headers are inspected
     * beyond the magic; GGUF metadata is left to the engine.
     */
    private fun inspectDit(context: Context, uri: Uri, kind: Kind): WeightFormat {
        val input = context.contentResolver.openInputStream(uri)
            ?: throw ImportException(context.getString(R.string.cannot_open_file))
        return input.use { stream ->
            val head = readFully(stream, 8)
                ?: throw ImportException(context.getString(R.string.dit_import_error_format))
            if (head.copyOfRange(0, 4).contentEquals("GGUF".toByteArray(Charsets.US_ASCII))) {
                return@use WeightFormat.GGUF
            }

            val headerSize = ByteBuffer.wrap(head).order(ByteOrder.LITTLE_ENDIAN).long
            if (headerSize < 2 || headerSize > MAX_SAFETENSORS_HEADER_BYTES) {
                throw ImportException(context.getString(R.string.dit_import_error_format))
            }
            val headerBytes = readFully(stream, headerSize.toInt())
                ?: throw ImportException(context.getString(R.string.dit_import_error_format))
            val header = try {
                JSONObject(String(headerBytes, Charsets.UTF_8))
            } catch (_: JSONException) {
                throw ImportException(context.getString(R.string.dit_import_error_format))
            }
            checkSafetensorsHeader(context, header, kind)
            WeightFormat.SAFETENSORS
        }
    }

    private fun checkSafetensorsHeader(context: Context, header: JSONObject, kind: Kind) {
        var hasE5m2 = false
        var bundled = false
        val detected = mutableSetOf<Kind>()
        for (key in header.keys()) {
            if (key == "__metadata__") continue
            if (header.optJSONObject(key)?.optString("dtype") == "F8_E5M2") hasE5m2 = true
            if (BUNDLED_PREFIXES.any { key.startsWith(it) }) bundled = true
            Kind.entries.filterTo(detected) { key.contains(it.signature) }
        }
        if (hasE5m2) throw ImportException(context.getString(R.string.dit_import_error_e5m2))
        if (bundled) throw ImportException(context.getString(R.string.dit_import_error_bundled))
        // Only reject a clear mismatch: an unrecognized layout is left for the
        // engine to identify, since it knows more key conventions than we do.
        if (detected.isNotEmpty() && kind !in detected) {
            throw ImportException(
                context.getString(
                    R.string.dit_import_error_kind,
                    detected.first().displayName,
                    kind.displayName,
                ),
            )
        }
    }

    private fun readFully(input: InputStream, size: Int): ByteArray? {
        val buffer = ByteArray(size)
        var offset = 0
        while (offset < size) {
            val read = input.read(buffer, offset, size - offset)
            if (read < 0) return null
            offset += read
        }
        return buffer
    }

    private fun sizeOf(context: Context, uri: Uri): Long = try {
        context.contentResolver.openAssetFileDescriptor(uri, "r")?.use { it.length } ?: -1L
    } catch (_: Exception) {
        -1L
    }

    private suspend fun copyUri(
        context: Context,
        uri: Uri,
        target: File,
        onCopied: (Long) -> Unit,
    ) {
        val input = context.contentResolver.openInputStream(uri)
            ?: throw ImportException(context.getString(R.string.cannot_open_file))
        input.use { source ->
            target.outputStream().use { sink ->
                val buffer = ByteArray(COPY_BUFFER_BYTES)
                while (true) {
                    currentCoroutineContext().ensureActive()
                    val read = source.read(buffer)
                    if (read < 0) break
                    sink.write(buffer, 0, read)
                    onCopied(read.toLong())
                }
            }
        }
    }

    // A hard link keeps the data alive even if the built-in package is deleted
    // later; a symlink is the fallback where the filesystem refuses links.
    private fun linkShared(source: File, target: File) {
        try {
            Os.link(source.absolutePath, target.absolutePath)
            return
        } catch (e: ErrnoException) {
            Log.i(TAG, "hard link of ${source.name} failed (${e.message}); using a symlink")
        }
        try {
            Os.symlink(source.absolutePath, target.absolutePath)
        } catch (e: ErrnoException) {
            throw IOException("cannot link ${source.name}: ${e.message}", e)
        }
    }
}
