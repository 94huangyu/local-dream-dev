package io.github.xororz.localdream.data

import android.annotation.SuppressLint
import android.content.Context
import android.content.Intent
import android.os.Build
import android.util.Log
import androidx.compose.runtime.Immutable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import io.github.xororz.localdream.R
import io.github.xororz.localdream.service.ModelDownloadService
import java.io.File
import kotlin.coroutines.cancellation.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import org.json.JSONObject

@Immutable
data class Resolution(val width: Int, val height: Int) {
    val isSquare: Boolean get() = width == height

    override fun toString(): String = if (isSquare) {
        "$width×$width"
    } else {
        "$width×$height"
    }
}

object PatchScanner {
    private val squarePatchPattern = Regex("""^(\d+)\.patch$""")
    private val rectangularPatchPattern = Regex("""^(\d+)x(\d+)\.patch$""")

    fun scanAvailableResolutions(context: Context, modelId: String): List<Resolution> {
        val modelDir = File(Model.getModelsDir(context), modelId)
        if (!modelDir.exists() || !modelDir.isDirectory) {
            return emptyList()
        }

        val resolutions = mutableListOf<Resolution>()

        modelDir.listFiles()?.forEach { file ->
            if (!file.isFile) return@forEach

            squarePatchPattern.matchEntire(file.name)?.let { match ->
                val size = match.groupValues[1].toIntOrNull()
                if (size != null && size > 0) {
                    resolutions.add(Resolution(size, size))
                }
            }

            rectangularPatchPattern.matchEntire(file.name)?.let { match ->
                val width = match.groupValues[1].toIntOrNull()
                val height = match.groupValues[2].toIntOrNull()
                if (width != null && height != null && width > 0 && height > 0) {
                    resolutions.add(Resolution(width, height))
                }
            }
        }

        return resolutions.distinct().sortedBy { it.width * it.height }
    }
}

private fun getDeviceSoc(): String = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
    Build.SOC_MODEL
} else {
    "CPU"
}

@Immutable
data class DownloadProgress(val progress: Float, val downloadedBytes: Long, val totalBytes: Long)

val chipsetModelSuffixes = mapOf(
    "SM8475" to "8gen1",
    "SM8450" to "8gen1",
    "SM8550" to "8gen2",
    "SM8550P" to "8gen2",
    "QCS8550" to "8gen2",
    "QCM8550" to "8gen2",
    "SM8650" to "8gen2",
    "SM8650P" to "8gen2",
    "SM8750" to "8gen2",
    "SM8750P" to "8gen2",
    "SM8850" to "8gen2",
    "SM8850P" to "8gen2",
    "SM8735" to "8gen2",
    "SM8845" to "8gen2",
)

sealed class DownloadResult {
    data object Success : DownloadResult()
    data class Error(val message: String) : DownloadResult()
    data class Progress(val progress: DownloadProgress) : DownloadResult()
}

sealed class RenameResult {
    data object Success : RenameResult()

    // Caller decides messaging; BlankName/Reserved/Exists are recoverable input
    // errors, Io means the on-disk move failed.
    enum class Reason { BlankName, Reserved, Exists, Io }
    data class Error(val reason: Reason) : RenameResult()
}

@Immutable
data class Model(
    val id: String,
    val name: String,
    val description: String,
    val baseUrl: String,
    val fileUri: String = "",
    val generationSize: Int = 512,
    val approximateSize: String = "1GB",
    val isDownloaded: Boolean = false,
    val needsUpgrade: Boolean = false,
    // Defaults written in code for this model; only the fields it cares about.
    val codeDefaults: ModelConfig = ModelConfig(),
    // Defaults read from config.json in the model directory, if present.
    val configDefaults: ModelConfig = ModelConfig(),
    val runOnCpu: Boolean = false,
    val isCustom: Boolean = false,
    val isSdxl: Boolean = false,
    val isAnima: Boolean = false,
    val isZImage: Boolean = false,

) {
    // Per-field priority: code defaults > config.json > global defaults.
    val defaults: GenerationDefaults
        get() = codeDefaults.withFallback(configDefaults).resolve()

    // SDXL, Anima, and Z-Image render on a fixed 1024 canvas and reach non-1:1
    // outputs via aspect-ratio inpaint padding; the run screen treats them
    // alike for default size and aspect-ratio handling (ultrafix stays
    // SDXL-only). SD1.5 NPU/CPU use their own sizes / resolution patches.
    val usesFixedCanvas: Boolean
        get() = isSdxl || isAnima || isZImage

    // Backend --type value; each type implies the full model file layout.
    val backendType: String
        get() = when {
            isZImage -> "zimage"
            isAnima -> "anima"
            isSdxl -> "sdxl"
            runOnCpu -> "sd15cpu"
            else -> "sd15npu"
        }

    fun startDownload(context: Context) {
        if (isCustom || fileUri.isEmpty()) return

        val intent = Intent(context, ModelDownloadService::class.java).apply {
            action = ModelDownloadService.ACTION_START_DOWNLOAD
            putExtra(ModelDownloadService.EXTRA_MODEL_ID, id)
            putExtra(ModelDownloadService.EXTRA_MODEL_NAME, name)
            putExtra(ModelDownloadService.EXTRA_FILE_URL, "${baseUrl.removeSuffix("/")}/$fileUri")
            putExtra(ModelDownloadService.EXTRA_IS_ZIP, fileUri.endsWith(".zip"))
            putExtra(ModelDownloadService.EXTRA_IS_NPU, !runOnCpu)
            putExtra(ModelDownloadService.EXTRA_MODEL_TYPE, "sd")
        }

        context.startForegroundService(intent)
    }

    suspend fun deleteModel(context: Context, keepHistory: Boolean = true): Boolean = withContext(Dispatchers.IO) {
        try {
            val modelDir = File(getModelsDir(context), id)
            val generationPreferences = GenerationPreferences(context)

            if (!keepHistory) {
                HistoryManager(context).clearHistoryForModel(id)
            }
            generationPreferences.clearPreferencesForModel(id)
            PinnedModels.unpin(context, listOf(id))

            if (modelDir.exists() && modelDir.isDirectory) {
                val deleted = modelDir.deleteRecursively()
                Log.d("Model", "Delete model $id: $deleted")
                deleted
            } else {
                Log.d("Model", "Model does not exist: $id")
                false
            }
        } catch (e: Exception) {
            Log.e("Model", "error: ${e.message}")
            false
        }
    }

    // Rename a custom model, migrating every artifact keyed by its id: the
    // model directory, history (files + DB rows), per-model preferences and the
    // pinned list. The model directory is moved first because scanCustomModels()
    // keys off it; if that move fails nothing else is touched.
    suspend fun rename(context: Context, newName: String): RenameResult = withContext(Dispatchers.IO) {
        val newId = newName.replace(" ", "")
        when {
            newId.isEmpty() -> return@withContext RenameResult.Error(RenameResult.Reason.BlankName)

            newId == id -> return@withContext RenameResult.Success

            ModelRepository.isReservedModelId(newId) ->
                return@withContext RenameResult.Error(RenameResult.Reason.Reserved)
        }

        val modelsDir = getModelsDir(context)
        val oldDir = File(modelsDir, id)
        val newDir = File(modelsDir, newId)
        if (!oldDir.exists()) return@withContext RenameResult.Error(RenameResult.Reason.Io)
        if (newDir.exists()) return@withContext RenameResult.Error(RenameResult.Reason.Exists)

        if (!oldDir.renameTo(newDir)) {
            return@withContext RenameResult.Error(RenameResult.Reason.Io)
        }

        // The directory move is the commit point: the model is now usable under
        // its new id. Migrate the remaining artifacts best-effort. A rare failure
        // here degrades gracefully (saved params/history may not carry over) but
        // must not be reported as a failed rename, since the model HAS been
        // renamed. Cancellation must still propagate.
        try {
            HistoryManager(context).renameModel(id, newId)
            GenerationPreferences(context).migratePreferencesForModel(id, newId)
            PinnedModels.rename(context, id, newId)
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            Log.e("Model", "rename post-move migration partly failed: ${e.message}")
        }
        RenameResult.Success
    }

    companion object {
        private const val MODELS_DIR = "models"

        fun isDeviceSupported(): Boolean {
            val soc = getDeviceSoc()
            return getChipsetSuffix(soc) != null
        }

        fun isQualcommDevice(): Boolean {
            val soc = getDeviceSoc().uppercase()
            val prefixes = listOf(
                "SM", "QCS", "QCM", "CQ", "IPQ", "SXR", "AIC", "SSG",
                "SC", "SA", "SDM", "MSM", "QRB", "X1E", "X1P",
            )
            return prefixes.any { soc.startsWith(it) }
        }

        fun getChipsetSuffix(soc: String): String? {
            if (soc in chipsetModelSuffixes) {
                return chipsetModelSuffixes[soc]
            }
            if (soc.startsWith("SM")) {
                return "min"
            }
            return null
        }

        fun getModelsDir(context: Context): File = File(context.filesDir, MODELS_DIR).apply {
            if (!exists()) mkdirs()
        }

        fun isModelDownloaded(context: Context, modelId: String, isCustom: Boolean = false): Boolean {
            if (isCustom) {
                return true
            }

            val modelDir = File(getModelsDir(context), modelId)
            if (!modelDir.exists() || !modelDir.isDirectory) {
                return false
            }

            val files = modelDir.listFiles()
            return files != null && files.isNotEmpty()
        }

        // Upscalers store a single raw weight file; existence must match what
        // performUpscale() actually loads, not just a non-empty directory.
        const val UPSCALER_FILE_NAME = "upscaler.bin"

        fun isUpscalerDownloaded(context: Context, upscalerId: String): Boolean {
            val file = File(File(getModelsDir(context), upscalerId), UPSCALER_FILE_NAME)
            return file.exists() && file.length() > 0
        }

        fun needsModelUpgrade(context: Context, modelId: String, isNpu: Boolean): Boolean {
            if (!isNpu) return false

            val modelDir = File(getModelsDir(context), modelId)
            if (!modelDir.exists()) return false

            val vFile = File(modelDir, "v3")
            return !vFile.exists()
        }
    }
}

@Immutable
data class UpscalerModel(
    val id: String,
    val name: String,
    val description: String,
    val baseUrl: String,
    val fileUri: String,
    val isDownloaded: Boolean = false,
) {
    fun startDownload(context: Context) {
        val intent = Intent(context, ModelDownloadService::class.java).apply {
            action = ModelDownloadService.ACTION_START_DOWNLOAD
            putExtra(ModelDownloadService.EXTRA_MODEL_ID, id)
            putExtra(ModelDownloadService.EXTRA_MODEL_NAME, name)
            putExtra(ModelDownloadService.EXTRA_FILE_URL, "${baseUrl.removeSuffix("/")}/$fileUri")
            putExtra(ModelDownloadService.EXTRA_IS_ZIP, false)
            putExtra(ModelDownloadService.EXTRA_IS_NPU, false)
            putExtra(ModelDownloadService.EXTRA_MODEL_TYPE, "upscaler")
        }

        context.startForegroundService(intent)
    }
}

class UpscalerRepository private constructor(private val context: Context) {
    private val generationPreferences = GenerationPreferences(context)
    private val refreshMutex = Mutex()

    var upscalers by mutableStateOf<List<UpscalerModel>>(emptyList())
        private set

    private var isLoaded = false

    suspend fun ensureLoaded() {
        if (isLoaded) return
        refreshMutex.withLock {
            if (isLoaded) return
            val baseUrl = generationPreferences.getBaseUrl()
            upscalers = withContext(Dispatchers.IO) { initializeUpscalers(baseUrl) }
            isLoaded = true
        }
    }

    private fun initializeUpscalers(baseUrl: String): List<UpscalerModel> {
        val soc = getDeviceSoc()
        val suffix = Model.getChipsetSuffix(soc) ?: "min"

        return listOf(
            createAnimeUpscaler(baseUrl, suffix),
            createRealisticUpscaler(baseUrl, suffix),
        )
    }

    private fun createAnimeUpscaler(baseUrl: String, suffix: String): UpscalerModel {
        val id = "upscaler_anime"
        val fileUri =
            "xororz/upscaler/resolve/main/realesrgan_x4plus_anime_6b/upscaler_$suffix.bin"

        val isDownloaded = Model.isUpscalerDownloaded(context, id)

        return UpscalerModel(
            id = id,
            name = context.getString(R.string.upscaler_anime),
            description = context.getString(R.string.upscaler_anime_desc),
            baseUrl = baseUrl,
            fileUri = fileUri,
            isDownloaded = isDownloaded,
        )
    }

    private fun createRealisticUpscaler(baseUrl: String, suffix: String): UpscalerModel {
        val id = "upscaler_realistic"
        val fileUri = "xororz/upscaler/resolve/main/4x_UltraSharpV2_Lite/upscaler_$suffix.bin"

        val isDownloaded = Model.isUpscalerDownloaded(context, id)

        return UpscalerModel(
            id = id,
            name = context.getString(R.string.upscaler_realistic),
            description = context.getString(R.string.upscaler_realistic_desc),
            baseUrl = baseUrl,
            fileUri = fileUri,
            isDownloaded = isDownloaded,
        )
    }

    // Re-read the base URL and rebuild the upscaler list so a base-URL change
    // in settings takes effect without an app restart. Mirrors
    // ModelRepository.refreshAllModels(); the singleton otherwise caches the
    // URL captured at first ensureLoaded().
    suspend fun refreshBaseUrl() {
        refreshMutex.withLock {
            if (!isLoaded) return
            val baseUrl = generationPreferences.getBaseUrl()
            upscalers = withContext(Dispatchers.IO) { initializeUpscalers(baseUrl) }
        }
    }

    suspend fun refreshUpscalerState(upscalerId: String) {
        refreshMutex.withLock {
            val current = upscalers
            upscalers = withContext(Dispatchers.IO) {
                current.map { upscaler ->
                    if (upscaler.id == upscalerId) {
                        val isDownloaded = Model.isUpscalerDownloaded(context, upscaler.id)
                        upscaler.copy(isDownloaded = isDownloaded)
                    } else {
                        upscaler
                    }
                }
            }
        }
    }

    companion object {
        @SuppressLint("StaticFieldLeak")
        @Volatile
        private var instance: UpscalerRepository? = null

        fun getInstance(context: Context): UpscalerRepository = instance ?: synchronized(this) {
            instance ?: UpscalerRepository(context.applicationContext).also { instance = it }
        }
    }
}

class ModelRepository private constructor(private val context: Context) {
    private val generationPreferences = GenerationPreferences(context)
    private val refreshMutex = Mutex()

    // Read by the create*Model() builders during a scan; refreshed from
    // preferences at the start of every refresh, always under refreshMutex.
    private var baseUrl = "https://huggingface.co/"

    var models by mutableStateOf<List<Model>>(emptyList())
        private set

    // False until the first disk scan completes; lets the UI tell "still
    // loading" apart from "genuinely no models".
    var isLoaded by mutableStateOf(false)
        private set

    suspend fun ensureLoaded() {
        if (isLoaded) return
        refreshAllModels()
    }

    private fun scanCustomModels(): List<Model> {
        val modelsDir = Model.getModelsDir(context)
        val customModels = mutableListOf<Model>()

        if (modelsDir.exists() && modelsDir.isDirectory) {
            modelsDir.listFiles()?.forEach { dir ->
                if (!dir.isDirectory) return@forEach

                val modelId = dir.name
                if (modelId in RESERVED_MODEL_IDS) {
                    Log.w(
                        "ModelRepository",
                        "skip custom model '$modelId': id conflicts with a built-in model",
                    )
                    return@forEach
                }

                val finishedFile = File(dir, "finished")
                val npuCustomFile = File(dir, "npucustom")
                val sdxlFile = File(dir, "SDXL")
                val animaFile = File(dir, "ANIMA")
                val zimageFile = File(dir, "ZIMAGE")

                when {
                    zimageFile.exists() && isCompleteZImageBundle(dir) ->
                        customModels.add(createCustomModel(dir, isNpu = true, isZImage = true))

                    zimageFile.exists() ->
                        Log.w(
                            "ModelRepository",
                            "skip incomplete Z-Image bundle '$modelId': QNN contract or required graph is missing",
                        )

                    animaFile.exists() ->
                        customModels.add(createCustomModel(dir, isNpu = true, isAnima = true))

                    sdxlFile.exists() ->
                        customModels.add(createCustomModel(dir, isNpu = true, isSdxl = true))

                    finishedFile.exists() ->
                        customModels.add(createCustomModel(dir, isNpu = false))

                    npuCustomFile.exists() ->
                        customModels.add(createCustomModel(dir, isNpu = true))
                }
            }
        }

        return customModels.sortedBy { it.name.lowercase() }
    }

    // A ZIMAGE marker alone is intentionally insufficient. It prevents an
    // interrupted copy/conversion from appearing in the model list and later
    // failing as a misleading localhost backend error.
    private fun isCompleteZImageBundle(dir: File): Boolean {
        val tokenizer = File(dir, "tokenizer/tokenizer.json").takeIf { it.isFile }
            ?: File(dir, "tokenizer.json").takeIf { it.isFile }
        if (tokenizer == null) {
            Log.w("ModelTest", "isCompleteZImageBundle failed: tokenizer not found in ${dir.canonicalPath}")
            return false
        }

        val contract = File(dir, "final_qnn_contract.json")
        if (!contract.isFile) {
            Log.w("ModelTest", "isCompleteZImageBundle failed: final_qnn_contract.json not found in ${dir.canonicalPath}")
            return false
        }

        // Published bundles have no qnn_runtime_libs/ (the APK ships the QNN
        // runtime, see BackendService). If a bundle does carry one, it must
        // be complete, so a half-copied runtime is still rejected here.
        val qnnRuntimeDir = File(dir, "qnn_runtime_libs/aarch64-android")
        val missingSo = listOf("libQnnHtp.so", "libQnnSystem.so", "libQnnHtpV79Stub.so")
            .takeIf { qnnRuntimeDir.isDirectory }
            ?.firstOrNull { !File(qnnRuntimeDir, it).isFile }
        if (missingSo != null) {
            Log.w("ModelTest", "isCompleteZImageBundle failed: missing SO file $missingSo")
            return false
        }

        return runCatching {
            val json = JSONObject(contract.readText())
            // Only the "models" schema is runnable. This used to also accept a
            // "graphs" array, which made a bundle pass validation here and then
            // fail later inside the native pipeline: ZImageQnnContract routes a
            // contract without "models" to the legacy path, which registers just
            // four graph names (text_encoder, transformer_part1, transformer_part2,
            // vae_decoder) while PipelineZImage asks for text_encoder_part1..4 and
            // transformer_part1a/1b — so the very first loadGraph() throws
            // "Unknown Z-Image graph". Rejecting it here turns a confusing
            // mid-generation crash into a clear "wrong schema" message.
            val models = json.optJSONArray("models")
            if (models == null) {
                val legacy = if (json.has("graphs")) " (found legacy \"graphs\" schema, which the native pipeline cannot load)" else ""
                Log.w("ModelTest", "isCompleteZImageBundle failed: contract has no \"models\" array$legacy")
                throw org.json.JSONException("Z-Image contract must use the \"models\" schema$legacy")
            }

            // Two accepted transformer layouts. The 4-segment one exists
            // because the per-row quantized single transformer_part2 context
            // (3535MB) exceeds the device's unsigned PD ceiling and fails to
            // load, so part2 ships as two ~1.9GB halves. Exact-set equality is
            // kept per layout: a bundle carrying part2 *and* part2a would be
            // ambiguous about which one the pipeline should run.
            val commonGraphs = setOf(
                "text_encoder_part1", "text_encoder_part2", "text_encoder_part3",
                "text_encoder_part4", "transformer_part1a", "transformer_part1b",
                "vae_decoder",
            )
            val singlePart2 = commonGraphs + "transformer_part2"
            val splitPart2 = commonGraphs + setOf("transformer_part2a", "transformer_part2b")

            val binaries = mutableMapOf<String, String>()
            for (index in 0 until models.length()) {
                val model = models.getJSONObject(index)
                binaries[model.getString("internal_graph_name")] =
                    model.getString("context_binary")
            }

            val requiredGraphs = if (binaries.keys.contains("transformer_part2a")) {
                splitPart2
            } else {
                singlePart2
            }

            if (binaries.keys != requiredGraphs) {
                Log.w("ModelTest", "isCompleteZImageBundle failed: keys mismatch. Expected: $requiredGraphs, Got: ${binaries.keys}")
                return@runCatching false
            }

            var allValid = true
            for ((key, relativePath) in binaries) {
                if (File(relativePath).isAbsolute) {
                    Log.w("ModelTest", "isCompleteZImageBundle failed: relativePath is absolute ($relativePath)")
                    allValid = false
                    continue
                }
                val candidate = File(dir, relativePath)
                val startsWith = candidate.canonicalPath.startsWith(dir.canonicalPath + File.separator)
                val isFile = candidate.isFile
                if (!startsWith || !isFile) {
                    Log.w("ModelTest", "isCompleteZImageBundle failed at candidate $relativePath. startsWith=$startsWith, isFile=$isFile, canonicalPath=${candidate.canonicalPath}")
                    allValid = false
                }
            }
            allValid
        }.onFailure { e ->
            Log.w("ModelTest", "isCompleteZImageBundle failed with exception: ${e.message}", e)
        }.getOrDefault(false)
    }

    private fun createCustomModel(
        modelDir: File,
        isNpu: Boolean = false,
        isSdxl: Boolean = false,
        isAnima: Boolean = false,
        isZImage: Boolean = false,
    ): Model {
        val modelId = modelDir.name
        // Imported models have no code-level defaults: config.json (if
        // bundled in the zip) wins, the generic placeholder prompts below
        // only fill what it leaves unset.
        val placeholders = ModelConfig(
            prompt = "masterpiece, best quality, a cat sat on a mat,",
            negativePrompt = "lowres, bad anatomy, bad hands, missing fingers, extra fingers, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, three crus, fused feet, fused thigh, extra crus, ugly fingers, horn, huge eyes, worst face, 2girl, long fingers, disconnected limbs,",
        )
        val config = ModelConfig.read(modelDir) ?: ModelConfig()

        return Model(
            id = modelId,
            name = modelId,
            description = context.getString(R.string.custom_model),
            baseUrl = "",
            generationSize = if (isSdxl || isAnima || isZImage) 1024 else 512,
            approximateSize = "Custom",
            isDownloaded = true,
            // Turbo is a distilled CFG-free pipeline.  These code defaults
            // deliberately win over a copied config.json and old per-model
            // preferences; the native side repeats the same enforcement.
            codeDefaults = if (isZImage) {
                ModelConfig(
                    steps = 8f,
                    cfg = 0f,
                    scheduler = "zimage_flowmatch",
                )
            } else {
                ModelConfig()
            },
            configDefaults = config.withFallback(placeholders),
            runOnCpu = !isNpu,
            isCustom = true,
            isSdxl = isSdxl,
            isAnima = isAnima,
            isZImage = isZImage,
        )
    }

    // MVP: this build only ships Z-Image Turbo. The full predefined SD1.5/SDXL
    // catalog (download entries for Illustrious, CyberRealistic, AnythingV5,
    // etc.) is intentionally omitted so the model list shows just whatever
    // Z-Image bundle the user has imported.
    private fun initializeModels(): List<Model> {
        return scanCustomModels()
    }

    // Load config.json shipped inside the model's downloaded files, keeping
    // any values already merged into configDefaults (e.g. the custom model
    // placeholders) as fallback.
    private fun applyConfigDefaults(model: Model): Model {
        val config = ModelConfig.read(File(Model.getModelsDir(context), model.id)) ?: return model
        return model.copy(configDefaults = config.withFallback(model.configDefaults))
    }

    suspend fun refreshModelState(modelId: String) {
        refreshMutex.withLock {
            val current = models
            models = withContext(Dispatchers.IO) {
                current.map { model ->
                    if (model.id == modelId) {
                        val isDownloaded =
                            Model.isModelDownloaded(context, modelId, model.isCustom)
                        val needsUpgrade = if (!model.runOnCpu) {
                            Model.needsModelUpgrade(context, modelId, true)
                        } else {
                            false
                        }
                        applyConfigDefaults(
                            model.copy(
                                isDownloaded = isDownloaded,
                                needsUpgrade = needsUpgrade,
                            ),
                        )
                    } else {
                        model
                    }
                }
            }
        }
    }

    suspend fun refreshAllModels() {
        refreshMutex.withLock {
            baseUrl = generationPreferences.getBaseUrl()
            models = withContext(Dispatchers.IO) { initializeModels() }
            isLoaded = true
        }
    }

    companion object {
        // IDs reserved by built-in models and upscalers. Custom model
        // directories that match one of these would collide with the built-in
        // entry on disk and in the UI list, so they are skipped during scan.
        // Keep in sync with the create*Model() functions and UpscalerRepository.
        private val RESERVED_MODEL_IDS = setOf(
            // SDXL (NPU)
            "illustrious_v16", "illustrious_v16_dmd2",
            "cyber_realistic_v10", "cyber_realistic_v10_dmd2",
            // SD 1.5 NPU
            "anythingv5", "qteamix", "cuteyukimix", "absolutereality", "chilloutmix",
            // SD 1.5 CPU
            "anythingv5cpu", "qteamixcpu", "cuteyukimixcpu",
            "absoluterealitycpu", "chilloutmixcpu",
        )

        fun isReservedModelId(id: String): Boolean = id in RESERVED_MODEL_IDS

        @SuppressLint("StaticFieldLeak")
        @Volatile
        private var instance: ModelRepository? = null

        fun getInstance(context: Context): ModelRepository = instance ?: synchronized(this) {
            instance ?: ModelRepository(context.applicationContext).also { instance = it }
        }
    }
}
