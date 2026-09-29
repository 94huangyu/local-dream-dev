package io.github.xororz.localdream.data

import android.util.Log
import java.io.EOFException
import java.io.File
import java.io.IOException
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Locale
import java.util.concurrent.ConcurrentHashMap
import kotlin.math.abs
import kotlin.math.roundToLong
import org.json.JSONException
import org.json.JSONObject

/**
 * Figures shown on an imported model's card in the same form the built-in
 * models hard-code: the package size ("8.8GB") and, for DiTs, the parameter
 * count ("6B") that goes into "6B DiT, 8 steps.".
 */
object ModelPackageInfo {
    private const val TAG = "ModelPackageInfo"
    private const val BYTES_PER_GB = 1_000_000_000.0
    private const val BYTES_PER_MB = 1_000_000.0
    private const val PARAMS_PER_BILLION = 1_000_000_000.0
    private const val PARAMS_PER_MILLION = 1_000_000.0

    // A safetensors header is a JSON index; anything larger is not one.
    private const val MAX_SAFETENSORS_HEADER_BYTES = 64L shl 20

    // Bounds for GGUF metadata, so a corrupt file cannot make us allocate or
    // loop without end while skipping it.
    private const val MAX_GGUF_STRING_BYTES = 64L shl 20
    private const val MAX_GGUF_COUNT = 1L shl 32
    private const val MAX_GGUF_DIMS = 8

    // Parsing a header is cheap but not free and the list rescans on every
    // refresh, so remember the result per file version.
    private data class CacheKey(val path: String, val length: Long, val modified: Long)
    private val paramCache = ConcurrentHashMap<CacheKey, Long>()

    /** Total size of the files making up a model directory, e.g. "8.8GB". */
    fun packageSizeLabel(modelDir: File): String {
        // File.length() follows links, so files shared with a built-in package
        // count in full: the size of the package, as for built-in models.
        val bytes = modelDir.walkTopDown().filter { it.isFile }.sumOf { it.length() }
        return if (bytes >= BYTES_PER_GB) {
            String.format(Locale.ROOT, "%.1fGB", bytes / BYTES_PER_GB)
        } else {
            String.format(Locale.ROOT, "%.0fMB", bytes / BYTES_PER_MB)
        }
    }

    /**
     * Parameter count of the DiT in [modelDir] formatted like the built-in
     * descriptions ("6B", "9B", "2.5B"), or null when it cannot be read.
     */
    fun ditParamLabel(modelDir: File): String? {
        val file = listOf("dit.safetensors", "dit.gguf")
            .map { File(modelDir, it) }
            .firstOrNull { it.isFile }
            ?: return null
        val key = CacheKey(file.absolutePath, file.length(), file.lastModified())
        val params = paramCache[key] ?: countParams(file)?.also { paramCache[key] = it } ?: return null
        return formatBillions(params)
    }

    private fun formatBillions(params: Long): String {
        val billions = params / PARAMS_PER_BILLION
        if (billions < 0.1) return "${(params / PARAMS_PER_MILLION).roundToLong()}M"
        // Whole numbers as the built-in cards do ("6B" for 6.15B), one decimal
        // only where rounding would mislead (2.5B, 0.6B).
        return if (billions >= 10 || (billions >= 1 && abs(billions - billions.roundToLong()) < 0.3)) {
            "${billions.roundToLong()}B"
        } else {
            String.format(Locale.ROOT, "%.1fB", billions)
        }
    }

    private fun countParams(file: File): Long? = try {
        file.inputStream().buffered().use { input ->
            val magic = readBytes(input, 4)
            if (magic.contentEquals("GGUF".toByteArray(Charsets.US_ASCII))) {
                countGgufParams(input)
            } else {
                countSafetensorsParams(magic, input)
            }
        }
    } catch (e: IOException) {
        Log.w(TAG, "cannot read ${file.name}: ${e.message}")
        null
    } catch (e: JSONException) {
        Log.w(TAG, "bad safetensors header in ${file.name}: ${e.message}")
        null
    }

    private fun countSafetensorsParams(firstBytes: ByteArray, input: InputStream): Long? {
        val head = firstBytes + readBytes(input, 4)
        val headerSize = ByteBuffer.wrap(head).order(ByteOrder.LITTLE_ENDIAN).long
        if (headerSize < 2 || headerSize > MAX_SAFETENSORS_HEADER_BYTES) return null
        val header = JSONObject(String(readBytes(input, headerSize.toInt()), Charsets.UTF_8))
        var total = 0L
        for (key in header.keys()) {
            if (key == "__metadata__") continue
            val shape = header.optJSONObject(key)?.optJSONArray("shape") ?: continue
            var count = 1L
            for (i in 0 until shape.length()) count *= shape.optLong(i)
            total += count
        }
        return total
    }

    // GGUF v2/v3: header, metadata key/values (skipped), then tensor infos
    // whose dimensions give the parameter count without touching the data.
    private fun countGgufParams(input: InputStream): Long? {
        val version = readU32(input)
        if (version < 2) return null
        val tensorCount = readU64(input)
        val kvCount = readU64(input)
        if (tensorCount !in 0..MAX_GGUF_COUNT || kvCount !in 0..MAX_GGUF_COUNT) return null
        repeat(kvCount.toInt()) {
            skipString(input)
            skipValue(input, readU32(input).toInt())
        }
        var total = 0L
        repeat(tensorCount.toInt()) {
            skipString(input)
            val dims = readU32(input)
            if (dims !in 0..MAX_GGUF_DIMS) return null
            var count = 1L
            repeat(dims.toInt()) { count *= readU64(input) }
            readU32(input) // ggml type
            readU64(input) // data offset
            total += count
        }
        return total
    }

    private fun skipValue(input: InputStream, type: Int) {
        when (type) {
            0, 1, 7 -> skipFully(input, 1)

            // u8, i8, bool
            2, 3 -> skipFully(input, 2)

            // u16, i16
            4, 5, 6 -> skipFully(input, 4)

            // u32, i32, f32
            10, 11, 12 -> skipFully(input, 8)

            // u64, i64, f64
            8 -> skipString(input)

            9 -> {
                val elementType = readU32(input).toInt()
                val count = readU64(input)
                if (count !in 0..MAX_GGUF_COUNT) throw IOException("bad GGUF array length")
                val fixedSize = when (elementType) {
                    0, 1, 7 -> 1L
                    2, 3 -> 2L
                    4, 5, 6 -> 4L
                    10, 11, 12 -> 8L
                    else -> 0L
                }
                if (fixedSize > 0) {
                    skipFully(input, fixedSize * count)
                } else {
                    repeat(count.toInt()) { skipValue(input, elementType) }
                }
            }

            else -> throw IOException("unknown GGUF value type $type")
        }
    }

    private fun skipString(input: InputStream) {
        val length = readU64(input)
        if (length !in 0..MAX_GGUF_STRING_BYTES) throw IOException("bad GGUF string length")
        skipFully(input, length)
    }

    private fun readU32(input: InputStream): Long = ByteBuffer.wrap(readBytes(input, 4)).order(ByteOrder.LITTLE_ENDIAN).int.toLong() and 0xFFFFFFFFL

    private fun readU64(input: InputStream): Long = ByteBuffer.wrap(readBytes(input, 8)).order(ByteOrder.LITTLE_ENDIAN).long

    private fun readBytes(input: InputStream, size: Int): ByteArray {
        val buffer = ByteArray(size)
        var offset = 0
        while (offset < size) {
            val read = input.read(buffer, offset, size - offset)
            if (read < 0) throw EOFException()
            offset += read
        }
        return buffer
    }

    private fun skipFully(input: InputStream, count: Long) {
        var remaining = count
        while (remaining > 0) {
            val skipped = input.skip(remaining)
            if (skipped > 0) {
                remaining -= skipped
            } else {
                if (input.read() < 0) throw EOFException()
                remaining--
            }
        }
    }
}
