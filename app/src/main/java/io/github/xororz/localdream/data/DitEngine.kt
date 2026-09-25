package io.github.xororz.localdream.data

import android.app.ActivityManager
import android.content.Context
import android.content.SharedPreferences
import android.os.Build
import java.io.File

/**
 * The DiT engine (libdit_engine.so) that runs Z-Image Turbo, FLUX.2/Klein and
 * Qwen Image 2.1.
 *
 * It ships inside the APK alongside the backend executable, so it lands in
 * nativeLibraryDir and the core dlopens it from there. Its FastRPC skels are
 * APK assets copied to the shared runtime directory by BackendService.
 */
object DitEngine {
    const val ENGINE_LIB = "libdit_engine.so"

    // The optimized FP8 path has been validated from SM8750 onward. Comparing
    // the numeric part also admits newer SM-series chips without maintaining a
    // hard-coded allowlist; suffixes such as SM8750P naturally map to 8750.
    private const val FIRST_SUPPORTED_PART_NUMBER = 8750

    fun isSupportedDevice(): Boolean {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.S) return false
        val soc = Build.SOC_MODEL.uppercase()
        if (!soc.startsWith("SM")) return false
        val partNumber = soc.dropWhile { !it.isDigit() }.takeWhile { it.isDigit() }.toIntOrNull()
        return partNumber != null && partNumber >= FIRST_SUPPORTED_PART_NUMBER
    }

    /** Directory holding the engine, i.e. the app's native library directory. */
    fun dir(context: Context): File = File(context.applicationInfo.nativeLibraryDir)

    fun isInstalled(context: Context): Boolean = File(dir(context), ENGINE_LIB).exists()

    /** Preference key (in the same prefs as sdxl_lowram) for the DiT low-RAM mode. */
    const val LOW_RAM_PREF = "dit_lowram"

    // A "16GB" phone reports roughly 14-15.5e9 bytes of total memory and a
    // "12GB" one roughly 11-11.5e9, so this splits the two classes.
    private const val RESIDENT_DIT_MIN_TOTAL_MEM = 13_000_000_000L

    /**
     * Default for [LOW_RAM_PREF]. Off, the DiT and VAE weights stay resident in
     * the backend between generations and only the text encoder is reloaded per
     * image, which saves re-reading and re-packing ~6 GB each time. That peaks
     * around 9-10 GB at 1024², so it is only the default with 16 GB of RAM.
     */
    fun defaultLowRam(context: Context): Boolean {
        val activityManager = context.getSystemService(ActivityManager::class.java) ?: return true
        val memoryInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memoryInfo)
        return memoryInfo.totalMem < RESIDENT_DIT_MIN_TOTAL_MEM
    }

    fun isLowRam(context: Context, preferences: SharedPreferences): Boolean = preferences.getBoolean(LOW_RAM_PREF, defaultLowRam(context))
}
