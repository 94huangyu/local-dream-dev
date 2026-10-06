package io.github.xororz.localdream.utils

fun schedulerDisplayName(id: String?): String = when (id) {
    "dpm" -> "DPM++ 2M"
    "dpm_karras" -> "DPM++ 2M Karras"
    "euler_a" -> "Euler A"
    "euler_a_karras" -> "Euler A Karras"
    "lcm" -> "LCM"
    "euler" -> "Euler"
    "euler_karras" -> "Euler Karras"
    "euler_beta" -> "Euler Beta"
    "euler_a_beta" -> "Euler A Beta"
    "dpm_sde" -> "DPM++ 2M SDE"
    "dpm_sde_karras" -> "DPM++ 2M SDE Karras"
    null -> ""
    else -> id
}
