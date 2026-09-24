package io.github.xororz.localdream.ui.screens

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Folder
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateMapOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import io.github.xororz.localdream.R
import io.github.xororz.localdream.data.DitModelImport
import io.github.xororz.localdream.data.DitModelImport.Component
import io.github.xororz.localdream.data.DitModelImport.Kind
import io.github.xororz.localdream.data.ModelRepository

/** What the user chose in [DitImportDialog]; see [DitModelImport.import]. */
data class DitImportRequest(
    val modelName: String,
    val kind: Kind,
    val ditUri: Uri,
    val pickedComponents: Map<Component, Uri>,
)

/**
 * Dialog for importing DiT weights (e.g. a Z-Image Turbo fine-tune) as a
 * custom model. Only the DiT file is required: the text encoder, VAE and
 * tokenizer default to the installed built-in package of the same kind, and
 * any of them can be replaced by a picked file.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DitImportDialog(existingIds: Set<String>, onDismiss: () -> Unit, onImport: (DitImportRequest) -> Unit) {
    val context = LocalContext.current
    var modelName by remember { mutableStateOf("") }
    var kind by remember { mutableStateOf(Kind.Z_IMAGE) }
    var ditUri by remember { mutableStateOf<Uri?>(null) }
    var ditFileName by remember { mutableStateOf<String?>(null) }
    val picked = remember { mutableStateMapOf<Component, Uri>() }
    val pickedNames = remember { mutableStateMapOf<Component, String>() }
    var pendingComponent by remember { mutableStateOf<Component?>(null) }

    val modelId = modelName.replace(" ", "")
    val nameError = when {
        ModelRepository.isReservedModelId(modelId) -> stringResource(R.string.custom_model_id_reserved)
        modelId in existingIds -> stringResource(R.string.rename_name_exists)
        else -> null
    }
    // Built-in availability depends only on disk state, which cannot change
    // while this dialog is open.
    val builtinAvailable = remember(kind) {
        kind.components.associateWith { DitModelImport.builtinFile(context, kind, it) != null }
    }
    val componentsReady = kind.components.all { it in picked || builtinAvailable[it] == true }
    val canImport = modelId.isNotEmpty() && nameError == null && ditUri != null && componentsReady

    val ditPicker = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri == null) return@rememberLauncherForActivityResult
        ditUri = uri
        val name = displayName(context, uri)
        ditFileName = name
        if (modelName.isBlank() && name != null) modelName = name.substringBeforeLast('.')
    }
    val componentPicker = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        val component = pendingComponent
        pendingComponent = null
        if (uri == null || component == null) return@rememberLauncherForActivityResult
        picked[component] = uri
        pickedNames[component] = displayName(context, uri) ?: component.fileName
    }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(stringResource(R.string.import_dit_model)) },
        text = {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Text(
                    text = stringResource(R.string.dit_import_hint),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )

                OutlinedTextField(
                    value = modelName,
                    onValueChange = { modelName = it },
                    label = { Text(stringResource(R.string.custom_model_name)) },
                    placeholder = { Text(stringResource(R.string.custom_model_name_hint)) },
                    modifier = Modifier.fillMaxWidth(),
                    singleLine = true,
                    isError = nameError != null,
                    supportingText = nameError?.let { { Text(it) } },
                )

                Text(
                    text = stringResource(R.string.dit_import_kind),
                    style = MaterialTheme.typography.labelLarge,
                )
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Kind.entries.forEach { option ->
                        FilterChip(
                            selected = kind == option,
                            onClick = {
                                kind = option
                                // Drop picks for components the new kind does not use.
                                picked.keys.retainAll(option.components.toSet())
                                pickedNames.keys.retainAll(option.components.toSet())
                            },
                            label = { Text(shortKindLabel(option)) },
                        )
                    }
                }

                FilledTonalButton(
                    onClick = { ditPicker.launch("*/*") },
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Icon(
                        imageVector = Icons.Default.Folder,
                        contentDescription = null,
                        modifier = Modifier.size(18.dp),
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = ditFileName ?: stringResource(R.string.dit_import_select_dit),
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                }

                kind.components.forEach { component ->
                    val pickedName = pickedNames[component]
                    val status = when {
                        pickedName != null -> pickedName

                        builtinAvailable[component] == true ->
                            stringResource(R.string.dit_component_from_builtin, kind.displayName)

                        else -> stringResource(R.string.dit_component_missing, kind.displayName)
                    }
                    val missing = pickedName == null && builtinAvailable[component] != true
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                text = stringResource(component.labelRes),
                                style = MaterialTheme.typography.bodyMedium,
                            )
                            Text(
                                text = status,
                                style = MaterialTheme.typography.bodySmall,
                                color = if (missing) {
                                    MaterialTheme.colorScheme.error
                                } else {
                                    MaterialTheme.colorScheme.onSurfaceVariant
                                },
                                maxLines = 2,
                                overflow = TextOverflow.Ellipsis,
                            )
                        }
                        if (pickedName != null) {
                            TextButton(
                                onClick = {
                                    picked.remove(component)
                                    pickedNames.remove(component)
                                },
                            ) {
                                Text(stringResource(R.string.dit_component_reset))
                            }
                        } else {
                            TextButton(
                                onClick = {
                                    pendingComponent = component
                                    componentPicker.launch("*/*")
                                },
                            ) {
                                Text(stringResource(R.string.dit_component_choose))
                            }
                        }
                    }
                }
            }
        },
        confirmButton = {
            TextButton(
                onClick = {
                    val uri = ditUri
                    if (canImport && uri != null) {
                        onImport(DitImportRequest(modelName, kind, uri, picked.toMap()))
                    }
                },
                enabled = canImport,
            ) {
                Text(stringResource(R.string.add_model))
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text(stringResource(R.string.cancel))
            }
        },
    )
}

private fun shortKindLabel(kind: Kind): String = when (kind) {
    Kind.Z_IMAGE -> "Z-Image"
    Kind.FLUX2_KLEIN -> "Klein"
    Kind.QWEN_IMAGE_2_1 -> "Qwen 2.1"
}

private fun displayName(context: Context, uri: Uri): String? = try {
    context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
        ?.use { cursor -> if (cursor.moveToFirst()) cursor.getString(0) else null }
} catch (_: Exception) {
    null
} ?: uri.lastPathSegment?.substringAfterLast('/')
