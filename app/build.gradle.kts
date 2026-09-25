import java.util.Properties

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.ksp)
    alias(libs.plugins.ktlint)
    alias(libs.plugins.detekt)
}

ktlint {
    android.set(true)
    version.set("1.8.0")
    ignoreFailures.set(false)
    filter {
        exclude { it.file.path.contains("/build/") }
        exclude { it.file.path.contains("/cpp/3rdparty/") }
    }
}

detekt {
    toolVersion = "1.23.7"
    config.setFrom("$projectDir/detekt.yml")
    buildUponDefaultConfig = true
    parallel = true
    baseline = file("$projectDir/detekt-baseline.xml")
    source.setFrom(files("src/main/java", "src/main/kotlin"))
}

// Machine-specific build paths: Gradle property (-P / ~/.gradle/gradle.properties),
// then the untracked local.properties, then an environment variable. Nothing is
// hard-coded here; CMake stops with setup instructions when a path is missing.
val localProps =
    Properties().apply {
        rootProject.file("local.properties").takeIf { it.exists() }?.inputStream()?.use { load(it) }
    }

fun machinePath(
    key: String,
    env: String,
): String? =
    (project.findProperty(key) as String?)
        ?: localProps.getProperty(key)
        ?: System.getenv(env)

android {
    namespace = "io.github.xororz.localdream"
    compileSdk = 37
    ndkVersion = "28.2.13676358"

    defaultConfig {
        // Keep this Z-Image edition installable beside the upstream Local Dream
        // app, regardless of which key signed that other APK.
        applicationId = "io.github.xororz.localdream.zimage"
        minSdk = 28
//        minSdk = 31
        targetSdk = 36
        // Ahead of the version already installed on the test device (a hand-
        // patched build from an earlier session) so `adb install -r` upgrades
        // in place instead of tripping Android's downgrade guard.
        // 101: renamed to 本地梦-ZIT with its own icon, ports and export folder
        // so it can run beside upstream Local Dream (3.x).
        // 102: QNN v79 runtime ships in the APK, so published model bundles
        // can leave out qnn_runtime_libs/.
        versionCode = 102
        versionName = "2.8.1-zit2"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        vectorDrawables {
            useSupportLibrary = true
        }
        ndk {
            //noinspection ChromeOsAbiSupport
            abiFilters += "arm64-v8a"
        }
        externalNativeBuild {
            cmake {
                // qnn.sdk.dir: QAIRT SDK root (e.g. .../qairt/2.48.0.260626).
                // spm.protoc.exe: host-native protoc 35.1. sentencepiece's vendored
                // protobuf is built for the Android target only; without a host
                // protoc it tries to run the cross-compiled arm64-v8a protoc on
                // the build host to generate sentencepiece_model.pb.cc/.h.
                machinePath("qnn.sdk.dir", "QNN_SDK_ROOT")?.let { arguments("-DQNN_SDK_ROOT=$it") }
                machinePath("spm.protoc.exe", "SPM_PROTOC_EXECUTABLE")?.let {
                    arguments("-DSPM_PROTOC_EXECUTABLE=$it")
                }
            }
        }
    }

    signingConfigs {
        create("release") {
            storeFile = file(project.findProperty("RELEASE_STORE_FILE") as String? ?: "keystore.jks")
            storePassword = project.findProperty("RELEASE_STORE_PASSWORD") as String?
            keyAlias = project.findProperty("RELEASE_KEY_ALIAS") as String?
            keyPassword = project.findProperty("RELEASE_KEY_PASSWORD") as String?
        }
    }

    bundle {
        density {
            enableSplit = true
        }
        abi {
            enableSplit = true
        }
        language {
            enableSplit = false
        }
    }
    buildTypes {
        release {
            signingConfig = signingConfigs.getByName("release")
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
        debug {
//            signingConfig = signingConfigs.getByName("release")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    buildFeatures {
        compose = true
        buildConfig = true
    }
    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
        jniLibs {
            useLegacyPackaging = true
        }
    }
    flavorDimensions += "version"
    productFlavors {
        create("basic") {
            dimension = "version"
            versionNameSuffix = ""
        }
        create("filter") {
            dimension = "version"
            versionNameSuffix = "_with_filter"
        }
    }
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
}

// QNN runtime for Z-Image, packaged as assets/qnnlibs and extracted to the
// runtime dir by BackendService. The QAIRT license permits distributing these
// libraries only as part of an application, so they ship in the APK and the
// Z-Image model bundle no longer has to carry them. Z-Image context binaries
// are compiled for SM8750 (HTP v79) only, hence just the v79 set.
abstract class CopyQnnLibsTask : DefaultTask() {
    @get:InputFiles
    abstract val libs: ConfigurableFileCollection

    @get:OutputDirectory
    abstract val outputDir: DirectoryProperty

    @TaskAction
    fun copy() {
        val dest = outputDir.get().asFile.resolve("qnnlibs")
        dest.deleteRecursively()
        dest.mkdirs()
        libs.files.forEach { lib ->
            require(lib.isFile) {
                "QNN library missing: $lib. Set qnn.sdk.dir=<QAIRT SDK root> in local.properties."
            }
            lib.copyTo(File(dest, lib.name))
        }
    }
}

val zimageQnnLibs = listOf(
    "lib/aarch64-android/libQnnHtp.so",
    "lib/aarch64-android/libQnnSystem.so",
    "lib/aarch64-android/libQnnHtpV79Stub.so",
    "lib/hexagon-v79/unsigned/libQnnHtpV79.so",
    "lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so",
)

kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
    }
}

androidComponents {
    onVariants { variant ->
        val sdk = machinePath("qnn.sdk.dir", "QNN_SDK_ROOT") ?: "<qnn.sdk.dir unset>"
        val copyQnnLibs = tasks.register<CopyQnnLibsTask>(
            "copyQnnLibs${variant.name.replaceFirstChar { it.uppercase() }}",
        ) {
            libs.from(zimageQnnLibs.map { "$sdk/$it" })
        }
        variant.sources.assets?.addGeneratedSourceDirectory(copyQnnLibs, CopyQnnLibsTask::outputDir)

        variant.outputs.forEach { output ->
            val versionName = output.versionName.orNull
            if (output is com.android.build.api.variant.impl.VariantOutputImpl) {
                output.outputFileName.set("LocalDreamZImage_armv8a_$versionName.apk")
            }
        }
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.lifecycle.runtime.compose)
    implementation(libs.androidx.activity.compose)
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.ui)
    implementation(libs.androidx.ui.graphics)
    implementation(libs.androidx.ui.tooling.preview)
    implementation(libs.androidx.material3)
    implementation(libs.androidx.material3.adaptive)
    implementation(libs.androidx.material3.window.size)
    implementation(libs.androidx.graphics.shapes)
    implementation(libs.androidx.navigation.compose)
    implementation(libs.okhttp)
    implementation(libs.androidx.material.icons.core)
    implementation(libs.androidx.material.icons.extended)
    implementation(libs.androidx.datastore.preferences)
    implementation(libs.material3.xml)
    implementation(libs.coil.compose)
    implementation(libs.cropify)
    implementation(libs.androidx.room.runtime)
    implementation(libs.androidx.room.ktx)
    implementation(libs.androidx.room.paging)
    ksp(libs.androidx.room.compiler)
    implementation(libs.androidx.paging.runtime)
    implementation(libs.androidx.paging.compose)

    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(platform(libs.androidx.compose.bom))
    androidTestImplementation(libs.androidx.ui.test.junit4)
    debugImplementation(libs.androidx.ui.tooling)
    debugImplementation(libs.androidx.ui.test.manifest)

    // Adds the ktlint-rule wrappers to detekt; we only enable UnusedImports
    // (the standalone ktlint plugin's no-unused-imports does not flag them).
    detektPlugins(libs.detekt.formatting)
}
