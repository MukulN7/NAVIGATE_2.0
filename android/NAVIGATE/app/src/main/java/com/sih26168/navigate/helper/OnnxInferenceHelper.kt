package com.sih26168.navigate.helper

import android.content.Context
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Pure Kotlin / ONNX Runtime helper for NAVIGATE 2.0 AI Inference.
 * Loads and runs velocity_model_v2.onnx and attitude_model.onnx.
 */
class OnnxInferenceHelper(context: Context? = null) {

    companion object {
        private const val TAG = "OnnxInferenceHelper"

        const val VELOCITY_MODEL_ASSET = "models/velocity_model_v2.onnx"
        const val ATTITUDE_MODEL_ASSET = "models/attitude_model.onnx"

        // VelocityModel V2 Normalization Stats (Inspected from PyTorch checkpoint velocity_model_v2.pt)
        val VEL_IMU_MEAN = floatArrayOf(
            0.03711839f, -0.051446695f, 9.688787f,
            0.00036940127f, -0.0051379497f, 0.0010106001f
        )
        val VEL_IMU_STD = floatArrayOf(
            1.6385568f, 1.572975f, 0.81974626f,
            0.12874067f, 0.22483817f, 0.14009948f
        )
        const val VEL_MEAN = 11.496023f
        const val VEL_STD = 8.599233f

        // AttitudeModel Normalization Stats (Inspected from PyTorch checkpoint attitude_model.pt)
        val ATT_IMU_MEAN = floatArrayOf(
            -0.028630607f, -0.050927904f, 9.874502f,
            -0.0013529458f, -0.0023137017f, -0.00097746286f
        )
        val ATT_IMU_STD = floatArrayOf(
            1.7627163f, 1.6538998f, 0.7382193f,
            0.102856115f, 0.24616703f, 0.14654571f
        )
    }

    data class InferenceResult(
        val speedMs: Float,
        val quaternion: FloatArray, // [qw, qx, qy, qz]
        val quatNorm: Float,
        val latencyMs: Long
    ) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is InferenceResult) return false
            return speedMs == other.speedMs &&
                    quaternion.contentEquals(other.quaternion) &&
                    quatNorm == other.quatNorm &&
                    latencyMs == other.latencyMs
        }

        override fun hashCode(): Int {
            var result = speedMs.hashCode()
            result = 31 * result + quaternion.contentHashCode()
            result = 31 * result + quatNorm.hashCode()
            result = 31 * result + latencyMs.hashCode()
            return result
        }
    }

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var velSession: OrtSession? = null
    private var attSession: OrtSession? = null

    init {
        if (context != null) {
            try {
                val velBytes = context.assets.open(VELOCITY_MODEL_ASSET).readBytes()
                velSession = env.createSession(velBytes)
                Log.d(TAG, "VelocityModel V2 ONNX session loaded successfully (${velBytes.size} bytes)")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load VelocityModel V2 ONNX session: ${e.message}", e)
            }

            try {
                val attBytes = context.assets.open(ATTITUDE_MODEL_ASSET).readBytes()
                attSession = env.createSession(attBytes)
                Log.d(TAG, "AttitudeModel ONNX session loaded successfully (${attBytes.size} bytes)")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load AttitudeModel ONNX session: ${e.message}", e)
            }
        }
    }

    fun loadSessionsFromBytes(velBytes: ByteArray, attBytes: ByteArray) {
        velSession = env.createSession(velBytes)
        attSession = env.createSession(attBytes)
    }

    fun isInitialized(): Boolean {
        return velSession != null && attSession != null
    }

    /**
     * Standardizes a raw [50][6] window using specific mean/std vectors.
     * Returns a FloatBuffer ready for OnnxTensor creation [1, 50, 6].
     */
    fun preprocessWindow(
        window: List<FloatArray>,
        mean: FloatArray,
        std: FloatArray
    ): FloatBuffer {
        require(window.size == 50) { "Input window must have exactly 50 samples, got ${window.size}" }
        val buffer = FloatBuffer.allocate(50 * 6)
        for (i in 0 until 50) {
            val sample = window[i]
            require(sample.size == 6) { "Sample $i must have 6 channels, got ${sample.size}" }
            for (c in 0 until 6) {
                val s = std[c]
                val safeStd = if (s == 0f) 1f else s
                val normVal = (sample[c] - mean[c]) / safeStd
                buffer.put(normVal)
            }
        }
        buffer.flip()
        return buffer
    }

    /**
     * Runs velocity model inference on a raw IMU window [50][6].
     * Returns denormalized speed in m/s.
     */
    fun predictVelocity(window: List<FloatArray>): Float {
        val session = velSession ?: throw IllegalStateException("Velocity session not initialized")
        val inputBuffer = preprocessWindow(window, VEL_IMU_MEAN, VEL_IMU_STD)
        val tensorShape = longArrayOf(1L, 50L, 6L)

        OnnxTensor.createTensor(env, inputBuffer, tensorShape).use { inputTensor ->
            val inputs = mapOf("imu_input" to inputTensor)
            session.run(inputs).use { results ->
                val speedTensor = results.get(0) as OnnxTensor
                val normSpeed = speedTensor.floatBuffer.get(0)
                // Denormalize speed
                val physicalSpeed = max(0f, normSpeed * VEL_STD + VEL_MEAN)
                return physicalSpeed
            }
        }
    }

    /**
     * Runs attitude model inference on a raw IMU window [50][6].
     * Returns relative quaternion [qw, qx, qy, qz] (qw >= 0).
     */
    fun predictAttitude(window: List<FloatArray>): Pair<FloatArray, Float> {
        val session = attSession ?: throw IllegalStateException("Attitude session not initialized")
        val inputBuffer = preprocessWindow(window, ATT_IMU_MEAN, ATT_IMU_STD)
        val tensorShape = longArrayOf(1L, 50L, 6L)

        OnnxTensor.createTensor(env, inputBuffer, tensorShape).use { inputTensor ->
            val inputs = mapOf("imu_input" to inputTensor)
            session.run(inputs).use { results ->
                val quatTensor = results.get(0) as OnnxTensor
                val fb = quatTensor.floatBuffer
                var qw = fb.get(0)
                var qx = fb.get(1)
                var qy = fb.get(2)
                var qz = fb.get(3)

                // Canonicalize to non-negative qw
                if (qw < 0.0f) {
                    qw = -qw
                    qx = -qx
                    qy = -qy
                    qz = -qz
                }

                val norm = sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
                return Pair(floatArrayOf(qw, qx, qy, qz), norm)
            }
        }
    }

    /**
     * Runs both velocity and attitude inference on a single 50-sample IMU window.
     */
    @Synchronized
    fun runInference(window: List<FloatArray>): InferenceResult {
        val startNs = System.nanoTime()
        val speed = predictVelocity(window)
        val (quat, norm) = predictAttitude(window)
        val latencyMs = (System.nanoTime() - startNs) / 1_000_000L

        return InferenceResult(
            speedMs = speed,
            quaternion = quat,
            quatNorm = norm,
            latencyMs = latencyMs
        )
    }

    fun close() {
        try {
            velSession?.close()
            attSession?.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error closing ONNX sessions: ${e.message}", e)
        }
    }
}
