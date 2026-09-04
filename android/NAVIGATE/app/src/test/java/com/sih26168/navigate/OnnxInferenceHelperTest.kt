package com.sih26168.navigate

import com.sih26168.navigate.helper.OnnxInferenceHelper
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import kotlin.math.abs

class OnnxInferenceHelperTest {

    @Test
    fun testPreprocessWindowShapeAndMath() {
        val helper = OnnxInferenceHelper()
        val dummyWindow = List(50) {
            floatArrayOf(0.1f, -0.2f, 9.8f, 0.01f, -0.05f, 0.02f)
        }

        val mean = floatArrayOf(0f, 0f, 9.8f, 0f, 0f, 0f)
        val std = floatArrayOf(1f, 1f, 1f, 1f, 1f, 1f)

        val buffer = helper.preprocessWindow(dummyWindow, mean, std)

        assertEquals(300, buffer.capacity())
        assertEquals(300, buffer.remaining())

        // Check sample 0 channel 2 (z-accel: (9.8 - 9.8)/1.0 = 0.0)
        assertEquals(0.1f, buffer.get(0), 1e-5f)
        assertEquals(-0.2f, buffer.get(1), 1e-5f)
        assertEquals(0.0f, buffer.get(2), 1e-5f)
    }

    @Test(expected = IllegalArgumentException::class)
    fun testPreprocessWindowInvalidWindowSizeThrows() {
        val helper = OnnxInferenceHelper()
        val invalidWindow = List(49) { floatArrayOf(0f, 0f, 0f, 0f, 0f, 0f) }
        val mean = FloatArray(6) { 0f }
        val std = FloatArray(6) { 1f }
        helper.preprocessWindow(invalidWindow, mean, std)
    }

    @Test(expected = IllegalArgumentException::class)
    fun testPreprocessWindowInvalidChannelCountThrows() {
        val helper = OnnxInferenceHelper()
        val invalidWindow = List(50) { floatArrayOf(0f, 0f, 0f) }
        val mean = FloatArray(6) { 0f }
        val std = FloatArray(6) { 1f }
        helper.preprocessWindow(invalidWindow, mean, std)
    }

    @Test
    fun testVelocityDenormalizationMath() {
        val normSpeed = 0.5f
        val physicalSpeed = normSpeed * OnnxInferenceHelper.VEL_STD + OnnxInferenceHelper.VEL_MEAN
        val expected = 0.5f * 8.599233f + 11.496023f
        assertEquals(expected, physicalSpeed, 1e-4f)
    }

    @Test
    fun testQuaternionNormAndCanonicalization() {
        // Test canonicalization (negative qw becomes positive)
        var qw = -0.8f
        var qx = 0.6f
        var qy = 0.0f
        var qz = 0.0f

        if (qw < 0.0f) {
            qw = -qw
            qx = -qx
            qy = -qy
            qz = -qz
        }

        assertEquals(0.8f, qw, 1e-5f)
        assertEquals(-0.6f, qx, 1e-5f)

        val norm = kotlin.math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        assertEquals(1.0f, norm, 1e-5f)
    }

    @Test
    fun testFullOnnxInferenceWithModelFiles() {
        val velFile = File("src/main/assets/models/velocity_model_v2.onnx")
        val attFile = File("src/main/assets/models/attitude_model.onnx")

        assertTrue("Velocity ONNX model file must exist", velFile.exists())
        assertTrue("Attitude ONNX model file must exist", attFile.exists())

        val helper = OnnxInferenceHelper()
        helper.loadSessionsFromBytes(velFile.readBytes(), attFile.readBytes())

        assertTrue(helper.isInitialized())

        // Create a fixed deterministic 50-sample window
        val window = List(50) { i ->
            floatArrayOf(
                0.01f * (i % 5),
                -0.02f * (i % 3),
                9.81f + 0.05f * (i % 2),
                0.001f * i,
                -0.002f * i,
                0.0005f * i
            )
        }

        val result = helper.runInference(window)

        assertNotNull(result)
        assertTrue("Velocity must be finite and non-negative", result.speedMs >= 0.0f && !result.speedMs.isNaN())
        assertEquals(4, result.quaternion.size)
        assertTrue("Quaternion qw must be non-negative", result.quaternion[0] >= 0.0f)
        for (q in result.quaternion) {
            assertTrue("Quaternion components must be finite", !q.isNaN() && !q.isInfinite())
        }

        // Quaternion L2 norm check
        assertEquals("Quaternion norm must be approximately 1.0", 1.0f, result.quatNorm, 1e-4f)

        // Exact numerical cross-validation against Python ONNX Runtime:
        // Python: Velocity = 0.757857 m/s, Quaternion = [0.999916, 0.012857, -0.000164, -0.001587]
        assertEquals(0.757857f, result.speedMs, 1e-3f)
        assertEquals(0.999916f, result.quaternion[0], 1e-3f)
        assertEquals(0.012857f, result.quaternion[1], 1e-3f)
        assertEquals(-0.000164f, result.quaternion[2], 1e-3f)
        assertEquals(-0.001587f, result.quaternion[3], 1e-3f)

        helper.close()
    }
}
