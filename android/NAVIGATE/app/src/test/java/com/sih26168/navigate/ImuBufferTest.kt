package com.sih26168.navigate

import com.sih26168.navigate.helper.ImuBuffer
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test

class ImuBufferTest {

    private lateinit var imuBuffer: ImuBuffer

    @Before
    fun setUp() {
        imuBuffer = ImuBuffer()
    }

    @Test
    fun testEmptyBuffer() {
        assertEquals(0, imuBuffer.getWindowSize())
        assertFalse(imuBuffer.isReady())
        assertTrue(imuBuffer.getWindow().isEmpty())
    }

    @Test
    fun testFewerThan50Samples() {
        // Feed 10 samples at 100ms intervals
        val startNs = 1_000_000_000L
        for (i in 0 until 10) {
            val tNs = startNs + i * 100_000_000L
            imuBuffer.addAccel(tNs, floatArrayOf(1f, 2f, 3f))
            imuBuffer.addGyro(tNs, floatArrayOf(0.1f, 0.2f, 0.3f))
        }

        assertTrue(imuBuffer.getWindowSize() < 50)
        assertFalse(imuBuffer.isReady())
        assertEquals(10, imuBuffer.getWindowSize())
    }

    @Test
    fun testExactly50Samples() {
        val startNs = 1_000_000_000L
        // Feed 50 samples at 100ms intervals
        for (i in 0 until 50) {
            val tNs = startNs + i * 100_000_000L
            imuBuffer.addAccel(tNs, floatArrayOf(9.8f, 0.1f, 0.2f))
            imuBuffer.addGyro(tNs, floatArrayOf(0.01f, 0.02f, 0.03f))
        }

        assertEquals(50, imuBuffer.getWindowSize())
        assertTrue(imuBuffer.isReady())
    }

    @Test
    fun testMoreThan50SamplesRetainsOnlyNewest50() {
        val startNs = 1_000_000_000L
        // Feed 80 samples with index-encoded values
        for (i in 0 until 80) {
            val tNs = startNs + i * 100_000_000L
            imuBuffer.addAccel(tNs, floatArrayOf(i.toFloat(), 0f, 0f))
            imuBuffer.addGyro(tNs, floatArrayOf(0f, 0f, 0f))
        }

        val window = imuBuffer.getWindow()
        assertEquals(50, imuBuffer.getWindowSize())
        assertTrue(imuBuffer.isReady())

        // The oldest sample in window should be sample #30 (index 30.0f)
        assertEquals(30.0f, window.first()[0], 0.001f)
        // The newest sample in window should be sample #79 (index 79.0f)
        assertEquals(79.0f, window.last()[0], 0.001f)
    }

    @Test
    fun testEachSampleHasSixValues() {
        val startNs = 1_000_000_000L
        for (i in 0 until 5) {
            val tNs = startNs + i * 100_000_000L
            imuBuffer.addAccel(tNs, floatArrayOf(1.0f, 2.0f, 3.0f))
            imuBuffer.addGyro(tNs, floatArrayOf(4.0f, 5.0f, 6.0f))
        }

        val window = imuBuffer.getWindow()
        assertFalse(window.isEmpty())
        for (sample in window) {
            assertEquals(6, sample.size)
        }
    }

    @Test
    fun testInvalidSampleHandling() {
        val tNs = 1_000_000_000L

        // Null sample check
        assertFalse(imuBuffer.isValidSample(null))

        // Size != 3
        assertFalse(imuBuffer.isValidSample(floatArrayOf(1f, 2f)))
        assertFalse(imuBuffer.isValidSample(floatArrayOf(1f, 2f, 3f, 4f)))

        // NaN value
        assertFalse(imuBuffer.isValidSample(floatArrayOf(Float.NaN, 1f, 2f)))

        // Infinite value
        assertFalse(imuBuffer.isValidSample(floatArrayOf(1f, Float.POSITIVE_INFINITY, 2f)))

        // Valid 3-value sample
        assertTrue(imuBuffer.isValidSample(floatArrayOf(1f, 2f, 3f)))

        // Adding invalid sample to buffer returns false and does not crash
        assertFalse(imuBuffer.addAccel(tNs, floatArrayOf(Float.NaN, 0f, 0f)))
        assertFalse(imuBuffer.addGyro(tNs, floatArrayOf(0f, Float.NEGATIVE_INFINITY, 0f)))
        assertEquals(0, imuBuffer.getWindowSize())
    }

    @Test
    fun test10HzResamplingAndInterpolation() {
        val startNs = 1_000_000_000L

        // Feed higher-rate sensors (50 Hz, 20ms spacing)
        // From 0ms to 500ms -> 26 samples
        for (i in 0..25) {
            val tNs = startNs + i * 20_000_000L // 20ms
            // Accel ramps from 0.0 to 2.5
            val accelVal = (i * 0.1).toFloat()
            imuBuffer.addAccel(tNs, floatArrayOf(accelVal, 0f, 0f))
            // Gyro constant
            imuBuffer.addGyro(tNs, floatArrayOf(0.5f, 0.5f, 0.5f))
        }

        val window = imuBuffer.getWindow()
        // 500ms span at 10 Hz (100ms interval) should yield 6 samples (t=0, 100, 200, 300, 400, 500ms)
        assertEquals(6, window.size)

        // Verify values interpolated properly: at 100ms (sample index 5 in 20ms steps, accelVal = 0.5)
        assertEquals(0.5f, window[1][0], 0.05f)
        // Verify gyro values intact
        assertEquals(0.5f, window[1][3], 0.001f)
    }
}
