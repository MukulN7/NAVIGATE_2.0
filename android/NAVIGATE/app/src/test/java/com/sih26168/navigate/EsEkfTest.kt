package com.sih26168.navigate

import com.sih26168.navigate.helper.EsEkf
import com.sih26168.navigate.helper.EsEkfMath
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import kotlin.math.abs

class EsEkfTest {

    private lateinit var ekf: EsEkf

    @Before
    fun setUp() {
        ekf = EsEkf(accelNoiseStd = 0.2, gyroNoiseStd = 0.02)
    }

    @Test
    fun testQuaternionNormalizationAndIdentity() {
        val qZero = doubleArrayOf(0.0, 0.0, 0.0, 0.0)
        val qNorm = EsEkfMath.quatNormalize(qZero)
        assertArrayEquals(doubleArrayOf(1.0, 0.0, 0.0, 0.0), qNorm, 1e-6)

        val qUnnorm = doubleArrayOf(-2.0, 0.0, 0.0, 0.0)
        val qNormCanonical = EsEkfMath.quatNormalize(qUnnorm)
        assertEquals(1.0, qNormCanonical[0], 1e-6)
    }

    @Test
    fun testStationaryImuPropagationAndGravityHandling() {
        // Initialize at lat 12.9716, lon 77.5946
        ekf.initialize(
            latDeg = 12.9716,
            lonDeg = 77.5946,
            altM = 900.0,
            timestampSec = 0.0,
            headingDeg = 0.0
        )

        assertTrue(ekf.isInitialized())

        // Stationary IMU: accel measures gravity up in body frame (+9.80665 m/s^2), gyro = 0
        // Because gravity vector in ENU is [0, 0, -9.80665], specific force R*a_b + g_n should sum to ~0
        val stationaryAccel = doubleArrayOf(0.0, 0.0, 9.80665)
        val stationaryGyro = doubleArrayOf(0.0, 0.0, 0.0)

        for (i in 1..10) {
            ekf.predict(dt = 0.1, accelB = stationaryAccel, gyroB = stationaryGyro)
        }

        val pos = ekf.getPosEnu()
        val vel = ekf.getVelEnu()

        // Position & velocity should remain close to zero
        assertEquals(0.0, pos[0], 1e-2)
        assertEquals(0.0, pos[1], 1e-2)
        assertEquals(0.0, pos[2], 1e-2)
        assertEquals(0.0, vel[0], 1e-2)
        assertEquals(0.0, vel[1], 1e-2)
        assertEquals(0.0, vel[2], 1e-2)
    }

    @Test
    fun testVelocityMeasurementUpdate() {
        ekf.initialize(12.9716, 77.5946, 900.0, 0.0, headingDeg = 90.0) // Facing East

        // Apply learned forward velocity update of 5.0 m/s
        ekf.updateVelocity(forwardSpeedMs = 5.0)

        val vel = ekf.getVelEnu()
        // Facing East (heading 90 deg), velocity along East (vel[0]) should be positive
        assertTrue("East velocity should increase after velocity update", vel[0] > 0.0)
    }

    @Test
    fun testNhcUpdate() {
        ekf.initialize(12.9716, 77.5946, 900.0, 0.0, headingDeg = 0.0)

        // Set non-zero lateral velocity manually by predicting or updating
        ekf.updateNhc()

        val vel = ekf.getVelEnu()
        assertFalse(vel[0].isNaN())
        assertFalse(vel[1].isNaN())
        assertFalse(vel[2].isNaN())
    }

    @Test
    fun testGnssPositionUpdate() {
        ekf.initialize(12.9716, 77.5946, 900.0, 0.0)

        val gnssEnuMeas = doubleArrayOf(10.0, 20.0, 0.0)
        ekf.updateGnssPosition(gnssEnuMeas)

        val pos = ekf.getPosEnu()
        // Position should shift towards (10, 20, 0)
        assertTrue("East position should update", pos[0] > 0.0)
        assertTrue("North position should update", pos[1] > 0.0)
    }

    @Test
    fun testInvalidMeasurementRejectionAndFiniteState() {
        ekf.initialize(12.9716, 77.5946, 900.0, 0.0)

        // 1. NaN measurement rejection
        val nanGnss = doubleArrayOf(Double.NaN, 20.0, 0.0)
        ekf.updateGnssPosition(nanGnss)

        val posAfterNan = ekf.getPosEnu()
        assertFalse(posAfterNan[0].isNaN())

        // 2. Outrageously large position jump (> 1000m) rejection
        val hugeGnss = doubleArrayOf(50000.0, 50000.0, 0.0)
        ekf.updateGnssPosition(hugeGnss)

        val posAfterHuge = ekf.getPosEnu()
        assertTrue("Huge position jump should be rejected", posAfterHuge[0] < 100.0)

        // 3. Outrageously large velocity rejection (> 100m/s)
        ekf.updateVelocity(forwardSpeedMs = 500.0)
        val speed = ekf.getSpeedMs()
        assertTrue("Huge velocity update should be rejected", speed < 50.0)
    }

    @Test
    fun testFilterNumericalStabilityMultiStep() {
        ekf.initialize(12.9716, 77.5946, 900.0, 0.0, headingDeg = 45.0)

        val accel = doubleArrayOf(0.1, 0.0, 9.80665)
        val gyro = doubleArrayOf(0.01, -0.01, 0.02)

        for (step in 1..200) {
            ekf.predict(0.1, accel, gyro)
            if (step % 10 == 0) {
                ekf.updateVelocity(forwardSpeedMs = 2.5)
                ekf.updateNhc()
            }
            if (step % 50 == 0) {
                val currentPos = ekf.getPosEnu()
                ekf.updateGnssPosition(currentPos)
            }
        }

        val pos = ekf.getPosEnu()
        val vel = ekf.getVelEnu()
        val quat = ekf.getQuat()

        for (v in pos) {
            assertFalse(v.isNaN())
            assertFalse(v.isInfinite())
        }
        for (v in vel) {
            assertFalse(v.isNaN())
            assertFalse(v.isInfinite())
        }
        for (q in quat) {
            assertFalse(q.isNaN())
            assertFalse(q.isInfinite())
        }

        // Quaternion must remain unit length
        val norm = Math.sqrt(quat[0] * quat[0] + quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3])
        assertEquals(1.0, norm, 1e-4)
    }
}
