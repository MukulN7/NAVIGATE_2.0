package com.sih26168.navigate

import com.sih26168.navigate.helper.EsEkf
import com.sih26168.navigate.helper.GnssOutageManager
import com.sih26168.navigate.helper.GnssState
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test

class GnssOutageTest {

    private lateinit var ekf: EsEkf
    private lateinit var outageManager: GnssOutageManager

    @Before
    fun setUp() {
        ekf = EsEkf(accelNoiseStd = 0.2, gyroNoiseStd = 0.02)
        outageManager = GnssOutageManager()
        ekf.initialize(12.9716, 77.5946, 900.0, 0.0)
    }

    @Test
    fun testGnssUpdatesEnabled_MeasurementAccepted() {
        assertTrue(outageManager.gnssUpdatesEnabled.get())
        assertEquals(GnssState.FIX, outageManager.currentState)

        val measurement = doubleArrayOf(10.0, 10.0, 0.0)
        if (outageManager.gnssUpdatesEnabled.get()) {
            ekf.updateGnssPosition(measurement)
            outageManager.onGnssUpdateApplied()
        }

        val pos = ekf.getPosEnu()
        assertTrue("East position should be updated", pos[0] > 0.0)
        assertTrue("North position should be updated", pos[1] > 0.0)
        assertEquals(GnssState.FIX, outageManager.currentState)
    }

    @Test
    fun testGnssUpdatesDisabled_MeasurementRejected() {
        outageManager.startOutage(nowMs = 1000L)
        assertFalse(outageManager.gnssUpdatesEnabled.get())
        assertEquals(GnssState.OUTAGE, outageManager.currentState)

        val initialPos = ekf.getPosEnu()
        val measurement = doubleArrayOf(50.0, 50.0, 0.0)

        if (outageManager.gnssUpdatesEnabled.get()) {
            ekf.updateGnssPosition(measurement)
            outageManager.onGnssUpdateApplied()
        } else {
            outageManager.onGnssUpdateSkipped()
        }

        val posAfterSkipped = ekf.getPosEnu()
        assertEquals(initialPos[0], posAfterSkipped[0], 1e-6)
        assertEquals(initialPos[1], posAfterSkipped[1], 1e-6)
        assertEquals(1, outageManager.skippedUpdateCount)
    }

    @Test
    fun testEsEkfPropagation_ContinuesWhileGnssDisabled() {
        outageManager.startOutage(nowMs = 1000L)
        assertFalse(outageManager.gnssUpdatesEnabled.get())

        val stationaryAccel = doubleArrayOf(0.0, 0.0, 9.80665)
        val stationaryGyro = doubleArrayOf(0.0, 0.0, 0.0)

        for (step in 1..20) {
            ekf.predict(0.1, stationaryAccel, stationaryGyro)
            ekf.updateVelocity(forwardSpeedMs = 2.0)
            ekf.updateNhc()
        }

        val pos = ekf.getPosEnu()
        val vel = ekf.getVelEnu()

        for (p in pos) assertFalse(p.isNaN() || p.isInfinite())
        for (v in vel) assertFalse(v.isNaN() || v.isInfinite())
        assertTrue("Speed should update from velocity measurements during outage", ekf.getSpeedMs() > 0.0)
    }

    @Test
    fun testReenablingGnss_AllowsMeasurementUpdatesAgain() {
        outageManager.startOutage(nowMs = 1000L)
        assertFalse(outageManager.gnssUpdatesEnabled.get())

        // Re-enable GNSS
        outageManager.startRecovery()
        assertTrue(outageManager.gnssUpdatesEnabled.get())
        assertEquals(GnssState.RECOVERING, outageManager.currentState)

        val measurement = doubleArrayOf(20.0, 20.0, 0.0)
        if (outageManager.gnssUpdatesEnabled.get()) {
            ekf.updateGnssPosition(measurement)
            outageManager.onGnssUpdateApplied()
        }

        val pos = ekf.getPosEnu()
        assertTrue("Position should update once GNSS is re-enabled", pos[0] > 0.0)
        assertEquals(GnssState.RECOVERED, outageManager.currentState)
    }

    @Test
    fun testStateTransitions_Deterministic() {
        assertEquals(GnssState.FIX, outageManager.currentState)

        outageManager.startOutage(1000L)
        assertEquals(GnssState.OUTAGE, outageManager.currentState)
        assertTrue(outageManager.isOutageActive())

        outageManager.onGnssUpdateSkipped()
        outageManager.onGnssUpdateSkipped()
        assertEquals(2, outageManager.skippedUpdateCount)

        outageManager.startRecovery()
        assertEquals(GnssState.RECOVERING, outageManager.currentState)
        assertFalse(outageManager.isOutageActive())

        outageManager.onGnssUpdateApplied()
        assertEquals(GnssState.RECOVERED, outageManager.currentState)
    }
}
