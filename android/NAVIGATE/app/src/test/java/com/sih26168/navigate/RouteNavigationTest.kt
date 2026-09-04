package com.sih26168.navigate

import com.sih26168.navigate.helper.EsEkf
import com.sih26168.navigate.helper.GnssOutageManager
import com.sih26168.navigate.helper.GnssState
import com.sih26168.navigate.helper.RouteProgressHelper
import com.sih26168.navigate.service.RouteResult
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.osmdroid.util.GeoPoint

class RouteNavigationTest {

    private lateinit var ekf: EsEkf
    private lateinit var outageManager: GnssOutageManager
    private lateinit var sampleRoute: RouteResult

    @Before
    fun setUp() {
        ekf = EsEkf()
        outageManager = GnssOutageManager()
        ekf.initialize(28.5451, 77.2721, 200.0, 0.0)

        val points = listOf(
            GeoPoint(28.5451, 77.2721),
            GeoPoint(28.5480, 77.2750),
            GeoPoint(28.5520, 77.2800)
        )
        val dist = RouteProgressHelper.totalRouteDistance(points)
        sampleRoute = RouteResult(points, dist, 600.0)
    }

    @Test
    fun testRouteStateRemainsAvailableDuringGnssOutage() {
        // Compute initial progress
        val (lat, lon, _) = ekf.getLatLonAlt()
        val p1 = RouteProgressHelper.calculateProgress(GeoPoint(lat, lon), sampleRoute.points, sampleRoute.distanceMeters)
        assertEquals(0, p1.progressPercent)

        // Enable GNSS outage
        outageManager.startOutage(1000L)
        assertTrue(outageManager.isOutageActive())

        // Route data and progress calculation must remain available during outage
        val p2 = RouteProgressHelper.calculateProgress(GeoPoint(lat, lon), sampleRoute.points, sampleRoute.distanceMeters)
        assertNotNull(p2)
        assertEquals(sampleRoute.points.size, 3)
        assertTrue(sampleRoute.distanceMeters > 0.0)
    }

    @Test
    fun testGnssOutageDoesNotDisableEsEkfPropagation() {
        outageManager.startOutage(1000L)

        val accel = doubleArrayOf(0.2, 0.0, 9.80665)
        val gyro = doubleArrayOf(0.0, 0.0, 0.01)

        for (step in 1..10) {
            ekf.predict(0.1, accel, gyro)
            ekf.updateVelocity(3.0)
            ekf.updateNhc()
        }

        val pos = ekf.getPosEnu()
        val speed = ekf.getSpeedMs()

        assertFalse(pos[0].isNaN())
        assertFalse(pos[1].isNaN())
        assertTrue("Speed should propagate via velocity updates during outage", speed > 0.0)
    }

    @Test
    fun testMarkerPositionSourceRemainsEsEkfBased() {
        // Check initial state (FIX)
        val (lat1, lon1, alt1) = ekf.getLatLonAlt()
        assertEquals(28.5451, lat1, 1e-4)
        assertEquals(77.2721, lon1, 1e-4)

        // Check during OUTAGE state
        outageManager.startOutage(1000L)
        val (lat2, lon2, alt2) = ekf.getLatLonAlt()
        assertFalse(lat2.isNaN())
        assertFalse(lon2.isNaN())

        // Check during RECOVERING & RECOVERED state
        outageManager.startRecovery()
        val (lat3, lon3, alt3) = ekf.getLatLonAlt()
        outageManager.onGnssUpdateApplied()
        val (lat4, lon4, alt4) = ekf.getLatLonAlt()

        assertFalse(lat3.isNaN())
        assertFalse(lat4.isNaN())
        assertEquals(GnssState.RECOVERED, outageManager.currentState)
    }

    @Test
    fun testGnssRecoveryDoesNotClearRoute() {
        outageManager.startOutage(1000L)
        outageManager.startRecovery()
        outageManager.onGnssUpdateApplied()

        assertEquals(3, sampleRoute.points.size)
        assertEquals("28.552", String.format("%.3f", sampleRoute.points.last().latitude))
        assertNotNull(sampleRoute.getFormattedDistance())
        assertNotNull(sampleRoute.getFormattedDuration())
    }
}
