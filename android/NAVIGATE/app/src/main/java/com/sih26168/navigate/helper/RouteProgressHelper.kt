package com.sih26168.navigate.helper

import org.osmdroid.util.GeoPoint
import java.util.Locale
import kotlin.math.*

data class RouteProgress(
    val remainingDistanceMeters: Double,
    val progressPercent: Int,
    val formattedRemaining: String,
    val formattedProgress: String
)

object RouteProgressHelper {

    private const val EARTH_RADIUS_M = 6371000.0

    fun distanceMeters(p1: GeoPoint, p2: GeoPoint): Double {
        val lat1Rad = Math.toRadians(p1.latitude)
        val lat2Rad = Math.toRadians(p2.latitude)
        val deltaLat = Math.toRadians(p2.latitude - p1.latitude)
        val deltaLon = Math.toRadians(p2.longitude - p1.longitude)

        val a = sin(deltaLat / 2.0).pow(2) +
                cos(lat1Rad) * cos(lat2Rad) * sin(deltaLon / 2.0).pow(2)
        val c = 2.0 * atan2(sqrt(a), sqrt(1.0 - a))
        return EARTH_RADIUS_M * c
    }

    fun totalRouteDistance(points: List<GeoPoint>): Double {
        if (points.size < 2) return 0.0
        var dist = 0.0
        for (i in 0 until points.size - 1) {
            dist += distanceMeters(points[i], points[i + 1])
        }
        return dist
    }

    fun calculateProgress(currentPos: GeoPoint, points: List<GeoPoint>, totalDistanceM: Double): RouteProgress {
        if (points.isEmpty()) {
            return RouteProgress(0.0, 0, "--", "Progress: --")
        }

        val totalDist = if (totalDistanceM > 0) totalDistanceM else totalRouteDistance(points)
        if (totalDist <= 0.0) {
            return RouteProgress(0.0, 100, "0 m", "Progress: 100%")
        }

        var minDistance = Double.MAX_VALUE
        var nearestIdx = 0

        for (i in points.indices) {
            val d = distanceMeters(currentPos, points[i])
            if (d < minDistance) {
                minDistance = d
                nearestIdx = i
            }
        }

        var remainingM = distanceMeters(currentPos, points[nearestIdx])
        for (i in nearestIdx until points.size - 1) {
            remainingM += distanceMeters(points[i], points[i + 1])
        }
        remainingM = min(totalDist, max(0.0, remainingM))

        val traveledM = max(0.0, totalDist - remainingM)
        val percent = min(100, max(0, ((traveledM / totalDist) * 100).toInt()))

        val formattedRem = if (remainingM >= 1000) {
            String.format(Locale.US, "%.1f km", remainingM / 1000.0)
        } else {
            String.format(Locale.US, "%d m", remainingM.toInt())
        }

        val formattedProg = String.format(Locale.US, "Progress: %d%% (Rem: %s)", percent, formattedRem)

        return RouteProgress(remainingM, percent, formattedRem, formattedProg)
    }
}
