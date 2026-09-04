package com.sih26168.navigate.service

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import org.osmdroid.util.GeoPoint
import java.net.HttpURLConnection
import java.net.URL
import java.util.Locale

data class RouteResult(
    val points: List<GeoPoint>,
    val distanceMeters: Double,
    val durationSeconds: Double
) {
    fun getFormattedDistance(): String {
        return if (distanceMeters >= 1000) {
            String.format(Locale.US, "%.1f km", distanceMeters / 1000.0)
        } else {
            String.format(Locale.US, "%d m", distanceMeters.toInt())
        }
    }

    fun getFormattedDuration(): String {
        val totalMinutes = (durationSeconds / 60).toInt()
        val hours = totalMinutes / 60
        val mins = totalMinutes % 60
        return if (hours > 0) {
            "${hours}h ${mins}m"
        } else {
            "${mins} mins"
        }
    }
}

object RoutingService {

    private const val USER_AGENT = "NAVIGATE_2.0_Android_App/1.0 (com.sih26168.navigate)"

    suspend fun calculateRoute(start: GeoPoint, destination: GeoPoint): RouteResult? = withContext(Dispatchers.IO) {
        try {
            val urlString = String.format(
                Locale.US,
                "https://router.project-osrm.org/route/v1/driving/%.6f,%.6f;%.6f,%.6f?overview=full&geometries=geojson",
                start.longitude, start.latitude,
                destination.longitude, destination.latitude
            )
            val url = URL(urlString)

            val connection = url.openConnection() as HttpURLConnection
            connection.requestMethod = "GET"
            connection.setRequestProperty("User-Agent", USER_AGENT)
            connection.connectTimeout = 8000
            connection.readTimeout = 8000

            if (connection.responseCode == 200) {
                val jsonText = connection.inputStream.bufferedReader().use { it.readText() }
                val jsonObject = JSONObject(jsonText)
                val code = jsonObject.optString("code")
                if (code == "Ok") {
                    val routes = jsonObject.getJSONArray("routes")
                    if (routes.length() > 0) {
                        val route = routes.getJSONObject(0)
                        val distance = route.getDouble("distance")
                        val duration = route.getDouble("duration")

                        val geometry = route.getJSONObject("geometry")
                        val coordinates = geometry.getJSONArray("coordinates")
                        val points = mutableListOf<GeoPoint>()

                        for (i in 0 until coordinates.length()) {
                            val coord = coordinates.getJSONArray(i)
                            val lon = coord.getDouble(0)
                            val lat = coord.getDouble(1)
                            points.add(GeoPoint(lat, lon))
                        }

                        return@withContext RouteResult(points, distance, duration)
                    }
                }
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
        return@withContext null
    }
}
