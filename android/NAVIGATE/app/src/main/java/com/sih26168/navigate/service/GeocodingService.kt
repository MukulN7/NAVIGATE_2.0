package com.sih26168.navigate.service

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder

data class GeocodedPlace(
    val latitude: Double,
    val longitude: Double,
    val displayName: String
)

object GeocodingService {

    private const val USER_AGENT = "NAVIGATE_2.0_Android_App/1.0 (com.sih26168.navigate)"

    suspend fun searchPlace(query: String): GeocodedPlace? = withContext(Dispatchers.IO) {
        if (query.isBlank()) return@withContext null

        try {
            val encodedQuery = URLEncoder.encode(query.trim(), "UTF-8")
            val urlString = "https://nominatim.openstreetmap.org/search?q=$encodedQuery&format=json&limit=1"
            val url = URL(urlString)

            val connection = url.openConnection() as HttpURLConnection
            connection.requestMethod = "GET"
            connection.setRequestProperty("User-Agent", USER_AGENT)
            connection.connectTimeout = 8000
            connection.readTimeout = 8000

            if (connection.responseCode == 200) {
                val jsonText = connection.inputStream.bufferedReader().use { it.readText() }
                val jsonArray = JSONArray(jsonText)
                if (jsonArray.length() > 0) {
                    val firstResult = jsonArray.getJSONObject(0)
                    val lat = firstResult.getString("lat").toDouble()
                    val lon = firstResult.getString("lon").toDouble()
                    val name = firstResult.optString("display_name", query)
                    return@withContext GeocodedPlace(lat, lon, name)
                }
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
        return@withContext null
    }
}
