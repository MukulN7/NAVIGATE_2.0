package com.sih26168.navigate.helper

import android.annotation.SuppressLint
import android.content.Context
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Bundle

class LocationHelper(
    private val context: Context,
    private val onLocationUpdated: (Location) -> Unit,
    private val onStatusChanged: (String) -> Unit
) : LocationListener {

    private val locationManager = context.getSystemService(Context.LOCATION_SERVICE) as LocationManager
    private var isListening = false

    @SuppressLint("MissingPermission")
    fun startLocationUpdates() {
        if (isListening) return

        try {
            val isGpsEnabled = locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER)
            val isNetworkEnabled = locationManager.isProviderEnabled(LocationManager.NETWORK_PROVIDER)

            if (!isGpsEnabled && !isNetworkEnabled) {
                onStatusChanged("GPS & Network Disabled")
                return
            }

            onStatusChanged("Acquiring GPS...")

            if (isGpsEnabled) {
                locationManager.requestLocationUpdates(
                    LocationManager.GPS_PROVIDER,
                    2000L,
                    3.0f,
                    this
                )
            }

            if (isNetworkEnabled) {
                locationManager.requestLocationUpdates(
                    LocationManager.NETWORK_PROVIDER,
                    2000L,
                    3.0f,
                    this
                )
            }

            // Get last known location as immediate fallback
            val lastGps = locationManager.getLastKnownLocation(LocationManager.GPS_PROVIDER)
            val lastNetwork = locationManager.getLastKnownLocation(LocationManager.NETWORK_PROVIDER)
            val bestLast = when {
                lastGps != null && lastNetwork != null -> if (lastGps.time > lastNetwork.time) lastGps else lastNetwork
                lastGps != null -> lastGps
                else -> lastNetwork
            }

            bestLast?.let {
                onLocationUpdated(it)
                onStatusChanged("GPS Active")
            }

            isListening = true
        } catch (e: Exception) {
            e.printStackTrace()
            onStatusChanged("Location Error")
        }
    }

    fun stopLocationUpdates() {
        if (!isListening) return
        try {
            locationManager.removeUpdates(this)
            isListening = false
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    override fun onLocationChanged(location: Location) {
        onLocationUpdated(location)
        onStatusChanged("GPS Active")
    }

    @Deprecated("Deprecated in Java")
    override fun onStatusChanged(provider: String?, status: Int, extras: Bundle?) {}
    override fun onProviderEnabled(provider: String) {
        onStatusChanged("GPS Active")
    }
    override fun onProviderDisabled(provider: String) {
        onStatusChanged("GPS Disabled")
    }
}
