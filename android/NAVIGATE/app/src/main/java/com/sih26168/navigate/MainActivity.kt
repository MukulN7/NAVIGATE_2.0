package com.sih26168.navigate

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Color
import android.location.Location
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.view.View
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputMethodManager
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.sih26168.navigate.databinding.ActivityMainBinding
import com.sih26168.navigate.helper.EsEkf
import com.sih26168.navigate.helper.EsEkfMath
import com.sih26168.navigate.helper.GnssOutageManager
import com.sih26168.navigate.helper.GnssState
import com.sih26168.navigate.helper.ImuHelper
import com.sih26168.navigate.helper.LocationHelper
import com.sih26168.navigate.helper.OnnxInferenceHelper
import com.sih26168.navigate.service.GeocodingService
import com.sih26168.navigate.service.RoutingService
import kotlinx.coroutines.launch
import org.osmdroid.config.Configuration
import org.osmdroid.tileprovider.tilesource.TileSourceFactory
import org.osmdroid.util.BoundingBox
import org.osmdroid.util.GeoPoint
import org.osmdroid.views.overlay.Marker
import org.osmdroid.views.overlay.Polyline
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var locationHelper: LocationHelper
    private lateinit var imuHelper: ImuHelper
    private lateinit var onnxInferenceHelper: OnnxInferenceHelper
    private val esEkf = EsEkf()
    private val gnssOutageManager = GnssOutageManager()

    private val navigationExecutor = Executors.newSingleThreadExecutor()
    private val isInferenceRunning = AtomicBoolean(false)

    private var currentGeoPoint: GeoPoint? = null
    private var destinationGeoPoint: GeoPoint? = null

    private var currentLocationMarker: Marker? = null
    private var destinationMarker: Marker? = null
    private var routePolyline: Polyline? = null

    private var isMapCenteredOnFirstFix = false

    // Diagnostics & Tracking
    private var lastProcessedSampleCount = -1L
    private var inferenceCount = 0
    private var lastAiRateCalcTimeMs = 0L
    private var currentAiRateHz = 0.0
    private var lastEkfLogTimeMs = 0L

    // Attitude History Buffer for 5-second relative attitude update
    private val attitudeHistory = mutableListOf<Pair<Double, DoubleArray>>()

    companion object {
        private const val LOCATION_PERMISSION_REQUEST_CODE = 1001
        private const val USER_AGENT = "NAVIGATE_2.0_Android_App/1.0 (com.sih26168.navigate)"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O_MR1) {
            setShowWhenLocked(true)
            setTurnScreenOn(true)
        }
        window.addFlags(
            android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON or
                    android.view.WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED or
                    android.view.WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON
        )

        val ctx = applicationContext
        Configuration.getInstance().load(ctx, getSharedPreferences("osmdroid", Context.MODE_PRIVATE))
        Configuration.getInstance().userAgentValue = USER_AGENT

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        onnxInferenceHelper = OnnxInferenceHelper(this)

        setupMapView()
        setupListeners()
        setupLocationHelper()
        setupImuHelper()

        checkLocationPermissions()
    }

    private fun setupImuHelper() {
        imuHelper = ImuHelper(this) { statusText, isReady, sampleRateHz, _ ->
            runOnUiThread {
                binding.tvImuStatus.text = statusText
                binding.tvImuRate.text = if (sampleRateHz > 0) String.format("%.1f Hz", sampleRateHz) else "Rate: -- Hz"

                if (!isReady) {
                    binding.tvAiStatus.text = "AI: Waiting for IMU"
                }
            }

            if (isReady && onnxInferenceHelper.isInitialized()) {
                val currentSampleCount = imuHelper.imuBuffer.getTotalSampleCount()
                if (currentSampleCount > lastProcessedSampleCount) {
                    val countDiff = if (lastProcessedSampleCount < 0) 1 else (currentSampleCount - lastProcessedSampleCount)
                    lastProcessedSampleCount = currentSampleCount

                    // Process 10 Hz IMU propagation & trigger AI inference on background worker
                    processImuSampleAndInference(countDiff.toInt())
                }
            }
        }
    }

    private fun processImuSampleAndInference(newSampleCount: Int) {
        navigationExecutor.execute {
            try {
                val window = imuHelper.imuBuffer.getWindow()
                if (window.isEmpty()) return@execute

                val latestSample = window.last()
                val nowSec = SystemClock.elapsedRealtime() / 1000.0

                // 1. IMU Propagation & NHC at 10 Hz
                if (esEkf.isInitialized()) {
                    val accel = doubleArrayOf(
                        latestSample[0].toDouble(),
                        latestSample[1].toDouble(),
                        latestSample[2].toDouble()
                    )
                    val gyro = doubleArrayOf(
                        latestSample[3].toDouble(),
                        latestSample[4].toDouble(),
                        latestSample[5].toDouble()
                    )

                    // Propagate dt = 0.1s for each new synchronized sample
                    for (i in 0 until newSampleCount) {
                        esEkf.predict(0.1, accel, gyro)
                        esEkf.updateNhc()
                    }

                    // Store attitude snapshot for relative attitude history
                    attitudeHistory.add(Pair(nowSec, esEkf.getQuat()))
                    if (attitudeHistory.size > 100) {
                        attitudeHistory.removeAt(0)
                    }
                }

                // 2. Trigger AI Inference if buffer ready
                if (window.size == 50 && isInferenceRunning.compareAndSet(false, true)) {
                    try {
                        val res = onnxInferenceHelper.runInference(window)
                        val nowMs = SystemClock.elapsedRealtime()

                        inferenceCount++
                        if (lastAiRateCalcTimeMs == 0L) {
                            lastAiRateCalcTimeMs = nowMs
                        } else {
                            val elapsedMs = nowMs - lastAiRateCalcTimeMs
                            if (elapsedMs >= 1000L) {
                                currentAiRateHz = (inferenceCount.toDouble() * 1000.0) / elapsedMs.toDouble()
                                inferenceCount = 0
                                lastAiRateCalcTimeMs = nowMs
                            }
                        }

                        // Apply AI Velocity & Relative Attitude updates to ES-EKF
                        if (esEkf.isInitialized()) {
                            esEkf.updateVelocity(res.speedMs.toDouble())

                            // Find best qStart for 5-second relative attitude window
                            val targetStartT = nowSec - 5.0
                            if (attitudeHistory.isNotEmpty()) {
                                var bestQStart = attitudeHistory[0].second
                                var minDt = kotlin.math.abs(attitudeHistory[0].first - targetStartT)
                                for (pair in attitudeHistory) {
                                    val diffT = kotlin.math.abs(pair.first - targetStartT)
                                    if (diffT < minDt) {
                                        minDt = diffT
                                        bestQStart = pair.second
                                    }
                                }
                                val qRelNet = res.quaternion.map { it.toDouble() }.toDoubleArray()
                                esEkf.updateRelativeAttitude(qRelNet, bestQStart)
                            }
                        }

                        // Periodic diagnostic logging ~once per second
                        if (nowMs - lastEkfLogTimeMs >= 1000L) {
                            lastEkfLogTimeMs = nowMs
                            val posEnu = if (esEkf.isInitialized()) esEkf.getPosEnu() else doubleArrayOf(0.0, 0.0, 0.0)
                            val velEnu = if (esEkf.isInitialized()) esEkf.getVelEnu() else doubleArrayOf(0.0, 0.0, 0.0)

                            if (gnssOutageManager.isOutageActive()) {
                                val elapsedSec = (nowMs - gnssOutageManager.outageStartTimeMs) / 1000.0
                                val (lat, lon, _) = if (esEkf.isInitialized()) esEkf.getLatLonAlt() else Triple(0.0, 0.0, 0.0)
                                Log.d(
                                    "MainActivity",
                                    String.format(
                                        "GNSS OUTAGE ACTIVE (%.1fs) | Lat: %.6f, Lon: %.6f | vel ENU: [%.2f, %.2f, %.2f] | speed: %.2f m/s | skipped updates: %d",
                                        elapsedSec, lat, lon, velEnu[0], velEnu[1], velEnu[2],
                                        if (esEkf.isInitialized()) esEkf.getSpeedMs() else 0.0,
                                        gnssOutageManager.skippedUpdateCount
                                    )
                                )
                            } else {
                                Log.d(
                                    "MainActivity",
                                    String.format(
                                        "ES-EKF active | pos ENU: [%.2f, %.2f, %.2f] | vel ENU: [%.2f, %.2f, %.2f] | speed: %.2f m/s | heading: %.1f° | GNSS: %s | AI speed: %.2f m/s | rate: %.1f Hz",
                                        posEnu[0], posEnu[1], posEnu[2],
                                        velEnu[0], velEnu[1], velEnu[2],
                                        if (esEkf.isInitialized()) esEkf.getSpeedMs() else 0.0,
                                        if (esEkf.isInitialized()) esEkf.getHeadingDeg() else 0.0,
                                        gnssOutageManager.currentState.name,
                                        res.speedMs,
                                        currentAiRateHz
                                    )
                                )
                            }
                        }

                        // Update UI on main thread
                        runOnUiThread {
                            binding.tvAiStatus.text = "AI: READY"
                            binding.tvAiVelocity.text = String.format("Velocity: %.2f m/s", if (esEkf.isInitialized()) esEkf.getSpeedMs() else res.speedMs.toDouble())
                            binding.tvEkfHeading.text = String.format("Heading: %.0f°", if (esEkf.isInitialized()) esEkf.getHeadingDeg() else 0.0)
                            binding.tvAiAttitude.text = String.format(
                                "Attitude: [%.3f, %.3f, %.3f, %.3f]",
                                res.quaternion[0], res.quaternion[1], res.quaternion[2], res.quaternion[3]
                            )
                            binding.tvAiRate.text = if (currentAiRateHz > 0) {
                                String.format("Inference rate: %.1f Hz (%d ms)", currentAiRateHz, res.latencyMs)
                            } else {
                                String.format("Inference latency: %d ms", res.latencyMs)
                            }

                            if (esEkf.isInitialized()) {
                                updateMapVehicleMarker()
                            }
                        }

                    } finally {
                        isInferenceRunning.set(false)
                    }
                }

            } catch (e: Exception) {
                Log.e("MainActivity", "Error in navigation loop: ${e.message}", e)
            }
        }
    }

    private fun updateMapVehicleMarker() {
        if (!esEkf.isInitialized()) return
        val (lat, lon, _) = esEkf.getLatLonAlt()
        val ekfGeoPoint = GeoPoint(lat, lon)
        currentGeoPoint = ekfGeoPoint

        if (currentLocationMarker == null) {
            currentLocationMarker = Marker(binding.mapView).apply {
                title = "ES-EKF Vehicle Position"
                setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_BOTTOM)
                position = ekfGeoPoint
            }
            binding.mapView.overlays.add(currentLocationMarker)
        } else {
            currentLocationMarker?.position = ekfGeoPoint
        }

        if (!isMapCenteredOnFirstFix) {
            binding.mapView.controller.animateTo(ekfGeoPoint)
            binding.mapView.controller.setZoom(16.5)
            isMapCenteredOnFirstFix = true
        }

        binding.mapView.invalidate()
    }

    private fun setupMapView() {
        binding.mapView.apply {
            setTileSource(TileSourceFactory.MAPNIK)
            setMultiTouchControls(true)
            controller.setZoom(15.0)
            controller.setCenter(GeoPoint(20.5937, 78.9629))
        }
    }

    private fun setupListeners() {
        binding.btnStartNavigation.setOnClickListener {
            performNavigationSearch()
        }

        binding.etDestination.setOnEditorActionListener { _, actionId, _ ->
            if (actionId == EditorInfo.IME_ACTION_SEARCH) {
                performNavigationSearch()
                true
            } else {
                false
            }
        }

        binding.btnToggleGnssOutage.setOnClickListener {
            if (gnssOutageManager.isOutageActive()) {
                // Restore GNSS
                gnssOutageManager.startRecovery()
                Log.d("MainActivity", "GNSS RECOVERY START")
                binding.btnToggleGnssOutage.text = getString(R.string.btn_simulate_gnss_outage)
                binding.btnToggleGnssOutage.setBackgroundColor(Color.parseColor("#DC2626"))
                binding.tvGpsStatus.text = getString(R.string.gps_status_recovering)
                binding.tvGpsStatus.setTextColor(Color.parseColor("#F59E0B"))
            } else {
                // Start Outage
                gnssOutageManager.startOutage(SystemClock.elapsedRealtime())
                Log.d("MainActivity", "GNSS OUTAGE START")
                binding.btnToggleGnssOutage.text = getString(R.string.btn_restore_gnss)
                binding.btnToggleGnssOutage.setBackgroundColor(Color.parseColor("#16A34A"))
                binding.tvGpsStatus.text = getString(R.string.gps_status_outage)
                binding.tvGpsStatus.setTextColor(Color.parseColor("#EF4444"))
            }
        }
    }

    private fun setupLocationHelper() {
        locationHelper = LocationHelper(
            context = this,
            onLocationUpdated = { location ->
                handleLocationUpdate(location)
            },
            onStatusChanged = { statusText ->
                if (!gnssOutageManager.isOutageActive() && gnssOutageManager.currentState != GnssState.RECOVERING) {
                    binding.tvGpsStatus.text = statusText
                }
            }
        )
    }

    private fun handleLocationUpdate(location: Location) {
        val nowSec = SystemClock.elapsedRealtime() / 1000.0

        navigationExecutor.execute {
            if (!esEkf.isInitialized()) {
                esEkf.initialize(location.latitude, location.longitude, location.altitude, nowSec)
                Log.d("MainActivity", "ES-EKF Initialized at Lat: ${location.latitude}, Lon: ${location.longitude}")
                Log.d("MainActivity", "GNSS FIX")
                runOnUiThread {
                    binding.tvGpsStatus.text = getString(R.string.gps_status_fix)
                    binding.tvGpsStatus.setTextColor(ContextCompat.getColor(this, R.color.accent_green))
                    binding.tvNavigationMode.text = "Navigation: ES-EKF ACTIVE"
                    updateMapVehicleMarker()
                }
                return@execute
            }

            if (!gnssOutageManager.gnssUpdatesEnabled.get()) {
                gnssOutageManager.onGnssUpdateSkipped()
                Log.d("MainActivity", "GNSS UPDATE SKIPPED")
                return@execute
            }

            val posEnuMeas = EsEkfMath.latLonToEnu(
                location.latitude,
                location.longitude,
                esEkf.getLatLonAlt().first,
                esEkf.getLatLonAlt().second
            )
            esEkf.updateGnssPosition(posEnuMeas)

            val isRecoveryUpdate = (gnssOutageManager.currentState == GnssState.RECOVERING)
            gnssOutageManager.onGnssUpdateApplied()

            if (isRecoveryUpdate) {
                Log.d("MainActivity", "GNSS UPDATE APPLIED")
                Log.d("MainActivity", "GNSS RECOVERED")
            }

            runOnUiThread {
                when (gnssOutageManager.currentState) {
                    GnssState.OUTAGE -> {
                        binding.tvGpsStatus.text = getString(R.string.gps_status_outage)
                        binding.tvGpsStatus.setTextColor(Color.parseColor("#EF4444"))
                    }
                    GnssState.RECOVERING -> {
                        binding.tvGpsStatus.text = getString(R.string.gps_status_recovering)
                        binding.tvGpsStatus.setTextColor(Color.parseColor("#F59E0B"))
                    }
                    GnssState.RECOVERED -> {
                        binding.tvGpsStatus.text = getString(R.string.gps_status_recovered)
                        binding.tvGpsStatus.setTextColor(ContextCompat.getColor(this, R.color.accent_green))
                    }
                    else -> {
                        binding.tvGpsStatus.text = getString(R.string.gps_status_fix)
                        binding.tvGpsStatus.setTextColor(ContextCompat.getColor(this, R.color.accent_green))
                    }
                }
                binding.tvNavigationMode.text = "Navigation: ES-EKF ACTIVE"
                updateMapVehicleMarker()
            }
        }
    }

    private fun performNavigationSearch() {
        hideKeyboard()

        val destText = binding.etDestination.text?.toString()?.trim()
        if (destText.isNullOrEmpty()) {
            Toast.makeText(this, "Please enter a destination", Toast.LENGTH_SHORT).show()
            return
        }

        val startPoint = currentGeoPoint
        if (startPoint == null) {
            Toast.makeText(this, "Acquiring current GPS location... Please wait.", Toast.LENGTH_LONG).show()
            return
        }

        binding.progressBar.visibility = View.VISIBLE
        binding.btnStartNavigation.isEnabled = false

        lifecycleScope.launch {
            try {
                val geocoded = GeocodingService.searchPlace(destText)
                if (geocoded == null) {
                    binding.progressBar.visibility = View.GONE
                    binding.btnStartNavigation.isEnabled = true
                    Toast.makeText(this@MainActivity, "Destination not found. Try a different address.", Toast.LENGTH_LONG).show()
                    return@launch
                }

                val destPoint = GeoPoint(geocoded.latitude, geocoded.longitude)
                destinationGeoPoint = destPoint

                if (destinationMarker == null) {
                    destinationMarker = Marker(binding.mapView).apply {
                        title = geocoded.displayName
                        setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_BOTTOM)
                        position = destPoint
                    }
                    binding.mapView.overlays.add(destinationMarker)
                } else {
                    destinationMarker?.title = geocoded.displayName
                    destinationMarker?.position = destPoint
                }

                val routeResult = RoutingService.calculateRoute(startPoint, destPoint)
                binding.progressBar.visibility = View.GONE
                binding.btnStartNavigation.isEnabled = true

                if (routeResult == null || routeResult.points.isEmpty()) {
                    Toast.makeText(this@MainActivity, "Unable to calculate route. Check internet connection.", Toast.LENGTH_LONG).show()
                    return@launch
                }

                if (routePolyline != null) {
                    binding.mapView.overlays.remove(routePolyline)
                }

                val polyline = Polyline().apply {
                    setPoints(routeResult.points)
                    outlinePaint.color = Color.parseColor("#2563EB")
                    outlinePaint.strokeWidth = 12.0f
                }
                routePolyline = polyline
                binding.mapView.overlays.add(polyline)

                binding.tvRouteDistance.text = "Distance: ${routeResult.getFormattedDistance()}"
                binding.tvRouteEta.text = "ETA: ${routeResult.getFormattedDuration()}"

                val boundingBox = BoundingBox.fromGeoPoints(routeResult.points)
                binding.mapView.zoomToBoundingBox(boundingBox, true, 80)
                binding.mapView.invalidate()

                Toast.makeText(this@MainActivity, "Route loaded successfully", Toast.LENGTH_SHORT).show()

            } catch (e: Exception) {
                e.printStackTrace()
                binding.progressBar.visibility = View.GONE
                binding.btnStartNavigation.isEnabled = true
                Toast.makeText(this@MainActivity, "Navigation error: ${e.localizedMessage}", Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun checkLocationPermissions() {
        val fineLocation = ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
        val coarseLocation = ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION)

        if (fineLocation != PackageManager.PERMISSION_GRANTED || coarseLocation != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION),
                LOCATION_PERMISSION_REQUEST_CODE
            )
        } else {
            locationHelper.startLocationUpdates()
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == LOCATION_PERMISSION_REQUEST_CODE) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                locationHelper.startLocationUpdates()
            } else {
                binding.tvGpsStatus.text = "Permission Denied"
                Toast.makeText(this, "Location permission is required for navigation", Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun hideKeyboard() {
        val view = currentFocus ?: binding.root
        val imm = getSystemService(Context.INPUT_METHOD_SERVICE) as InputMethodManager
        imm.hideSoftInputFromWindow(view.windowToken, 0)
    }

    override fun onResume() {
        super.onResume()
        binding.mapView.onResume()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED) {
            locationHelper.startLocationUpdates()
        }
        imuHelper.start()
    }

    override fun onPause() {
        super.onPause()
        binding.mapView.onPause()
        locationHelper.stopLocationUpdates()
        imuHelper.stop()
    }

    override fun onDestroy() {
        super.onDestroy()
        navigationExecutor.shutdown()
        if (::onnxInferenceHelper.isInitialized) {
            onnxInferenceHelper.close()
        }
    }
}
