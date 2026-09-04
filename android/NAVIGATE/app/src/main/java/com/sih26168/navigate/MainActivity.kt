package com.sih26168.navigate

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Color
import android.location.Location
import android.os.Bundle
import android.view.View
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputMethodManager
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import android.os.SystemClock
import android.util.Log
import com.sih26168.navigate.databinding.ActivityMainBinding
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

    private val inferenceExecutor = Executors.newSingleThreadExecutor()
    private val isInferenceRunning = AtomicBoolean(false)

    private var currentGeoPoint: GeoPoint? = null
    private var destinationGeoPoint: GeoPoint? = null

    private var currentLocationMarker: Marker? = null
    private var destinationMarker: Marker? = null
    private var routePolyline: Polyline? = null

    private var isMapCenteredOnFirstFix = false

    // AI Inference Diagnostics
    private var lastProcessedSampleCount = -1L
    private var inferenceCount = 0
    private var lastAiRateCalcTimeMs = 0L
    private var currentAiRateHz = 0.0
    private var lastAiLogTimeMs = 0L

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
        window.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON or android.view.WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED or android.view.WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON)

        // Initialize osmdroid configuration
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
                    lastProcessedSampleCount = currentSampleCount
                    triggerAiInference()
                }
            }
        }
    }

    private fun triggerAiInference() {
        if (!isInferenceRunning.compareAndSet(false, true)) {
            // Drop redundant trigger if previous inference is still executing
            return
        }

        val window = imuHelper.imuBuffer.getWindow()
        if (window.size != 50) {
            isInferenceRunning.set(false)
            return
        }

        inferenceExecutor.execute {
            try {
                val res = onnxInferenceHelper.runInference(window)
                val nowMs = SystemClock.elapsedRealtime()

                // Calculate AI Inference Rate (Hz)
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

                // Periodic diagnostic logging ~once per second
                if (nowMs - lastAiLogTimeMs >= 1000L) {
                    lastAiLogTimeMs = nowMs
                    Log.d(
                        "MainActivity",
                        String.format(
                            "AI velocity: %.2f m/s | AI quaternion: [%.3f, %.3f, %.3f, %.3f] (norm=%.4f) | AI inference: %.1f Hz | latency: %d ms",
                            res.speedMs,
                            res.quaternion[0], res.quaternion[1], res.quaternion[2], res.quaternion[3],
                            res.quatNorm,
                            currentAiRateHz,
                            res.latencyMs
                        )
                    )
                }

                // Update UI elements on main thread
                runOnUiThread {
                    binding.tvAiStatus.text = "AI: Inference ready"
                    binding.tvAiVelocity.text = String.format("Velocity: %.2f m/s", res.speedMs)
                    binding.tvAiAttitude.text = String.format(
                        "Attitude: [%.3f, %.3f, %.3f, %.3f]",
                        res.quaternion[0], res.quaternion[1], res.quaternion[2], res.quaternion[3]
                    )
                    binding.tvAiRate.text = if (currentAiRateHz > 0) {
                        String.format("Inference rate: %.1f Hz (%d ms)", currentAiRateHz, res.latencyMs)
                    } else {
                        String.format("Inference latency: %d ms", res.latencyMs)
                    }
                }
            } catch (e: Exception) {
                Log.e("MainActivity", "Error during AI inference: ${e.message}", e)
            } finally {
                isInferenceRunning.set(false)
            }
        }
    }

    private fun setupMapView() {
        binding.mapView.apply {
            setTileSource(TileSourceFactory.MAPNIK)
            setMultiTouchControls(true)
            controller.setZoom(15.0)
            // Default center fallback (India center)
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
    }

    private fun setupLocationHelper() {
        locationHelper = LocationHelper(
            context = this,
            onLocationUpdated = { location ->
                handleLocationUpdate(location)
            },
            onStatusChanged = { statusText ->
                binding.tvGpsStatus.text = statusText
            }
        )
    }

    private fun handleLocationUpdate(location: Location) {
        val geoPoint = GeoPoint(location.latitude, location.longitude)
        currentGeoPoint = geoPoint

        if (currentLocationMarker == null) {
            currentLocationMarker = Marker(binding.mapView).apply {
                title = "Current Location"
                setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_BOTTOM)
                position = geoPoint
            }
            binding.mapView.overlays.add(currentLocationMarker)
        } else {
            currentLocationMarker?.position = geoPoint
        }

        if (!isMapCenteredOnFirstFix) {
            binding.mapView.controller.animateTo(geoPoint)
            binding.mapView.controller.setZoom(16.5)
            isMapCenteredOnFirstFix = true
        }

        binding.mapView.invalidate()
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
                // 1. Geocode Destination
                val geocoded = GeocodingService.searchPlace(destText)
                if (geocoded == null) {
                    binding.progressBar.visibility = View.GONE
                    binding.btnStartNavigation.isEnabled = true
                    Toast.makeText(this@MainActivity, "Destination not found. Try a different address.", Toast.LENGTH_LONG).show()
                    return@launch
                }

                val destPoint = GeoPoint(geocoded.latitude, geocoded.longitude)
                destinationGeoPoint = destPoint

                // Update Destination Marker
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

                // 2. Fetch OSRM Route
                val routeResult = RoutingService.calculateRoute(startPoint, destPoint)
                binding.progressBar.visibility = View.GONE
                binding.btnStartNavigation.isEnabled = true

                if (routeResult == null || routeResult.points.isEmpty()) {
                    Toast.makeText(this@MainActivity, "Unable to calculate route. Check internet connection.", Toast.LENGTH_LONG).show()
                    return@launch
                }

                // Draw Polyline
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

                // Update UI Status Bar
                binding.tvRouteDistance.text = "Distance: ${routeResult.getFormattedDistance()}"
                binding.tvRouteEta.text = "ETA: ${routeResult.getFormattedDuration()}"

                // Zoom Map to Fit Route
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
        inferenceExecutor.shutdown()
        if (::onnxInferenceHelper.isInitialized) {
            onnxInferenceHelper.close()
        }
    }
}
