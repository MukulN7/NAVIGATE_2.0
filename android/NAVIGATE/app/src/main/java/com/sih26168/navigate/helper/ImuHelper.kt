package com.sih26168.navigate.helper

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.SystemClock
import android.util.Log

class ImuHelper(
    context: Context,
    private val onImuStateUpdated: (statusText: String, isReady: Boolean, sampleRateHz: Double, windowCount: Int) -> Unit
) : SensorEventListener {

    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val accelSensor: Sensor? = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val gyroSensor: Sensor? = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

    val imuBuffer = ImuBuffer()

    private var isListening = false
    private var lastDiagnosticLogTimeMs = 0L

    val isSensorAvailable: Boolean
        get() = accelSensor != null && gyroSensor != null

    fun start() {
        if (!isSensorAvailable) {
            onImuStateUpdated("IMU: Unavailable", false, 0.0, 0)
            Log.w("ImuHelper", "IMU sensors unavailable on device (Accel present: ${accelSensor != null}, Gyro present: ${gyroSensor != null})")
            return
        }

        if (isListening) return

        imuBuffer.clear()

        val accelRegistered = sensorManager.registerListener(
            this,
            accelSensor,
            SensorManager.SENSOR_DELAY_GAME
        )
        val gyroRegistered = sensorManager.registerListener(
            this,
            gyroSensor,
            SensorManager.SENSOR_DELAY_GAME
        )

        if (accelRegistered && gyroRegistered) {
            isListening = true
            onImuStateUpdated("IMU: Initializing", false, 0.0, 0)
            Log.d("ImuHelper", "IMU sensors registered successfully (Accelerometer + Gyroscope)")
        } else {
            sensorManager.unregisterListener(this)
            isListening = false
            onImuStateUpdated("IMU: Unavailable", false, 0.0, 0)
            Log.e("ImuHelper", "Failed to register IMU sensor listeners")
        }
    }

    fun stop() {
        if (isListening) {
            sensorManager.unregisterListener(this)
            isListening = false
            Log.d("ImuHelper", "IMU sensors unregistered")
        }
    }

    override fun onSensorChanged(event: SensorEvent?) {
        if (event == null || !isListening) return

        val timestampNs = event.timestamp
        val values = event.values

        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                imuBuffer.addAccel(timestampNs, values)
            }
            Sensor.TYPE_GYROSCOPE -> {
                imuBuffer.addGyro(timestampNs, values)
            }
        }

        val windowSize = imuBuffer.getWindowSize()
        val isReady = imuBuffer.isReady()
        val sampleRate = imuBuffer.getSampleRateHz()

        val statusText = when {
            isReady -> "IMU: Ready (50/50)"
            windowSize > 0 -> "IMU: Collecting ($windowSize/50)"
            else -> "IMU: Initializing"
        }

        onImuStateUpdated(statusText, isReady, sampleRate, windowSize)

        // Periodic diagnostic log once per second
        val nowMs = SystemClock.elapsedRealtime()
        if (nowMs - lastDiagnosticLogTimeMs >= 1000L) {
            lastDiagnosticLogTimeMs = nowMs
            val rateFormatted = String.format("%.1f", sampleRate)
            Log.d("ImuHelper", "IMU rate: $rateFormatted Hz, window: $windowSize/50")
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // No action needed for accuracy changes
    }
}
