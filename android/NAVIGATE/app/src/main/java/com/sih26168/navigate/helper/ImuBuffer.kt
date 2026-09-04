package com.sih26168.navigate.helper

import kotlin.math.max

/**
 * Pure Kotlin helper for buffering raw Accelerometer and Gyroscope sensor samples,
 * interpolating/synchronizing them to a 10 Hz (100 ms) output grid using sensor timestamps,
 * and maintaining a rolling window of the most recent 50 samples ([50][6]).
 *
 * Sensor Coordinate System: Raw Android Sensor Frame.
 * Units:
 * - Accelerometer: m/s^2 (Android SensorEvent values)
 * - Gyroscope: rad/s (Android SensorEvent values)
 * Format of each synchronized sample:
 * [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
 */
class ImuBuffer(
    val targetIntervalNs: Long = 100_000_000L, // 100 ms = 10 Hz
    val windowCapacity: Int = 50
) {
    data class SensorSample(
        val timestampNs: Long,
        val values: FloatArray
    ) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is SensorSample) return false
            return timestampNs == other.timestampNs && values.contentEquals(other.values)
        }

        override fun hashCode(): Int {
            var result = timestampNs.hashCode()
            result = 31 * result + values.contentHashCode()
            return result
        }
    }

    private val accelQueue = ArrayDeque<SensorSample>()
    private val gyroQueue = ArrayDeque<SensorSample>()
    private val rollingWindow = ArrayDeque<FloatArray>()

    private var nextTargetTimestampNs: Long = -1L
    private var isInitialized = false

    // Diagnostic metrics
    private var generatedSampleCount = 0
    private var totalSynchronizedSamples = 0L
    private var lastRateCalcTimeNs = -1L
    private var currentRateHz = 0.0

    /**
     * Validates raw sensor values.
     * Rejects null, wrong size (!= 3), NaN, or Infinite values.
     */
    fun isValidSample(values: FloatArray?): Boolean {
        if (values == null || values.size != 3) return false
        for (v in values) {
            if (v.isNaN() || v.isInfinite()) return false
        }
        return true
    }

    /**
     * Add accelerometer sample [x, y, z] with timestamp in nanoseconds.
     */
    @Synchronized
    fun addAccel(timestampNs: Long, values: FloatArray): Boolean {
        if (!isValidSample(values)) return false
        accelQueue.addLast(SensorSample(timestampNs, values.clone()))
        processSynchronization()
        return true
    }

    /**
     * Add gyroscope sample [x, y, z] with timestamp in nanoseconds.
     */
    @Synchronized
    fun addGyro(timestampNs: Long, values: FloatArray): Boolean {
        if (!isValidSample(values)) return false
        gyroQueue.addLast(SensorSample(timestampNs, values.clone()))
        processSynchronization()
        return true
    }

    /**
     * Core synchronization loop.
     * Aligns accel & gyro data onto 10 Hz target grid using monotonic timestamps.
     */
    private fun processSynchronization() {
        if (accelQueue.isEmpty() || gyroQueue.isEmpty()) return

        if (!isInitialized) {
            val startNs = maxOf(accelQueue.first().timestampNs, gyroQueue.first().timestampNs)
            nextTargetTimestampNs = startNs
            isInitialized = true
        }

        while (isInitialized) {
            val latestAccelNs = accelQueue.last().timestampNs
            val latestGyroNs = gyroQueue.last().timestampNs

            // We can only produce a target sample if both streams reach or exceed nextTargetTimestampNs
            if (latestAccelNs < nextTargetTimestampNs || latestGyroNs < nextTargetTimestampNs) {
                break
            }

            val accelInterp = interpolate(accelQueue, nextTargetTimestampNs)
            val gyroInterp = interpolate(gyroQueue, nextTargetTimestampNs)

            val sample6d = floatArrayOf(
                accelInterp[0], accelInterp[1], accelInterp[2],
                gyroInterp[0], gyroInterp[1], gyroInterp[2]
            )

            // Add to rolling window
            rollingWindow.addLast(sample6d)
            if (rollingWindow.size > windowCapacity) {
                rollingWindow.removeFirst()
            }

            // Rate tracking
            generatedSampleCount++
            totalSynchronizedSamples++
            if (lastRateCalcTimeNs < 0) {
                lastRateCalcTimeNs = nextTargetTimestampNs
            } else {
                val elapsedNs = nextTargetTimestampNs - lastRateCalcTimeNs
                if (elapsedNs >= 1_000_000_000L) {
                    currentRateHz = (generatedSampleCount.toDouble() * 1_000_000_000.0) / elapsedNs.toDouble()
                    generatedSampleCount = 0
                    lastRateCalcTimeNs = nextTargetTimestampNs
                }
            }

            nextTargetTimestampNs += targetIntervalNs

            // Prune old samples from queues (keep 1 sample before nextTargetTimestampNs)
            pruneQueue(accelQueue, nextTargetTimestampNs - targetIntervalNs)
            pruneQueue(gyroQueue, nextTargetTimestampNs - targetIntervalNs)
        }
    }

    private fun interpolate(queue: ArrayDeque<SensorSample>, targetNs: Long): FloatArray {
        if (queue.isEmpty()) return floatArrayOf(0f, 0f, 0f)
        if (targetNs <= queue.first().timestampNs) return queue.first().values.clone()
        if (targetNs >= queue.last().timestampNs) return queue.last().values.clone()

        for (i in 0 until queue.size - 1) {
            val s1 = queue[i]
            val s2 = queue[i + 1]
            if (s1.timestampNs <= targetNs && s2.timestampNs >= targetNs) {
                val dt = (s2.timestampNs - s1.timestampNs).toDouble()
                if (dt <= 0.0) return s1.values.clone()
                val alpha = ((targetNs - s1.timestampNs).toDouble() / dt).toFloat()
                return floatArrayOf(
                    s1.values[0] + alpha * (s2.values[0] - s1.values[0]),
                    s1.values[1] + alpha * (s2.values[1] - s1.values[1]),
                    s1.values[2] + alpha * (s2.values[2] - s1.values[2])
                )
            }
        }
        return queue.last().values.clone()
    }

    private fun pruneQueue(queue: ArrayDeque<SensorSample>, thresholdNs: Long) {
        while (queue.size > 1 && queue[1].timestampNs <= thresholdNs) {
            queue.removeFirst()
        }
    }

    @Synchronized
    fun getWindow(): List<FloatArray> {
        return rollingWindow.map { it.clone() }
    }

    @Synchronized
    fun getWindowSize(): Int {
        return rollingWindow.size
    }

    @Synchronized
    fun isReady(): Boolean {
        return rollingWindow.size == windowCapacity
    }

    @Synchronized
    fun getSampleRateHz(): Double {
        return currentRateHz
    }

    @Synchronized
    fun getTotalSampleCount(): Long {
        return totalSynchronizedSamples
    }

    @Synchronized
    fun clear() {
        accelQueue.clear()
        gyroQueue.clear()
        rollingWindow.clear()
        nextTargetTimestampNs = -1L
        isInitialized = false
        generatedSampleCount = 0
        totalSynchronizedSamples = 0L
        lastRateCalcTimeNs = -1L
        currentRateHz = 0.0
    }
}
