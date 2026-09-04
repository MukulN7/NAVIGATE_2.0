package com.sih26168.navigate.helper

import java.util.concurrent.atomic.AtomicBoolean

enum class GnssState {
    FIX,
    OUTAGE,
    RECOVERING,
    RECOVERED
}

class GnssOutageManager {

    val gnssUpdatesEnabled = AtomicBoolean(true)

    var currentState: GnssState = GnssState.FIX
        private set

    var outageStartTimeMs: Long = 0L
        private set

    var skippedUpdateCount: Int = 0
        private set

    fun startOutage(nowMs: Long) {
        gnssUpdatesEnabled.set(false)
        currentState = GnssState.OUTAGE
        outageStartTimeMs = nowMs
        skippedUpdateCount = 0
    }

    fun startRecovery() {
        gnssUpdatesEnabled.set(true)
        currentState = GnssState.RECOVERING
    }

    fun onGnssUpdateSkipped() {
        skippedUpdateCount++
    }

    fun onGnssUpdateApplied() {
        if (currentState == GnssState.RECOVERING) {
            currentState = GnssState.RECOVERED
        } else if (currentState != GnssState.RECOVERED) {
            currentState = GnssState.FIX
        }
    }

    fun isOutageActive(): Boolean = !gnssUpdatesEnabled.get()
}
