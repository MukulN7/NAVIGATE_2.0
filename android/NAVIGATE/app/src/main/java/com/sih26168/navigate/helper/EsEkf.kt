package com.sih26168.navigate.helper

import com.sih26168.navigate.helper.EsEkfMath.DEG2RAD
import com.sih26168.navigate.helper.EsEkfMath.STANDARD_GRAVITY
import com.sih26168.navigate.helper.EsEkfMath.enuToLatLon
import com.sih26168.navigate.helper.EsEkfMath.eye
import com.sih26168.navigate.helper.EsEkfMath.headingDegToQuat
import com.sih26168.navigate.helper.EsEkfMath.latLonToEnu
import com.sih26168.navigate.helper.EsEkfMath.matAdd
import com.sih26168.navigate.helper.EsEkfMath.matInvertSmall
import com.sih26168.navigate.helper.EsEkfMath.matMul
import com.sih26168.navigate.helper.EsEkfMath.matMulVec
import com.sih26168.navigate.helper.EsEkfMath.matSub
import com.sih26168.navigate.helper.EsEkfMath.matTranspose
import com.sih26168.navigate.helper.EsEkfMath.quatInverse
import com.sih26168.navigate.helper.EsEkfMath.quatMultiply
import com.sih26168.navigate.helper.EsEkfMath.quatNormalize
import com.sih26168.navigate.helper.EsEkfMath.quatToHeadingDeg
import com.sih26168.navigate.helper.EsEkfMath.quatToRotmat
import com.sih26168.navigate.helper.EsEkfMath.quatToRotvec
import com.sih26168.navigate.helper.EsEkfMath.rotvecToQuat
import com.sih26168.navigate.helper.EsEkfMath.skewSymmetric
import kotlin.math.abs
import kotlin.math.sqrt

data class EsEkfState(
    var posEnu: DoubleArray = DoubleArray(3),
    var velEnu: DoubleArray = DoubleArray(3),
    var quat: DoubleArray = doubleArrayOf(1.0, 0.0, 0.0, 0.0),
    var cov: Array<DoubleArray> = eye(9),
    var timestamp: Double = 0.0,
    var refLatDeg: Double = 0.0,
    var refLonDeg: Double = 0.0,
    var refAltM: Double = 0.0,
    var isInitialized: Boolean = false
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as EsEkfState

        if (!posEnu.contentEquals(other.posEnu)) return false
        if (!velEnu.contentEquals(other.velEnu)) return false
        if (!quat.contentEquals(other.quat)) return false
        if (timestamp != other.timestamp) return false
        if (isInitialized != other.isInitialized) return false

        return true
    }

    override fun hashCode(): Int {
        var result = posEnu.contentHashCode()
        result = 31 * result + velEnu.contentHashCode()
        result = 31 * result + quat.contentHashCode()
        result = 31 * result + timestamp.hashCode()
        result = 31 * result + isInitialized.hashCode()
        return result
    }
}

class EsEkf(
    var accelNoiseStd: Double = 0.2,
    var gyroNoiseStd: Double = 0.02
) {

    private val state = EsEkfState()

    fun isInitialized(): Boolean = state.isInitialized

    fun initialize(
        latDeg: Double,
        lonDeg: Double,
        altM: Double,
        timestampSec: Double,
        headingDeg: Double = 0.0,
        stdPosInit: Double = 1.0,
        stdVelInit: Double = 0.5,
        stdAttInitDeg: Double = 2.0
    ) {
        state.refLatDeg = latDeg
        state.refLonDeg = lonDeg
        state.refAltM = altM

        state.posEnu = DoubleArray(3) // [0.0, 0.0, 0.0]
        state.velEnu = DoubleArray(3) // [0.0, 0.0, 0.0]
        state.quat = headingDegToQuat(headingDeg)
        state.timestamp = timestampSec

        val cov = eye(9)
        val varPos = stdPosInit * stdPosInit
        val varVel = stdVelInit * stdVelInit
        val stdAttRad = stdAttInitDeg * DEG2RAD
        val varAtt = stdAttRad * stdAttRad

        for (i in 0..2) cov[i][i] = varPos
        for (i in 3..5) cov[i][i] = varVel
        for (i in 6..8) cov[i][i] = varAtt

        state.cov = cov
        state.isInitialized = true
    }

    /**
     * IMU Propagation step at 10 Hz.
     */
    fun predict(dt: Double, accelB: DoubleArray, gyroB: DoubleArray) {
        if (!state.isInitialized || dt <= 0.0) return
        if (accelB.any { it.isNaN() || it.isInfinite() } || gyroB.any { it.isNaN() || it.isInfinite() }) return

        // 1. Propagate nominal quaternion: q = q (x) delta_q(gyro * dt)
        val deltaQ = rotvecToQuat(doubleArrayOf(gyroB[0] * dt, gyroB[1] * dt, gyroB[2] * dt))
        state.quat = quatNormalize(quatMultiply(state.quat, deltaQ))

        // 2. Rotate specific force to navigation frame & compensate gravity
        val Rcurr = quatToRotmat(state.quat)
        val gN = doubleArrayOf(0.0, 0.0, -STANDARD_GRAVITY)
        val accelRot = matMulVec(Rcurr, accelB)
        val accelN = doubleArrayOf(accelRot[0] + gN[0], accelRot[1] + gN[1], accelRot[2] + gN[2])

        // 3. Propagate position and velocity
        for (i in 0..2) {
            state.posEnu[i] += state.velEnu[i] * dt + 0.5 * accelN[i] * dt * dt
            state.velEnu[i] += accelN[i] * dt
        }

        // 4. Error State Transition Matrix F (9x9)
        val F = eye(9)
        for (i in 0..2) {
            F[i][i + 3] = dt
        }
        val skewA = skewSymmetric(accelRot)
        for (r in 0..2) {
            for (c in 0..2) {
                F[r][c + 6] = -0.5 * skewA[r][c] * dt * dt
                F[r + 3][c + 6] = -skewA[r][c] * dt
            }
        }

        // 5. Discrete Process Noise Matrix Q (9x9)
        val Q = eye(9)
        for (i in 0..8) Q[i][i] = 0.0
        val varA = accelNoiseStd * accelNoiseStd * dt
        val varG = gyroNoiseStd * gyroNoiseStd * dt

        for (i in 0..2) {
            Q[i][i] = (1.0 / 3.0) * varA * dt * dt
            Q[i][i + 3] = 0.5 * varA * dt
            Q[i + 3][i] = 0.5 * varA * dt
            Q[i + 3][i + 3] = varA
            Q[i + 6][i + 6] = varG
        }

        // 6. Propagate Covariance: P = F * P * F^T + Q
        val FP = matMul(F, state.cov)
        val FPFt = matMul(FP, matTranspose(F))
        val newCov = matAdd(FPFt, Q)

        // Make symmetric
        for (r in 0..8) {
            for (c in 0..8) {
                val avg = 0.5 * (newCov[r][c] + newCov[c][r])
                newCov[r][c] = avg
                newCov[c][r] = avg
            }
        }
        state.cov = newCov
    }

    /**
     * Generic measurement update via error state injection.
     */
    private fun applyErrorStateUpdate(residual: DoubleArray, H: Array<DoubleArray>, Rcov: Array<DoubleArray>) {
        if (residual.any { it.isNaN() || it.isInfinite() }) return
        val P = state.cov

        // S = H * P * H^T + R (m x m)
        val HP = matMul(H, P)
        val HPHt = matMul(HP, matTranspose(H))
        val S = matAdd(HPHt, Rcov)

        val invS = try {
            matInvertSmall(S)
        } catch (e: Exception) {
            return
        }

        // K = P * H^T * invS (9 x m)
        val Ht = matTranspose(H)
        val PHt = matMul(P, Ht)
        val K = matMul(PHt, invS)

        // delta_x = K * residual (9D)
        val deltaX = matMulVec(K, residual)
        if (deltaX.any { it.isNaN() || it.isInfinite() }) return

        // Joseph form covariance update: P = (I - K*H) * P * (I - K*H)^T + K * R * K^T
        val KH = matMul(K, H)
        val IKH = matSub(eye(9), KH)
        val IKHP = matMul(IKH, P)
        val IKHP_IKHt = matMul(IKHP, matTranspose(IKH))

        val KR = matMul(K, Rcov)
        val KRKt = matMul(KR, matTranspose(K))
        val Pupdated = matAdd(IKHP_IKHt, KRKt)

        for (r in 0..8) {
            for (c in 0..8) {
                val avg = 0.5 * (Pupdated[r][c] + Pupdated[c][r])
                Pupdated[r][c] = avg
                Pupdated[c][r] = avg
            }
        }
        state.cov = Pupdated

        // Error injection
        for (i in 0..2) {
            state.posEnu[i] += deltaX[i]
            state.velEnu[i] += deltaX[i + 3]
        }
        val deltaQ = rotvecToQuat(doubleArrayOf(deltaX[6], deltaX[7], deltaX[8]))
        state.quat = quatNormalize(quatMultiply(deltaQ, state.quat))
    }

    /**
     * Fuses scalar forward speed estimate from VelocityModel V2 into body X axis.
     */
    fun updateVelocity(forwardSpeedMs: Double, covSpeed: Double = 0.25 * 0.25) {
        if (!state.isInitialized || forwardSpeedMs.isNaN() || forwardSpeedMs.isInfinite()) return

        val RbN = quatToRotmat(state.quat)
        val RnB = matTranspose(RbN)
        val vB = matMulVec(RnB, state.velEnu)

        val residualVal = forwardSpeedMs - vB[0]
        // Reject unreasonable velocity residual (> 100 m/s)
        if (abs(residualVal) > 100.0) return

        val residual = doubleArrayOf(residualVal)
        val H = Array(1) { DoubleArray(9) }
        for (c in 0..2) {
            H[0][c + 3] = RnB[0][c]
        }

        val vSkew = skewSymmetric(state.velEnu)
        val RnB_vSkew = matMul(RnB, vSkew)
        for (c in 0..2) {
            H[0][c + 6] = RnB_vSkew[0][c]
        }

        val Rcov = arrayOf(doubleArrayOf(covSpeed))
        applyErrorStateUpdate(residual, H, Rcov)
    }

    /**
     * Enforces Non-Holonomic Constraints (lateral & vertical body velocity = 0).
     */
    fun updateNhc(covLateral: Double = 0.05 * 0.05, covVertical: Double = 0.05 * 0.05) {
        if (!state.isInitialized) return

        val RbN = quatToRotmat(state.quat)
        val RnB = matTranspose(RbN)
        val vB = matMulVec(RnB, state.velEnu)

        val residual = doubleArrayOf(-vB[1], -vB[2])
        val H = Array(2) { DoubleArray(9) }
        for (r in 0..1) {
            for (c in 0..2) {
                H[r][c + 3] = RnB[r + 1][c]
            }
        }

        val vSkew = skewSymmetric(state.velEnu)
        val RnB_vSkew = matMul(RnB, vSkew)
        for (r in 0..1) {
            for (c in 0..2) {
                H[r][c + 6] = RnB_vSkew[r + 1][c]
            }
        }

        val Rcov = arrayOf(
            doubleArrayOf(covLateral, 0.0),
            doubleArrayOf(0.0, covVertical)
        )
        applyErrorStateUpdate(residual, H, Rcov)
    }

    /**
     * Relative attitude measurement update from AttitudeModel over a 5-second window.
     */
    fun updateRelativeAttitude(
        qRelNetwork: DoubleArray,
        qStart: DoubleArray,
        covAttRad: Double = (5.0 * DEG2RAD) * (5.0 * DEG2RAD)
    ) {
        if (!state.isInitialized) return
        if (qRelNetwork.any { it.isNaN() || it.isInfinite() } || qStart.any { it.isNaN() || it.isInfinite() }) return

        val qStartNorm = quatNormalize(qStart)
        val qCurr = state.quat
        val qRelEkf = quatMultiply(quatInverse(qStartNorm), qCurr)

        val qNetNorm = quatNormalize(qRelNetwork)
        var qErr = quatMultiply(quatInverse(qNetNorm), qRelEkf)

        // Antipodal check
        if (qErr[0] < 0.0) {
            qErr = doubleArrayOf(-qErr[0], -qErr[1], -qErr[2], -qErr[3])
        }

        val rotvecB = quatToRotvec(qErr)
        val Rcurr = quatToRotmat(qCurr)
        val rotvecN = matMulVec(Rcurr, rotvecB)

        val residual = doubleArrayOf(-rotvecN[0], -rotvecN[1], -rotvecN[2])
        val H = Array(3) { DoubleArray(9) }
        for (i in 0..2) {
            H[i][i + 6] = 1.0
        }

        val Rcov = Array(3) { r ->
            DoubleArray(3) { c ->
                if (r == c) covAttRad else 0.0
            }
        }
        applyErrorStateUpdate(residual, H, Rcov)
    }

    /**
     * Fuses ENU GNSS position measurement.
     */
    fun updateGnssPosition(posEnuMeas: DoubleArray, covPos: Double = 1.0 * 1.0) {
        if (!state.isInitialized) return
        if (posEnuMeas.any { it.isNaN() || it.isInfinite() }) return

        val residual = doubleArrayOf(
            posEnuMeas[0] - state.posEnu[0],
            posEnuMeas[1] - state.posEnu[1],
            posEnuMeas[2] - state.posEnu[2]
        )

        // Reject unreasonably large position jump (> 1000m)
        val distSquare = residual[0] * residual[0] + residual[1] * residual[1] + residual[2] * residual[2]
        if (distSquare > 1000.0 * 1000.0) return

        val H = Array(3) { DoubleArray(9) }
        for (i in 0..2) {
            H[i][i] = 1.0
        }

        val Rcov = Array(3) { r ->
            DoubleArray(3) { c ->
                if (r == c) covPos else 0.0
            }
        }
        applyErrorStateUpdate(residual, H, Rcov)
    }

    // ================================================================== //
    //  Getters
    // ================================================================== //

    fun getLatLonAlt(): Triple<Double, Double, Double> {
        val (lat, lon) = enuToLatLon(state.posEnu[0], state.posEnu[1], state.refLatDeg, state.refLonDeg)
        val alt = state.refAltM + state.posEnu[2]
        return Triple(lat, lon, alt)
    }

    fun getHeadingDeg(): Double {
        return quatToHeadingDeg(state.quat)
    }

    fun getSpeedMs(): Double {
        val vx = state.velEnu[0]
        val vy = state.velEnu[1]
        val vz = state.velEnu[2]
        return sqrt(vx * vx + vy * vy + vz * vz)
    }

    fun getPosEnu(): DoubleArray = state.posEnu.clone()
    fun getVelEnu(): DoubleArray = state.velEnu.clone()
    fun getQuat(): DoubleArray = state.quat.clone()
    fun getTimestamp(): Double = state.timestamp
}
