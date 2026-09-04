package com.sih26168.navigate.helper

import kotlin.math.*

object EsEkfMath {

    const val EARTH_RADIUS_M = 6371000.0
    const val STANDARD_GRAVITY = 9.80665
    const val DEG2RAD = Math.PI / 180.0
    const val RAD2DEG = 180.0 / Math.PI

    // ================================================================== //
    //  Quaternion & Rotation Utilities
    // ================================================================== //

    /**
     * Normalizes a quaternion to unit length with canonical non-negative scalar (qw >= 0).
     * Format: [qw, qx, qy, qz]
     */
    fun quatNormalize(q: DoubleArray): DoubleArray {
        val norm = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
        if (norm < 1e-12) {
            return doubleArrayOf(1.0, 0.0, 0.0, 0.0)
        }
        var qw = q[0] / norm
        var qx = q[1] / norm
        var qy = q[2] / norm
        var qz = q[3] / norm

        if (qw < 0.0) {
            qw = -qw
            qx = -qx
            qy = -qy
            qz = -qz
        }
        return doubleArrayOf(qw, qx, qy, qz)
    }

    /**
     * Hamilton product of two quaternions q1 and q2.
     * Format: [qw, qx, qy, qz]
     */
    fun quatMultiply(q1: DoubleArray, q2: DoubleArray): DoubleArray {
        val w1 = q1[0]; val x1 = q1[1]; val y1 = q1[2]; val z1 = q1[3]
        val w2 = q2[0]; val x2 = q2[1]; val y2 = q2[2]; val z2 = q2[3]

        return doubleArrayOf(
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        )
    }

    /**
     * Returns the conjugate / inverse of a unit quaternion.
     */
    fun quatInverse(q: DoubleArray): DoubleArray {
        val qNorm = quatNormalize(q)
        return doubleArrayOf(qNorm[0], -qNorm[1], -qNorm[2], -qNorm[3])
    }

    /**
     * Converts a 3D rotation vector (angle * axis) to a unit quaternion [qw, qx, qy, qz].
     */
    fun rotvecToQuat(rotvec: DoubleArray): DoubleArray {
        val rx = rotvec[0]
        val ry = rotvec[1]
        val rz = rotvec[2]
        val angle = sqrt(rx * rx + ry * ry + rz * rz)

        if (angle < 1e-8) {
            val qw = 1.0 - (angle * angle) / 8.0
            val scale = 0.5 - (angle * angle) / 48.0
            return quatNormalize(doubleArrayOf(qw, rx * scale, ry * scale, rz * scale))
        }

        val halfAngle = angle * 0.5
        val scale = sin(halfAngle) / angle
        return quatNormalize(doubleArrayOf(cos(halfAngle), rx * scale, ry * scale, rz * scale))
    }

    /**
     * Converts a unit quaternion [qw, qx, qy, qz] to a 3D rotation vector.
     */
    fun quatToRotvec(q: DoubleArray): DoubleArray {
        val qNorm = quatNormalize(q)
        val qw = qNorm[0].coerceIn(-1.0, 1.0)
        val rx = qNorm[1]
        val ry = qNorm[2]
        val rz = qNorm[3]
        val vecNorm = sqrt(rx * rx + ry * ry + rz * rz)

        if (vecNorm < 1e-8) {
            return doubleArrayOf(2.0 * rx, 2.0 * ry, 2.0 * rz)
        }
        val angle = 2.0 * atan2(vecNorm, qw)
        val scale = angle / vecNorm
        return doubleArrayOf(rx * scale, ry * scale, rz * scale)
    }

    /**
     * Converts unit quaternion [qw, qx, qy, qz] to 3x3 direction cosine matrix R_b^n.
     * Transforms vectors from body frame to navigation frame: v^n = R_b^n * v^b.
     */
    fun quatToRotmat(q: DoubleArray): Array<DoubleArray> {
        val qN = quatNormalize(q)
        val w = qN[0]; val x = qN[1]; val y = qN[2]; val z = qN[3]

        return arrayOf(
            doubleArrayOf(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)),
            doubleArrayOf(2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)),
            doubleArrayOf(2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y))
        )
    }

    /**
     * Converts a 3x3 rotation matrix to a unit quaternion [qw, qx, qy, qz].
     */
    fun rotmatToQuat(R: Array<DoubleArray>): DoubleArray {
        val tr = R[0][0] + R[1][1] + R[2][2]
        val qw: Double; val qx: Double; val qy: Double; val qz: Double

        if (tr > 0.0) {
            val s = 0.5 / sqrt(tr + 1.0)
            qw = 0.25 / s
            qx = (R[2][1] - R[1][2]) * s
            qy = (R[0][2] - R[2][0]) * s
            qz = (R[1][0] - R[0][1]) * s
        } else if ((R[0][0] > R[1][1]) && (R[0][0] > R[2][2])) {
            val s = 2.0 * sqrt(1.0 + R[0][0] - R[1][1] - R[2][2])
            qw = (R[2][1] - R[1][2]) / s
            qx = 0.25 * s
            qy = (R[0][1] + R[1][0]) / s
            qz = (R[0][2] + R[2][0]) / s
        } else if (R[1][1] > R[2][2]) {
            val s = 2.0 * sqrt(1.0 + R[1][1] - R[0][0] - R[2][2])
            qw = (R[0][2] - R[2][0]) / s
            qx = (R[0][1] + R[1][0]) / s
            qy = 0.25 * s
            qz = (R[1][2] + R[2][1]) / s
        } else {
            val s = 2.0 * sqrt(1.0 + R[2][2] - R[0][0] - R[1][1])
            qw = (R[1][0] - R[0][1]) / s
            qx = (R[0][2] + R[2][0]) / s
            qy = (R[1][2] + R[2][1]) / s
            qz = 0.25 * s
        }
        return quatNormalize(doubleArrayOf(qw, qx, qy, qz))
    }

    /**
     * Constructs orientation quaternion from vehicle heading (clockwise from North in ENU tangent plane).
     */
    fun headingDegToQuat(headingDeg: Double, pitchDeg: Double = 0.0, rollDeg: Double = 0.0): DoubleArray {
        val psi = headingDeg * DEG2RAD
        val theta = pitchDeg * DEG2RAD
        val phi = rollDeg * DEG2RAD

        val RYaw = arrayOf(
            doubleArrayOf(sin(psi), -cos(psi), 0.0),
            doubleArrayOf(cos(psi), sin(psi), 0.0),
            doubleArrayOf(0.0, 0.0, 1.0)
        )

        if (abs(pitchDeg) < 1e-6 && abs(rollDeg) < 1e-6) {
            return rotmatToQuat(RYaw)
        }

        val RPitch = arrayOf(
            doubleArrayOf(cos(theta), 0.0, sin(theta)),
            doubleArrayOf(0.0, 1.0, 0.0),
            doubleArrayOf(-sin(theta), 0.0, cos(theta))
        )

        val RRoll = arrayOf(
            doubleArrayOf(1.0, 0.0, 0.0),
            doubleArrayOf(0.0, cos(phi), -sin(phi)),
            doubleArrayOf(0.0, sin(phi), cos(phi))
        )

        val RTotal = matMul(matMul(RYaw, RPitch), RRoll)
        return rotmatToQuat(RTotal)
    }

    /**
     * Extracts heading (clockwise from North, [0, 360) deg) from quaternion R_b^n.
     */
    fun quatToHeadingDeg(q: DoubleArray): Double {
        val R = quatToRotmat(q)
        val fwdEast = R[0][0]
        val fwdNorth = R[1][0]
        var headingRad = atan2(fwdEast, fwdNorth)
        if (headingRad < 0.0) {
            headingRad += 2.0 * Math.PI
        }
        val deg = headingRad * RAD2DEG
        return (deg % 360.0 + 360.0) % 360.0
    }

    /**
     * Returns 3x3 skew-symmetric cross-product matrix [v]x.
     */
    fun skewSymmetric(v: DoubleArray): Array<DoubleArray> {
        return arrayOf(
            doubleArrayOf(0.0, -v[2], v[1]),
            doubleArrayOf(v[2], 0.0, -v[0]),
            doubleArrayOf(-v[1], v[0], 0.0)
        )
    }

    // ================================================================== //
    //  WGS84 & ENU Coordinate Transformations
    // ================================================================== //

    fun latLonToEnu(
        latDeg: Double,
        lonDeg: Double,
        refLatDeg: Double,
        refLonDeg: Double
    ): DoubleArray {
        val dLat = (latDeg - refLatDeg) * DEG2RAD
        val dLon = (lonDeg - refLonDeg) * DEG2RAD
        val refLatRad = refLatDeg * DEG2RAD

        val northM = dLat * EARTH_RADIUS_M
        val eastM = dLon * EARTH_RADIUS_M * cos(refLatRad)
        return doubleArrayOf(eastM, northM, 0.0)
    }

    fun enuToLatLon(
        eastM: Double,
        northM: Double,
        refLatDeg: Double,
        refLonDeg: Double
    ): Pair<Double, Double> {
        val refLatRad = refLatDeg * DEG2RAD
        val latDeg = refLatDeg + (northM / EARTH_RADIUS_M) * RAD2DEG
        val lonDeg = refLonDeg + (eastM / (EARTH_RADIUS_M * cos(refLatRad))) * RAD2DEG
        return Pair(latDeg, lonDeg)
    }

    fun haversineDistanceM(
        lat1Deg: Double, lon1Deg: Double,
        lat2Deg: Double, lon2Deg: Double
    ): Double {
        val lat1 = lat1Deg * DEG2RAD
        val lat2 = lat2Deg * DEG2RAD
        val dlat = lat2 - lat1
        val dlon = (lon2Deg - lon1Deg) * DEG2RAD
        val a = sin(dlat / 2.0).pow(2) + cos(lat1) * cos(lat2) * sin(dlon / 2.0).pow(2)
        return 2.0 * EARTH_RADIUS_M * asin(sqrt(a))
    }

    // ================================================================== //
    //  Matrix Operations
    // ================================================================== //

    fun matMul(A: Array<DoubleArray>, B: Array<DoubleArray>): Array<DoubleArray> {
        val rowsA = A.size
        val colsA = A[0].size
        val rowsB = B.size
        val colsB = B[0].size
        require(colsA == rowsB) { "Matrix dimensions mismatch for multiplication: $colsA vs $rowsB" }

        val C = Array(rowsA) { DoubleArray(colsB) }
        for (i in 0 until rowsA) {
            for (j in 0 until colsB) {
                var sum = 0.0
                for (k in 0 until colsA) {
                    sum += A[i][k] * B[k][j]
                }
                C[i][j] = sum
            }
        }
        return C
    }

    fun matMulVec(A: Array<DoubleArray>, x: DoubleArray): DoubleArray {
        val rows = A.size
        val cols = A[0].size
        require(cols == x.size) { "Matrix-vector dimension mismatch: $cols vs ${x.size}" }

        val y = DoubleArray(rows)
        for (i in 0 until rows) {
            var sum = 0.0
            for (j in 0 until cols) {
                sum += A[i][j] * x[j]
            }
            y[i] = sum
        }
        return y
    }

    fun matAdd(A: Array<DoubleArray>, B: Array<DoubleArray>): Array<DoubleArray> {
        val rows = A.size
        val cols = A[0].size
        val C = Array(rows) { DoubleArray(cols) }
        for (i in 0 until rows) {
            for (j in 0 until cols) {
                C[i][j] = A[i][j] + B[i][j]
            }
        }
        return C
    }

    fun matSub(A: Array<DoubleArray>, B: Array<DoubleArray>): Array<DoubleArray> {
        val rows = A.size
        val cols = A[0].size
        val C = Array(rows) { DoubleArray(cols) }
        for (i in 0 until rows) {
            for (j in 0 until cols) {
                C[i][j] = A[i][j] - B[i][j]
            }
        }
        return C
    }

    fun matTranspose(A: Array<DoubleArray>): Array<DoubleArray> {
        val rows = A.size
        val cols = A[0].size
        val At = Array(cols) { DoubleArray(rows) }
        for (i in 0 until rows) {
            for (j in 0 until cols) {
                At[j][i] = A[i][j]
            }
        }
        return At
    }

    fun eye(n: Int): Array<DoubleArray> {
        val I = Array(n) { DoubleArray(n) }
        for (i in 0 until n) {
            I[i][i] = 1.0
        }
        return I
    }

    fun matInvert1x1(A: Array<DoubleArray>): Array<DoubleArray> {
        val det = A[0][0]
        require(abs(det) > 1e-15) { "Singular 1x1 matrix" }
        return arrayOf(doubleArrayOf(1.0 / det))
    }

    fun matInvert2x2(A: Array<DoubleArray>): Array<DoubleArray> {
        val det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
        require(abs(det) > 1e-15) { "Singular 2x2 matrix" }
        val invDet = 1.0 / det
        return arrayOf(
            doubleArrayOf(A[1][1] * invDet, -A[0][1] * invDet),
            doubleArrayOf(-A[1][0] * invDet, A[0][0] * invDet)
        )
    }

    fun matInvert3x3(A: Array<DoubleArray>): Array<DoubleArray> {
        val a = A[0][0]; val b = A[0][1]; val c = A[0][2]
        val d = A[1][0]; val e = A[1][1]; val f = A[1][2]
        val g = A[2][0]; val h = A[2][1]; val i = A[2][2]

        val A00 = e * i - f * h
        val A01 = -(d * i - f * g)
        val A02 = d * h - e * g
        val A10 = -(b * i - c * h)
        val A11 = a * i - c * g
        val A12 = -(a * h - b * g)
        val A20 = b * f - c * e
        val A21 = -(a * f - c * d)
        val A22 = a * e - b * d

        val det = a * A00 + b * A01 + c * A02
        require(abs(det) > 1e-15) { "Singular 3x3 matrix" }
        val invDet = 1.0 / det

        return arrayOf(
            doubleArrayOf(A00 * invDet, A10 * invDet, A20 * invDet),
            doubleArrayOf(A01 * invDet, A11 * invDet, A21 * invDet),
            doubleArrayOf(A02 * invDet, A12 * invDet, A22 * invDet)
        )
    }

    fun matInvertSmall(A: Array<DoubleArray>): Array<DoubleArray> {
        return when (A.size) {
            1 -> matInvert1x1(A)
            2 -> matInvert2x2(A)
            3 -> matInvert3x3(A)
            else -> throw IllegalArgumentException("Unsupported matrix size for small inversion: ${A.size}")
        }
    }
}
