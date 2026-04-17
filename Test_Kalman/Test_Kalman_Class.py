import math
import numpy as np

import matplotlib

# matplotlib.use("TkAgg")  # à placer avant import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

from dataclasses import dataclass, field
from typing import Optional

np.set_printoptions(precision=4, suppress=True)

from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

import sys
import time


# Class_7_v2.5->Class_8_v1 : toutes vares et tous calculs en radians
# Class_8_v1->Class_8_v2 : rotation alpha autour de y -> rotation alpha autour de u unitaire qcq
# Class_8_v2->Class_8_v3 : ajout représentation graphique attitude
# Class_8_v3->Class : Gestion Git+GitHub


# https://www.ferdinandpiette.com/blog/2011/04/exemple-dutilisation-du-filtre-de-kalman/


# =============================================================================
# Fonctions utilitaires (indépendantes de tout contexte)
# =============================================================================


def progress_bar(current, total, prefix="", bar_length=40, start_time=None):
    if total <= 0:
        total = 1

    fraction = current / total
    filled_length = int(bar_length * fraction)
    bar = "#" * filled_length + "-" * (bar_length - filled_length)
    percent = 100.0 * fraction

    # if start_time is not None and current > 0:
    #     elapsed = time.time() - start_time
    #     eta = elapsed * (total - current) / current
    #     message = (f"\r{prefix} [{bar}] {percent:6.2f}% "
    #                f"({current}/{total}) | écoulé : {elapsed:6.1f}s | ETA : {eta:6.1f}s")
    # else:
    #     message = f"\r{prefix} [{bar}] {percent:6.2f}% ({current}/{total})"

    message = f"\r{prefix} [{bar}] {percent:6.2f}% ({current:3}/{total:3})"
    if start_time is not None and current > 0:
        elapsed = time.time() - start_time
        eta = elapsed * (total - current) / current
        message += f" | écoulé : {elapsed:6.1f}s | ETA : {eta:6.1f}s"

    print(message, end="", flush=True)

    if current >= total:
        print()


def plotsProgress(init=False, total=None, full=False):
    if not hasattr(plotsProgress, "current"):
        plotsProgress.current = 0
    if full:
        plotsProgress.current = plotsProgress.total
    if init:
        plotsProgress.startTime = time.time()
        plotsProgress.current = 0
    if total is not None:
        plotsProgress.total = total
    progress_bar(plotsProgress.current, plotsProgress.total, prefix="Plots", start_time=plotsProgress.startTime)
    plotsProgress.current += 1


def loadCSVRecord(filename: str) -> np.ndarray:
    (
        timeArray,
        measuredAccelXArray,
        measuredAccelYArray,
        measuredAccelZArray,
        measuredGyroXArray,
        measuredGyroYArray,
        measuredGyroZArray,
        measuredMagnetoXArray,
        measuredMagnetoYArray,
        measuredMagnetoZArray
    ) = np.loadtxt(
        filename,
        delimiter=",",
        skiprows=1,
        unpack=True

    )


def normalizeVector(v):
    v = np.asarray(v, dtype=float)
    vNorm = np.linalg.norm(v)
    if vNorm <= 0.0:
        return np.array([0.0, 1.0, 0.0], dtype=float)
    return v / vNorm


# def normalizeQuaternion(q):
#     q = np.asarray(q, dtype=float)
#     qNorm = np.linalg.norm(q)
#     if qNorm <= 0.0:
#         return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
#     return q / qNorm

def normalizeQuaternion(q):
    q = np.asarray(q, dtype=float)

    # ADDED: validation de la forme
    if q.shape[-1] != 4:
        raise ValueError("Input quaternion must have a last dimension of size 4.")

    # MODIFIED: norme calculée sur le dernier axe pour gérer (4,) et (N, 4)
    qNorm = np.linalg.norm(q, axis=-1, keepdims=True)

    # MODIFIED: gestion du cas quaternion unique nul
    if q.ndim == 1:
        if qNorm[0] <= 0.0:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        return q / qNorm[0]

    # ADDED: gestion vectorisée des quaternions nuls dans un tableau 2D
    safeQNorm = np.where(qNorm <= 0.0, 1.0, qNorm)
    normalizedQ = q / safeQNorm

    # ADDED: remplacement des lignes nulles par le quaternion identité
    zeroMask = (qNorm[..., 0] <= 0.0)
    normalizedQ[zeroMask] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    return normalizedQ


def quaternionConjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quaternionMultiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ], dtype=float)


def rotateVectorWorldToBody(q, vWorld):
    q = normalizeQuaternion(q)
    vQuat = np.array([0.0, vWorld[0], vWorld[1], vWorld[2]], dtype=float)
    qConj = quaternionConjugate(q)
    vBodyQuat = quaternionMultiply(
        quaternionMultiply(qConj, vQuat),
        q
    )
    return vBodyQuat[1:]


# v2->v3
def rotateVectorBodyToWorld(q, vBody):
    q = normalizeQuaternion(q)
    vQuat = np.array([0.0, vBody[0], vBody[1], vBody[2]], dtype=float)
    qConj = quaternionConjugate(q)
    vWorldQuat = quaternionMultiply(
        quaternionMultiply(q, vQuat),
        qConj
    )
    return vWorldQuat[1:]


def quaternionToEuler(q):
    q = np.asarray(q, dtype=float)
    q = normalizeQuaternion(q)

    # w, x, y, z = q
    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    # RollPitchYaw = np.array([roll, pitch, yaw], dtype=float)
    RollPitchYaw = np.stack((roll, pitch, yaw), axis=-1)
    return RollPitchYaw


def setEqual3DAxes(ax, lim):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])


def runTriadAnimation(
        quaternionArrayToPlot,
        timeArray,
        gravity,
        labelOrientation="orientation",
        axisLength=2.5,
        exportGif=False,
        gifFileName="triadre_imu.gif",
        exportVideo=False,
        videoFileName="triadre_imu.mp4",
        fps=25,
        dpi=120,
        showProgress=False,
):
    gravityVectorWorld = np.array([0.0, 0.0, -gravity], dtype=float)

    figTriad = plt.figure(figsize=(9, 9))
    axTriad = figTriad.add_subplot(111, projection='3d')

    frameStride = max(1, len(timeArray) // 500)
    frameIndices = np.arange(0, len(timeArray), frameStride)

    plotLimit = 1.2 * max(axisLength, gravity)

    if exportGif and exportVideo:
        raise ValueError(
            "Les résultat ne peut être exporté que dans 1 seul format :"
            "exportGif et exportVideo ne peuvent pas être True en même temps!"
        )

    # if showProgress:
    # startTime = time.time()

    def updateTriad(frameNumber):
        frameIndex = frameIndices[frameNumber]

        # if showProgress:
        #     percent = 100.0 * (frameNumber + 1) / len(frameIndices)
        #     if frameNumber == 0 or (frameNumber + 1) % 10 == 0 or (frameNumber + 1) == len(frameIndices):
        #         print(f"Trièdre {labelOrientation} : frame {frameNumber + 1}/{len(frameIndices)} "
        #           f"frames ({percent:5.1f} %)", end="")
        # if showProgress and (frameNumber == 0 or (frameNumber + 1) % 10 == 0 or (frameNumber + 1) == len(frameIndices)):
        #     progress_bar(frameNumber + 1,
        #                  len(frameIndices),
        #                  prefix=f"Animation {labelOrientation}",
        #                  # startTime=startTime,
        #                  )
        axTriad.cla()

        q = normalizeQuaternion(quaternionArrayToPlot[frameIndex])

        ex_world = rotateVectorBodyToWorld(q, np.array([1.0, 0.0, 0.0])) * axisLength
        ey_world = rotateVectorBodyToWorld(q, np.array([0.0, 1.0, 0.0])) * axisLength
        ez_world = rotateVectorBodyToWorld(q, np.array([0.0, 0.0, 1.0])) * axisLength

        # vecteur gravité
        axTriad.quiver(
            0, 0, 0,
            gravityVectorWorld[0], gravityVectorWorld[1], gravityVectorWorld[2],
            color='k', linewidth=2.5, arrow_length_ratio=0.08
        )
        axTriad.text(
            gravityVectorWorld[0], gravityVectorWorld[1], gravityVectorWorld[2],
            "g", color='k', fontsize=12
        )

        # trièdre IMU
        axTriad.quiver(0, 0, 0, ex_world[0], ex_world[1], ex_world[2],
                       color='r', linewidth=2, arrow_length_ratio=0.08)
        axTriad.quiver(0, 0, 0, ey_world[0], ey_world[1], ey_world[2],
                       color='g', linewidth=2, arrow_length_ratio=0.08)
        axTriad.quiver(0, 0, 0, ez_world[0], ez_world[1], ez_world[2],
                       color='b', linewidth=2, arrow_length_ratio=0.08)

        axTriad.text(ex_world[0], ex_world[1], ex_world[2], "x", color='r', fontsize=12)
        axTriad.text(ey_world[0], ey_world[1], ey_world[2], "y", color='g', fontsize=12)
        axTriad.text(ez_world[0], ez_world[1], ez_world[2], "z", color='b', fontsize=12)

        setEqual3DAxes(axTriad, plotLimit)
        axTriad.set_xlabel("X monde")
        axTriad.set_ylabel("Y monde")
        axTriad.set_zlabel("Z monde")
        axTriad.set_title(
            f"Trièdre IMU + gravité\n"
            f"{labelOrientation} | t = {timeArray[frameIndex]:.2f} s"
        )
        axTriad.grid(True)

    aniTriad = FuncAnimation(
        figTriad,
        updateTriad,
        frames=len(frameIndices),
        interval=30,
        repeat=True,
        blit=False
    )

    if exportGif:
        writer = PillowWriter(fps=25)

        if showProgress:
            start_time = time.time()

            def save_progress_callback(i, n):
                progress_bar(i + 1,
                             n,
                             prefix=f"Export GIF {labelOrientation}",
                             start_time=start_time)

            aniTriad.save(
                gifFileName,
                writer=writer,
                progress_callback=save_progress_callback
            )
        else:
            aniTriad.save(gifFileName, writer=writer)

    if exportVideo:
        writer = FFMpegWriter(
            fps=fps,
            codec="libx264",
            bitrate=1800,
            metadata={"artist": "Python / Matplotlib"}
        )

        if showProgress:
            start_time = time.time()

            def save_progress_callback(i, n):
                progress_bar(i + 1, n, prefix="Export MP4 trièdre", start_time=start_time)

            aniTriad.save(
                videoFileName,
                writer=writer,
                dpi=dpi,
                progress_callback=save_progress_callback
            )
        else:
            aniTriad.save(
                videoFileName,
                writer=writer,
                dpi=dpi
            )

    return figTriad, aniTriad


# =============================================================================
# Classe SimulationConfig  [INCHANGÉE]
# Rôle : décrire la réalité physique simulée et générer les mesures bruitées.
# =============================================================================

class SimulationConfig:

    def __init__(
            self,
            totalTime: float = None,
            timeStep: float = None,
            sampleSize: int = None,
            randomSeed: int = 123,
            trueInitialAlpha: float = 0.0,
            trueInitialAlphadot: float = 0.0,
            trueInitialBiasX: float = 0.0,
            trueInitialBiasY: float = 0.0,
            trueInitialBiasZ: float = 0.0,
            measurementAccelNoiseStd: float = 0.2,
            measurementGyroNoiseStd: float = 1.0,
            gravity: float = 9.81,
            rotationAxis: np.ndarray = None,
    ):
        if totalTime is None:
            self.timeStep = timeStep
            self.sampleSize = sampleSize
            self.totalTime = self.timeStep * self.sampleSize
        elif timeStep is None:
            self.totalTime = totalTime
            self.sampleSize = sampleSize
            self.timeStep = float(self.totalTime / self.sampleSize)
        elif sampleSize is None:
            self.totalTime = totalTime
            self.timeStep = timeStep
            self.sampleSize = int(self.totalTime / self.timeStep)
        else:
            raise ValueError(
                "Founir 2 valeurs parmi (totalTime, timeStep, sampleSize)!"
            )
        self.randomSeed = randomSeed

        self.trueInitialAlpha = trueInitialAlpha
        self.trueInitialAlphaDot = trueInitialAlphadot
        self.trueInitialBiasX = trueInitialBiasX
        self.trueInitialBiasY = trueInitialBiasY
        self.trueInitialBiasZ = trueInitialBiasZ

        self.measurementAccelNoiseStd = measurementAccelNoiseStd

        self.measurementGyroNoiseStd = measurementGyroNoiseStd
        # if self.angleUnitIsDegree:
        #     self.measurementGyroNoiseStd = np.deg2rad(measurementGyroNoiseStd)
        # else:
        #     self.measurementGyroNoiseStd = measurementGyroNoiseStd

        self.gravity = gravity

        if rotationAxis is None:
            rotationAxis = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            rotationAxis = np.array(rotationAxis, dtype=float)

        axisNorm = np.linalg.norm(rotationAxis)
        if axisNorm <= 0.0:
            raise ValueError("rotationAxis ne doit pas être nul")
        self.rotationAxis = rotationAxis / axisNorm

        np.random.seed(self.randomSeed)
        self.setAngularAccelerationProfile()

    def alphaToQuaternion(self, alpha):
        """
        Convertit l'angle vrai alpha en quaternion [qw, qx, qy, qz].
        Ici on suppose une rotation pure autour de l'axe Y,
        cohérente avec accelX = -g*sin(alpha), accelZ = g*cos(alpha).
        """
        # alphaArray = np.asarray(alpha, dtype=float)
        alphaArray = np.atleast_1d(np.asarray(alpha, dtype=float))

        halfAlphaArray = 0.5 * alphaArray

        quaternionArray = np.zeros((alphaArray.size, 4), dtype=float)
        # quaternionArray[:, 0] = np.cos(halfAlphaArray)  # qw
        # quaternionArray[:, 1] = 0.0  # qx
        # quaternionArray[:, 2] = np.sin(halfAlphaArray)  # qy
        # quaternionArray[:, 3] = 0.0  # qz

        ux, uy, uz = self.rotationAxis
        quaternionArray[:, 0] = np.cos(halfAlphaArray)
        quaternionArray[:, 1] = ux * np.sin(halfAlphaArray)
        quaternionArray[:, 2] = uy * np.sin(halfAlphaArray)
        quaternionArray[:, 3] = uz * np.sin(halfAlphaArray)

        if np.isscalar(alpha):
            return quaternionArray[0]
        return quaternionArray

    def setAngularAccelerationProfile(self, alphaAccelerationProfile: np.ndarray = [[0.0, 0.0]]):
        alphaAccelerationProfile = alphaAccelerationProfile + [[1.0, 0.0]]
        alphaAccelerationProfile = np.array(alphaAccelerationProfile, dtype=float)

        self.alphaAccelerationProfile = alphaAccelerationProfile

    def generateTrueValuesAndMeasurements(self):
        """
        Génère les valeurs vraies (alpha, alphadot) et les mesures bruitées
        (accelX, accelY, alphadot gyro).
        Retourne : (timeArray, trueAlphaArray, trueAlphaDotArray,
                    measuredAccelXArray, measuredAccelYArray, measuredAlphadotArray)
        """
        print(f"TimeStep : {self.timeStep}")

        timeArray = np.arange(self.sampleSize, dtype=float) * self.timeStep
        trueAlphaArray = np.zeros(self.sampleSize, dtype=float)
        trueAlphaDotArray = np.zeros(self.sampleSize, dtype=float)

        currentAlpha = self.trueInitialAlpha
        currentAlphaDot = self.trueInitialAlphaDot

        alphadotDots = self.alphaAccelerationProfile

        percentIndex = 0
        for indexTime in range(self.sampleSize):
            currentTime = timeArray[indexTime]
            timeRatio = currentTime / self.totalTime

            if timeRatio > alphadotDots[percentIndex + 1][0]:
                percentIndex = percentIndex + 1
            currentAlphaDotDot = alphadotDots[percentIndex][1]

            currentAlphaDot = currentAlphaDot + currentAlphaDotDot * self.timeStep
            currentAlpha = currentAlpha + currentAlphaDot * self.timeStep
            trueAlphaArray[indexTime] = currentAlpha
            trueAlphaDotArray[indexTime] = currentAlphaDot

        # v1 -> v2
        trueQuaternionArray = self.alphaToQuaternion(trueAlphaArray)

        gravityWorld = np.array([0.0, 0.0, self.gravity], dtype=float)
        trueAccelArray = np.array([
            rotateVectorWorldToBody(q, gravityWorld)
            for q in trueQuaternionArray
        ]).T

        # accelNoiseX = np.random.normal(0.0, self.measurementAccelNoiseStd, self.sampleSize)
        # accelNoiseY = np.random.normal(0.0, self.measurementAccelNoiseStd, self.sampleSize)
        # accelNoiseZ = np.random.normal(0.0, self.measurementAccelNoiseStd, self.sampleSize)
        # accelNoise = np.array(
        #     [
        #         accelNoiseX,
        #         accelNoiseY,
        #         accelNoiseZ,
        #     ]
        # )
        accelNoise = np.random.normal(
            0.0,
            self.measurementAccelNoiseStd,
            size=(3, self.sampleSize)
        )

        # measuredAccelXArray = -self.gravity * np.sin(trueAlphaArray) + accelNoiseX
        # measuredAccelYArray = np.zeros(self.sampleSize, dtype=float) + accelNoiseY
        # measuredAccelZArray = self.gravity * np.cos(trueAlphaArray) + accelNoiseZ
        measuredAccelArray = trueAccelArray + accelNoise

        #

        # v1 -> v2
        # trueGyroArray = (trueAlphaDotArray[:, None] * self.rotationAxis[None, :]).T
        trueGyroArray = (trueAlphaDotArray[None, :] * self.rotationAxis[:, None])

        # gyroNoiseX = np.random.normal(0.0, self.measurementGyroNoiseStd, self.sampleSize)
        # gyroNoiseY = np.random.normal(0.0, self.measurementGyroNoiseStd, self.sampleSize)
        # gyroNoiseZ = np.random.normal(0.0, self.measurementGyroNoiseStd, self.sampleSize)
        # gyroNoise = np.array(
        #     [
        #         gyroNoiseX,
        #         gyroNoiseY,
        #         gyroNoiseZ,
        #     ]
        # )
        gyroNoise = np.random.normal(
            0.0,
            self.measurementGyroNoiseStd,
            size=(3, self.sampleSize)
        )

        gyroBias = np.array([
            self.trueInitialBiasX,
            self.trueInitialBiasY,
            self.trueInitialBiasZ,
        ], dtype=float)

        # measuredGyroXArray = 0.0 + self.trueInitialBiasX + gyroNoiseX
        # measuredGyroYArray = trueAlphaDotArray + self.trueInitialBiasY + gyroNoiseY
        # measuredGyroZArray = 0.0 + self.trueInitialBiasZ + gyroNoiseZ
        measuredGyroArray = trueGyroArray + gyroBias[:, None] + gyroNoise

        # v1 -> v2
        return (
            timeArray,
            trueAlphaArray,
            trueAlphaDotArray,
            trueQuaternionArray,
            measuredAccelArray[0, :],
            measuredAccelArray[1, :],
            measuredAccelArray[2, :],
            measuredGyroArray[0, :],
            measuredGyroArray[1, :],
            measuredGyroArray[2, :],
        )

        # return (
        #     timeArray,
        #     trueAlphaArray,
        #     trueAlphaDotArray,
        #     trueQuaternionArray,
        #     measuredAccelXArray,
        #     measuredAccelYArray,
        #     measuredAccelZArray,
        #     measuredGyroXArray,
        #     measuredGyroYArray,
        #     measuredGyroZArray,
        # )


# =============================================================================
# Classe UkfModel  [NOUVEAU — extrait de l'ancienne UkfConfig]
# Rôle : décrire la structure invariante du filtre :
#        dimensions, pas de temps, fonctions de modèle physique (fx, hx),
#        paramètres sigma points.
#        Ne contient AUCUN réglage numérique (Q, R, P, état initial).
#        Une seule instance suffit pour tous les runs sur la même physique.
# =============================================================================

class UkfModel:

    def __init__(
            self,
            simConfig: SimulationConfig,
            sigmaAlpha: float = 0.1,  # paramètre alpha des sigma-points Merwe
            sigmaBeta: float = 2.0,  # paramètre beta
            sigmaKappa: float = 0.0,  # paramètre kappa
    ):
        self.timeStep = simConfig.timeStep
        self.gravity = simConfig.gravity
        self.sigmaAlpha = sigmaAlpha
        self.sigmaBeta = sigmaBeta
        self.sigmaKappa = sigmaKappa
        self.lastGyroInput = np.zeros(3, dtype=float)

    def setCurrentGyroMeasurement(self, gyroMeas):
        self.lastGyroInput = np.array(gyroMeas, dtype=float)

    # v1->v2
    # def normalizeQuaternion(self, q):
    #     q = np.asarray(q, dtype=float)
    #     qNorm = np.linalg.norm(q)
    #     if qNorm <= 0.0:
    #         return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    #     return q / qNorm
    #
    # def quaternionConjugate(self, q):
    #     return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)
    #
    # def quaternionMultiply(self, q1, q2):
    #     w1, x1, y1, z1 = q1
    #     w2, x2, y2, z2 = q2
    #     return np.array([
    #         w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    #         w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
    #         w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
    #         w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    #     ], dtype=float)
    #
    # def rotateVectorWorldToBody(self, q, vWorld):
    #     q = self.normalizeQuaternion(q)
    #     vQuat = np.array([0.0, vWorld[0], vWorld[1], vWorld[2]], dtype=float)
    #     qConj = self.quaternionConjugate(q)
    #     vBodyQuat = self.quaternionMultiply(
    #         self.quaternionMultiply(qConj, vQuat),
    #         q
    #     )
    #     return vBodyQuat[1:]
    #
    # def quaternionToEuler(self, q):
    #     q = self.normalizeQuaternion(q)
    #     w, x, y, z = q
    #
    #     sinr_cosp = 2.0 * (w * x + y * z)
    #     cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    #     roll = np.arctan2(sinr_cosp, cosr_cosp)
    #
    #     sinp = 2.0 * (w * y - z * x)
    #     sinp = np.clip(sinp, -1.0, 1.0)
    #     pitch = np.arcsin(sinp)
    #
    #     siny_cosp = 2.0 * (w * z + x * y)
    #     cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    #     yaw = np.arctan2(siny_cosp, cosy_cosp)
    #
    #     RollPitchYaw = np.array([roll, pitch, yaw], dtype=float)
    #     return RollPitchYaw

    # --- Fonctions de modèle (dépendent uniquement de la physique) ---

    # f(x, dt)
    def stateTransitionFunction(self, x, dt):
        # v1->v2
        # q = self.normalizeQuaternion(x[0:4])
        q = normalizeQuaternion(x[0:4])
        b = x[4:7]

        omega = self.lastGyroInput - b
        wx, wy, wz = omega

        Omega = np.array([
            [0.0, -wx, -wy, -wz],
            [wx, 0.0, wz, -wy],
            [wy, -wz, 0.0, wx],
            [wz, wy, -wx, 0.0],
        ], dtype=float)

        qDot = 0.5 * Omega @ q
        # v1->v2
        # qNext = self.normalizeQuaternion(q + qDot * dt)
        qNext = normalizeQuaternion(q + qDot * dt)

        return np.hstack((qNext, b))

    # h(x)
    def measurementFunction(self, x):
        # v1->v2
        # q = self.normalizeQuaternion(x[0:4])
        q = normalizeQuaternion(x[0:4])
        b = x[4:7]

        gravityWorld = np.array([0.0, 0.0, self.gravity], dtype=float)
        # v1->v2
        # gravityBody = self.rotateVectorWorldToBody(q, gravityWorld)
        gravityBody = rotateVectorWorldToBody(q, gravityWorld)
        accelPred = gravityBody
        omegaBody = self.lastGyroInput - b
        gyroPred = omegaBody + b

        return np.hstack((accelPred, gyroPred))
        # return np.hstack((gravityBody, gyroMeasPred))


# =============================================================================
# Classe UkfParams  [NOUVEAU — extrait de l'ancienne UkfConfig]
# Rôle : regrouper les réglages numériques du filtre (Q, R, P₀, état initial
#        supposé). Modifiable à volonté sans toucher au modèle.
#        Chaque instance représente un "scénario de réglage" différent.
# =============================================================================

class UkfParams:

    def __init__(
            self,
            simConfig: SimulationConfig,
            supposedInitialQuaternion: np.ndarray = None,
            # supposedInitialAlphadot: float = 0.0,  # (°/s)
            supposedInitialBiasX: float = 0.0,
            supposedInitialBiasY: float = 0.0,
            supposedInitialBiasZ: float = 0.0,
            processQuaternionNoiseStd: float = 0.01,
            # processAlphadotNoiseStd: float = 1.0,  # => Q
            processBiasNoiseStd: float = 0.01,  # => Q
            processInitialConfidenceStd: float = 300.0,  # => P₀
            label: str = "",
    ):
        self.simConfig = simConfig
        self.supposedInitialQuaternion = (
            np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
            if supposedInitialQuaternion is None
            else np.array(supposedInitialQuaternion, dtype=float)
        )
        # self.supposedInitialAlphadot = supposedInitialAlphadot

        self.supposedInitialBiasX = supposedInitialBiasX
        self.supposedInitialBiasY = supposedInitialBiasY
        self.supposedInitialBiasZ = supposedInitialBiasZ

        self.measurementAccelNoiseStd = simConfig.measurementAccelNoiseStd
        self.measurementGyroNoiseStd = simConfig.measurementGyroNoiseStd

        self.processQuaternionNoiseStd = processQuaternionNoiseStd

        self.processBiasNoiseStd = processBiasNoiseStd

        self.processInitialConfidenceStd = processInitialConfidenceStd
        self.label = label

    @classmethod
    def getConstructorAttrs(cls, base: "UkfParams"):
        constructorAttrs = {
            "simConfig": base.simConfig,
            "supposedInitialQuaternion": base.supposedInitialQuaternion,
            # "supposedInitialAlphadot": base.supposedInitialAlphadot,
            "supposedInitialBiasX": base.supposedInitialBiasX,
            "supposedInitialBiasY": base.supposedInitialBiasY,
            "supposedInitialBiasZ": base.supposedInitialBiasZ,
            "processQuaternionNoiseStd": base.processQuaternionNoiseStd,
            # "processAlphadotNoiseStd": base.processAlphadotNoiseStd,
            "processBiasNoiseStd": base.processBiasNoiseStd,
            "processInitialConfidenceStd": base.processInitialConfidenceStd,
            "label": base.label,
        }
        return constructorAttrs

    @classmethod
    def fromBase(cls, base: "UkfParams", **overrides) -> "UkfParams":
        """
        Construit un UkfParams identique à `base`,
        en remplaçant uniquement les paramètres fournis dans overrides.
        Exemple :
            params2 = UkfParams.fromBase(params1, processBiasNoiseStd=0.1)
            !!! ne pas tenter de modifier 'simConfig' !!!
        """

        constructorAttrs = cls.getConstructorAttrs(base)

        validKeys = set(constructorAttrs.keys())
        unknownKeys = set(overrides.keys()) - validKeys
        if unknownKeys:
            raise ValueError(
                f"UkfParams.fromBase() : paramètres inconnus : {unknownKeys}"
            )

        constructorAttrs.update(overrides)
        if "label" in overrides:
            constructorAttrs.update({"label": base.label + "[" + overrides["label"] + "]"})
        return cls(**constructorAttrs)

    @classmethod
    def createSweepParams(
            cls,
            base: "UkfParams",
            paramName: str,
            paramValues,
    ) -> list["UkfParams"]:
        """                                                                 
        Construit une liste de UkfParams à partir de `base`,                
        en faisant varier uniquement le paramètre `paramName`               
        selon les valeurs fournies dans `paramValues`.                      
                                                                            
        Exemple :                                                           
        paramsList = UkfParams.createSweepParams(                           
            params1,                                                        
            "processBiasNoiseStd",
            [1.0, 0.5, 0.1, 0.01]                                           
        )                                                                   
        """
        constructorAttrs = cls.getConstructorAttrs(base)

        if paramName not in constructorAttrs:
            raise ValueError(
                f"UkfParams.createSweepParams() : paramètre inconnu ou non balayable : '{paramName}'"
            )

        paramsList = []
        for paramValue in paramValues:
            labelName = "label"
            labelValue = f"{paramName} = {paramValue}"
            paramsList.append(
                cls.fromBase(base, **{paramName: paramValue, labelName: labelValue})
            )

        return paramsList


# =============================================================================
# Classe UkfResult  [NOUVEAU]
# Rôle : conteneur nommé pour les arrays de sortie d'un run.
#        Conserve une référence aux UkfParams utilisés et un label libre,
#        ce qui permet d'accumuler plusieurs résultats dans une liste
#        et de les identifier facilement pour le tracé.
# =============================================================================

@dataclass
class UkfResult:
    label: str  # identifiant libre du run
    params: UkfParams  # réglages ayant produit ce résultat
    estimatedQuaternionArray: np.ndarray
    estimatedEulerArray: np.ndarray
    estimatedBiasArray: np.ndarray  # biais estimés


# =============================================================================
# Classe UkfRunner  [MODIFIÉE]
# Rôle : assembler UkfModel + UkfParams pour construire un filtre frais,
#        exécuter la boucle predict/update, retourner un UkfResult.
#        Sans état propre : peut être réutilisé pour n'importe quelle
#        combinaison (model, params, mesures).
# =============================================================================

class UkfRunner:

    def run(
            self,
            model: UkfModel,
            params: UkfParams,
            measuredAccelXArray: np.ndarray,
            measuredAccelYArray: np.ndarray,
            measuredAccelZArray: np.ndarray,
            measuredGyroXArray: np.ndarray,
            measuredGyroYArray: np.ndarray,
            measuredGyroZArray: np.ndarray,
            label: str = "run",
    ) -> UkfResult:
        """
        Construit un filtre UKF neuf à partir de (model, params),
        exécute la boucle predict/update sur les mesures fournies,
        et retourne un UkfResult étiqueté.
        """
        # --- Construction du filtre ---
        sigmaPoints = MerweScaledSigmaPoints(
            n=7,
            alpha=model.sigmaAlpha,
            beta=model.sigmaBeta,
            kappa=model.sigmaKappa,
        )

        ukf = UnscentedKalmanFilter(
            dim_x=7,
            dim_z=6,
            dt=model.timeStep,
            fx=model.stateTransitionFunction,
            hx=model.measurementFunction,
            points=sigmaPoints,
        )

        # v1->v2
        # initialQuaternion = model.normalizeQuaternion(params.supposedInitialQuaternion)
        initialQuaternion = normalizeQuaternion(params.supposedInitialQuaternion)

        ukf.x = np.hstack((
            initialQuaternion,
            np.array([
                params.supposedInitialBiasX,
                params.supposedInitialBiasY,
                params.supposedInitialBiasZ,
            ], dtype=float)
        ))

        ukf.P = np.diag([params.processInitialConfidenceStd ** 2] * 7)

        # measurementGyroNoiseStd = User2Rad(params.angleUnitIsDegree, params.measurementGyroNoiseStd)
        ukf.R = np.diag([
            params.measurementAccelNoiseStd ** 2,
            params.measurementAccelNoiseStd ** 2,
            params.measurementAccelNoiseStd ** 2,
            params.measurementGyroNoiseStd ** 2,
            params.measurementGyroNoiseStd ** 2,
            params.measurementGyroNoiseStd ** 2,
        ])

        ukf.Q = np.diag([
            params.processQuaternionNoiseStd ** 2,
            params.processQuaternionNoiseStd ** 2,
            params.processQuaternionNoiseStd ** 2,
            params.processQuaternionNoiseStd ** 2,
            params.processBiasNoiseStd ** 2,
            params.processBiasNoiseStd ** 2,
            params.processBiasNoiseStd ** 2,
        ])

        # --- Boucle predict / update ---
        estimatedQuaternionList = []
        estimatedEulerList = []
        estimatedBiasList = []

        for accelX, accelY, accelZ, gyroX, gyroY, gyroZ in zip(
                measuredAccelXArray,
                measuredAccelYArray,
                measuredAccelZArray,
                measuredGyroXArray,
                measuredGyroYArray,
                measuredGyroZArray,
        ):
            model.setCurrentGyroMeasurement([gyroX, gyroY, gyroZ])

            ukf.predict()
            # ukf.x[0:4] = model.normalizeQuaternion(ukf.x[0:4])

            ukf.update(np.array([accelX, accelY, accelZ, gyroX, gyroY, gyroZ], dtype=float))
            # v1->v2
            # ukf.x[0:4] = model.normalizeQuaternion(ukf.x[0:4])
            ukf.x[0:4] = normalizeQuaternion(ukf.x[0:4])

            qEst = ukf.x[0:4].copy()
            bEst = ukf.x[4:7].copy()
            # v1->v2
            # eulerEst = model.quaternionToEuler(qEst)
            eulerEst = quaternionToEuler(qEst)

            estimatedQuaternionList.append(qEst)
            estimatedEulerList.append(eulerEst)
            estimatedBiasList.append(bEst)

        return UkfResult(
            label=label,
            params=params,
            estimatedQuaternionArray=np.array(estimatedQuaternionList),
            estimatedEulerArray=np.array(estimatedEulerList),
            estimatedBiasArray=np.array(estimatedBiasList),
        )


if __name__ == "__main__":

    # =============================================================================
    # EXEMPLE D'UTILISATION (cellules Jupyter)
    # =============================================================================

    # --- Cellule 1 : simulation ---
    # simConfig = SimulationConfig(
    #     totalTime=100.0,
    #     timeStep=0.1,
    #     angleUnitIsDegree = True,
    #     trueInitialAlpha=-45.0,
    #     trueInitialAlphaDot=0.0,
    #     trueInitialBiasX=10.0,
    #     trueInitialBiasY=15.0,
    #     trueInitialBiasZ=20.0,
    #     measurementAccelNoiseStd=0.1,
    #     measurementGyroNoiseStd=0.01,
    # )

    totalTime = 100.0
    timeStep = 0.01

    print("Config")
    simConfig = SimulationConfig(
        totalTime=totalTime,
        timeStep=timeStep,
        trueInitialAlpha=np.deg2rad(-45.0),
        trueInitialAlphadot=np.deg2rad(0.0),
        trueInitialBiasX=np.deg2rad(10.0),
        trueInitialBiasY=np.deg2rad(15.0),
        trueInitialBiasZ=np.deg2rad(20.0),
        measurementAccelNoiseStd=0.2,
        measurementGyroNoiseStd=np.deg2rad(10.0),
        rotationAxis=normalizeVector([1.0, 1.0, 0.0]),
    )

    # trueBiases = np.full(simConfig.sampleSize,[[simConfig.trueInitialBiasX, simConfig.trueInitialBiasY,simConfig.trueInitialBiasZ]])
    trueBiases = np.tile(
        [
            simConfig.trueInitialBiasX,
            simConfig.trueInitialBiasY,
            simConfig.trueInitialBiasZ,
        ],
        (simConfig.sampleSize, 1)
    )

    trueAngularAccel = 0.30
    simConfig.setAngularAccelerationProfile(
        [
            [0.0, np.deg2rad(0.0)],
            [.1, np.deg2rad(trueAngularAccel)],
            [.3, np.deg2rad(-trueAngularAccel)],
            [.5, np.deg2rad(-trueAngularAccel)],
            [.7, np.deg2rad(trueAngularAccel)],
            [.9, np.deg2rad(0.0)],
        ]
    )

    print("True Values")
    (
        timeArray,
        trueAlphaArray,
        trueAlphaDotArray,
        trueQuaternionArray,
        measuredAccelXArray,
        measuredAccelYArray,
        measuredAccelZArray,
        measuredGyroXArray,
        measuredGyroYArray,
        measuredGyroZArray,
    ) = simConfig.generateTrueValuesAndMeasurements()

    # trueEulerArray = np.array([quaternionToEuler(trueQuaternion) for trueQuaternion in trueQuaternionArray[:,]))
    trueEulerArray = quaternionToEuler(trueQuaternionArray)

    # --- Cellule 2 : modèle physique (une seule fois) ---
    print("Model + Runner")
    ukfModel = UkfModel(simConfig)
    runner = UkfRunner()

    # Réglage de référence
    print("Param Réf")
    paramsRef = UkfParams(
        simConfig,
        supposedInitialQuaternion=None,
        supposedInitialBiasX=0.0,
        supposedInitialBiasY=0.0,
        supposedInitialBiasZ=0.0,
        processQuaternionNoiseStd=0.001,
        processBiasNoiseStd=0.001,  # => Q
        processInitialConfidenceStd=1.0,  # => P₀
        label="Référence",
    )

    # Réglage de base
    print("Param Base")
    paramsBase = UkfParams.fromBase(
        paramsRef,
        processQuaternionNoiseStd=0.01,
        processBiasNoiseStd=0.001,  # => Q
        processInitialConfidenceStd=0.1,  # => P₀
        label="Base",
    )
    # paramsSweep = UkfParams.createSweepParams(
    #     paramsRef,
    #     "processBiasNoiseStd",
    #     [0.001, 0.01, 0.1]
    # )

    # paramsSweep = UkfParams.createSweepParams(
    #     paramsBase,
    #     "processInitialConfidenceStd",
    #     [0.01, 0.1, 1.0]
    # )

    print("Params multiples")
    paramsSweep = UkfParams.createSweepParams(
        paramsBase,
        "processInitialConfidenceStd",
        [0.01, 0.1, 1.0]
    )

    print("Run multiples")
    results = []
    for currentParams in paramsSweep:
        currentLabel = currentParams.label
        results.append(
            runner.run(
                ukfModel,
                currentParams,
                measuredAccelXArray,
                measuredAccelYArray,
                measuredAccelZArray,
                measuredGyroXArray,
                measuredGyroYArray,
                measuredGyroZArray,
                label=currentLabel
            )
        )

    # --- Cellule 4 : affichage ---
    print("Plots multiples", flush=True)
    plotsProgress(init=True, total=50)
    fig, axes = plt.subplots(11, 1, figsize=(18, 36), sharex=True)
    plotsProgress()
    for res in results:
        axes[0].plot(timeArray, res.estimatedQuaternionArray[:, 0], label="estimatedQuaternionArray " + res.label)
        plotsProgress()
        axes[1].plot(timeArray, res.estimatedQuaternionArray[:, 1], label="estimatedQuaternionArray " + res.label)
        plotsProgress()
        axes[2].plot(timeArray, res.estimatedQuaternionArray[:, 2], label="estimatedQuaternionArray " + res.label)
        plotsProgress()
        axes[3].plot(timeArray, res.estimatedQuaternionArray[:, 3], label="estimatedQuaternionArray " + res.label)
        plotsProgress()
        axes[4].plot(timeArray, np.linalg.norm(res.estimatedQuaternionArray, axis=-1),
                     label="estimatedQuaternionArray " + res.label)
        plotsProgress()
        axes[5].plot(timeArray, np.rad2deg(res.estimatedEulerArray[:, 0]), label="estimatedEulerArray " + res.label)
        plotsProgress()
        axes[6].plot(timeArray, np.rad2deg(res.estimatedEulerArray[:, 1]), label="estimatedEulerArray " + res.label)
        plotsProgress()
        axes[7].plot(timeArray, np.rad2deg(res.estimatedEulerArray[:, 2]), label="estimatedEulerArray " + res.label)
        plotsProgress()
        axes[8].plot(timeArray, np.rad2deg(res.estimatedBiasArray[:, 0]), label="estimatedBiasArray " + res.label)
        plotsProgress()
        axes[9].plot(timeArray, np.rad2deg(res.estimatedBiasArray[:, 1]), label="estimatedBiasArray " + res.label)
        plotsProgress()
        axes[10].plot(timeArray, np.rad2deg(res.estimatedBiasArray[:, 2]), label="estimatedBiasArray " + res.label)
        plotsProgress()

    axes[0].plot(timeArray, trueQuaternionArray[:, 0], 'k--', label="trueQuaternion qw")
    plotsProgress()
    axes[1].plot(timeArray, trueQuaternionArray[:, 1], 'k--', label="trueQuaternion qx")
    plotsProgress()
    axes[2].plot(timeArray, trueQuaternionArray[:, 2], 'k--', label="trueQuaternion qy")
    plotsProgress()
    axes[3].plot(timeArray, trueQuaternionArray[:, 3], 'k--', label="trueQuaternion qz")
    plotsProgress()
    axes[4].plot(timeArray, np.linalg.norm(trueQuaternionArray, axis=-1), 'k--', label="trueQuaternion Norm")
    plotsProgress()

    # axes[5].plot(timeArray, np.rad2deg([0] * simConfig.sampleSize), 'k--', label="Roll")
    axes[5].plot(timeArray, np.rad2deg(trueEulerArray[:, 0]), 'k--', label="Roll")
    plotsProgress()
    axes[6].plot(timeArray, np.rad2deg(trueEulerArray[:, 1]), 'k--', label="Pitch")
    plotsProgress()
    axes[7].plot(timeArray, np.rad2deg(trueEulerArray[:, 2]), 'k--', label="Yaw")
    plotsProgress()
    # axes[7].plot(timeArray, np.rad2deg([0] * simConfig.sampleSize), 'k--', label="Yaw")
    axes[8].plot(timeArray, np.rad2deg(trueBiases[:, 0]), 'k--', label="trueBiases")
    plotsProgress()
    axes[9].plot(timeArray, np.rad2deg(trueBiases[:, 1]), 'k--', label="trueBiases")
    plotsProgress()
    axes[10].plot(timeArray, np.rad2deg(trueBiases[:, 2]), 'k--', label="trueBiases")
    plotsProgress()

    axes[0].set_ylim(-1.0, 1.0)
    axes[1].set_ylim(-1.0, 1.0)
    axes[2].set_ylim(-1.0, 1.0)
    axes[3].set_ylim(-1.0, 1.0)
    axes[4].set_ylim(-0.1, 1.1)
    axes[5].set_ylim(-200.0, 200.0)
    axes[6].set_ylim(-200.0, 200.0)
    axes[7].set_ylim(-200.0, 200.0)
    axes[8].set_ylim(-200.0, 200.0)
    axes[9].set_ylim(-200.0, 200.0)
    axes[10].set_ylim(-200.0, 200.0)
    plotsProgress(full=True)

    # titles = [np.reshape(["Quaternions (-)"]*4, (4,1)), np.reshape(["EulerDeg (°)"]*3, (3,1)), np.reshape(["Biais (°/s)"]*3, (3,1))]
    # titles = np.reshape(titles, (10, 1, 1))
    titles = ["Quaternions (-)"] * 4 + ["Module Quat"] + ["EulerRPY (°)"] * 3 + ["Biais (°/s)"] * 3
    for ax, title in zip(axes, titles):
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True)

    axes[10].set_xlabel("Temps (s)")
    plt.tight_layout()
    plt.show()

print("Fin plots")
exit(0)

# =============================================================================
# Bloc A : représentation 3D du trièdre IMU + gravité
# Version PyCharm-compatible
# =============================================================================

# Animation de la réalité
print("Anim trueQuaternionArray")
figTriad, aniTriad = runTriadAnimation(
    quaternionArrayToPlot=trueQuaternionArray,
    timeArray=timeArray,
    gravity=simConfig.gravity,
    labelOrientation="trueQuaternionArray",
    # exportGif=True,
    # gifFileName="trièdre_trueQuaternionArray.gif",
    exportVideo=True,
    videoFileName="trièdre_trueQuaternionArray.mp4",
    fps=25,
    dpi=120,
    showProgress=True,
)

# Animation de l'estimation
print("Anim estimatedQuaternionArray")
figTriad, aniTriad = runTriadAnimation(
    quaternionArrayToPlot=results[0].estimatedQuaternionArray,
    timeArray=timeArray,
    gravity=simConfig.gravity,
    labelOrientation=results[0].label,
    # exportGif=True,
    # gifFileName="trièdre_estimatedQuaternionArray.gif",
    exportVideo=True,
    videoFileName="trièdre_estimatedQuaternionArray.mp4",
    fps=25,
    dpi=120,
    showProgress=True,
)

# plt.show()
