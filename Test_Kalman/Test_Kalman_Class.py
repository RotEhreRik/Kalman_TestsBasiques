import math
import numpy as np

import matplotlib

# matplotlib.use("TkAgg") # à placer avant import matplotlib.pyplot as plt
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


def normalizeVector(v):
    v = np.asarray(v, dtype=float)
    vNorm = np.linalg.norm(v)
    if vNorm <= 0.0:
        return np.array([0.0, 1.0, 0.0], dtype=float)
    return v / vNorm


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

    RollPitchYaw = np.stack((roll, pitch, yaw), axis=-1)
    return RollPitchYaw


def angleAxisToQuaternion(alpha, rotationAxis):
    alphaArray = np.atleast_1d(np.asarray(alpha, dtype=float))
    halfAlphaArray = 0.5 * alphaArray

    quaternionArray = np.zeros((alphaArray.size, 4), dtype=float)

    ux, uy, uz = rotationAxis
    quaternionArray[:, 0] = np.cos(halfAlphaArray)
    quaternionArray[:, 1] = ux * np.sin(halfAlphaArray)
    quaternionArray[:, 2] = uy * np.sin(halfAlphaArray)
    quaternionArray[:, 3] = uz * np.sin(halfAlphaArray)

    if np.isscalar(alpha):
        return quaternionArray[0]
    return quaternionArray


# =============================================================================
# ADDED : structures de données communes
# =============================================================================

@dataclass
class MeasurementSequence:
    TimeArray: np.ndarray
    MeasuredAccelArray: np.ndarray  # shape = (SampleSize, 3)
    MeasuredGyroArray: np.ndarray  # shape = (SampleSize, 3)

    def __post_init__(self):
        self.TimeArray = np.asarray(self.TimeArray, dtype=float)
        self.MeasuredAccelArray = np.asarray(self.MeasuredAccelArray, dtype=float)
        self.MeasuredGyroArray = np.asarray(self.MeasuredGyroArray, dtype=float)

        if self.MeasuredAccelArray.ndim != 2 or self.MeasuredAccelArray.shape[1] != 3:
            raise ValueError("MeasuredAccelArray doit être de forme (SampleSize, 3).")

        if self.MeasuredGyroArray.ndim != 2 or self.MeasuredGyroArray.shape[1] != 3:
            raise ValueError("MeasuredGyroArray doit être de forme (SampleSize, 3).")

        if len(self.TimeArray) != self.MeasuredAccelArray.shape[0]:
            raise ValueError("TimeArray et MeasuredAccelArray n'ont pas la même longueur.")

        if len(self.TimeArray) != self.MeasuredGyroArray.shape[0]:
            raise ValueError("TimeArray et MeasuredGyroArray n'ont pas la même longueur.")

    @property
    def SampleSize(self):
        return len(self.TimeArray)

    # ADDED : alias de compatibilité / confort
    @property
    def MeasuredAccelXArray(self):
        return self.MeasuredAccelArray[:, 0]

    @property
    def MeasuredAccelYArray(self):
        return self.MeasuredAccelArray[:, 1]

    @property
    def MeasuredAccelZArray(self):
        return self.MeasuredAccelArray[:, 2]

    @property
    def MeasuredGyroXArray(self):
        return self.MeasuredGyroArray[:, 0]

    @property
    def MeasuredGyroYArray(self):
        return self.MeasuredGyroArray[:, 1]

    @property
    def MeasuredGyroZArray(self):
        return self.MeasuredGyroArray[:, 2]


@dataclass
class SimulationTruthData:
    TimeArray: np.ndarray
    TrueAlphaArray: np.ndarray
    TrueAlphaDotArray: np.ndarray
    TrueQuaternionArray: np.ndarray
    TrueBiasArray: np.ndarray


# =============================================================================
# MODIFIED : chargement CSV -> MeasurementSequence
# =============================================================================

def loadCSVRecord(fileName: str) -> MeasurementSequence:
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
        fileName,
        delimiter=",",
        skiprows=1,
        unpack=True
    )

    measuredAccelArray = np.column_stack((
        measuredAccelXArray,
        measuredAccelYArray,
        measuredAccelZArray,
    ))

    measuredGyroArray = np.column_stack((
        measuredGyroXArray,
        measuredGyroYArray,
        measuredGyroZArray,
    ))

    return MeasurementSequence(
        TimeArray=timeArray,
        MeasuredAccelArray=measuredAccelArray,
        MeasuredGyroArray=measuredGyroArray,
    )


def estimateStaticImuCharacteristics(
    measurementSequence: MeasurementSequence,
    ddof: int = 1,
):
    """
    Estime les caractéristiques d'une IMU à partir d'une séquence acquise
    lorsque l'IMU est immobile.

    Estimations retournées :
    - measurementAccelNoiseStd
    - measurementGyroNoiseStd
    - accelNoiseStdPerAxis
    - gyroNoiseStdPerAxis
    - meanStaticAccel
    - meanStaticGyro
    - estimatedGravity
    - estimatedInitialBiasX
    - estimatedInitialBiasY
    - estimatedInitialBiasZ
    - sampleSize
    """

    accel = np.asarray(measurementSequence.MeasuredAccelArray, dtype=float)
    gyro = np.asarray(measurementSequence.MeasuredGyroArray, dtype=float)

    if accel.ndim != 2 or accel.shape[1] != 3:
        raise ValueError("MeasuredAccelArray doit être de forme (N, 3).")

    if gyro.ndim != 2 or gyro.shape[1] != 3:
        raise ValueError("MeasuredGyroArray doit être de forme (N, 3).")

    sampleSize = accel.shape[0]
    if sampleSize < 2:
        raise ValueError("Il faut au moins 2 échantillons pour estimer les caractéristiques.")

    meanStaticAccel = np.mean(accel, axis=0)
    meanStaticGyro = np.mean(gyro, axis=0)

    accelResiduals = accel - meanStaticAccel
    gyroResiduals = gyro - meanStaticGyro

    accelNoiseStdPerAxis = np.std(accelResiduals, axis=0, ddof=ddof)
    gyroNoiseStdPerAxis = np.std(gyroResiduals, axis=0, ddof=ddof)

    measurementAccelNoiseStd = float(np.sqrt(np.mean(accelNoiseStdPerAxis ** 2)))
    measurementGyroNoiseStd = float(np.sqrt(np.mean(gyroNoiseStdPerAxis ** 2)))

    estimatedGravity = float(np.linalg.norm(meanStaticAccel))

    return {
        "measurementAccelNoiseStd": measurementAccelNoiseStd,
        "measurementGyroNoiseStd": measurementGyroNoiseStd,
        "accelNoiseStdPerAxis": accelNoiseStdPerAxis,
        "gyroNoiseStdPerAxis": gyroNoiseStdPerAxis,
        "meanStaticAccel": meanStaticAccel,
        "meanStaticGyro": meanStaticGyro,
        "estimatedGravity": estimatedGravity,
        "estimatedInitialBiasX": float(meanStaticGyro[0]),
        "estimatedInitialBiasY": float(meanStaticGyro[1]),
        "estimatedInitialBiasZ": float(meanStaticGyro[2]),
        "sampleSize": sampleSize,
    }

# =============================================================================
# Configurations IMU
# =============================================================================

class BaseImuConfig:
    def __init__(
        self,
        timeStep: float,
        sampleSize: int,
        measurementAccelNoiseStd: float = 0.2,
        measurementGyroNoiseStd: float = 1.0,
        gravity: float = 9.81,
        # sourceFileName: str = "",
    ):
        self.timeStep = float(timeStep)
        self.sampleSize = int(sampleSize)
        self.totalTime = self.timeStep * self.sampleSize

        self.measurementAccelNoiseStd = float(measurementAccelNoiseStd)
        self.measurementGyroNoiseStd = float(measurementGyroNoiseStd)
        self.gravity = float(gravity)


# =============================================================================


class MeasurementConfig(BaseImuConfig):
    def __init__(
        self,
        timeStep: float,
        sampleSize: int,
        measurementAccelNoiseStd: float = 0.2,
        measurementGyroNoiseStd: float = 1.0,
        gravity: float = 9.81,
        estimatedInitialBiasX: float = 0.0,
        estimatedInitialBiasY: float = 0.0,
        estimatedInitialBiasZ: float = 0.0,
        # staticSampleSize: int = 0,
        # sourceFileName: str = "",
    ):
        super().__init__(
            timeStep=timeStep,
            sampleSize=sampleSize,
            measurementAccelNoiseStd=measurementAccelNoiseStd,
            measurementGyroNoiseStd=measurementGyroNoiseStd,
            gravity=gravity,
        )

        self.estimatedInitialBiasX = float(estimatedInitialBiasX)
        self.estimatedInitialBiasY = float(estimatedInitialBiasY)
        self.estimatedInitialBiasZ = float(estimatedInitialBiasZ)
        # self.staticSampleSize = int(staticSampleSize)
        # self.sourceFileName = sourceFileName

    @classmethod
    def fromStaticMeasurements(
        cls,
        measurementSequence: MeasurementSequence,
        # sourceFileName: str = "",
        ddof: int = 1,
        verbose: bool = True,
    ):
        staticStats = estimateStaticImuCharacteristics(
            measurementSequence=measurementSequence,
            ddof=ddof,
        )

        dtArray = np.diff(measurementSequence.TimeArray)
        if len(dtArray) == 0:
            raise ValueError("Il faut au moins 2 échantillons temporels pour estimer timeStep.")
        timeStep = float(np.mean(dtArray))

        obj = cls(
            timeStep=timeStep,
            sampleSize=measurementSequence.SampleSize,
            measurementAccelNoiseStd=staticStats["measurementAccelNoiseStd"],
            measurementGyroNoiseStd=staticStats["measurementGyroNoiseStd"],
            gravity=staticStats["estimatedGravity"],
            estimatedInitialBiasX=staticStats["estimatedInitialBiasX"],
            estimatedInitialBiasY=staticStats["estimatedInitialBiasY"],
            estimatedInitialBiasZ=staticStats["estimatedInitialBiasZ"],
            # sourceFileName=sourceFileName,
            # staticSampleSize=measurementSequence.SampleSize,
        )

        if verbose:
            print("=== MeasurementConfig depuis séquence fixe ===")
            # print(f"sourceFileName             = {obj.sourceFileName}")
            print(f"timeStep                   = {obj.timeStep}")
            print(f"sampleSize                 = {obj.sampleSize}")
            # print(f"staticSampleSize           = {obj.staticSampleSize}")
            print(f"measurementAccelNoiseStd   = {obj.measurementAccelNoiseStd}")
            print(f"measurementGyroNoiseStd    = {obj.measurementGyroNoiseStd}")
            print(f"gravity                    = {obj.gravity}")
            print(f"estimatedInitialBiasX      = {obj.estimatedInitialBiasX}")
            print(f"estimatedInitialBiasY      = {obj.estimatedInitialBiasY}")
            print(f"estimatedInitialBiasZ      = {obj.estimatedInitialBiasZ}")

        return obj

# =============================================================================

class SimulationConfig(BaseImuConfig):
    def __init__(
        self,
        totalTime: float = None,
        timeStep: float = None,
        sampleSize: int = None,
        randomSeed: int = 123,
        trueInitialAlpha: float = 0.0,
        trueInitialAlphadot: float = 0.0,
        rotationAxis: np.ndarray = None,
        measurementAccelNoiseStd: float = 0.2,
        measurementGyroNoiseStd: float = 1.0,
        gravity: float = 9.81,
        trueInitialBiasX: float = 0.0,
        trueInitialBiasY: float = 0.0,
        trueInitialBiasZ: float = 0.0,
        # sourceFileName: str = "",
    ):
        noneCount = [totalTime, timeStep, sampleSize].count(None)
        if noneCount != 1:
            raise ValueError(
                "Fournir exactement 2 valeurs parmi (totalTime, timeStep, sampleSize) !"
            )

        if totalTime is None:
            resolvedTimeStep = timeStep
            resolvedSampleSize = sampleSize
        elif timeStep is None:
            resolvedSampleSize = sampleSize
            resolvedTimeStep = float(totalTime / resolvedSampleSize)
        else:
            resolvedTimeStep = timeStep
            resolvedSampleSize = int(totalTime / resolvedTimeStep)

        super().__init__(
            timeStep=resolvedTimeStep,
            sampleSize=resolvedSampleSize,
            measurementAccelNoiseStd=measurementAccelNoiseStd,
            measurementGyroNoiseStd=measurementGyroNoiseStd,
            gravity=gravity,
        )

        self.randomSeed = randomSeed

        self.trueInitialAlpha = trueInitialAlpha
        self.trueInitialAlphaDot = trueInitialAlphadot
        self.trueInitialBiasX = trueInitialBiasX
        self.trueInitialBiasY = trueInitialBiasY
        self.trueInitialBiasZ = trueInitialBiasZ
        # self.sourceFileName = sourceFileName

        np.random.seed(self.randomSeed)
        self.setAngularAccelerationProfile()

        if rotationAxis is None:
            rotationAxis = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            rotationAxis = np.array(rotationAxis, dtype=float)

        axisNorm = np.linalg.norm(rotationAxis)
        if axisNorm <= 0.0:
            raise ValueError("rotationAxis ne doit pas être nul")

        self.rotationAxis = rotationAxis / axisNorm

    @classmethod
    def fromStaticMeasurements(
        cls,
        measurementSequence: MeasurementSequence,
        totalTime: float = None,
        timeStep: float = None,
        sampleSize: int = None,
        randomSeed: int = 123,
        trueInitialAlpha: float = 0.0,
        trueInitialAlphadot: float = 0.0,
        trueInitialBiasX: float = None,
        trueInitialBiasY: float = None,
        trueInitialBiasZ: float = None,
        measurementAccelNoiseStd: float = None,
        measurementGyroNoiseStd: float = None,
        gravity: float = None,
        rotationAxis: np.ndarray = None,
        ddof: int = 1,
        verbose: bool = True,
    ):
        staticStats = estimateStaticImuCharacteristics(
            measurementSequence=measurementSequence,
            ddof=ddof,
        )

        # dtArray = np.diff(measurementSequence.TimeArray)
        # if len(dtArray) == 0:
        #     raise ValueError("Il faut au moins 2 échantillons temporels pour estimer timeStep.")
        # timeStep = float(np.mean(dtArray))

        if measurementAccelNoiseStd is None:
            measurementAccelNoiseStd = staticStats["measurementAccelNoiseStd"]

        if measurementGyroNoiseStd is None:
            measurementGyroNoiseStd = staticStats["measurementGyroNoiseStd"]

        if gravity is None:
            gravity = staticStats["estimatedGravity"]

        if trueInitialBiasX is None:
            trueInitialBiasX = staticStats["estimatedInitialBiasX"]

        if trueInitialBiasY is None:
            trueInitialBiasY = staticStats["estimatedInitialBiasY"]

        if trueInitialBiasZ is None:
            trueInitialBiasZ = staticStats["estimatedInitialBiasZ"]

        obj = cls(
            totalTime=totalTime,
            timeStep=timeStep,
            sampleSize=sampleSize,
            randomSeed=randomSeed,
            trueInitialAlpha=trueInitialAlpha,
            trueInitialAlphadot=trueInitialAlphadot,
            rotationAxis=rotationAxis,
            trueInitialBiasX=trueInitialBiasX,
            trueInitialBiasY=trueInitialBiasY,
            trueInitialBiasZ=trueInitialBiasZ,
            measurementAccelNoiseStd=measurementAccelNoiseStd,
            measurementGyroNoiseStd=measurementGyroNoiseStd,
            gravity=gravity,
            # sourceFileName: str = "",
        )

        if verbose:
            print("=== MeasurementConfig depuis séquence fixe ===")
            # print(f"sourceFileName             = {obj.sourceFileName}")
            print(f"timeStep                   = {obj.timeStep}")
            print(f"sampleSize                 = {obj.sampleSize}")
            # print(f"staticSampleSize           = {obj.staticSampleSize}")
            print(f"measurementAccelNoiseStd   = {obj.measurementAccelNoiseStd}")
            print(f"measurementGyroNoiseStd    = {obj.measurementGyroNoiseStd}")
            print(f"gravity                    = {obj.gravity}")
            print(f"trueInitialBiasX           = {obj.trueInitialBiasX}")
            print(f"trueInitialBiasY           = {obj.trueInitialBiasY}")
            print(f"trueInitialBiasZ           = {obj.trueInitialBiasZ}")

        return obj

    def setAngularAccelerationProfile(self, alphaAccelerationProfile: np.ndarray = [[0.0, 0.0]]):
        alphaAccelerationProfile = alphaAccelerationProfile + [[1.0, 0.0]]
        alphaAccelerationProfile = np.array(alphaAccelerationProfile, dtype=float)
        self.alphaAccelerationProfile = alphaAccelerationProfile

    def generateTrueValues(self):
        print(f"TimeStep : {self.timeStep}")

        timeArray = np.arange(self.sampleSize, dtype=float) * self.timeStep
        trueAlphaArray = np.zeros(self.sampleSize, dtype=float)
        trueAlphaDotArray = np.zeros(self.sampleSize, dtype=float)

        currentAlpha = self.trueInitialAlpha
        currentAlphaDot = self.trueInitialAlphaDot

        alphaDotDots = self.alphaAccelerationProfile

        percentIndex = 0
        for indexTime in range(self.sampleSize):
            currentTime = timeArray[indexTime]
            timeRatio = currentTime / self.totalTime

            if timeRatio > alphaDotDots[percentIndex + 1][0]:
                percentIndex = percentIndex + 1

            currentAlphaDotDot = alphaDotDots[percentIndex][1]
            currentAlphaDot = currentAlphaDot + currentAlphaDotDot * self.timeStep
            currentAlpha = currentAlpha + currentAlphaDot * self.timeStep

            trueAlphaArray[indexTime] = currentAlpha
            trueAlphaDotArray[indexTime] = currentAlphaDot

        trueQuaternionArray = angleAxisToQuaternion(trueAlphaArray, self.rotationAxis)

        gyroBias = np.array([
            self.trueInitialBiasX,
            self.trueInitialBiasY,
            self.trueInitialBiasZ,
        ], dtype=float)

        trueBiasArray = np.tile(gyroBias, (self.sampleSize, 1))

        return SimulationTruthData(
            TimeArray=timeArray,
            TrueAlphaArray=trueAlphaArray,
            TrueAlphaDotArray=trueAlphaDotArray,
            TrueQuaternionArray=trueQuaternionArray,
            TrueBiasArray=trueBiasArray,
        )

    def generateMeasurements(self, truthData: SimulationTruthData):
        gravityWorld = np.array([0.0, 0.0, self.gravity], dtype=float)

        trueAccelArray = np.array([
            rotateVectorWorldToBody(q, gravityWorld)
            for q in truthData.TrueQuaternionArray
        ])

        accelNoise = np.random.normal(
            0.0,
            self.measurementAccelNoiseStd,
            size=(self.sampleSize, 3)
        )
        measuredAccelArray = trueAccelArray + accelNoise

        trueGyroArray = truthData.TrueAlphaDotArray[:, None] * self.rotationAxis[None, :]

        gyroNoise = np.random.normal(
            0.0,
            self.measurementGyroNoiseStd,
            size=(self.sampleSize, 3)
        )

        gyroBias = truthData.TrueBiasArray
        measuredGyroArray = trueGyroArray + gyroBias + gyroNoise

        return MeasurementSequence(
            TimeArray=truthData.TimeArray,
            MeasuredAccelArray=measuredAccelArray,
            MeasuredGyroArray=measuredGyroArray,
        )

    def generateTrueValuesAndMeasurements(self):
        truthData = self.generateTrueValues()
        measurementSequence = self.generateMeasurements(truthData)
        return truthData, measurementSequence




# =============================================================================
# Classe UkfModel
# =============================================================================

class UkfModel:

    def __init__(
            self,
            imuConfig: BaseImuConfig,
            sigmaAlpha: float = 0.1,
            sigmaBeta: float = 2.0,
            sigmaKappa: float = 0.0,
    ):
        self.timeStep = imuConfig.timeStep
        self.gravity = imuConfig.gravity
        self.sigmaAlpha = sigmaAlpha
        self.sigmaBeta = sigmaBeta
        self.sigmaKappa = sigmaKappa
        self.lastGyroInput = np.zeros(3, dtype=float)

    def setCurrentGyroMeasurement(self, gyroMeas):
        self.lastGyroInput = np.array(gyroMeas, dtype=float)

    def stateTransitionFunction(self, x, dt):
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
        qNext = normalizeQuaternion(q + qDot * dt)

        return np.hstack((qNext, b))

    def measurementFunction(self, x):
        q = normalizeQuaternion(x[0:4])
        b = x[4:7]

        gravityWorld = np.array([0.0, 0.0, self.gravity], dtype=float)
        gravityBody = rotateVectorWorldToBody(q, gravityWorld)
        accelPred = gravityBody
        omegaBody = self.lastGyroInput - b
        gyroPred = omegaBody + b

        return np.hstack((accelPred, gyroPred))


# =============================================================================
# Classe UkfParams
# =============================================================================

class UkfParams:

    def __init__(
            self,
            imuConfig: BaseImuConfig,
            supposedInitialQuaternion: np.ndarray = None,
            supposedInitialBiasX: float = 0.0,
            supposedInitialBiasY: float = 0.0,
            supposedInitialBiasZ: float = 0.0,
            processQuaternionNoiseStd: float = 0.01,
            processBiasNoiseStd: float = 0.01,
            processInitialConfidenceStd: float = 300.0,
            label: str = "",
    ):
        self.imuConfig = imuConfig
        self.supposedInitialQuaternion = (
            np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
            if supposedInitialQuaternion is None
            else np.array(supposedInitialQuaternion, dtype=float)
        )

        self.supposedInitialBiasX = supposedInitialBiasX
        self.supposedInitialBiasY = supposedInitialBiasY
        self.supposedInitialBiasZ = supposedInitialBiasZ

        self.measurementAccelNoiseStd = imuConfig.measurementAccelNoiseStd
        self.measurementGyroNoiseStd = imuConfig.measurementGyroNoiseStd

        self.processQuaternionNoiseStd = processQuaternionNoiseStd
        self.processBiasNoiseStd = processBiasNoiseStd
        self.processInitialConfidenceStd = processInitialConfidenceStd
        self.label = label

    @classmethod
    def getConstructorAttrs(cls, base: "UkfParams"):
        constructorAttrs = {
            "imuConfig": base.imuConfig,
            "supposedInitialQuaternion": base.supposedInitialQuaternion,
            "supposedInitialBiasX": base.supposedInitialBiasX,
            "supposedInitialBiasY": base.supposedInitialBiasY,
            "supposedInitialBiasZ": base.supposedInitialBiasZ,
            "processQuaternionNoiseStd": base.processQuaternionNoiseStd,
            "processBiasNoiseStd": base.processBiasNoiseStd,
            "processInitialConfidenceStd": base.processInitialConfidenceStd,
            "label": base.label,
        }
        return constructorAttrs

    # @classmethod
    # def fromStaticMeasurements(
    #     cls,
    #     simConfig: SimulationConfig,
    #     measurementSequence: MeasurementSequence,
    #     supposedInitialQuaternion: np.ndarray = None,
    #     processQuaternionNoiseStd: float = 0.01,
    #     processBiasNoiseStd: float = 0.01,
    #     processInitialConfidenceStd: float = 300.0,
    #     label: str = "",
    #     ddof: int = 1,
    #     verbose: bool = True,
    # ) -> "UkfParams":
    #     """
    #     Construit des paramètres UKF en initialisant automatiquement
    #     le biais gyro supposé à partir d'une séquence IMU fixe.
    #     """
    #
    #     staticStats = estimateStaticImuCharacteristics(
    #         measurementSequence=measurementSequence,
    #         ddof=ddof,
    #     )
    #
    #     supposedInitialBiasX = staticStats["estimatedInitialBiasX"]
    #     supposedInitialBiasY = staticStats["estimatedInitialBiasY"]
    #     supposedInitialBiasZ = staticStats["estimatedInitialBiasZ"]
    #
    #     if verbose:
    #         print("=== Initialisation UkfParams depuis séquence fixe ===")
    #         print(f"supposedInitialBiasX = {supposedInitialBiasX}")
    #         print(f"supposedInitialBiasY = {supposedInitialBiasY}")
    #         print(f"supposedInitialBiasZ = {supposedInitialBiasZ}")
    #
    #     return cls(
    #         simConfig=simConfig,
    #         supposedInitialQuaternion=supposedInitialQuaternion,
    #         supposedInitialBiasX=supposedInitialBiasX,
    #         supposedInitialBiasY=supposedInitialBiasY,
    #         supposedInitialBiasZ=supposedInitialBiasZ,
    #         processQuaternionNoiseStd=processQuaternionNoiseStd,
    #         processBiasNoiseStd=processBiasNoiseStd,
    #         processInitialConfidenceStd=processInitialConfidenceStd,
    #         label=label,
    #     )

    @classmethod
    def fromBase(cls, base: "UkfParams", **overrides) -> "UkfParams":
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

    @classmethod
    def fromMeasurementConfig(
        cls,
        measurementConfig: MeasurementConfig,
        supposedInitialQuaternion: np.ndarray = None,
        processQuaternionNoiseStd: float = 0.01,
        processBiasNoiseStd: float = 0.01,
        processInitialConfidenceStd: float = 300.0,
        label: str = "",
    ):
        return cls(
            imuConfig=measurementConfig,
            supposedInitialQuaternion=supposedInitialQuaternion,
            supposedInitialBiasX=measurementConfig.estimatedInitialBiasX,
            supposedInitialBiasY=measurementConfig.estimatedInitialBiasY,
            supposedInitialBiasZ=measurementConfig.estimatedInitialBiasZ,
            processQuaternionNoiseStd=processQuaternionNoiseStd,
            processBiasNoiseStd=processBiasNoiseStd,
            processInitialConfidenceStd=processInitialConfidenceStd,
            label=label,
        )


# =============================================================================
# Classe UkfResult
# =============================================================================

@dataclass
class UkfResult:
    label: str
    params: UkfParams
    estimatedQuaternionArray: np.ndarray
    estimatedEulerArray: np.ndarray
    estimatedBiasArray: np.ndarray


# =============================================================================
# Classe UkfRunner [MODIFIÉE]
# =============================================================================

class UkfRunner:

    def run(
            self,
            model: UkfModel,
            params: UkfParams,
            measurementSequence: MeasurementSequence,  # MODIFIED
            label: str = "run",
    ) -> UkfResult:
        """
        Construit un filtre UKF neuf à partir de (model, params),
        exécute la boucle predict/update sur les mesures fournies,
        et retourne un UkfResult étiqueté.
        """
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

        estimatedQuaternionList = []
        estimatedEulerList = []
        estimatedBiasList = []

        # MODIFIED : boucle sur la séquence de mesures
        for measuredAccel, measuredGyro in zip(
                measurementSequence.MeasuredAccelArray,
                measurementSequence.MeasuredGyroArray,
        ):
            model.setCurrentGyroMeasurement(measuredGyro)

            ukf.predict()

            ukf.update(np.hstack((measuredAccel, measuredGyro)).astype(float))
            ukf.x[0:4] = normalizeQuaternion(ukf.x[0:4])

            qEst = ukf.x[0:4].copy()
            bEst = ukf.x[4:7].copy()
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


# =============================================================================
# =============================================================================
# =============================================================================
#
# EXEMPLE D'UTILISATION
#
# =============================================================================
# =============================================================================
# =============================================================================

if __name__ == "__main__":

    measurementSequenceStatic = loadCSVRecord("../Real_Data_Files/imu_data_static.csv")

    noiseStats = estimateStaticImuCharacteristics(measurementSequenceStatic)

    RUN_SIMULATION_SEULE = False
    RUN_SIMULATION_CALIBREE = False
    RUN_TRAITEMENT_REEL = True

    # ============================================================================
    # CAS 1 : simulation seule
    # ============================================================================
    if RUN_SIMULATION_SEULE:

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

        # simConfig = SimulationConfig.fromStaticMeasurements(
        #     measurementSequence=measurementSequenceStatic,
        #     totalTime=totalTime,
        #     timeStep=timeStep,
        #     trueInitialAlpha=np.deg2rad(-45.0),
        #     trueInitialAlphadot=np.deg2rad(0.0),
        #     # trueInitialBiasX=np.deg2rad(10.0),
        #     # trueInitialBiasY=np.deg2rad(15.0),
        #     # trueInitialBiasZ=np.deg2rad(20.0),
        #     rotationAxis=normalizeVector([1.0, 1.0, 0.0]),
        #     verbose=True,
        # )

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
        truthData, measurementSequence = simConfig.generateTrueValuesAndMeasurements()  # MODIFIED

        trueEulerArray = quaternionToEuler(truthData.TrueQuaternionArray)  # MODIFIED

        print("Model + Runner")
        ukfModel = UkfModel(simConfig)
        runner = UkfRunner()

        # print("Param Mesure Statique")
        # paramsMesStat = UkfParams.fromStaticMeasurements(
        #     imuConfig=simConfig,
        #     measurementSequence=measurementSequenceStatic,
        #     supposedInitialQuaternion=None,
        #     processQuaternionNoiseStd=0.001,
        #     processBiasNoiseStd=0.001,
        #     processInitialConfidenceStd=1.0,
        #     label="Depuis Mesure Statique",
        #     verbose=True,
        # )

        print("Param Réf")
        paramsRef = UkfParams(
            imuConfig = simConfig,
            supposedInitialQuaternion=None,
            supposedInitialBiasX=0.0,
            supposedInitialBiasY=0.0,
            supposedInitialBiasZ=0.0,
            processQuaternionNoiseStd=0.001,
            processBiasNoiseStd=0.001,
            processInitialConfidenceStd=1.0,
            label="Référence",
        )

        print("Param Base")
        paramsBase = UkfParams.fromBase(
            # paramsMesStat,
            base=paramsRef,
            processQuaternionNoiseStd=0.01,
            processBiasNoiseStd=0.001,
            processInitialConfidenceStd=0.1,
            label="Base",
        )

    # ============================================================================
    # CAS 2 : simulation calibrée
    # ============================================================================
    if RUN_SIMULATION_CALIBREE:

        totalTime = 100.0
        timeStep = 0.01

        print("Config")
        # simConfig = SimulationConfig(
        #     totalTime=totalTime,
        #     timeStep=timeStep,
        #     trueInitialAlpha=np.deg2rad(-45.0),
        #     trueInitialAlphadot=np.deg2rad(0.0),
        #     trueInitialBiasX=np.deg2rad(10.0),
        #     trueInitialBiasY=np.deg2rad(15.0),
        #     trueInitialBiasZ=np.deg2rad(20.0),
        #     measurementAccelNoiseStd=0.2,
        #     measurementGyroNoiseStd=np.deg2rad(10.0),
        #     rotationAxis=normalizeVector([1.0, 1.0, 0.0]),
        # )

        simConfig = SimulationConfig.fromStaticMeasurements(
            measurementSequence=measurementSequenceStatic,
            totalTime=totalTime,
            timeStep=timeStep,
            trueInitialAlpha=np.deg2rad(-45.0),
            trueInitialAlphadot=np.deg2rad(0.0),
            # trueInitialBiasX=np.deg2rad(10.0),
            # trueInitialBiasY=np.deg2rad(15.0),
            # trueInitialBiasZ=np.deg2rad(20.0),
            rotationAxis=normalizeVector([1.0, 1.0, 0.0]),
            verbose=True,
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
        truthData, measurementSequence = simConfig.generateTrueValuesAndMeasurements()  # MODIFIED

        trueEulerArray = quaternionToEuler(truthData.TrueQuaternionArray)  # MODIFIED

        print("Model + Runner")
        ukfModel = UkfModel(simConfig)
        runner = UkfRunner()

        # print("Param Mesure Statique")
        # paramsMesStat = UkfParams.fromStaticMeasurements(
        #     imuConfig=simConfig,
        #     measurementSequence=measurementSequenceStatic,
        #     supposedInitialQuaternion=None,
        #     processQuaternionNoiseStd=0.001,
        #     processBiasNoiseStd=0.001,
        #     processInitialConfidenceStd=1.0,
        #     label="Depuis Mesure Statique",
        #     verbose=True,
        # )

        print("Param Mesure Statique")
        paramsMesStat = UkfParams(
            imuConfig=simConfig,
            # measurementSequence=measurementSequenceStatic,
            supposedInitialQuaternion=None,
            processQuaternionNoiseStd=0.001,
            processBiasNoiseStd=0.001,
            processInitialConfidenceStd=1.0,
            label="Depuis Mesure Statique",
            # verbose=True,
        )


        print("Param Réf")
        paramsRef = UkfParams(
            imuConfig = simConfig,
            supposedInitialQuaternion=None,
            supposedInitialBiasX=0.0,
            supposedInitialBiasY=0.0,
            supposedInitialBiasZ=0.0,
            processQuaternionNoiseStd=0.001,
            processBiasNoiseStd=0.001,
            processInitialConfidenceStd=1.0,
            label="Référence",
        )

        print("Param Base")
        paramsBase = UkfParams.fromBase(
            base=paramsMesStat,
            # paramsRef,
            processQuaternionNoiseStd=0.01,
            processBiasNoiseStd=0.001,
            processInitialConfidenceStd=0.1,
            label="Base",
        )

    # ============================================================================
    # CAS 3 : traitement réel
    # ============================================================================
    if RUN_TRAITEMENT_REEL:
        measurementSequence = loadCSVRecord("../Real_Data_Files/imu_data_rotation.csv")

        calibConfig = MeasurementConfig.fromStaticMeasurements(
            measurementSequence=measurementSequenceStatic,
            # rotationAxis=normalizeVector([1.0, 1.0, 0.0]),
            # sourceFileName="mesures_statiques.csv",
            verbose=True,
        )

        mesConfig = MeasurementConfig(
            timeStep=calibConfig.timeStep,
            sampleSize=measurementSequence.SampleSize,
            measurementAccelNoiseStd=calibConfig.measurementAccelNoiseStd,
            measurementGyroNoiseStd=calibConfig.measurementGyroNoiseStd,
            gravity=calibConfig.gravity,
            # rotationAxis=calibConfig.rotationAxis,
            estimatedInitialBiasX=calibConfig.estimatedInitialBiasX,
            estimatedInitialBiasY=calibConfig.estimatedInitialBiasY,
            estimatedInitialBiasZ=calibConfig.estimatedInitialBiasZ,
            # sourceFileName="mesures_reelles.csv",
            # staticSampleSize=calibConfig.staticSampleSize,
        )

        ukfModel = UkfModel(mesConfig)
        runner = UkfRunner()

        paramsBase = UkfParams.fromMeasurementConfig(
            measurementConfig=mesConfig,
            supposedInitialQuaternion=None,
            processQuaternionNoiseStd=0.01,
            processBiasNoiseStd=0.001,
            processInitialConfidenceStd=0.1,
            label="Traitement réel avec calibration statique séparée",
        )

    print("Params multiples")
    paramsSweep = UkfParams.createSweepParams(
        base= paramsBase,
        paramName="processInitialConfidenceStd",
        paramValues=[.01, 0.1, 1.0],
    )

    print("Run multiples")
    results = []
    for currentParams in paramsSweep:
        currentLabel = currentParams.label
        results.append(
            runner.run(
                model=ukfModel,
                params=currentParams,
                measurementSequence=measurementSequence,  # MODIFIED
                label=currentLabel
            )
        )

    print("Plots multiples", flush=True)
    plotsProgress(init=True, total=50)
    fig, axes = plt.subplots(11, 1, figsize=(18, 36), sharex=True)
    plotsProgress()

    for res in results:
        axes[0].plot(measurementSequence.TimeArray, res.estimatedQuaternionArray[:, 0],
                     label="estimatedQuaternionArray " + res.label)
        plotsProgress()
        axes[1].plot(measurementSequence.TimeArray, res.estimatedQuaternionArray[:, 1],
                     label="estimatedQuaternionArray " + res.label)
        plotsProgress()
        axes[2].plot(measurementSequence.TimeArray, res.estimatedQuaternionArray[:, 2],
                     label="estimatedQuaternionArray " + res.label)
        plotsProgress()
        axes[3].plot(measurementSequence.TimeArray, res.estimatedQuaternionArray[:, 3],
                     label="estimatedQuaternionArray " + res.label)
        plotsProgress()
        axes[4].plot(measurementSequence.TimeArray, np.linalg.norm(res.estimatedQuaternionArray, axis=-1),
                     label="estimatedQuaternionArray " + res.label)
        plotsProgress()
        axes[5].plot(measurementSequence.TimeArray, np.rad2deg(res.estimatedEulerArray[:, 0]),
                     label="estimatedEulerArray " + res.label)
        plotsProgress()
        axes[6].plot(measurementSequence.TimeArray, np.rad2deg(res.estimatedEulerArray[:, 1]),
                     label="estimatedEulerArray " + res.label)
        plotsProgress()
        axes[7].plot(measurementSequence.TimeArray, np.rad2deg(res.estimatedEulerArray[:, 2]),
                     label="estimatedEulerArray " + res.label)
        plotsProgress()
        axes[8].plot(measurementSequence.TimeArray, np.rad2deg(res.estimatedBiasArray[:, 0]),
                     label="estimatedBiasArray " + res.label)
        plotsProgress()
        axes[9].plot(measurementSequence.TimeArray, np.rad2deg(res.estimatedBiasArray[:, 1]),
                     label="estimatedBiasArray " + res.label)
        plotsProgress()
        axes[10].plot(measurementSequence.TimeArray, np.rad2deg(res.estimatedBiasArray[:, 2]),
                      label="estimatedBiasArray " + res.label)
        plotsProgress()


    # axes[0].plot(truthData.TimeArray, truthData.TrueQuaternionArray[:, 0], 'k--', label="trueQuaternion qw")
    # plotsProgress()
    # axes[1].plot(truthData.TimeArray, truthData.TrueQuaternionArray[:, 1], 'k--', label="trueQuaternion qx")
    # plotsProgress()
    # axes[2].plot(truthData.TimeArray, truthData.TrueQuaternionArray[:, 2], 'k--', label="trueQuaternion qy")
    # plotsProgress()
    # axes[3].plot(truthData.TimeArray, truthData.TrueQuaternionArray[:, 3], 'k--', label="trueQuaternion qz")
    # plotsProgress()
    # axes[4].plot(truthData.TimeArray, np.linalg.norm(truthData.TrueQuaternionArray, axis=-1), 'k--',
    #              label="trueQuaternion Norm")
    # plotsProgress()
    #
    # axes[5].plot(truthData.TimeArray, np.rad2deg(trueEulerArray[:, 0]), 'k--', label="Roll")
    # plotsProgress()
    # axes[6].plot(truthData.TimeArray, np.rad2deg(trueEulerArray[:, 1]), 'k--', label="Pitch")
    # plotsProgress()
    # axes[7].plot(truthData.TimeArray, np.rad2deg(trueEulerArray[:, 2]), 'k--', label="Yaw")
    # plotsProgress()
    # axes[8].plot(truthData.TimeArray, np.rad2deg(truthData.TrueBiasArray[:, 0]), 'k--', label="trueBiases")
    # plotsProgress()
    # axes[9].plot(truthData.TimeArray, np.rad2deg(truthData.TrueBiasArray[:, 1]), 'k--', label="trueBiases")
    # plotsProgress()
    # axes[10].plot(truthData.TimeArray, np.rad2deg(truthData.TrueBiasArray[:, 2]), 'k--', label="trueBiases")
    # plotsProgress()

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
