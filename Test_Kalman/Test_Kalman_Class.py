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


def setEqual3DAxes(ax, lim):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])


# =============================================================================
# ADDED : configs élémentaires
# =============================================================================

@dataclass
class PhysicalModelConfig:
    TimeStep: float
    Gravity: float = 9.81
    RotationAxis: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0], dtype=float))

    def __post_init__(self):
        self.RotationAxis = normalizeVector(self.RotationAxis)


@dataclass
class TimeConfig:
    TotalTime: Optional[float] = None
    TimeStep: Optional[float] = None
    SampleSize: Optional[int] = None

    def __post_init__(self):
        ProvidedCount = sum(value is None for value in [self.TotalTime, self.TimeStep, self.SampleSize])
        # MODIFIED : on impose qu'une seule valeur manque, comme dans votre code initial
        if ProvidedCount != 1:
            raise ValueError("Fournir exactement 2 valeurs parmi (TotalTime, TimeStep, SampleSize).")

        if self.TotalTime is None:
            self.TotalTime = self.TimeStep * self.SampleSize
        elif self.TimeStep is None:
            self.TimeStep = float(self.TotalTime / self.SampleSize)
        elif self.SampleSize is None:
            self.SampleSize = int(self.TotalTime / self.TimeStep)


@dataclass
class SimulationTruthConfig:
    TrueInitialAlpha: float = 0.0
    TrueInitialAlphaDot: float = 0.0
    TrueInitialBiasX: float = 0.0
    TrueInitialBiasY: float = 0.0
    TrueInitialBiasZ: float = 0.0
    AlphaAccelerationProfile: list = field(default_factory=lambda: [[0.0, 0.0]])


@dataclass
class MeasurementNoiseConfig:
    MeasurementAccelNoiseStd: float = 0.2
    MeasurementGyroNoiseStd: float = 1.0


@dataclass
class UkfInitialStateConfig:
    SupposedInitialQuaternion: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=float))
    SupposedInitialBiasX: float = 0.0
    SupposedInitialBiasY: float = 0.0
    SupposedInitialBiasZ: float = 0.0

    def __post_init__(self):
        self.SupposedInitialQuaternion = normalizeQuaternion(self.SupposedInitialQuaternion)


@dataclass
class UkfMeasurementNoiseConfig:
    MeasurementAccelNoiseStd: float = 0.2
    MeasurementGyroNoiseStd: float = 1.0


@dataclass
class UkfProcessNoiseConfig:
    ProcessQuaternionNoiseStd: float = 0.01
    ProcessBiasNoiseStd: float = 0.01


@dataclass
class UkfInitialCovarianceConfig:
    ProcessInitialConfidenceStd: float = 300.0


# =============================================================================
# Classe SimulationConfig [MODIFIÉE]
# Rôle : agrégat orienté simulation.
# Elle assemble la base de temps, la physique commune, la vérité simulée,
# et le bruit injecté dans les mesures simulées.
# =============================================================================

class SimulationConfig:

    def __init__(
        self,
        timeConfig: TimeConfig,
        physicalConfig: PhysicalModelConfig,
        truthConfig: SimulationTruthConfig,
        measurementNoiseConfig: MeasurementNoiseConfig,
        randomSeed: int = 123,
    ):
        # ADDED
        self.timeConfig = timeConfig
        self.physicalConfig = physicalConfig
        self.truthConfig = truthConfig
        self.measurementNoiseConfig = measurementNoiseConfig
        self.randomSeed = randomSeed

        # ADDED : alias de compatibilité pour minimiser les modifications
        self.totalTime = self.timeConfig.TotalTime
        self.timeStep = self.timeConfig.TimeStep
        self.sampleSize = self.timeConfig.SampleSize

        self.gravity = self.physicalConfig.Gravity
        self.rotationAxis = self.physicalConfig.RotationAxis

        self.trueInitialAlpha = self.truthConfig.TrueInitialAlpha
        self.trueInitialAlphaDot = self.truthConfig.TrueInitialAlphaDot
        self.trueInitialBiasX = self.truthConfig.TrueInitialBiasX
        self.trueInitialBiasY = self.truthConfig.TrueInitialBiasY
        self.trueInitialBiasZ = self.truthConfig.TrueInitialBiasZ

        self.measurementAccelNoiseStd = self.measurementNoiseConfig.MeasurementAccelNoiseStd
        self.measurementGyroNoiseStd = self.measurementNoiseConfig.MeasurementGyroNoiseStd

        np.random.seed(self.randomSeed)
        self.setAngularAccelerationProfile(self.truthConfig.AlphaAccelerationProfile)

    def alphaToQuaternion(self, alpha):
        alphaArray = np.atleast_1d(np.asarray(alpha, dtype=float))
        halfAlphaArray = 0.5 * alphaArray

        quaternionArray = np.zeros((alphaArray.size, 4), dtype=float)

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

        trueQuaternionArray = self.alphaToQuaternion(trueAlphaArray)

        gravityWorld = np.array([0.0, 0.0, self.gravity], dtype=float)
        trueAccelArray = np.array([
            rotateVectorWorldToBody(q, gravityWorld)
            for q in trueQuaternionArray
        ]).T

        accelNoise = np.random.normal(
            0.0,
            self.measurementAccelNoiseStd,
            size=(3, self.sampleSize)
        )
        measuredAccelArray = trueAccelArray + accelNoise

        trueGyroArray = (trueAlphaDotArray[None, :] * self.rotationAxis[:, None])

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

        measuredGyroArray = trueGyroArray + gyroBias[:, None] + gyroNoise

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


# =============================================================================
# Classe UkfModel [MODIFIÉE]
# Rôle : modèle physique invariant du filtre.
# =============================================================================

class UkfModel:

    def __init__(
        self,
        physicalConfig: PhysicalModelConfig,  # MODIFIED
        sigmaAlpha: float = 0.1,
        sigmaBeta: float = 2.0,
        sigmaKappa: float = 0.0,
    ):
        self.timeStep = physicalConfig.TimeStep  # MODIFIED
        self.gravity = physicalConfig.Gravity  # MODIFIED
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
# Classe UkfParams [MODIFIÉE]
# Rôle : agrégat des réglages numériques du filtre, indépendant de la simulation.
# =============================================================================

class UkfParams:

    def __init__(
        self,
        initialStateConfig: UkfInitialStateConfig,  # ADDED
        measurementNoiseConfig: UkfMeasurementNoiseConfig,  # ADDED
        processNoiseConfig: UkfProcessNoiseConfig,  # ADDED
        initialCovarianceConfig: UkfInitialCovarianceConfig,  # ADDED
        label: str = "",
    ):
        # ADDED
        self.initialStateConfig = initialStateConfig
        self.measurementNoiseConfig = measurementNoiseConfig
        self.processNoiseConfig = processNoiseConfig
        self.initialCovarianceConfig = initialCovarianceConfig
        self.label = label

        # ADDED : alias de compatibilité pour minimiser les modifications
        self.supposedInitialQuaternion = self.initialStateConfig.SupposedInitialQuaternion
        self.supposedInitialBiasX = self.initialStateConfig.SupposedInitialBiasX
        self.supposedInitialBiasY = self.initialStateConfig.SupposedInitialBiasY
        self.supposedInitialBiasZ = self.initialStateConfig.SupposedInitialBiasZ

        self.measurementAccelNoiseStd = self.measurementNoiseConfig.MeasurementAccelNoiseStd
        self.measurementGyroNoiseStd = self.measurementNoiseConfig.MeasurementGyroNoiseStd

        self.processQuaternionNoiseStd = self.processNoiseConfig.ProcessQuaternionNoiseStd
        self.processBiasNoiseStd = self.processNoiseConfig.ProcessBiasNoiseStd

        self.processInitialConfidenceStd = self.initialCovarianceConfig.ProcessInitialConfidenceStd

    @classmethod
    def getConstructorAttrs(cls, base: "UkfParams"):
        constructorAttrs = {
            "initialStateConfig": base.initialStateConfig,  # MODIFIED
            "measurementNoiseConfig": base.measurementNoiseConfig,  # MODIFIED
            "processNoiseConfig": base.processNoiseConfig,  # MODIFIED
            "initialCovarianceConfig": base.initialCovarianceConfig,  # MODIFIED
            "label": base.label,
        }
        return constructorAttrs

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

        paramsList = []

        for paramValue in paramValues:
            # MODIFIED : balayage ciblé sur quelques champs fréquents
            newInitialStateConfig = base.initialStateConfig
            newMeasurementNoiseConfig = base.measurementNoiseConfig
            newProcessNoiseConfig = base.processNoiseConfig
            newInitialCovarianceConfig = base.initialCovarianceConfig

            if paramName == "ProcessQuaternionNoiseStd":
                newProcessNoiseConfig = UkfProcessNoiseConfig(
                    ProcessQuaternionNoiseStd=paramValue,
                    ProcessBiasNoiseStd=base.processNoiseConfig.ProcessBiasNoiseStd,
                )
            elif paramName == "ProcessBiasNoiseStd":
                newProcessNoiseConfig = UkfProcessNoiseConfig(
                    ProcessQuaternionNoiseStd=base.processNoiseConfig.ProcessQuaternionNoiseStd,
                    ProcessBiasNoiseStd=paramValue,
                )
            elif paramName == "ProcessInitialConfidenceStd":
                newInitialCovarianceConfig = UkfInitialCovarianceConfig(
                    ProcessInitialConfidenceStd=paramValue
                )
            elif paramName == "MeasurementAccelNoiseStd":
                newMeasurementNoiseConfig = UkfMeasurementNoiseConfig(
                    MeasurementAccelNoiseStd=paramValue,
                    MeasurementGyroNoiseStd=base.measurementNoiseConfig.MeasurementGyroNoiseStd,
                )
            elif paramName == "MeasurementGyroNoiseStd":
                newMeasurementNoiseConfig = UkfMeasurementNoiseConfig(
                    MeasurementAccelNoiseStd=base.measurementNoiseConfig.MeasurementAccelNoiseStd,
                    MeasurementGyroNoiseStd=paramValue,
                )
            else:
                raise ValueError(
                    f"UkfParams.createSweepParams() : paramètre inconnu ou non balayable : '{paramName}'"
                )

            paramsList.append(
                cls(
                    initialStateConfig=newInitialStateConfig,
                    measurementNoiseConfig=newMeasurementNoiseConfig,
                    processNoiseConfig=newProcessNoiseConfig,
                    initialCovarianceConfig=newInitialCovarianceConfig,
                    label=base.label + f"[{paramName} = {paramValue}]",
                )
            )

        return paramsList


# =============================================================================
# Classe UkfResult [INCHANGÉE]
# =============================================================================

@dataclass
class UkfResult:
    label: str
    params: UkfParams
    estimatedQuaternionArray: np.ndarray
    estimatedEulerArray: np.ndarray
    estimatedBiasArray: np.ndarray


# =============================================================================
# Classe UkfRunner [quasi inchangée]
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
            ukf.update(np.array([accelX, accelY, accelZ, gyroX, gyroY, gyroZ], dtype=float))
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
# *********************************************************************************************************************
#
#
#
#
#
#
# *********************************************************************************************************************

if __name__ == "__main__":

    totalTime = 100.0
    timeStep = 0.01

    print("Config")

    # ADDED : nouveau découpage explicite
    timeConfig = TimeConfig(
        TotalTime=totalTime,
        TimeStep=timeStep,
    )

    physicalConfig = PhysicalModelConfig(
        TimeStep=timeConfig.TimeStep,
        Gravity=9.81,
        RotationAxis=normalizeVector([1.0, 1.0, 0.0]),
    )

    truthConfig = SimulationTruthConfig(
        TrueInitialAlpha=np.deg2rad(-45.0),
        TrueInitialAlphaDot=np.deg2rad(0.0),
        TrueInitialBiasX=np.deg2rad(10.0),
        TrueInitialBiasY=np.deg2rad(15.0),
        TrueInitialBiasZ=np.deg2rad(20.0),
        AlphaAccelerationProfile=[
            [0.0, np.deg2rad(0.0)],
            [0.1, np.deg2rad(0.30)],
            [0.3, np.deg2rad(-0.30)],
            [0.5, np.deg2rad(-0.30)],
            [0.7, np.deg2rad(0.30)],
            [0.9, np.deg2rad(0.0)],
        ],
    )

    measurementNoiseConfig = MeasurementNoiseConfig(
        MeasurementAccelNoiseStd=0.2,
        MeasurementGyroNoiseStd=np.deg2rad(10.0),
    )

    simConfig = SimulationConfig(
        timeConfig=timeConfig,
        physicalConfig=physicalConfig,
        truthConfig=truthConfig,
        measurementNoiseConfig=measurementNoiseConfig,
        randomSeed=123,
    )

    trueBiases = np.tile(
        [
            simConfig.trueInitialBiasX,
            simConfig.trueInitialBiasY,
            simConfig.trueInitialBiasZ,
        ],
        (simConfig.sampleSize, 1)
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

    trueEulerArray = quaternionToEuler(trueQuaternionArray)

    print("Model + Runner")
    ukfModel = UkfModel(physicalConfig)  # MODIFIED
    runner = UkfRunner()

    print("Param Réf")
    paramsRef = UkfParams(
        initialStateConfig=UkfInitialStateConfig(
            SupposedInitialQuaternion=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            SupposedInitialBiasX=0.0,
            SupposedInitialBiasY=0.0,
            SupposedInitialBiasZ=0.0,
        ),
        measurementNoiseConfig=UkfMeasurementNoiseConfig(
            MeasurementAccelNoiseStd=measurementNoiseConfig.MeasurementAccelNoiseStd,
            MeasurementGyroNoiseStd=measurementNoiseConfig.MeasurementGyroNoiseStd,
        ),
        processNoiseConfig=UkfProcessNoiseConfig(
            ProcessQuaternionNoiseStd=0.001,
            ProcessBiasNoiseStd=0.001,
        ),
        initialCovarianceConfig=UkfInitialCovarianceConfig(
            ProcessInitialConfidenceStd=1.0,
        ),
        label="Référence",
    )

    print("Param Base")
    paramsBase = UkfParams(
        initialStateConfig=paramsRef.initialStateConfig,
        measurementNoiseConfig=paramsRef.measurementNoiseConfig,
        processNoiseConfig=UkfProcessNoiseConfig(
            ProcessQuaternionNoiseStd=0.01,
            ProcessBiasNoiseStd=0.001,
        ),
        initialCovarianceConfig=UkfInitialCovarianceConfig(
            ProcessInitialConfidenceStd=0.1,
        ),
        label="Base",
    )

    print("Params multiples")
    paramsSweep = UkfParams.createSweepParams(
        paramsBase,
        "ProcessInitialConfidenceStd",
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
