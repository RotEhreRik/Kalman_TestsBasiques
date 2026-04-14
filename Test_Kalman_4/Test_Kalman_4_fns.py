# %%
import math
import numpy as np
import matplotlib.pyplot as plt

# Optionnel : configuration d'affichage
np.set_printoptions(precision=4, suppress=True)

from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints

TotalTime = 1
SampleSize = 10  # nombre d'instants de mesure  # <<<
TimeStep = float(TotalTime / SampleSize)  # période d'échantillonage (s)  # <<<

RandomSeed = 123
np.random.seed(RandomSeed)

# Paramètres du modèle de mouvement  # <<<

TrueInitialAlpha = 0.0  # angle initial vrai (°)        # <<<
TrueInitialAlphadot = 0.0  # vitesse angulaire initiale vraie (°/s)        # <<<
TrueInitialBias = 1.0  # biais de vitesse angulaire vrai (°/s)             # <<<

SupposedInitialAlpha = 0.0  # angle initial supposé (°)        # <<<
SupposedInitialAlphadot = 0.0  # vitesse angulaire initiale supposé (°/s)        # <<<
SupposedInitialBias = 0.0  # biais de vitesse angulaire supposé (°/s)             # <<<

MeasurementAlphaNoiseStd = 1.0  # => R écart-type du bruit de mesure sur alpha  # <<<
MeasurementAlphadotNoiseStd = 1.0  # => R écart-type du bruit de mesure sur alphadot   # <<<

ProcessAlphaNoiseStd = 1.0  # => Q bruit de processus (sur alpha)  # <<<
ProcessAlphadotNoiseStd = 1.0  # => Q bruit de processus (sur alphadot)  # <<<
ProcessBiasNoiseStd = 1.0  # => Q bruit de processus (sur bias)  # <<<

ProcessInitialConfidenceStd = 300.0

def StateTransitionFunction(x, dt):
    alpha, alphadot, bias = x
    return np.array([
        alpha + alphadot * dt,
        alphadot,
        bias
    ])


def MeasurementFunction(x):
    alpha, alphadot, bias = x
    return np.array([
        alpha,
        alphadot + bias
    ])


def GenerateTrueValuesAndMeasurements():
    """
    Generate values with piecewise constant acceleration
    and noisy position measurements.
    """
    print(f"TimeStep : {TimeStep}")

    timeArray = np.arange(SampleSize, dtype=float) * TimeStep
    # print(f"timeArray : {timeArray}, len : {len(timeArray)}")
    trueAlphaArray = np.zeros(SampleSize, dtype=float)
    trueAlphadotArray = np.zeros(SampleSize, dtype=float)

    currentAlpha = TrueInitialAlpha
    currentAlphadot = TrueInitialAlphadot

    for indexTime in range(SampleSize):
        currentTime = timeArray[indexTime]

        # Acceleration profile (piecewise)
        if currentTime < 20.0:
            currentAlphadotdot = 0.0
        elif currentTime < 40.0:
            # currentAlphadotdot = 0.5
            currentAlphadotdot = 5.0
        elif currentTime < 60.0:
            # currentAlphadotdot = -0.3
            currentAlphadotdot = -4.8
        else:
            currentAlphadotdot = 0.0

        print(f"currentAlphadotdot: {currentAlphadotdot}")
        currentAlphadot = currentAlphadot + currentAlphadotdot * TimeStep
        currentAlpha = currentAlpha + currentAlphadot * TimeStep

        trueAlphaArray[indexTime] = currentAlpha
        trueAlphadotArray[indexTime] = currentAlphadot

    # print(f"trueAlphaArray : {trueAlphaArray}, len : {len(trueAlphaArray)}")
    # print(f"trueAlphaDotArray : {trueAlphaDotArray}, len : {len(trueAlphaDotArray)}")

    measurementNoiseArray = np.random.normal(
        loc=0.0,
        scale=MeasurementAlphaNoiseStd,
        size=SampleSize
    )
    measuredAlphaArray = trueAlphaArray + measurementNoiseArray

    measurementNoiseArray = np.random.normal(
        loc=0.0,
        scale=MeasurementAlphadotNoiseStd,
        size=SampleSize
    )
    measuredAlphadotArray = trueAlphadotArray + TrueInitialBias + measurementNoiseArray

    return timeArray, trueAlphaArray, trueAlphadotArray, measuredAlphaArray, measuredAlphadotArray


def CreateKalmanFilter():
    """

    """
    kf = KalmanFilter(dim_x=3, dim_z=2)

    kf.x = np.array([
        SupposedInitialAlpha,
        SupposedInitialAlphadot,
        SupposedInitialBias
    ], dtype=float)

    kf.F = np.array([
        [1.0, TimeStep, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=float)

    kf.H = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0]
    ], dtype=float)

    kf.P = np.diag([
        ProcessInitialConfidenceStd ** 2,
        ProcessInitialConfidenceStd ** 2,
        ProcessInitialConfidenceStd ** 2
    ])

    # kf.P = np.array([
    #     [ProcessInitialConfidenceStd**2, 0.0, 0.0],
    #     [0.0, 500.0, 0.0],
    #     [0.0, 0.0, 500.0]
    # ], dtype=float)

    kf.R = np.diag([
        MeasurementAlphaNoiseStd ** 2,
        MeasurementAlphadotNoiseStd ** 2
    ])

    # q = ProcessNoiseStdAcceleration ** 2
    # dt = TimeStep
    # kf.Q = np.array([
    #     [0.25 * dt**4 * q, 0.5 * dt**3 * q],
    #     [0.5 * dt**3 * q,  dt**2 * q]
    # ])

    kf.Q = np.diag([
        ProcessAlphaNoiseStd ** 2,
        ProcessAlphadotNoiseStd ** 2,
        ProcessBiasNoiseStd ** 2
    ])

    return kf


def CreateUnscentedKalmanFilter():
    sigmaPoints = MerweScaledSigmaPoints(
        n=3,
        alpha=0.1,
        beta=2.0,
        kappa=0.0
    )

    ukf = UnscentedKalmanFilter(
        dim_x=3,
        dim_z=2,
        dt=TimeStep,
        fx=StateTransitionFunction,
        hx=MeasurementFunction,
        points=sigmaPoints
    )

    ukf.x = np.array([
        SupposedInitialAlpha,
        SupposedInitialAlphadot,
        SupposedInitialBias
    ], dtype=float)

    ukf.P = np.diag([
        ProcessInitialConfidenceStd ** 2,
        ProcessInitialConfidenceStd ** 2,
        ProcessInitialConfidenceStd ** 2
    ])

    # ukf.P = np.array([
    #     [ProcessInitialConfidenceStd**2, 0.0, 0.0],
    #     [0.0, 500.0, 0.0],
    #     [0.0, 0.0, 500.0]
    # ], dtype=float)

    ukf.R = np.diag([
        MeasurementAlphaNoiseStd ** 2,
        MeasurementAlphadotNoiseStd ** 2
    ])

    ukf.Q = np.diag([
        ProcessAlphaNoiseStd ** 2,
        ProcessAlphadotNoiseStd ** 2,
        ProcessBiasNoiseStd ** 2
    ])

    return ukf


def ApplyKalmanFilterOnMeasurements(
        kalmanFilter: KalmanFilter,
        measuredAlphaArray: np.ndarray,
        measuredAlphadotArray: np.ndarray,
):
    """
    Apply the Kalman filter step-by-step on noisy measurements
    and return estimated position and velocity arrays.
    """
    estimatedAlphaList = []
    estimatedAlphadotList = []
    estimatedBiasList = []

    for measuredAlpha, measuredAlphadot in zip(measuredAlphaArray, measuredAlphadotArray):
        kalmanFilter.predict()
        kalmanFilter.update(np.array([measuredAlpha, measuredAlphadot], dtype=float))

        estimatedAlphaList.append(float(kalmanFilter.x[0]))
        estimatedAlphadotList.append(float(kalmanFilter.x[1]))
        estimatedBiasList.append(float(kalmanFilter.x[2]))

    estimatedAlphaArray = np.array(estimatedAlphaList)
    estimatedAlphadotArray = np.array(estimatedAlphadotList)
    estimatedBiasArray = np.array(estimatedBiasList)

    return estimatedAlphaArray, estimatedAlphadotArray, estimatedBiasArray


def ApplyUnscentedKalmanFilterOnMeasurements(
        kalmanFilter: UnscentedKalmanFilter,
        measuredAlphaArray: np.ndarray,
        measuredAlphadotArray: np.ndarray,
):
    estimatedAlphaList = []
    estimatedAlphadotList = []
    estimatedBiasList = []

    for measuredAlpha, measuredAlphadot in zip(measuredAlphaArray, measuredAlphadotArray):
        kalmanFilter.predict()
        kalmanFilter.update(np.array([measuredAlpha, measuredAlphadot], dtype=float))

        estimatedAlphaList.append(float(kalmanFilter.x[0]))
        estimatedAlphadotList.append(float(kalmanFilter.x[1]))
        estimatedBiasList.append(float(kalmanFilter.x[2]))

    estimatedAlphaArray = np.array(estimatedAlphaList)
    estimatedAlphadotArray = np.array(estimatedAlphadotList)
    estimatedBiasArray = np.array(estimatedBiasList)

    return estimatedAlphaArray, estimatedAlphadotArray, estimatedBiasArray
