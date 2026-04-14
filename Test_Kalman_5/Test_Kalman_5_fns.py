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

MeasurementAccelNoiseStd = 0.2  # => R écart-type du bruit de mesure sur alpha  # <<<
MeasurementAlphadotNoiseStd = 1.0  # => R écart-type du bruit de mesure sur alphadot   # <<<

ProcessAlphaNoiseStd = 1.0  # => Q bruit de processus (sur alpha)  # <<<
ProcessAlphadotNoiseStd = 1.0  # => Q bruit de processus (sur alphadot)  # <<<
ProcessBiasNoiseStd = 1.0  # => Q bruit de processus (sur bias)  # <<<

ProcessInitialConfidenceStd = 300.0

Gravity = 9.81
AngleUnitIsDegree = True

def Roll(minVal, maxVal, val):
    return((val - minVal) % (maxVal-minVal) + minVal)

def Modulo(baseVal, modVal, val):
    return((val - baseVal) % modVal + baseVal)

def AngleModulo360(baseAngle, angle):
    return Modulo(-180,360, angle)

def StateTransitionFunction(x, dt):
    alpha, alphadot, bias = x
    return np.array([
        alpha + alphadot * dt,
        alphadot,
        bias
    ])


def MeasurementFunction_OLD(x):
    alpha, alphadot, bias = x
    return np.array([
        alpha,
        alphadot + bias
    ])


def MeasurementFunction(x):
    alpha, alphadot, bias = x

    if AngleUnitIsDegree:
        alphaRad = np.deg2rad(alpha)
    else:
        alphaRad = alpha

    return np.array([
        -Gravity * np.sin(alphaRad),
        alphadot + bias
    ], dtype=float)


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
            currentAlphadotdot = 0.5
        elif currentTime < 60.0:
            # currentAlphadotdot = -0.3
            currentAlphadotdot = -0.3
        else:
            currentAlphadotdot = 0.0

        # currentAlphadotdot = 0.0

        # print(f"currentAlphadotdot: {currentAlphadotdot}")

        currentAlphadot = currentAlphadot + currentAlphadotdot * TimeStep
        currentAlpha = currentAlpha + currentAlphadot * TimeStep

        trueAlphaArray[indexTime] = currentAlpha
        trueAlphadotArray[indexTime] = currentAlphadot

    if AngleUnitIsDegree:
        trueAlphaRadArray = np.deg2rad(trueAlphaArray)
    else:
        trueAlphaRadArray = trueAlphaArray

    # print(f"trueAlphaArray : {trueAlphaArray}, len : {len(trueAlphaArray)}")
    # print(f"trueAlphaDotArray : {trueAlphaDotArray}, len : {len(trueAlphaDotArray)}")

    accelNoiseArray = np.random.normal(
        loc=0.0,
        scale=MeasurementAccelNoiseStd,
        size=SampleSize
    )
    # measuredAlphaArray = trueAlphaArray + measurementNoiseArray
    measuredAccelArray = -Gravity * np.sin(trueAlphaRadArray) + accelNoiseArray

    gyroNoiseArray = np.random.normal(
        loc=0.0,
        scale=MeasurementAlphadotNoiseStd,
        size=SampleSize
    )
    measuredAlphadotArray = trueAlphadotArray + TrueInitialBias + gyroNoiseArray

    return timeArray, trueAlphaArray, trueAlphadotArray, measuredAccelArray, measuredAlphadotArray


def CreateKalmanFilter_HID():
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

    # ukf.R = np.diag([
    #     MeasurementAlphaNoiseStd ** 2,
    #     MeasurementAlphadotNoiseStd ** 2
    # ])

    ukf.R = np.diag([
        MeasurementAccelNoiseStd ** 2,
        MeasurementAlphadotNoiseStd ** 2
    ])

    ukf.Q = np.diag([
        ProcessAlphaNoiseStd ** 2,
        ProcessAlphadotNoiseStd ** 2,
        ProcessBiasNoiseStd ** 2
    ])

    return ukf


def ApplyKalmanFilterOnMeasurements_HID(
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
        measuredAccelArray: np.ndarray,
        measuredAlphadotArray: np.ndarray,
):
    estimatedAlphaList = []
    estimatedAlphadotList = []
    estimatedBiasList = []

    for measuredAccel, measuredAlphadot in zip(measuredAccelArray, measuredAlphadotArray):
        kalmanFilter.predict()
        kalmanFilter.update(np.array([measuredAccel, measuredAlphadot], dtype=float))

        estimatedAlphaList.append(float(kalmanFilter.x[0]))
        # estimatedAlphaList.append(AngleModulo360(-180, float(kalmanFilter.x[0])))
        estimatedAlphadotList.append(float(kalmanFilter.x[1]))
        estimatedBiasList.append(float(kalmanFilter.x[2]))

    estimatedAlphaArray = np.array(estimatedAlphaList)
    estimatedAlphadotArray = np.array(estimatedAlphadotList)
    estimatedBiasArray = np.array(estimatedBiasList)

    return estimatedAlphaArray, estimatedAlphadotArray, estimatedBiasArray
