#%%
import math
import numpy as np
import matplotlib.pyplot as plt

# Optionnel : configuration d'affichage
np.set_printoptions(precision=4, suppress=True)

from filterpy.kalman import KalmanFilter





TotalTime = 1
SampleSize = 10  # nombre d'instants de mesure  # <<<
TimeStep = float(TotalTime / SampleSize)  # période d'échantillonage (s)  # <<<

RandomSeed = 123
np.random.seed(RandomSeed)

# Paramètres du modèle de mouvement  # <<<

TrueInitialAlpha = 0.0  # angle initial vrai (°)        # <<<
TrueInitialAlphadot = 0.0  # vitesse angulaire initiale vraie (°/s)        # <<<
TrueInitialBias = 1.0  # biais de vitesse angulaire vrai (°/s)             # <<<
MeasurementAlphaNoiseStd = 1.0  # => R écart-type du bruit de mesure sur alpha  # <<<
MeasurementAlphadotNoiseStd = 1.0  # => R écart-type du bruit de mesure sur alphadot   # <<<
ProcessAlphaNoiseStd = 1.0  # => Q bruit de processus (sur alpha)  # <<<
ProcessAlphadotNoiseStd = 1.0  # => Q bruit de processus (sur alphadot)  # <<<
ProcessBiasNoiseStd = 1.0  # => Q bruit de processus (sur bias)  # <<<




def GenerateTrueValuesAndMeasurements():
    """
    Generate values with piecewise constant acceleration
    and noisy position measurements.
    """
    # print(f"TimeStep : {TimeStep}")

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
            currentAlphadotdot = 0.5
        elif currentTime < 60.0:
            currentAlphadotdot = -0.3
        else:
            currentAlphadotdot = 0.0

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
    kalmanFilter = KalmanFilter(dim_x=3, dim_z=2)

    kalmanFilter.x = np.array([
        [TrueInitialAlpha],
        [TrueInitialAlphadot],
        [TrueInitialBias]
    ])

    kalmanFilter.F = np.array([
        [1.0, TimeStep, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])

    kalmanFilter.H = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0]
    ])

    kalmanFilter.P = np.array([
        [500.0, 0.0, 0.0],
        [0.0, 500.0, 0.0],
        [0.0, 0.0, 500.0]
    ])

    kalmanFilter.R = np.diag([
        MeasurementAlphaNoiseStd ** 2,
        MeasurementAlphadotNoiseStd ** 2
    ])

    # q = ProcessNoiseStdAcceleration ** 2
    # dt = TimeStep
    # kalmanFilter.Q = np.array([
    #     [0.25 * dt**4 * q, 0.5 * dt**3 * q],
    #     [0.5 * dt**3 * q,  dt**2 * q]
    # ])

    kalmanFilter.Q = np.diag([
        ProcessAlphaNoiseStd ** 2,
        ProcessAlphadotNoiseStd ** 2,
        ProcessBiasNoiseStd ** 2
    ])

    return kalmanFilter

def ApplyKalmanFilterOnMeasurements(kalmanFilter: KalmanFilter,
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
        kalmanFilter.update(np.array([[measuredAlpha, measuredAlphadot]]))

        estimatedAlphaList.append(float(kalmanFilter.x[0, 0]))
        estimatedAlphadotList.append(float(kalmanFilter.x[1, 0]))
        estimatedBiasList.append(float(kalmanFilter.x[2, 0]))

    estimatedAlphaArray = np.array(estimatedAlphaList)
    estimatedAlphadotArray = np.array(estimatedAlphadotList)
    estimatedBiasArray = np.array(estimatedBiasList)

    return estimatedAlphaArray, estimatedAlphadotArray, estimatedBiasArray

