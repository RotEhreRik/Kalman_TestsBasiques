import math
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)

from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints


# =============================================================================
# Fonctions utilitaires (indépendantes de tout contexte)
# =============================================================================

def Roll(minVal, maxVal, val):
    return ((val - minVal) % (maxVal - minVal) + minVal)

def Modulo(baseVal, modVal, val):
    return ((val - baseVal) % modVal + baseVal)

def AngleModulo360(baseAngle, angle):
    return Modulo(-180, 360, angle)


# =============================================================================
# Classe SimulationConfig
# Rôle : décrire la réalité physique simulée et générer les mesures bruitées.
# Usage : instancier avec les vrais paramètres du monde, puis appeler
#         generateTrueValuesAndMeasurements().
# =============================================================================

class SimulationConfig:

    def __init__(
        self,
        totalTime: float         = 1.0,
        sampleSize: int          = 10,
        randomSeed: int          = 123,
        trueInitialAlpha: float  = 0.0,    # angle initial vrai (°)
        trueInitialAlphadot: float = 0.0,  # vitesse angulaire initiale vraie (°/s)
        trueInitialBias: float   = 1.0,    # biais de vitesse angulaire vrai (°/s)
        measurementAccelNoiseStd: float    = 0.2,   # écart-type bruit accéléromètre
        measurementAlphadotNoiseStd: float = 1.0,   # écart-type bruit gyroscope
        gravity: float           = 9.81,
        angleUnitIsDegree: bool  = True,
    ):
        self.totalTime                  = totalTime
        self.sampleSize                 = sampleSize
        self.timeStep                   = float(totalTime / sampleSize)
        self.randomSeed                 = randomSeed
        self.trueInitialAlpha           = trueInitialAlpha
        self.trueInitialAlphadot        = trueInitialAlphadot
        self.trueInitialBias            = trueInitialBias
        self.measurementAccelNoiseStd   = measurementAccelNoiseStd
        self.measurementAlphadotNoiseStd = measurementAlphadotNoiseStd
        self.gravity                    = gravity
        self.angleUnitIsDegree          = angleUnitIsDegree

        np.random.seed(self.randomSeed)

    def generateTrueValuesAndMeasurements(self):
        """
        Génère les valeurs vraies (alpha, alphadot) et les mesures bruitées
        (accelX, accelY, alphadot gyro).
        Retourne : (timeArray, trueAlphaArray, trueAlphaDotArray,
                    measuredAccelXArray, measuredAccelYArray, measuredAlphadotArray)
        """
        print(f"TimeStep : {self.timeStep}")

        timeArray         = np.arange(self.sampleSize, dtype=float) * self.timeStep
        trueAlphaArray    = np.zeros(self.sampleSize, dtype=float)
        trueAlphadotArray = np.zeros(self.sampleSize, dtype=float)

        currentAlpha    = self.trueInitialAlpha
        currentAlphadot = self.trueInitialAlphadot

        for indexTime in range(self.sampleSize):
            currentTime = timeArray[indexTime]

            # Profil d'accélération angulaire par morceaux
            if currentTime < 20.0:
                currentAlphadotdot = 0.0
            elif currentTime < 40.0:
                currentAlphadotdot = 0.5
            elif currentTime < 60.0:
                currentAlphadotdot = -0.3
            else:
                currentAlphadotdot = 0.0

            currentAlphadot = currentAlphadot + currentAlphadotdot * self.timeStep
            currentAlpha    = currentAlpha    + currentAlphadot    * self.timeStep

            trueAlphaArray[indexTime]    = currentAlpha
            trueAlphadotArray[indexTime] = currentAlphadot

        if self.angleUnitIsDegree:
            trueAlphaRadArray = np.deg2rad(trueAlphaArray)
        else:
            trueAlphaRadArray = trueAlphaArray

        accelNoiseArray      = np.random.normal(0.0, self.measurementAccelNoiseStd, self.sampleSize)
        measuredAccelXArray  = -self.gravity * np.sin(trueAlphaRadArray) + accelNoiseArray

        accelNoiseArray      = np.random.normal(0.0, self.measurementAccelNoiseStd, self.sampleSize)
        measuredAccelYArray  =  self.gravity * np.cos(trueAlphaRadArray) + accelNoiseArray

        gyroNoiseArray        = np.random.normal(0.0, self.measurementAlphadotNoiseStd, self.sampleSize)
        measuredAlphadotArray = trueAlphadotArray + self.trueInitialBias + gyroNoiseArray

        return (timeArray, trueAlphaArray, trueAlphadotArray,
                measuredAccelXArray, measuredAccelYArray, measuredAlphadotArray)


# =============================================================================
# Classe UkfConfig
# Rôle : encapsuler les paramètres du filtre UKF et construire une instance
#        prête à l'emploi via buildFilter().
#        Contient aussi les fonctions de modèle (transition d'état, mesure)
#        car elles dépendent des paramètres gravity et angleUnitIsDegree.
# Usage : instancier avec les paramètres souhaités, appeler buildFilter().
#         Changer un paramètre et rappeler buildFilter() => nouveau filtre indépendant.
# =============================================================================

class UkfConfig:

    def __init__(
        self,
        simConfig: SimulationConfig,
        supposedInitialAlpha: float    = 0.0,   # angle initial supposé (°)
        supposedInitialAlphadot: float = 0.0,   # vitesse angulaire initiale supposée (°/s)
        supposedInitialBias: float     = 0.0,   # biais initial supposé (°/s)
        processAlphaNoiseStd: float    = 1.0,   # => Q sur alpha
        processAlphadotNoiseStd: float = 1.0,   # => Q sur alphadot
        processBiasNoiseStd: float     = 1.0,   # => Q sur biais
        processInitialConfidenceStd: float = 300.0,
    ):
        self.timeStep                       = simConfig.timeStep
        self.supposedInitialAlpha           = supposedInitialAlpha
        self.supposedInitialAlphadot        = supposedInitialAlphadot
        self.supposedInitialBias            = supposedInitialBias
        self.measurementAccelNoiseStd       = simConfig.measurementAccelNoiseStd
        self.measurementAlphadotNoiseStd    = simConfig.measurementAlphadotNoiseStd
        self.processAlphaNoiseStd           = processAlphaNoiseStd
        self.processAlphadotNoiseStd        = processAlphadotNoiseStd
        self.processBiasNoiseStd            = processBiasNoiseStd
        self.processInitialConfidenceStd    = processInitialConfidenceStd
        self.gravity                        = simConfig.gravity
        self.angleUnitIsDegree              = simConfig.angleUnitIsDegree

    # --- Fonctions de modèle (méthodes) ---

    def stateTransitionFunction(self, x, dt):
        """Modèle de transition d'état : alpha, alphadot, biais."""
        alpha, alphadot, bias = x
        return np.array([
            alpha + alphadot * dt,
            alphadot,
            bias
        ])

    def measurementFunction(self, x):
        """Modèle de mesure : accelX, accelY, alphadot_gyro."""
        alpha, alphadot, bias = x

        alphaRad = np.deg2rad(alpha) if self.angleUnitIsDegree else alpha

        return np.array([
            -self.gravity * np.sin(alphaRad),
             self.gravity * np.cos(alphaRad),
            alphadot + bias
        ], dtype=float)

    # --- Construction du filtre ---

    def buildFilter(self) -> UnscentedKalmanFilter:
        """
        Construit et retourne un UnscentedKalmanFilter configuré
        avec les paramètres courants de cette instance.
        Appels successifs produisent des filtres indépendants.
        """
        sigmaPoints = MerweScaledSigmaPoints(
            n=3,
            alpha=0.1,
            beta=2.0,
            kappa=0.0
        )

        ukf = UnscentedKalmanFilter(
            dim_x=3,
            dim_z=3,
            dt=self.timeStep,
            fx=self.stateTransitionFunction,
            hx=self.measurementFunction,
            points=sigmaPoints
        )

        ukf.x = np.array([
            self.supposedInitialAlpha,
            self.supposedInitialAlphadot,
            self.supposedInitialBias
        ], dtype=float)

        ukf.P = np.diag([
            self.processInitialConfidenceStd ** 2,
            self.processInitialConfidenceStd ** 2,
            self.processInitialConfidenceStd ** 2
        ])

        ukf.R = np.diag([
            self.measurementAccelNoiseStd    ** 2,
            self.measurementAccelNoiseStd    ** 2,
            self.measurementAlphadotNoiseStd ** 2
        ])

        ukf.Q = np.diag([
            self.processAlphaNoiseStd    ** 2,
            self.processAlphadotNoiseStd ** 2,
            self.processBiasNoiseStd     ** 2
        ])

        return ukf


# =============================================================================
# Classe UkfRunner
# Rôle : exécuter la boucle predict/update sur un filtre donné et des mesures
#        données. Complètement découplée des paramètres : elle reçoit un filtre
#        déjà construit (par UkfConfig.buildFilter()) et des arrays de mesures.
# Usage : instancier, appeler runOnMeasurements().
#         Permet d'appliquer le même filtre à plusieurs jeux de mesures,
#         ou plusieurs filtres au même jeu de mesures.
# =============================================================================

class UkfRunner:

    def runOnMeasurements(
        self,
        kalmanFilter: UnscentedKalmanFilter,
        measuredAccelXArray: np.ndarray,
        measuredAccelYArray: np.ndarray,
        measuredAlphadotArray: np.ndarray,
    ):
        """
        Exécute la boucle predict/update et retourne les arrays estimés
        (estimatedAlphaArray, estimatedAlphadotArray, estimatedBiasArray).
        """
        estimatedAlphaList    = []
        estimatedAlphadotList = []
        estimatedBiasList     = []

        for measuredAccelX, measuredAccelY, measuredAlphadot in zip(
            measuredAccelXArray, measuredAccelYArray, measuredAlphadotArray
        ):
            kalmanFilter.predict()
            kalmanFilter.update(
                np.array([measuredAccelX, measuredAccelY, measuredAlphadot], dtype=float)
            )

            estimatedAlphaList.append(float(kalmanFilter.x[0]))
            # estimatedAlphaList.append(AngleModulo360(-180, float(kalmanFilter.x[0])))
            estimatedAlphadotList.append(float(kalmanFilter.x[1]))
            estimatedBiasList.append(float(kalmanFilter.x[2]))

        estimatedAlphaArray    = np.array(estimatedAlphaList)
        estimatedAlphadotArray = np.array(estimatedAlphadotList)
        estimatedBiasArray     = np.array(estimatedBiasList)

        return estimatedAlphaArray, estimatedAlphadotArray, estimatedBiasArray


# =============================================================================
# EXEMPLE D'UTILISATION (cellule Jupyter)
# =============================================================================

# --- Simulation de base ---
simConfig = SimulationConfig(
    totalTime=100.0,
    sampleSize=200,
    trueInitialBias=1.0,
)
(timeArray, trueAlphaArray, trueAlphadotArray,
 measuredAccelXArray, measuredAccelYArray, measuredAlphadotArray) = \
    simConfig.generateTrueValuesAndMeasurements()

# --- Filtre avec paramètres Q conservateurs ---
ukfConfig = UkfConfig(
    timeStep=simConfig.timeStep,
    processAlphaNoiseStd=1.0,
    processAlphadotNoiseStd=1.0,
    processBiasNoiseStd=1.0,
)
runner = UkfRunner()

ukf1 = ukfConfig.buildFilter()
estimatedAlphaArray1, estimatedAlphadotArray1, estimatedBiasArray1 = \
    runner.runOnMeasurements(ukf1, measuredAccelXArray, measuredAccelYArray, measuredAlphadotArray)

# --- Même simulation, filtre avec Q différent (ex: bruit de biais plus faible) ---
ukfConfig.processBiasNoiseStd = 0.1          # on modifie un seul paramètre
ukf2 = ukfConfig.buildFilter()               # nouveau filtre indépendant
estimatedAlphaArray2, estimatedAlphadotArray2, estimatedBiasArray2 = \
    runner.runOnMeasurements(ukf2, measuredAccelXArray, measuredAccelYArray, measuredAlphadotArray)