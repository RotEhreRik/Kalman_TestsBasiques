import math
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional

np.set_printoptions(precision=4, suppress=True)

from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints


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
# Classe SimulationConfig  [INCHANGÉE]
# Rôle : décrire la réalité physique simulée et générer les mesures bruitées.
# =============================================================================

class SimulationConfig:

    def __init__(
            self,
            totalTime: float = 1.0,
            sampleSize: int = 10,
            randomSeed: int = 123,
            trueInitialAlpha: float = 0.0,
            trueInitialAlphadot: float = 0.0,
            trueInitialBias: float = 1.0,
            measurementAccelNoiseStd: float = 0.2,
            measurementAlphadotNoiseStd: float = 1.0,
            gravity: float = 9.81,
            angleUnitIsDegree: bool = True,
    ):
        self.totalTime = totalTime
        self.sampleSize = sampleSize
        self.timeStep = float(totalTime / sampleSize)
        self.randomSeed = randomSeed
        self.trueInitialAlpha = trueInitialAlpha
        self.trueInitialAlphadot = trueInitialAlphadot
        self.trueInitialBias = trueInitialBias
        self.measurementAccelNoiseStd = measurementAccelNoiseStd
        self.measurementAlphadotNoiseStd = measurementAlphadotNoiseStd
        self.gravity = gravity
        self.angleUnitIsDegree = angleUnitIsDegree

        np.random.seed(self.randomSeed)

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
        trueAlphadotArray = np.zeros(self.sampleSize, dtype=float)

        currentAlpha = self.trueInitialAlpha
        currentAlphadot = self.trueInitialAlphadot

        for indexTime in range(self.sampleSize):
            currentTime = timeArray[indexTime]

            if currentTime < 20.0:
                currentAlphadotdot = 0.0
            elif currentTime < 40.0:
                currentAlphadotdot = 0.5
            elif currentTime < 60.0:
                currentAlphadotdot = -0.3
            else:
                currentAlphadotdot = 0.0

            currentAlphadot = currentAlphadot + currentAlphadotdot * self.timeStep
            currentAlpha = currentAlpha + currentAlphadot * self.timeStep
            trueAlphaArray[indexTime] = currentAlpha
            trueAlphadotArray[indexTime] = currentAlphadot

        trueAlphaRadArray = (
            np.deg2rad(trueAlphaArray) if self.angleUnitIsDegree else trueAlphaArray
        )

        accelNoiseArray = np.random.normal(0.0, self.measurementAccelNoiseStd, self.sampleSize)
        measuredAccelXArray = -self.gravity * np.sin(trueAlphaRadArray) + accelNoiseArray

        accelNoiseArray = np.random.normal(0.0, self.measurementAccelNoiseStd, self.sampleSize)
        measuredAccelYArray = self.gravity * np.cos(trueAlphaRadArray) + accelNoiseArray

        gyroNoiseArray = np.random.normal(0.0, self.measurementAlphadotNoiseStd, self.sampleSize)
        measuredAlphadotArray = trueAlphadotArray + self.trueInitialBias + gyroNoiseArray

        return (timeArray, trueAlphaArray, trueAlphadotArray,
                measuredAccelXArray, measuredAccelYArray, measuredAlphadotArray)


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
        self.angleUnitIsDegree = simConfig.angleUnitIsDegree
        self.sigmaAlpha = sigmaAlpha
        self.sigmaBeta = sigmaBeta
        self.sigmaKappa = sigmaKappa

    # --- Fonctions de modèle (dépendent uniquement de la physique) ---

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
            supposedInitialAlpha: float = 0.0,  # état initial supposé (°)
            supposedInitialAlphadot: float = 0.0,  # (°/s)
            supposedInitialBias: float = 0.0,  # (°/s)
            processAlphaNoiseStd: float = 1.0,  # => Q
            processAlphadotNoiseStd: float = 1.0,  # => Q
            processBiasNoiseStd: float = 1.0,  # => Q
            processInitialConfidenceStd: float = 300.0,  # => P₀
    ):
        self.simConfig = simConfig
        self.supposedInitialAlpha = supposedInitialAlpha
        self.supposedInitialAlphadot = supposedInitialAlphadot
        self.supposedInitialBias = supposedInitialBias
        self.measurementAccelNoiseStd = simConfig.measurementAccelNoiseStd
        self.measurementAlphadotNoiseStd = simConfig.measurementAlphadotNoiseStd
        self.processAlphaNoiseStd = processAlphaNoiseStd
        self.processAlphadotNoiseStd = processAlphadotNoiseStd
        self.processBiasNoiseStd = processBiasNoiseStd
        self.processInitialConfidenceStd = processInitialConfidenceStd

    @classmethod
    def fromBase(cls, base: "UkfParams", **overrides) -> "UkfParams":
        """
        Construit un UkfParams identique à `base`,
        en remplaçant uniquement les paramètres fournis dans overrides.
        Exemple :
            params2 = UkfParams.fromBase(params1, processBiasNoiseStd=0.1)
        """
        constructorAttrs = {                                        # [AJOUT]
            "simConfig": base.simConfig,                            # [AJOUT]
            "supposedInitialAlpha": base.supposedInitialAlpha,      # [AJOUT]
            "supposedInitialAlphadot": base.supposedInitialAlphadot,# [AJOUT]
            "supposedInitialBias": base.supposedInitialBias,        # [AJOUT]
            "processAlphaNoiseStd": base.processAlphaNoiseStd,      # [AJOUT]
            "processAlphadotNoiseStd": base.processAlphadotNoiseStd,# [AJOUT]
            "processBiasNoiseStd": base.processBiasNoiseStd,        # [AJOUT]
            "processInitialConfidenceStd": base.processInitialConfidenceStd, # [AJOUT]
        }                                                           # [AJOUT]

        validKeys = set(constructorAttrs.keys())                    # [MODIFIÉ]
        unknownKeys = set(overrides.keys()) - validKeys             # [MODIFIÉ]
        if unknownKeys:
            raise ValueError(
                f"UkfParams.fromBase() : paramètres inconnus : {unknownKeys}"
            )

        constructorAttrs.update(overrides)                          # [AJOUT]
        return cls(**constructorAttrs)

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
    estimatedAlphaArray: np.ndarray  # angles estimés
    estimatedAlphadotArray: np.ndarray  # vitesses angulaires estimées
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
            measuredAlphadotArray: np.ndarray,
            label: str = "run",
    ) -> UkfResult:
        """
        Construit un filtre UKF neuf à partir de (model, params),
        exécute la boucle predict/update sur les mesures fournies,
        et retourne un UkfResult étiqueté.
        """
        # --- Construction du filtre ---
        sigmaPoints = MerweScaledSigmaPoints(
            n=3,
            alpha=model.sigmaAlpha,
            beta=model.sigmaBeta,
            kappa=model.sigmaKappa,
        )

        ukf = UnscentedKalmanFilter(
            dim_x=3,
            dim_z=3,
            dt=model.timeStep,
            fx=model.stateTransitionFunction,
            hx=model.measurementFunction,
            points=sigmaPoints,
        )

        ukf.x = np.array([
            params.supposedInitialAlpha,
            params.supposedInitialAlphadot,
            params.supposedInitialBias,
        ], dtype=float)

        ukf.P = np.diag([params.processInitialConfidenceStd ** 2] * 3)

        ukf.R = np.diag([
            params.measurementAccelNoiseStd ** 2,
            params.measurementAccelNoiseStd ** 2,
            params.measurementAlphadotNoiseStd ** 2,
        ])

        ukf.Q = np.diag([
            params.processAlphaNoiseStd ** 2,
            params.processAlphadotNoiseStd ** 2,
            params.processBiasNoiseStd ** 2,
        ])

        # --- Boucle predict / update ---
        estimatedAlphaList = []
        estimatedAlphadotList = []
        estimatedBiasList = []

        for accelX, accelY, alphadot in zip(
                measuredAccelXArray, measuredAccelYArray, measuredAlphadotArray
        ):
            ukf.predict()
            ukf.update(np.array([accelX, accelY, alphadot], dtype=float))

            estimatedAlphaList.append(float(ukf.x[0]))
            # estimatedAlphaList.append(AngleModulo360(-180, float(ukf.x[0])))
            estimatedAlphadotList.append(float(ukf.x[1]))
            estimatedBiasList.append(float(ukf.x[2]))

        return UkfResult(
            label=label,
            params=params,
            estimatedAlphaArray=np.array(estimatedAlphaList),
            estimatedAlphadotArray=np.array(estimatedAlphadotList),
            estimatedBiasArray=np.array(estimatedBiasList),
        )


if __name__ == "__main__":

    # =============================================================================
    # EXEMPLE D'UTILISATION (cellules Jupyter)
    # =============================================================================

    # --- Cellule 1 : simulation ---
    simConfig = SimulationConfig(totalTime=100.0, sampleSize=200, trueInitialBias=1.0)
    (timeArray, trueAlphaArray, trueAlphadotArray,
     measuredAccelXArray, measuredAccelYArray, measuredAlphadotArray) = \
        simConfig.generateTrueValuesAndMeasurements()

    # --- Cellule 2 : modèle physique (une seule fois) ---
    ukfModel = UkfModel(simConfig)
    runner = UkfRunner()

    # --- Cellule 3 : plusieurs réglages, même modèle, même mesures ---
    results = []

    # Réglage de référence
    params1 = UkfParams(simConfig, processBiasNoiseStd=1.0)
    results.append(runner.run(ukfModel, params1, measuredAccelXArray,
                              measuredAccelYArray, measuredAlphadotArray,
                              label="Q_bias=1.0"))

    # Biais supposé différent — on crée un nouvel objet UkfParams
    params2 = UkfParams.fromBase(params1, processBiasNoiseStd=0.1)
    results.append(runner.run(ukfModel, params2, measuredAccelXArray,
                              measuredAccelYArray, measuredAlphadotArray,
                              label="Q_bias=0.1"))

    # État initial supposé différent
    params3 = UkfParams.fromBase(params1,supposedInitialBias=0.5)
    results.append(runner.run(ukfModel, params3, measuredAccelXArray,
                              measuredAccelYArray, measuredAlphadotArray,
                              label="InitBias=0.5"))

    # --- Cellule 4 : affichage ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for res in results:
        axes[0].plot(timeArray, res.estimatedAlphaArray, label=res.label)
        axes[1].plot(timeArray, res.estimatedAlphadotArray, label=res.label)
        axes[2].plot(timeArray, res.estimatedBiasArray, label=res.label)

    axes[0].plot(timeArray, trueAlphaArray, 'k--', label="vrai")
    axes[1].plot(timeArray, trueAlphadotArray, 'k--', label="vrai")

    for ax, title in zip(axes, ["Alpha (°)", "Alphadot (°/s)", "Biais (°/s)"]):
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True)

    axes[2].set_xlabel("Temps (s)")
    plt.tight_layout()
    plt.show()
