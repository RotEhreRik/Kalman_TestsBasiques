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
            angleUnitIsDegree: bool = True,
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
        self.trueInitialAlphadot = trueInitialAlphadot
        self.trueInitialBiasX = trueInitialBiasX
        self.trueInitialBiasY = trueInitialBiasY
        self.trueInitialBiasZ = trueInitialBiasZ
        self.measurementAccelNoiseStd = measurementAccelNoiseStd
        self.measurementGyroNoiseStd = measurementGyroNoiseStd
        self.gravity = gravity
        self.angleUnitIsDegree = angleUnitIsDegree

        np.random.seed(self.randomSeed)
        self.setAngleAccelerationProfile()



    def alphaToQuaternion(self, alpha):
        """
        Convertit l'angle vrai alpha en quaternion [qw, qx, qy, qz].
        Ici on suppose une rotation pure autour de l'axe Y,
        cohérente avec accelX = -g*sin(alpha), accelZ = g*cos(alpha).
        """
        alphaArray = np.asarray(alpha, dtype=float)
        alphaRadArray = np.deg2rad(alphaArray) if self.angleUnitIsDegree else alphaArray
        halfAlphaArray = 0.5 * alphaRadArray

        quaternionArray = np.zeros((alphaArray.size, 4), dtype=float)
        quaternionArray[:, 0] = np.cos(halfAlphaArray)  # qw
        quaternionArray[:, 1] = 0.0                     # qx
        quaternionArray[:, 2] = np.sin(halfAlphaArray)  # qy
        quaternionArray[:, 3] = 0.0                     # qz

        if np.isscalar(alpha):
            return quaternionArray[0]
        return quaternionArray

    def setAngleAccelerationProfile(self, alphadotdots: np.ndarray = [[0.0, 0.0]]):
        alphadotdots += [[1.0, 0.0]]
        self.alphaAccelerationProfile = alphadotdots


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

        alphadotdots = self.alphaAccelerationProfile

        percentIndex = 0
        for indexTime in range(self.sampleSize):
            currentTime = timeArray[indexTime]
            ratio = currentTime / self.totalTime

            if ratio > alphadotdots[percentIndex + 1][0]:
                percentIndex = percentIndex + 1
            currentAlphadotdot = alphadotdots[percentIndex][1]

            if self.angleUnitIsDegree:
                currentAlphadotdot = np.deg2rad(currentAlphadotdot)


            currentAlphadot = currentAlphadot + currentAlphadotdot * self.timeStep
            currentAlpha = currentAlpha + currentAlphadot * self.timeStep
            trueAlphaArray[indexTime] = currentAlpha
            trueAlphadotArray[indexTime] = currentAlphadot

        trueQuaternionArray = self.alphaToQuaternion(trueAlphaArray)

        trueAlphaRadArray = (
            np.deg2rad(trueAlphaArray) if self.angleUnitIsDegree else trueAlphaArray
        )

        accelNoiseX = np.random.normal(0.0, self.measurementAccelNoiseStd, self.sampleSize)
        accelNoiseY = np.random.normal(0.0, self.measurementAccelNoiseStd, self.sampleSize)
        accelNoiseZ = np.random.normal(0.0, self.measurementAccelNoiseStd, self.sampleSize)

        measuredAccelXArray = -self.gravity * np.sin(trueAlphaRadArray) + accelNoiseX
        measuredAccelYArray = np.zeros(self.sampleSize, dtype=float) + accelNoiseY
        measuredAccelZArray = self.gravity * np.cos(trueAlphaRadArray) + accelNoiseZ

        gyroNoiseX = np.random.normal(0.0, self.measurementGyroNoiseStd, self.sampleSize)
        gyroNoiseY = np.random.normal(0.0, self.measurementGyroNoiseStd, self.sampleSize)
        gyroNoiseZ = np.random.normal(0.0, self.measurementGyroNoiseStd, self.sampleSize)

        measuredGyroXArray = 0.0 + self.trueInitialBiasX + gyroNoiseX
        measuredGyroYArray = trueAlphadotArray + self.trueInitialBiasY + gyroNoiseY
        measuredGyroZArray = 0.0 + self.trueInitialBiasZ + gyroNoiseZ

        return (
            timeArray,
            trueAlphaArray,
            trueAlphadotArray,
            trueQuaternionArray,
            measuredAccelXArray,
            measuredAccelYArray,
            measuredAccelZArray,
            measuredGyroXArray,
            measuredGyroYArray,
            measuredGyroZArray,
        )


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
        sigmaAlpha: float = 0.5, # paramètre alpha des sigma-points Merwe
        sigmaBeta: float = 2.0, # paramètre beta
        sigmaKappa: float = 0.0, # paramètre kappa
    ):
        self.timeStep = simConfig.timeStep
        self.gravity = simConfig.gravity
        self.angleUnitIsDegree = simConfig.angleUnitIsDegree
        self.sigmaAlpha = sigmaAlpha
        self.sigmaBeta = sigmaBeta
        self.sigmaKappa = sigmaKappa
        self.lastGyroInput = np.zeros(3, dtype=float)

    def setCurrentGyroMeasurement(self, gyroMeas):
        self.lastGyroInput = np.array(gyroMeas, dtype=float)

    def normalizeQuaternion(self, q):
        qNorm = np.linalg.norm(q)
        if qNorm <= 0.0:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        return q / qNorm

    def quaternionConjugate(self, q):
        return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)

    def quaternionMultiply(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        ], dtype=float)

    def rotateVectorWorldToBody(self, q, vWorld):
        q = self.normalizeQuaternion(q)
        vQuat = np.array([0.0, vWorld[0], vWorld[1], vWorld[2]], dtype=float)
        qConj = self.quaternionConjugate(q)
        vBodyQuat = self.quaternionMultiply(
            self.quaternionMultiply(qConj, vQuat),
            q
        )
        return vBodyQuat[1:]

    def quaternionToEulerDeg(self, q):
        q = self.normalizeQuaternion(q)
        w, x, y, z = q

        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp)

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.rad2deg(np.array([roll, pitch, yaw], dtype=float))

    # --- Fonctions de modèle (dépendent uniquement de la physique) ---

    # f(x, dt)
    def stateTransitionFunction(self, x, dt):
        q = self.normalizeQuaternion(x[0:4])
        b = x[4:7].copy()                                  # [MODIFIÉ]

        omega = self.lastGyroInput - b
        wx, wy, wz = omega

        if self.angleUnitIsDegree:
            wx, wy, wz = np.deg2rad([wx, wy, wz])

        Omega = np.array([
            [0.0, -wx, -wy, -wz],
            [wx, 0.0, wz, -wy],
            [wy, -wz, 0.0, wx],
            [wz, wy, -wx, 0.0],
        ], dtype=float)

        qDot = 0.5 * Omega @ q
        qNext = self.normalizeQuaternion(q + qDot * dt)

        return np.hstack((qNext, b))

    # h(x)
    def measurementFunction(self, x):
        q = self.normalizeQuaternion(x[0:4])              # [MODIFIÉ]
        # b = x[4:7]                                      # [SUPPRIMÉ / inutile ici]

        gravityWorld = np.array([0.0, 0.0, self.gravity], dtype=float)
        gravityBody = self.rotateVectorWorldToBody(q, gravityWorld)
        accelPred = gravityBody

        return accelPred                                  # [MODIFIÉ]

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
            processQuaternionNoiseStd: float = 0.001,
            # processAlphadotNoiseStd: float = 1.0,  # => Q
            processBiasNoiseStd: float = 0.01,  # => Q
            processInitialConfidenceStd: float = 1.0,  # => P₀
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
        # self.processAlphadotNoiseStd = processAlphadotNoiseStd
        self.processBiasNoiseStd = processBiasNoiseStd
        self.processInitialConfidenceStd = processInitialConfidenceStd
        self.label = label

    @classmethod
    def fromBase(cls, base: "UkfParams", **overrides) -> "UkfParams":
        """
        Construit un UkfParams identique à `base`,
        en remplaçant uniquement les paramètres fournis dans overrides.
        Exemple :
            params2 = UkfParams.fromBase(params1, processBiasNoiseStd=0.1)
        """
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

        validKeys = set(constructorAttrs.keys())  # [MODIFIÉ]
        unknownKeys = set(overrides.keys()) - validKeys  # [MODIFIÉ]
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
    estimatedEulerDegArray: np.ndarray
    estimatedBiasArray: np.ndarray  # biais estimés


# =============================================================================
# Classe UkfRunner  [MODIFIÉE]
# Rôle : assembler UkfModel + UkfParams pour construire un filtre frais,
#        exécuter la boucle predict/update, retourner un UkfResult.
#        Sans état propre : peut être réutilisé pour n'importe quelle
#        combinaison (model, params, mesures).
# =============================================================================

class UkfRunner:


    def ensurePositiveDefiniteCovariance(self, covarianceMatrix: np.ndarray) -> np.ndarray:
        """
        Force une covariance symétrique définie positive
        par symétrisation + léger jitter diagonal.
        """
        symmetricMatrix = 0.5 * (covarianceMatrix + covarianceMatrix.T)   # [AJOUT]
        jitterValue = 1e-9                                                # [AJOUT]
        identityMatrix = np.eye(symmetricMatrix.shape[0], dtype=float)    # [AJOUT]

        for _ in range(10):                                               # [AJOUT]
            try:
                np.linalg.cholesky(symmetricMatrix)
                return symmetricMatrix
            except np.linalg.LinAlgError:
                symmetricMatrix = symmetricMatrix + jitterValue * identityMatrix
                jitterValue *= 10.0

        raise np.linalg.LinAlgError(
            "Impossible de rendre la covariance définie positive"
        )

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
            dim_z=3,                                      # [MODIFIÉ]
            dt=model.timeStep,
            fx=model.stateTransitionFunction,
            hx=model.measurementFunction,
            points=sigmaPoints,
        )

        initialQuaternion = model.normalizeQuaternion(params.supposedInitialQuaternion)

        ukf.x = np.hstack((
            initialQuaternion,
            np.array([
                params.supposedInitialBiasX,
                params.supposedInitialBiasY,
                params.supposedInitialBiasZ,
            ], dtype=float)
        ))

        ukf.P = np.diag([params.processInitialConfidenceStd ** 2] * 7)

        ukf.R = np.diag([                                # [MODIFIÉ]
            params.measurementAccelNoiseStd ** 2,
            params.measurementAccelNoiseStd ** 2,
            params.measurementAccelNoiseStd ** 2,
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

        ukf.P = self.ensurePositiveDefiniteCovariance(ukf.P)              # [AJOUT]

        # --- Boucle predict / update ---
        estimatedQuaternionList = []
        estimatedEulerDegList = []
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

            # ukf.P = self.ensurePositiveDefiniteCovariance(ukf.P)  # [AJOUT]
            ukf.predict()
            # ukf.x[0:4] = model.normalizeQuaternion(ukf.x[0:4])      # [MODIFIÉ]

            # ukf.P = self.ensurePositiveDefiniteCovariance(ukf.P)  # [AJOUT]
            ukf.update(np.array([accelX, accelY, accelZ], dtype=float))  # [MODIFIÉ]
            ukf.x[0:4] = model.normalizeQuaternion(ukf.x[0:4])      # [MODIFIÉ]

            # ukf.P = self.ensurePositiveDefiniteCovariance(ukf.P)          # [AJOUT]

            qEst = ukf.x[0:4].copy()
            bEst = ukf.x[4:7].copy()
            eulerDegEst = model.quaternionToEulerDeg(qEst)

            estimatedQuaternionList.append(qEst)
            estimatedEulerDegList.append(eulerDegEst)
            estimatedBiasList.append(bEst)

        return UkfResult(
            label=label,
            params=params,
            estimatedQuaternionArray=np.array(estimatedQuaternionList),
            estimatedEulerDegArray=np.array(estimatedEulerDegList),
            estimatedBiasArray=np.array(estimatedBiasList),
        )


if __name__ == "__main__":

    # =============================================================================
    # EXEMPLE D'UTILISATION (cellules Jupyter)
    # =============================================================================

    # totalTime: float = 1.0,
    # sampleSize: int = 10,
    # randomSeed: int = 123,
    # trueInitialAlpha: float = 0.0,
    # trueInitialAlphaDot: float = 0.0,
    # trueInitialBiasX: float = 0.0,
    # trueInitialBiasY: float = 0.0,
    # trueInitialBiasZ: float = 0.0,
    # measurementAccelNoiseStd: float = 0.2,
    # measurementGyroNoiseStd: float = 1.0,
    # gravity: float = 9.81,
    # angleUnitIsDegree: bool = True,


    # --- Cellule 1 : simulation ---
    simConfig = SimulationConfig(
        totalTime=100.0,
        timeStep=0.5,
        trueInitialAlpha = 45.0,
        trueInitialAlphadot = 0.0,
        trueInitialBiasX=10.0,
        trueInitialBiasY=15.0,
        trueInitialBiasZ=20.0,
        measurementAccelNoiseStd = 0.01,
        measurementGyroNoiseStd = 0.01,
    )
    # trueBiases = np.full(simConfig.sampleSize,[[simConfig.trueInitialBiasX, simConfig.trueInitialBiasY,simConfig.trueInitialBiasZ]])
    trueBiases = np.tile(
        [simConfig.trueInitialBiasX,
         simConfig.trueInitialBiasY,
         simConfig.trueInitialBiasZ],
        (simConfig.sampleSize, 1))

    simConfig.setAngleAccelerationProfile(
        [
            [0.0, 0.0],
            [.1, 15.0],
            [.3, -15.0],
            [.5, -15.0],
            [.7, 15.0],
            [.9, 0.0],
        ]
    )

    (
        timeArray,
        trueAlphaArray,
        trueAlphadotArray,
        trueQuaternionArray,
        measuredAccelXArray,
        measuredAccelYArray,
        measuredAccelZArray,
        measuredGyroXArray,
        measuredGyroYArray,
        measuredGyroZArray,
    ) = simConfig.generateTrueValuesAndMeasurements()

    # --- Cellule 2 : modèle physique (une seule fois) ---
    ukfModel = UkfModel(simConfig)
    runner = UkfRunner()

    # Réglage de référence
    paramsRef = UkfParams(
        simConfig,
        # processQuaternionNoiseStd=0.000001,
        # processBiasNoiseStd=0.1
        processQuaternionNoiseStd=0.001,
        processBiasNoiseStd=0.001
    )

    # paramsSweep = UkfParams.createSweepParams(
    #     paramsRef,
    #     "processBiasNoiseStd",
    #     [0.001, 0.01, 0.1]
    # )

    paramsSweep = UkfParams.createSweepParams(
        paramsRef,
        "processBiasNoiseStd",
        [0.001]
    )

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

    # # --- Cellule 4 : affichage ---
    # fig, axes = plt.subplots(3, 1, figsize=(12, 24), sharex=True)
    # for res in results:
    #     axes[0].plot(timeArray, res.estimatedQuaternionArray, label="estimatedQuaternionArray "+res.label)
    #     axes[1].plot(timeArray, res.estimatedEulerArray, label="estimatedEulerArray "+res.label)
    #     axes[2].plot(timeArray, res.estimatedBiasArray, label="estimatedBiasArray "+res.label)
    #
    # axes[1].plot(timeArray, trueAlphaArray, 'k--', label="trueAlphaArray")
    # axes[2].plot(timeArray, trueBiases, 'k--', label="trueBiases")
    #
    # for ax, title in zip(axes, ["Quaternions (-)", "EulerDeg (°)", "Biais (°/s)"]):
    #     ax.set_ylabel(title)
    #     ax.legend()
    #     ax.grid(True)
    #
    # axes[2].set_xlabel("Temps (s)")
    # plt.tight_layout()
    # plt.show()

    # --- Cellule 4 : affichage ---
    fig, axes = plt.subplots(11, 1, figsize=(12, 24), sharex=True)
    for res in results:
        axes[0].plot(timeArray, res.estimatedQuaternionArray[:,0], label="estimatedQuaternionArray "+res.label)
        axes[1].plot(timeArray, res.estimatedQuaternionArray[:,1], label="estimatedQuaternionArray "+res.label)
        axes[2].plot(timeArray, res.estimatedQuaternionArray[:,2], label="estimatedQuaternionArray "+res.label)
        axes[3].plot(timeArray, res.estimatedQuaternionArray[:,3], label="estimatedQuaternionArray "+res.label)
        axes[4].plot(timeArray, np.linalg.norm(res.estimatedQuaternionArray, axis=-1), label="estimatedQuaternionArray "+res.label)
        axes[5].plot(timeArray, res.estimatedEulerDegArray[:,0], label="estimatedEulerArray "+res.label)
        axes[6].plot(timeArray, res.estimatedEulerDegArray[:,1], label="estimatedEulerArray "+res.label)
        axes[7].plot(timeArray, res.estimatedEulerDegArray[:,2], label="estimatedEulerArray "+res.label)
        axes[8].plot(timeArray, res.estimatedBiasArray[:,0], label="estimatedBiasArray "+res.label)
        axes[9].plot(timeArray, res.estimatedBiasArray[:,1], label="estimatedBiasArray "+res.label)
        axes[10].plot(timeArray, res.estimatedBiasArray[:,2], label="estimatedBiasArray "+res.label)

    axes[0].plot(timeArray, trueQuaternionArray[:,0], 'k--', label="trueQuaternion qw")
    axes[1].plot(timeArray, trueQuaternionArray[:,1], 'k--', label="trueQuaternion qx")
    axes[2].plot(timeArray, trueQuaternionArray[:,2], 'k--', label="trueQuaternion qy")
    axes[3].plot(timeArray, trueQuaternionArray[:,3], 'k--', label="trueQuaternion qz")
    axes[4].plot(timeArray, np.linalg.norm(trueQuaternionArray, axis=-1), 'k--', label="trueQuaternion Norm")

    axes[5].plot(timeArray, [0]*simConfig.sampleSize, 'k--', label="trueAlphaArray")
    axes[6].plot(timeArray, trueAlphaArray, 'k--', label="trueAlphaArray")
    axes[7].plot(timeArray, [0]*simConfig.sampleSize, 'k--', label="trueAlphaArray")
    axes[8].plot(timeArray, trueBiases[:,0], 'k--', label="trueBiases")
    axes[9].plot(timeArray, trueBiases[:,1], 'k--', label="trueBiases")
    axes[10].plot(timeArray, trueBiases[:,2], 'k--', label="trueBiases")

    axes[0].set_ylim(-1.0,1.0)
    axes[1].set_ylim(-1.0,1.0)
    axes[2].set_ylim(-1.0,1.0)
    axes[3].set_ylim(-1.0,1.0)
    axes[4].set_ylim(-0.1,1.1)
    axes[5].set_ylim(-180.0,180.0)
    axes[6].set_ylim(-180.0,180.0)
    axes[7].set_ylim(-180.0,180.0)
    axes[8].set_ylim(-180.0,180.0)
    axes[9].set_ylim(-180.0,180.0)
    axes[10].set_ylim(-180.0,180.0)

    # titles = [np.reshape(["Quaternions (-)"]*4, (4,1)), np.reshape(["EulerDeg (°)"]*3, (3,1)), np.reshape(["Biais (°/s)"]*3, (3,1))]
    # titles = np.reshape(titles, (10, 1, 1))
    titles = ["Quaternions (-)"]*4 + ["Module Quat"] + ["EulerDeg (°)"]*3 + ["Biais (°/s)"]*3
    for ax, title in zip(axes, titles):
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True)

    axes[10].set_xlabel("Temps (s)")
    plt.tight_layout()
    plt.show()
