import math
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional

np.set_printoptions(precision=4, suppress=True)

from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints


# Class_7_v2.5->Class_8_v1 : toutes vares et tous calculs en radians


# =============================================================================
# Fonctions utilitaires (indépendantes de tout contexte)
# =============================================================================

def User2Rad_HID(userUsesDegree: bool, angle: float) -> float:
    return np.deg2rad(angle) if userUsesDegree else angle


def Rad2User_HID(userUsesDegree: bool, angle: float) -> float:
    return (np.rad2deg(angle) if userUsesDegree else angle)


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
        self.trueInitialBiasX =  trueInitialBiasX
        self.trueInitialBiasY =  trueInitialBiasY
        self.trueInitialBiasZ =  trueInitialBiasZ

        self.measurementAccelNoiseStd = measurementAccelNoiseStd

        self.measurementGyroNoiseStd =  measurementGyroNoiseStd
        # if self.angleUnitIsDegree:
        #     self.measurementGyroNoiseStd = np.deg2rad(measurementGyroNoiseStd)
        # else:
        #     self.measurementGyroNoiseStd = measurementGyroNoiseStd

        self.gravity = gravity

        np.random.seed(self.randomSeed)
        self.setAngularAccelerationProfile()

    def alphaToQuaternion(self, alpha):
        """
        Convertit l'angle vrai alpha en quaternion [qw, qx, qy, qz].
        Ici on suppose une rotation pure autour de l'axe Y,
        cohérente avec accelX = -g*sin(alpha), accelZ = g*cos(alpha).
        """
        alphaArray = np.asarray(alpha, dtype=float)
        # alphaRadArray = np.deg2rad(alphaArray) if self.angleUnitIsDegree else alphaArray
        halfAlphaArray = 0.5 * alphaArray

        quaternionArray = np.zeros((alphaArray.size, 4), dtype=float)
        quaternionArray[:, 0] = np.cos(halfAlphaArray)  # qw
        quaternionArray[:, 1] = 0.0  # qx
        quaternionArray[:, 2] = np.sin(halfAlphaArray)  # qy
        quaternionArray[:, 3] = 0.0  # qz

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

        trueQuaternionArray = self.alphaToQuaternion(trueAlphaArray)

        accelNoiseX = np.random.normal(0.0, self.measurementAccelNoiseStd, self.sampleSize)
        accelNoiseY = np.random.normal(0.0, self.measurementAccelNoiseStd, self.sampleSize)
        accelNoiseZ = np.random.normal(0.0, self.measurementAccelNoiseStd, self.sampleSize)

        measuredAccelXArray = -self.gravity * np.sin(trueAlphaArray) + accelNoiseX
        measuredAccelYArray = np.zeros(self.sampleSize, dtype=float) + accelNoiseY
        measuredAccelZArray = self.gravity * np.cos(trueAlphaArray) + accelNoiseZ

        gyroNoiseX = np.random.normal(0.0, self.measurementGyroNoiseStd, self.sampleSize)
        gyroNoiseY = np.random.normal(0.0, self.measurementGyroNoiseStd, self.sampleSize)
        gyroNoiseZ = np.random.normal(0.0, self.measurementGyroNoiseStd, self.sampleSize)

        measuredGyroXArray = 0.0 + self.trueInitialBiasX + gyroNoiseX
        measuredGyroYArray = trueAlphaDotArray + self.trueInitialBiasY + gyroNoiseY
        measuredGyroZArray = 0.0 + self.trueInitialBiasZ + gyroNoiseZ

        trueAlphaArray = trueAlphaArray
        trueAlphaDotArray = trueAlphaDotArray


        return (
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

    def quaternionToEuler(self, q):
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

        RollPitchYaw = np.array([roll, pitch, yaw], dtype=float)
        return RollPitchYaw

    # --- Fonctions de modèle (dépendent uniquement de la physique) ---

    # f(x, dt)
    def stateTransitionFunction(self, x, dt):
        q = self.normalizeQuaternion(x[0:4])
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
        qNext = self.normalizeQuaternion(q + qDot * dt)

        return np.hstack((qNext, b))

    # h(x)
    def measurementFunction(self, x):
        q = self.normalizeQuaternion(x[0:4])
        b = x[4:7]

        gravityWorld = np.array([0.0, 0.0, self.gravity], dtype=float)
        gravityBody = self.rotateVectorWorldToBody(q, gravityWorld)
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
            "processBiasNoiseStd":  base.processBiasNoiseStd,
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
            ukf.x[0:4] = model.normalizeQuaternion(ukf.x[0:4])

            qEst = ukf.x[0:4].copy()
            bEstRad = ukf.x[4:7].copy()
            bEst = bEstRad
            eulerEst = model.quaternionToEuler(qEst)

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

    # --- Cellule 2 : modèle physique (une seule fois) ---
    ukfModel = UkfModel(simConfig)
    runner = UkfRunner()

    # Réglage de référence
    paramsRef = UkfParams(
        simConfig,
        supposedInitialQuaternion = None,
        supposedInitialBiasX = 0.0,
        supposedInitialBiasY = 0.0,
        supposedInitialBiasZ = 0.0,
        processQuaternionNoiseStd = 0.001,
        processBiasNoiseStd = 0.001,  # => Q
        processInitialConfidenceStd = 1.0,  # => P₀
        label = "Référence",
    )

    # Réglage de base
    paramsBase = UkfParams.fromBase(
        paramsRef,
        processQuaternionNoiseStd = 0.01,
        processBiasNoiseStd = 0.001,  # => Q
        processInitialConfidenceStd = 0.1,  # => P₀
        label = "Base",
    )
    # paramsSweep = UkfParams.createSweepParams(
    #     paramsRef,
    #     "processBiasNoiseStd",
    #     [0.001, 0.01, 0.1]
    # )

    paramsSweep = UkfParams.createSweepParams(
        paramsBase,
        "processInitialConfidenceStd",
        [0.01, 0.1, 1.0]
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


    # --- Cellule 4 : affichage ---
    fig, axes = plt.subplots(11, 1, figsize=(18, 36), sharex=True)
    for res in results:
        axes[0].plot(timeArray, res.estimatedQuaternionArray[:, 0], label="estimatedQuaternionArray " + res.label)
        axes[1].plot(timeArray, res.estimatedQuaternionArray[:, 1], label="estimatedQuaternionArray " + res.label)
        axes[2].plot(timeArray, res.estimatedQuaternionArray[:, 2], label="estimatedQuaternionArray " + res.label)
        axes[3].plot(timeArray, res.estimatedQuaternionArray[:, 3], label="estimatedQuaternionArray " + res.label)
        axes[4].plot(timeArray, np.linalg.norm(res.estimatedQuaternionArray, axis=-1),
                     label="estimatedQuaternionArray " + res.label)
        axes[5].plot(timeArray, np.rad2deg(res.estimatedEulerArray[:, 0]), label="estimatedEulerArray " + res.label)
        axes[6].plot(timeArray, np.rad2deg(res.estimatedEulerArray[:, 1]), label="estimatedEulerArray " + res.label)
        axes[7].plot(timeArray, np.rad2deg(res.estimatedEulerArray[:, 2]), label="estimatedEulerArray " + res.label)
        axes[8].plot(timeArray, np.rad2deg(res.estimatedBiasArray[:, 0]), label="estimatedBiasArray " + res.label)
        axes[9].plot(timeArray, np.rad2deg(res.estimatedBiasArray[:, 1]), label="estimatedBiasArray " + res.label)
        axes[10].plot(timeArray, np.rad2deg(res.estimatedBiasArray[:, 2]), label="estimatedBiasArray " + res.label)

    axes[0].plot(timeArray, trueQuaternionArray[:, 0], 'k--', label="trueQuaternion qw")
    axes[1].plot(timeArray, trueQuaternionArray[:, 1], 'k--', label="trueQuaternion qx")
    axes[2].plot(timeArray, trueQuaternionArray[:, 2], 'k--', label="trueQuaternion qy")
    axes[3].plot(timeArray, trueQuaternionArray[:, 3], 'k--', label="trueQuaternion qz")
    axes[4].plot(timeArray, np.linalg.norm(trueQuaternionArray, axis=-1), 'k--', label="trueQuaternion Norm")

    axes[5].plot(timeArray, np.rad2deg([0] * simConfig.sampleSize), 'k--', label="Roll")
    axes[6].plot(timeArray, np.rad2deg(trueAlphaArray), 'k--', label="Pitch")
    axes[7].plot(timeArray, np.rad2deg([0] * simConfig.sampleSize), 'k--', label="Yaw")
    axes[8].plot(timeArray, np.rad2deg(trueBiases[:, 0]), 'k--', label="trueBiases")
    axes[9].plot(timeArray, np.rad2deg(trueBiases[:, 1]), 'k--', label="trueBiases")
    axes[10].plot(timeArray, np.rad2deg(trueBiases[:, 2]), 'k--', label="trueBiases")

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
