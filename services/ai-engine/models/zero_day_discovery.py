"""
Zero-Day Vulnerability Discovery System
Uses ML-based anomaly detection and behavior analysis
FREE Models: Isolation Forest, LSTM Autoencoders, Graph Analysis
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class Vulnerability:
    id: str
    type: str  # buffer_overflow, injection, logic_flaw, race_condition
    severity: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    description: str
    affected_component: str
    detection_method: str
    evidence: List[Dict[str, Any]]
    discovered_at: datetime
    cvss_score: Optional[float] = None
    exploit_available: bool = False

@dataclass
class BehaviorAnomaly:
    id: str
    component: str
    anomaly_type: str  # memory, network, api_usage, execution_flow
    anomaly_score: float
    baseline_mean: float
    observed_value: float
    deviation: float
    timestamp: datetime
    context: Dict[str, Any]

class IsolationForestDetector:
    """
    Isolation Forest for anomaly detection in system behavior
    Uses scikit-learn (FREE)
    """

    def __init__(self, contamination: float = 0.1, n_estimators: int = 100):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.model = None
        self.is_trained = False

        try:
            from sklearn.ensemble import IsolationForest
            self.IsolationForest = IsolationForest
            logger.info("✅ Isolation Forest initialized")
        except ImportError:
            logger.error("❌ scikit-learn not installed. Install with: pip install scikit-learn")
            self.IsolationForest = None

    def train(self, normal_behavior_data: np.ndarray) -> bool:
        """
        Train on normal system behavior

        Args:
            normal_behavior_data: Shape (n_samples, n_features)
                Features: [cpu_usage, memory_usage, network_io, disk_io,
                          api_calls_per_sec, process_count, thread_count, ...]
        """
        if not self.IsolationForest:
            logger.error("Isolation Forest not available")
            return False

        try:
            self.model = self.IsolationForest(
                contamination=self.contamination,
                n_estimators=self.n_estimators,
                random_state=42,
                n_jobs=-1
            )

            self.model.fit(normal_behavior_data)
            self.is_trained = True

            logger.info(f"✅ Isolation Forest trained on {len(normal_behavior_data)} samples")
            return True

        except Exception as e:
            logger.error(f"Failed to train Isolation Forest: {e}")
            return False

    def detect_anomalies(self, behavior_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies in system behavior

        Returns:
            predictions: -1 for anomaly, 1 for normal
            scores: Anomaly scores (lower = more anomalous)
        """
        if not self.is_trained:
            logger.warning("Model not trained yet")
            return np.array([]), np.array([])

        try:
            predictions = self.model.predict(behavior_data)
            scores = self.model.score_samples(behavior_data)

            anomaly_count = np.sum(predictions == -1)
            logger.info(f"Detected {anomaly_count} anomalies out of {len(behavior_data)} samples")

            return predictions, scores

        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return np.array([]), np.array([])

class LSTMAutoencoderDetector:
    """
    LSTM Autoencoder for temporal anomaly detection
    Detects unusual execution sequences and attack patterns
    """

    def __init__(self, sequence_length: int = 50, latent_dim: int = 32):
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.model = None
        self.is_trained = False
        self.reconstruction_threshold = None

        try:
            import tensorflow as tf
            from tensorflow import keras
            self.tf = tf
            self.keras = keras
            logger.info("✅ TensorFlow/Keras available for LSTM Autoencoder")
        except ImportError:
            logger.warning("⚠️ TensorFlow not available. LSTM detector disabled.")
            self.keras = None

    def build_model(self, input_dim: int):
        """Build LSTM Autoencoder model"""
        if not self.keras:
            return None

        try:
            # Encoder
            encoder_inputs = self.keras.Input(shape=(self.sequence_length, input_dim))
            encoder_lstm = self.keras.layers.LSTM(self.latent_dim, return_sequences=False)(encoder_inputs)
            encoder_output = self.keras.layers.Dense(self.latent_dim, activation='relu')(encoder_lstm)

            # Decoder
            decoder_inputs = self.keras.layers.RepeatVector(self.sequence_length)(encoder_output)
            decoder_lstm = self.keras.layers.LSTM(self.latent_dim, return_sequences=True)(decoder_inputs)
            decoder_output = self.keras.layers.Dense(input_dim, activation='sigmoid')(decoder_lstm)

            # Autoencoder
            autoencoder = self.keras.Model(encoder_inputs, decoder_output)
            autoencoder.compile(optimizer='adam', loss='mse')

            logger.info("✅ LSTM Autoencoder model built")
            return autoencoder

        except Exception as e:
            logger.error(f"Failed to build LSTM Autoencoder: {e}")
            return None

    def train(self, sequence_data: np.ndarray, epochs: int = 50, batch_size: int = 32) -> bool:
        """
        Train LSTM Autoencoder on normal execution sequences

        Args:
            sequence_data: Shape (n_sequences, sequence_length, n_features)
        """
        if not self.keras:
            return False

        try:
            input_dim = sequence_data.shape[2]
            self.model = self.build_model(input_dim)

            if self.model is None:
                return False

            # Train
            self.model.fit(
                sequence_data,
                sequence_data,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.1,
                verbose=0
            )

            # Calculate reconstruction threshold
            reconstructions = self.model.predict(sequence_data, verbose=0)
            reconstruction_errors = np.mean(np.square(sequence_data - reconstructions), axis=(1, 2))
            self.reconstruction_threshold = np.percentile(reconstruction_errors, 95)

            self.is_trained = True
            logger.info(f"✅ LSTM Autoencoder trained. Threshold: {self.reconstruction_threshold:.4f}")
            return True

        except Exception as e:
            logger.error(f"Failed to train LSTM Autoencoder: {e}")
            return False

    def detect_anomalies(self, sequence_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalous execution sequences

        Returns:
            predictions: Boolean array (True = anomaly)
            scores: Reconstruction errors
        """
        if not self.is_trained:
            logger.warning("Model not trained yet")
            return np.array([]), np.array([])

        try:
            reconstructions = self.model.predict(sequence_data, verbose=0)
            reconstruction_errors = np.mean(np.square(sequence_data - reconstructions), axis=(1, 2))

            predictions = reconstruction_errors > self.reconstruction_threshold
            anomaly_count = np.sum(predictions)

            logger.info(f"Detected {anomaly_count} anomalous sequences out of {len(sequence_data)}")

            return predictions, reconstruction_errors

        except Exception as e:
            logger.error(f"Sequence anomaly detection failed: {e}")
            return np.array([]), np.array([])

class PatternAnalyzer:
    """
    Analyze attack patterns and identify potential zero-days
    """

    def __init__(self):
        self.known_patterns = defaultdict(list)
        self.pattern_frequency = defaultdict(int)
        self.novel_patterns = []

    def add_known_pattern(self, pattern_type: str, pattern: Dict[str, Any]):
        """Add known attack pattern to baseline"""
        self.known_patterns[pattern_type].append(pattern)
        self.pattern_frequency[pattern_type] += 1

    def analyze_pattern(self, observed_pattern: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze if pattern matches known attacks or is novel

        Returns:
            analysis with novelty score and classification
        """
        pattern_type = observed_pattern.get('type', 'unknown')

        # Check against known patterns
        if pattern_type in self.known_patterns:
            similarity_scores = []

            for known_pattern in self.known_patterns[pattern_type]:
                similarity = self._calculate_similarity(observed_pattern, known_pattern)
                similarity_scores.append(similarity)

            max_similarity = max(similarity_scores) if similarity_scores else 0.0
            novelty_score = 1.0 - max_similarity

            return {
                'is_novel': novelty_score > 0.7,
                'novelty_score': novelty_score,
                'max_similarity': max_similarity,
                'pattern_type': pattern_type,
                'classification': 'zero_day_candidate' if novelty_score > 0.7 else 'known_variant'
            }
        else:
            # Completely new pattern type
            self.novel_patterns.append(observed_pattern)

            return {
                'is_novel': True,
                'novelty_score': 1.0,
                'max_similarity': 0.0,
                'pattern_type': pattern_type,
                'classification': 'zero_day_candidate'
            }

    def _calculate_similarity(self, pattern1: Dict[str, Any], pattern2: Dict[str, Any]) -> float:
        """Calculate Jaccard similarity between two patterns"""
        # Convert patterns to sets of features
        features1 = set(str(k) + ":" + str(v) for k, v in pattern1.items() if k != 'timestamp')
        features2 = set(str(k) + ":" + str(v) for k, v in pattern2.items() if k != 'timestamp')

        if not features1 and not features2:
            return 1.0

        intersection = len(features1 & features2)
        union = len(features1 | features2)

        return intersection / union if union > 0 else 0.0

class ZeroDayDiscoveryEngine:
    """
    Main Zero-Day Discovery Engine
    Combines multiple detection methods
    """

    def __init__(self):
        self.isolation_forest = IsolationForestDetector()
        self.lstm_autoencoder = LSTMAutoencoderDetector()
        self.pattern_analyzer = PatternAnalyzer()
        self.discovered_vulnerabilities = []
        self.behavior_anomalies = []

    async def train_models(
        self,
        normal_behavior_data: np.ndarray,
        normal_sequences: Optional[np.ndarray] = None
    ) -> Dict[str, bool]:
        """Train all detection models on normal behavior"""
        results = {}

        # Train Isolation Forest
        results['isolation_forest'] = self.isolation_forest.train(normal_behavior_data)

        # Train LSTM Autoencoder if sequence data provided
        if normal_sequences is not None:
            results['lstm_autoencoder'] = self.lstm_autoencoder.train(normal_sequences)
        else:
            results['lstm_autoencoder'] = False

        logger.info(f"Training results: {results}")
        return results

    async def analyze_system_behavior(
        self,
        behavior_data: np.ndarray,
        metadata: List[Dict[str, Any]]
    ) -> List[BehaviorAnomaly]:
        """
        Analyze system behavior for anomalies

        Args:
            behavior_data: Real-time system metrics
            metadata: Context for each sample (component, timestamp, etc.)
        """
        anomalies = []

        # Detect with Isolation Forest
        predictions, scores = self.isolation_forest.detect_anomalies(behavior_data)

        if len(predictions) == 0:
            return anomalies

        # Process anomalies
        for i, (pred, score) in enumerate(zip(predictions, scores)):
            if pred == -1:  # Anomaly detected
                meta = metadata[i] if i < len(metadata) else {}

                anomaly = BehaviorAnomaly(
                    id=f"anomaly_{datetime.now().timestamp()}_{i}",
                    component=meta.get('component', 'unknown'),
                    anomaly_type=meta.get('type', 'behavior'),
                    anomaly_score=abs(score),
                    baseline_mean=meta.get('baseline_mean', 0.0),
                    observed_value=meta.get('observed_value', 0.0),
                    deviation=abs(meta.get('observed_value', 0.0) - meta.get('baseline_mean', 0.0)),
                    timestamp=datetime.now(),
                    context=meta
                )

                anomalies.append(anomaly)
                self.behavior_anomalies.append(anomaly)

        logger.info(f"Found {len(anomalies)} behavior anomalies")
        return anomalies

    async def analyze_execution_sequences(
        self,
        sequence_data: np.ndarray,
        metadata: List[Dict[str, Any]]
    ) -> List[BehaviorAnomaly]:
        """Analyze execution sequences for unusual patterns"""
        anomalies = []

        if not self.lstm_autoencoder.is_trained:
            logger.warning("LSTM Autoencoder not trained, skipping sequence analysis")
            return anomalies

        # Detect anomalous sequences
        predictions, scores = self.lstm_autoencoder.detect_anomalies(sequence_data)

        if len(predictions) == 0:
            return anomalies

        # Process anomalous sequences
        for i, (is_anomaly, score) in enumerate(zip(predictions, scores)):
            if is_anomaly:
                meta = metadata[i] if i < len(metadata) else {}

                anomaly = BehaviorAnomaly(
                    id=f"seq_anomaly_{datetime.now().timestamp()}_{i}",
                    component=meta.get('component', 'unknown'),
                    anomaly_type='execution_sequence',
                    anomaly_score=float(score),
                    baseline_mean=self.lstm_autoencoder.reconstruction_threshold,
                    observed_value=float(score),
                    deviation=float(score - self.lstm_autoencoder.reconstruction_threshold),
                    timestamp=datetime.now(),
                    context=meta
                )

                anomalies.append(anomaly)
                self.behavior_anomalies.append(anomaly)

        logger.info(f"Found {len(anomalies)} sequence anomalies")
        return anomalies

    async def discover_vulnerabilities(
        self,
        anomalies: List[BehaviorAnomaly],
        attack_patterns: List[Dict[str, Any]]
    ) -> List[Vulnerability]:
        """
        Analyze anomalies and patterns to discover potential zero-days
        """
        vulnerabilities = []

        # Analyze attack patterns
        for pattern in attack_patterns:
            analysis = self.pattern_analyzer.analyze_pattern(pattern)

            if analysis['is_novel']:
                # Potential zero-day discovered
                vuln = Vulnerability(
                    id=f"ZD_{datetime.now().timestamp()}",
                    type=pattern.get('attack_type', 'unknown'),
                    severity=self._calculate_severity(pattern, anomalies),
                    confidence=analysis['novelty_score'],
                    description=f"Novel attack pattern detected: {pattern.get('description', 'No description')}",
                    affected_component=pattern.get('target_component', 'unknown'),
                    detection_method='pattern_analysis',
                    evidence=[pattern, {'analysis': analysis}],
                    discovered_at=datetime.now(),
                    exploit_available=False
                )

                vulnerabilities.append(vuln)
                self.discovered_vulnerabilities.append(vuln)

        # Cluster anomalies to find vulnerability indicators
        clustered_vulns = self._cluster_anomalies_to_vulnerabilities(anomalies)
        vulnerabilities.extend(clustered_vulns)

        logger.info(f"✅ Discovered {len(vulnerabilities)} potential zero-day vulnerabilities")
        return vulnerabilities

    def _calculate_severity(
        self,
        pattern: Dict[str, Any],
        anomalies: List[BehaviorAnomaly]
    ) -> float:
        """Calculate vulnerability severity (0.0 to 1.0)"""
        severity = 0.5  # Base severity

        # Increase for privilege escalation
        if 'privilege' in pattern.get('attack_type', '').lower():
            severity += 0.3

        # Increase for remote execution
        if 'remote' in pattern.get('attack_type', '').lower():
            severity += 0.2

        # Increase based on anomaly scores
        if anomalies:
            avg_anomaly_score = np.mean([a.anomaly_score for a in anomalies])
            severity += min(avg_anomaly_score * 0.2, 0.3)

        return min(severity, 1.0)

    def _cluster_anomalies_to_vulnerabilities(
        self,
        anomalies: List[BehaviorAnomaly]
    ) -> List[Vulnerability]:
        """
        Cluster related anomalies to identify vulnerabilities
        """
        vulnerabilities = []

        # Group anomalies by component
        component_anomalies = defaultdict(list)
        for anomaly in anomalies:
            component_anomalies[anomaly.component].append(anomaly)

        # Analyze each component's anomalies
        for component, comp_anomalies in component_anomalies.items():
            if len(comp_anomalies) >= 3:  # Threshold for vulnerability indication
                # Calculate average severity
                avg_score = np.mean([a.anomaly_score for a in comp_anomalies])

                vuln = Vulnerability(
                    id=f"ZD_CLUSTER_{datetime.now().timestamp()}",
                    type='behavior_anomaly_cluster',
                    severity=min(avg_score * 0.8, 1.0),
                    confidence=0.6,
                    description=f"Cluster of {len(comp_anomalies)} anomalies in {component}",
                    affected_component=component,
                    detection_method='anomaly_clustering',
                    evidence=[a.__dict__ for a in comp_anomalies],
                    discovered_at=datetime.now(),
                    exploit_available=False
                )

                vulnerabilities.append(vuln)

        return vulnerabilities

    def get_discovered_vulnerabilities(
        self,
        min_severity: float = 0.0,
        min_confidence: float = 0.0
    ) -> List[Vulnerability]:
        """Get discovered vulnerabilities with filtering"""
        return [
            v for v in self.discovered_vulnerabilities
            if v.severity >= min_severity and v.confidence >= min_confidence
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """Get discovery statistics"""
        if not self.discovered_vulnerabilities:
            return {
                "total_vulnerabilities": 0,
                "total_anomalies": len(self.behavior_anomalies),
                "high_severity_count": 0,
                "models_trained": {
                    "isolation_forest": self.isolation_forest.is_trained,
                    "lstm_autoencoder": self.lstm_autoencoder.is_trained
                }
            }

        severities = [v.severity for v in self.discovered_vulnerabilities]

        return {
            "total_vulnerabilities": len(self.discovered_vulnerabilities),
            "total_anomalies": len(self.behavior_anomalies),
            "high_severity_count": sum(1 for s in severities if s >= 0.7),
            "medium_severity_count": sum(1 for s in severities if 0.4 <= s < 0.7),
            "low_severity_count": sum(1 for s in severities if s < 0.4),
            "avg_severity": np.mean(severities),
            "avg_confidence": np.mean([v.confidence for v in self.discovered_vulnerabilities]),
            "models_trained": {
                "isolation_forest": self.isolation_forest.is_trained,
                "lstm_autoencoder": self.lstm_autoencoder.is_trained
            }
        }
