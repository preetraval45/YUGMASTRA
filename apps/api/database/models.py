from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Episode(Base):
    __tablename__ = "episodes"

    id = Column(Integer, primary_key=True, index=True)
    episode_number = Column(Integer, unique=True, index=True)
    red_agent_id = Column(String)
    blue_agent_id = Column(String)
    red_reward = Column(Float)
    blue_reward = Column(Float)
    red_win = Column(Boolean)
    blue_detection_rate = Column(Float)
    timesteps = Column(Integer)
    phase = Column(String)
    difficulty = Column(Float)
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)


class Attack(Base):
    __tablename__ = "attacks"

    id = Column(Integer, primary_key=True, index=True)
    episode_id = Column(Integer, ForeignKey("episodes.id"))
    attack_type = Column(String, index=True)
    target = Column(String)
    success = Column(Boolean)
    detected = Column(Boolean)
    impact_score = Column(Float)
    detection_time = Column(Float, nullable=True)
    techniques = Column(JSON)
    metadata = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)

    episode = relationship("Episode", backref="attacks")


class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    episode_id = Column(Integer, ForeignKey("episodes.id"))
    attack_id = Column(Integer, ForeignKey("attacks.id"), nullable=True)
    detection_rule_id = Column(Integer, ForeignKey("detection_rules.id"))
    confidence = Column(Float)
    is_true_positive = Column(Boolean, nullable=True)
    response_time = Column(Float)
    metadata = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)

    episode = relationship("Episode", backref="detections")
    attack = relationship("Attack", backref="detections")


class DetectionRule(Base):
    __tablename__ = "detection_rules"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(String)
    rule_type = Column(String)
    confidence = Column(Float)
    false_positive_rate = Column(Float)
    true_positive_count = Column(Integer, default=0)
    false_positive_count = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    generated_by_ai = Column(Boolean, default=True)
    rule_content = Column(JSON)
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Strategy(Base):
    __tablename__ = "strategies"

    id = Column(Integer, primary_key=True, index=True)
    agent_type = Column(String, index=True)  # 'red' or 'blue'
    strategy_name = Column(String)
    description = Column(String)
    success_rate = Column(Float)
    usage_count = Column(Integer, default=0)
    avg_reward = Column(Float)
    strategy_vector = Column(JSON)  # Action distribution
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Metric(Base):
    __tablename__ = "metrics"

    id = Column(Integer, primary_key=True, index=True)
    metric_type = Column(String, index=True)
    metric_name = Column(String, index=True)
    value = Column(Float)
    episode_id = Column(Integer, ForeignKey("episodes.id"), nullable=True)
    metadata = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    episode = relationship("Episode", backref="metrics")
