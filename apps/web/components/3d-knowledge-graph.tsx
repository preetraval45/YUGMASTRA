'use client';

import React, { useRef, useMemo, useEffect, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text, Line, Sphere } from '@react-three/drei';
import * as THREE from 'three';

export interface GraphNode {
  id: string;
  type: 'attack' | 'defense' | 'vulnerability' | 'asset' | 'threat_actor' | 'technique';
  label: string;
  connections: string[];
  position: [number, number, number] | THREE.Vector3;
  velocity: THREE.Vector3;
  severity?: 'critical' | 'high' | 'medium' | 'low';
  metadata: {
    mitre?: string;
    cve?: string;
    score?: number;
    description?: string;
  };
}

interface NetworkGraphProps {
  nodes: GraphNode[];
  selectedNode: string | null;
  onNodeClick: (id: string) => void;
  running: boolean;
}

const NODE_COLORS = {
  attack: '#ef4444',
  defense: '#3b82f6',
  vulnerability: '#eab308',
  asset: '#22c55e',
  threat_actor: '#a855f7',
  technique: '#ec4899',
};

function Node({ node, isSelected, onClick }: { node: GraphNode; isSelected: boolean; onClick: () => void }) {
  const meshRef = useRef<any>(null);
  const [hovered, setHovered] = React.useState(false);

  useFrame(() => {
    if (meshRef.current) {
      meshRef.current.rotation.x += 0.01;
      meshRef.current.rotation.y += 0.01;
    }
  });

  const scale = isSelected ? 1.5 : hovered ? 1.2 : 1;
  const color = NODE_COLORS[node.type];

  const position = Array.isArray(node.position)
    ? node.position
    : [node.position.x, node.position.y, node.position.z] as [number, number, number];

  return (
    <group position={position}>
      <Sphere
        ref={meshRef}
        args={[0.3, 32, 32]}
        scale={scale}
        onClick={onClick}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
      >
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={isSelected ? 0.5 : hovered ? 0.3 : 0.1}
          metalness={0.8}
          roughness={0.2}
        />
      </Sphere>
      {(isSelected || hovered) && (
        <Text
          position={[0, 0.6, 0]}
          fontSize={0.2}
          color="white"
          anchorX="center"
          anchorY="middle"
          outlineWidth={0.02}
          outlineColor="#000000"
        >
          {node.label}
        </Text>
      )}
      {node.severity === 'critical' && (
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <ringGeometry args={[0.4, 0.5, 32]} />
          <meshBasicMaterial color="#ff0000" transparent opacity={0.3} />
        </mesh>
      )}
    </group>
  );
}

function Connection({ from, to, opacity = 0.3 }: {
  from: [number, number, number] | THREE.Vector3;
  to: [number, number, number] | THREE.Vector3;
  opacity?: number
}) {
  const points = useMemo(() => {
    const fromArray = Array.isArray(from) ? from : [from.x, from.y, from.z] as [number, number, number];
    const toArray = Array.isArray(to) ? to : [to.x, to.y, to.z] as [number, number, number];
    return [fromArray, toArray];
  }, [from, to]);

  return (
    <Line
      points={points}
      color="#888888"
      lineWidth={1}
      transparent
      opacity={opacity}
    />
  );
}

function NetworkGraph({ nodes, selectedNode, onNodeClick, running }: NetworkGraphProps) {
  useFrame(() => {
    if (!running) return;

    const SPRING_LENGTH = 3;
    const SPRING_STRENGTH = 0.01;
    const REPULSION_STRENGTH = 2;
    const DAMPING = 0.9;

    nodes.forEach(node => {
      const force = new THREE.Vector3(0, 0, 0);
      const nodePos = Array.isArray(node.position)
        ? new THREE.Vector3(...node.position)
        : node.position;

      nodes.forEach(other => {
        if (node.id !== other.id) {
          const otherPos = Array.isArray(other.position)
            ? new THREE.Vector3(...other.position)
            : other.position;
          const delta = new THREE.Vector3().subVectors(nodePos, otherPos);
          const distance = delta.length();
          if (distance > 0) {
            const repulsion = REPULSION_STRENGTH / (distance * distance);
            delta.normalize().multiplyScalar(repulsion);
            force.add(delta);
          }
        }
      });

      node.connections.forEach(connId => {
        const other = nodes.find(n => n.id === connId);
        if (other) {
          const otherPos = Array.isArray(other.position)
            ? new THREE.Vector3(...other.position)
            : other.position;
          const delta = new THREE.Vector3().subVectors(otherPos, nodePos);
          const distance = delta.length();
          const displacement = distance - SPRING_LENGTH;
          delta.normalize().multiplyScalar(displacement * SPRING_STRENGTH);
          force.add(delta);
        }
      });

      node.velocity.add(force);
      node.velocity.multiplyScalar(DAMPING);

      if (node.position instanceof THREE.Vector3) {
        node.position.add(node.velocity);

        const maxDist = 15;
        if (node.position.length() > maxDist) {
          node.position.normalize().multiplyScalar(maxDist);
          node.velocity.set(0, 0, 0);
        }
      }
    });
  });

  return (
    <>
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      <pointLight position={[-10, -10, -10]} intensity={0.5} />
      <gridHelper args={[20, 20, '#444444', '#222222']} />

      {nodes.map(node =>
        node.connections.map(connId => {
          const target = nodes.find(n => n.id === connId);
          if (target) {
            const isHighlighted = selectedNode === node.id || selectedNode === connId;
            return (
              <Connection
                key={`${node.id}-${connId}`}
                from={node.position}
                to={target.position}
                opacity={isHighlighted ? 0.8 : 0.3}
              />
            );
          }
          return null;
        })
      )}

      {nodes.map(node => (
        <Node
          key={node.id}
          node={node}
          isSelected={selectedNode === node.id}
          onClick={() => onNodeClick(node.id)}
        />
      ))}

      <OrbitControls
        enablePan
        enableZoom
        enableRotate
        maxDistance={30}
        minDistance={5}
      />
    </>
  );
}

export function ThreeDKnowledgeGraph({ nodes, selectedNode, onNodeClick, running }: NetworkGraphProps) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return null;
  }

  return (
    <Canvas
      camera={{ position: [0, 0, 15], fov: 60 }}
      gl={{
        preserveDrawingBuffer: true,
        antialias: true,
        alpha: false
      }}
      dpr={[1, 2]}
    >
      <color attach="background" args={['#000000']} />
      <fog attach="fog" args={['#000000', 10, 50]} />
      <NetworkGraph
        nodes={nodes}
        selectedNode={selectedNode}
        onNodeClick={onNodeClick}
        running={running}
      />
    </Canvas>
  );
}
