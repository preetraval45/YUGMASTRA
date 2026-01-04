'use client';

/**
 * 3D Network Topology Visualization with Three.js
 * Real-time attack path visualization in 3D space
 */

import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

interface Node {
  id: string;
  label: string;
  type: 'server' | 'database' | 'router' | 'firewall' | 'endpoint';
  position: { x: number; y: number; z: number };
  compromised: boolean;
}

interface Edge {
  source: string;
  target: string;
  type: 'attack' | 'normal' | 'blocked';
}

export default function NetworkTopology3D() {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);

  const [nodes] = useState<Node[]>([
    { id: 'fw1', label: 'Firewall', type: 'firewall', position: { x: 0, y: 0, z: 0 }, compromised: false },
    { id: 'web1', label: 'Web Server', type: 'server', position: { x: 5, y: 2, z: 0 }, compromised: true },
    { id: 'web2', label: 'Web Server 2', type: 'server', position: { x: -5, y: 2, z: 0 }, compromised: false },
    { id: 'db1', label: 'Database', type: 'database', position: { x: 0, y: 5, z: 3 }, compromised: false },
    { id: 'api1', label: 'API Gateway', type: 'server', position: { x: 3, y: -2, z: 2 }, compromised: true },
    { id: 'router1', label: 'Router', type: 'router', position: { x: -3, y: -3, z: -2 }, compromised: false },
  ]);

  const [edges] = useState<Edge[]>([
    { source: 'fw1', target: 'web1', type: 'attack' },
    { source: 'fw1', target: 'web2', type: 'normal' },
    { source: 'web1', target: 'db1', type: 'attack' },
    { source: 'web2', target: 'db1', type: 'blocked' },
    { source: 'fw1', target: 'api1', type: 'attack' },
    { source: 'router1', target: 'fw1', type: 'normal' },
  ]);

  useEffect(() => {
    if (!mountRef.current) return;

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0e1a);
    scene.fog = new THREE.Fog(0x0a0e1a, 10, 50);
    sceneRef.current = scene;

    // Camera setup
    const camera = new THREE.PerspectiveCamera(
      75,
      mountRef.current.clientWidth / mountRef.current.clientHeight,
      0.1,
      1000
    );
    camera.position.set(10, 10, 10);
    cameraRef.current = camera;

    // Renderer setup
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    mountRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controlsRef.current = controls;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(10, 10, 10);
    scene.add(directionalLight);

    // Add grid
    const gridHelper = new THREE.GridHelper(20, 20, 0x444444, 0x222222);
    scene.add(gridHelper);

    // Create nodes
    const nodeMeshes: { [key: string]: THREE.Mesh } = {};

    nodes.forEach((node) => {
      const geometry = new THREE.SphereGeometry(0.5, 32, 32);
      const material = new THREE.MeshPhongMaterial({
        color: node.compromised ? 0xff3333 : getNodeColor(node.type),
        emissive: node.compromised ? 0xff0000 : 0x000000,
        emissiveIntensity: node.compromised ? 0.5 : 0,
      });

      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(node.position.x, node.position.y, node.position.z);
      scene.add(mesh);
      nodeMeshes[node.id] = mesh;

      // Add label
      const canvas = document.createElement('canvas');
      const context = canvas.getContext('2d')!;
      canvas.width = 256;
      canvas.height = 64;
      context.fillStyle = 'white';
      context.font = 'Bold 20px Arial';
      context.fillText(node.label, 10, 35);

      const texture = new THREE.CanvasTexture(canvas);
      const spriteMaterial = new THREE.SpriteMaterial({ map: texture });
      const sprite = new THREE.Sprite(spriteMaterial);
      sprite.position.set(node.position.x, node.position.y + 1, node.position.z);
      sprite.scale.set(2, 0.5, 1);
      scene.add(sprite);

      // Pulsing animation for compromised nodes
      if (node.compromised) {
        const pulseGeometry = new THREE.SphereGeometry(0.7, 32, 32);
        const pulseMaterial = new THREE.MeshBasicMaterial({
          color: 0xff0000,
          transparent: true,
          opacity: 0.3,
        });
        const pulseMesh = new THREE.Mesh(pulseGeometry, pulseMaterial);
        pulseMesh.position.copy(mesh.position);
        scene.add(pulseMesh);

        // Animate pulse
        const animatePulse = () => {
          const scale = 1 + 0.3 * Math.sin(Date.now() * 0.005);
          pulseMesh.scale.setScalar(scale);
          pulseMaterial.opacity = 0.3 + 0.2 * Math.sin(Date.now() * 0.005);
        };
        (pulseMesh as any).animate = animatePulse;
      }
    });

    // Create edges
    edges.forEach((edge) => {
      const sourceNode = nodes.find((n) => n.id === edge.source);
      const targetNode = nodes.find((n) => n.id === edge.target);

      if (sourceNode && targetNode) {
        const points = [
          new THREE.Vector3(sourceNode.position.x, sourceNode.position.y, sourceNode.position.z),
          new THREE.Vector3(targetNode.position.x, targetNode.position.y, targetNode.position.z),
        ];

        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const material = new THREE.LineBasicMaterial({
          color: getEdgeColor(edge.type),
          linewidth: edge.type === 'attack' ? 3 : 1,
        });

        const line = new THREE.Line(geometry, material);
        scene.add(line);

        // Animated particles for attack paths
        if (edge.type === 'attack') {
          const particleGeometry = new THREE.SphereGeometry(0.1, 8, 8);
          const particleMaterial = new THREE.MeshBasicMaterial({ color: 0xff0000 });
          const particle = new THREE.Mesh(particleGeometry, particleMaterial);

          let progress = 0;
          const animateParticle = () => {
            progress += 0.01;
            if (progress > 1) progress = 0;

            particle.position.lerpVectors(
              new THREE.Vector3(sourceNode.position.x, sourceNode.position.y, sourceNode.position.z),
              new THREE.Vector3(targetNode.position.x, targetNode.position.y, targetNode.position.z),
              progress
            );
          };

          scene.add(particle);
          (particle as any).animate = animateParticle;
        }
      }
    });

    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate);

      // Animate all objects with animate function
      scene.traverse((obj) => {
        if ((obj as any).animate) {
          (obj as any).animate();
        }
      });

      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // Handle resize
    const handleResize = () => {
      if (!mountRef.current) return;

      camera.aspect = mountRef.current.clientWidth / mountRef.current.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
    };
    window.addEventListener('resize', handleResize);

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
      mountRef.current?.removeChild(renderer.domElement);
      renderer.dispose();
    };
  }, [nodes, edges]);

  return (
    <div className="relative w-full h-full">
      <div ref={mountRef} className="w-full h-[600px] rounded-lg overflow-hidden" />
      <div className="absolute top-4 left-4 bg-black/80 p-4 rounded-lg border border-gray-700">
        <h3 className="text-sm font-semibold mb-2">Network Legend</h3>
        <div className="space-y-1 text-xs">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-red-500" />
            <span>Compromised Node</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-blue-500" />
            <span>Server</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-purple-500" />
            <span>Database</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-1 bg-red-500" />
            <span>Attack Path</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-1 bg-green-500" />
            <span>Normal Traffic</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-1 bg-yellow-500" />
            <span>Blocked</span>
          </div>
        </div>
      </div>
      <div className="absolute top-4 right-4 bg-black/80 p-4 rounded-lg border border-gray-700">
        <p className="text-xs text-gray-400">WebSocket: {wsStatus === 'connected' ? '🟢 Connected' : '🔴 Disconnected'}</p>
      </div>
    </div>
  );
}

function getNodeColor(type: string): number {
  switch (type) {
    case 'server':
      return 0x3b82f6; // Blue
    case 'database':
      return 0x8b5cf6; // Purple
    case 'router':
      return 0x10b981; // Green
    case 'firewall':
      return 0xf59e0b; // Orange
    case 'endpoint':
      return 0x6b7280; // Gray
    default:
      return 0xffffff;
  }
}

function getEdgeColor(type: string): number {
  switch (type) {
    case 'attack':
      return 0xff0000; // Red
    case 'normal':
      return 0x10b981; // Green
    case 'blocked':
      return 0xfbbf24; // Yellow
    default:
      return 0x6b7280;
  }
}

// Add props for dynamic WebSocket status
interface NetworkTopology3DProps {
  wsStatus?: 'connected' | 'disconnected' | 'connecting' | 'error';
}

function NetworkTopology3DWithStatus({ wsStatus = 'disconnected' }: NetworkTopology3DProps) {
  return <NetworkTopology3D />;
}
