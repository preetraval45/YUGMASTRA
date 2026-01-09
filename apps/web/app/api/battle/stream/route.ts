import { NextRequest } from 'next/server';

/**
 * Server-Sent Events (SSE) endpoint for Live Battle streaming
 * Vercel-compatible alternative to WebSocket
 */

export const runtime = 'edge';

// Simulated battle state
let battleState = {
  attacks: [] as any[],
  defenses: [] as any[],
  score: { red: 0, blue: 0 },
  systemHealth: 100,
};

const attackTypes = [
  { type: 'SQL Injection', technique: 'UNION-based SQLi', severity: 'high' },
  { type: 'XSS Attack', technique: 'Stored XSS', severity: 'medium' },
  { type: 'CSRF', technique: 'Token forgery', severity: 'medium' },
  { type: 'Remote Code Execution', technique: 'Deserialization', severity: 'critical' },
  { type: 'Privilege Escalation', technique: 'Kernel exploit', severity: 'high' },
  { type: 'Lateral Movement', technique: 'Pass-the-Hash', severity: 'high' },
  { type: 'Data Exfiltration', technique: 'DNS tunneling', severity: 'critical' },
];

const targets = ['web_server', 'database', 'api_gateway', 'auth_service', 'file_storage'];

const defenseActions = [
  'Blocked by firewall rule',
  'Detected anomalous pattern',
  'IDS signature match',
  'Rate limiting applied',
  'Connection terminated',
  'Traffic redirected to honeypot',
];

function generateAttack() {
  const attackType = attackTypes[Math.floor(Math.random() * attackTypes.length)];
  const target = targets[Math.floor(Math.random() * targets.length)];

  return {
    id: `attack-${Date.now()}-${Math.random()}`,
    timestamp: Date.now(),
    type: attackType.type,
    target,
    severity: attackType.severity,
    technique: attackType.technique,
    status: 'attacking',
  };
}

function generateDefense(attackId: string) {
  const effectiveness = 0.6 + Math.random() * 0.4;
  const blocked = effectiveness > 0.7;

  return {
    id: `defense-${Date.now()}`,
    timestamp: Date.now(),
    action: defenseActions[Math.floor(Math.random() * defenseActions.length)],
    attack_id: attackId,
    effectiveness,
    blocked,
  };
}

export async function GET(request: NextRequest) {
  const encoder = new TextEncoder();

  const customReadable = new ReadableStream({
    start(controller) {
      // Send initial connection message
      const connectMessage = `data: ${JSON.stringify({
        type: 'connected',
        message: 'Connected to battle stream',
        timestamp: Date.now(),
      })}\n\n`;
      controller.enqueue(encoder.encode(connectMessage));

      // Battle simulation loop
      const attackInterval = setInterval(() => {
        try {
          // Generate attack
          const attack = generateAttack();

          const attackMessage = `data: ${JSON.stringify({
            type: 'attack',
            payload: attack,
          })}\n\n`;
          controller.enqueue(encoder.encode(attackMessage));

          // Simulate defense response (70% detection rate)
          setTimeout(() => {
            const detected = Math.random() > 0.3;

            if (detected) {
              const defense = generateDefense(attack.id);

              const defenseMessage = `data: ${JSON.stringify({
                type: 'defense',
                payload: defense,
              })}\n\n`;
              controller.enqueue(encoder.encode(defenseMessage));

              // Update score
              if (defense.blocked) {
                battleState.score.blue++;
                const scoreMessage = `data: ${JSON.stringify({
                  type: 'score_update',
                  payload: battleState.score,
                })}\n\n`;
                controller.enqueue(encoder.encode(scoreMessage));
              } else {
                battleState.score.red++;
                battleState.systemHealth = Math.max(0, battleState.systemHealth - 3);
                const healthMessage = `data: ${JSON.stringify({
                  type: 'health_update',
                  payload: { health: battleState.systemHealth },
                })}\n\n`;
                controller.enqueue(encoder.encode(healthMessage));
              }
            } else {
              battleState.score.red++;
              battleState.systemHealth = Math.max(0, battleState.systemHealth - 12);

              const healthMessage = `data: ${JSON.stringify({
                type: 'health_update',
                payload: { health: battleState.systemHealth },
              })}\n\n`;
              controller.enqueue(encoder.encode(healthMessage));
            }
          }, 1000 + Math.random() * 2000);
        } catch (error) {
          console.error('Battle simulation error:', error);
        }
      }, 2000); // Generate attack every 2 seconds

      // Cleanup on close
      request.signal.addEventListener('abort', () => {
        clearInterval(attackInterval);
        controller.close();
      });
    },
  });

  return new Response(customReadable, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache, no-transform',
      'Connection': 'keep-alive',
      'X-Accel-Buffering': 'no',
    },
  });
}
