import { logger } from './logger';

const RED_TEAM_URL = process.env.RED_TEAM_URL || 'http://yugmastra-red-team-ai:8000';
const BLUE_TEAM_URL = process.env.BLUE_TEAM_URL || 'http://yugmastra-blue-team-ai:8000';
const EVOLUTION_URL = process.env.EVOLUTION_URL || 'http://yugmastra-evolution-engine:8000';

interface MLResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
}

async function mlFetch<T>(url: string, options?: RequestInit): Promise<MLResponse<T>> {
  try {
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();
    return { success: true, data };
  } catch (error: any) {
    logger.error(`ML Service Error: ${url}`, error);
    return { success: false, error: error.message };
  }
}

// Red Team AI Service
export const redTeamAI = {
  async generateAttack(battleContext: any) {
    return mlFetch(`${RED_TEAM_URL}/api/generate-attack`, {
      method: 'POST',
      body: JSON.stringify(battleContext),
    });
  },

  async trainModel(data: any) {
    return mlFetch(`${RED_TEAM_URL}/api/train`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },

  async getModel() {
    return mlFetch(`${RED_TEAM_URL}/api/model`);
  },

  async updateStrategy(strategy: any) {
    return mlFetch(`${RED_TEAM_URL}/api/strategy`, {
      method: 'PUT',
      body: JSON.stringify(strategy),
    });
  },
};

// Blue Team AI Service
export const blueTeamAI = {
  async generateDefense(attackContext: any) {
    return mlFetch(`${BLUE_TEAM_URL}/api/generate-defense`, {
      method: 'POST',
      body: JSON.stringify(attackContext),
    });
  },

  async trainModel(data: any) {
    return mlFetch(`${BLUE_TEAM_URL}/api/train`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },

  async getModel() {
    return mlFetch(`${BLUE_TEAM_URL}/api/model`);
  },

  async updateStrategy(strategy: any) {
    return mlFetch(`${BLUE_TEAM_URL}/api/strategy`, {
      method: 'PUT',
      body: JSON.stringify(strategy),
    });
  },
};

// Evolution Engine Service
export const evolutionEngine = {
  async calculateNashEquilibrium(battleData: any) {
    return mlFetch(`${EVOLUTION_URL}/api/nash-equilibrium`, {
      method: 'POST',
      body: JSON.stringify(battleData),
    });
  },

  async evolve(generation: number, population: any[]) {
    return mlFetch(`${EVOLUTION_URL}/api/evolve`, {
      method: 'POST',
      body: JSON.stringify({ generation, population }),
    });
  },

  async getGenerationStats(generation: number) {
    return mlFetch(`${EVOLUTION_URL}/api/generation/${generation}`);
  },

  async coevolve(redPopulation: any[], bluePopulation: any[]) {
    return mlFetch(`${EVOLUTION_URL}/api/coevolve`, {
      method: 'POST',
      body: JSON.stringify({ redPopulation, bluePopulation }),
    });
  },
};

// Health check for all ML services
export async function checkMLServices() {
  const checks = await Promise.all([
    mlFetch(`${RED_TEAM_URL}/health`),
    mlFetch(`${BLUE_TEAM_URL}/health`),
    mlFetch(`${EVOLUTION_URL}/health`),
  ]);

  return {
    redTeam: checks[0].success,
    blueTeam: checks[1].success,
    evolution: checks[2].success,
    allHealthy: checks.every((c) => c.success),
  };
}
