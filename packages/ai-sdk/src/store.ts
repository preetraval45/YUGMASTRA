/**
 * Zustand State Store for AI SDK (Web, Mobile, Desktop)
 */

import { create } from 'zustand';
import type { Battle, ModelInfo, GraphNode, SIEMRule } from './types';

interface AIStore {
  // Battles
  battles: Battle[];
  activeBattle: Battle | null;
  setBattles: (battles: Battle[]) => void;
  setActiveBattle: (battle: Battle | null) => void;
  addBattle: (battle: Battle) => void;
  updateBattle: (id: string, updates: Partial<Battle>) => void;

  // Models
  models: ModelInfo[];
  setModels: (models: ModelInfo[]) => void;

  // Knowledge Graph
  graphNodes: GraphNode[];
  setGraphNodes: (nodes: GraphNode[]) => void;

  // SIEM Rules
  siemRules: SIEMRule[];
  setSIEMRules: (rules: SIEMRule[]) => void;
  addSIEMRule: (rule: SIEMRule) => void;

  // UI State
  sidebarOpen: boolean;
  toggleSidebar: () => void;
  theme: 'light' | 'dark';
  setTheme: (theme: 'light' | 'dark') => void;
}

export const useAIStore = create<AIStore>((set) => ({
  // Battles
  battles: [],
  activeBattle: null,
  setBattles: (battles) => set({ battles }),
  setActiveBattle: (battle) => set({ activeBattle: battle }),
  addBattle: (battle) => set((state) => ({ battles: [battle, ...state.battles] })),
  updateBattle: (id, updates) =>
    set((state) => ({
      battles: state.battles.map((b) => (b.id === id ? { ...b, ...updates } : b)),
    })),

  // Models
  models: [],
  setModels: (models) => set({ models }),

  // Knowledge Graph
  graphNodes: [],
  setGraphNodes: (nodes) => set({ graphNodes: nodes }),

  // SIEM Rules
  siemRules: [],
  setSIEMRules: (rules) => set({ siemRules: rules }),
  addSIEMRule: (rule) => set((state) => ({ siemRules: [rule, ...state.siemRules] })),

  // UI State
  sidebarOpen: true,
  toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
  theme: 'dark',
  setTheme: (theme) => set({ theme }),
}));
