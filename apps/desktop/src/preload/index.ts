import { contextBridge, ipcRenderer } from 'electron';

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld('electronAPI', {
  // App info
  getAppVersion: () => ipcRenderer.invoke('get-app-version'),
  getAppPath: () => ipcRenderer.invoke('get-app-path'),

  // Navigation
  onNavigate: (callback: (path: string) => void) => {
    ipcRenderer.on('navigate', (_, path) => callback(path));
  },

  // Deep links
  onDeepLink: (callback: (url: string) => void) => {
    ipcRenderer.on('deep-link', (_, url) => callback(url));
  },

  // Notifications
  showNotification: (title: string, body: string) => {
    new Notification(title, { body });
  },
});

// Type definitions for TypeScript
export interface ElectronAPI {
  getAppVersion: () => Promise<string>;
  getAppPath: () => Promise<string>;
  onNavigate: (callback: (path: string) => void) => void;
  onDeepLink: (callback: (url: string) => void) => void;
  showNotification: (title: string, body: string) => void;
}

declare global {
  interface Window {
    electronAPI: ElectronAPI;
  }
}
