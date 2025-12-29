'use client';

import { Save, User, Bell, Shield, Database } from 'lucide-react';

export default function SettingsPage() {
  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto p-8 max-w-4xl">
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2">Settings</h1>
          <p className="text-muted-foreground">Manage your account and preferences</p>
        </div>

        <div className="space-y-6">
          <div className="bg-card rounded-lg p-6 border">
            <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
              <User className="w-5 h-5" />
              Profile
            </h2>
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium mb-2">Full Name</label>
                  <input type="text" className="w-full px-3 py-2 bg-background border rounded-md" defaultValue="John Doe" />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">Email</label>
                  <input type="email" className="w-full px-3 py-2 bg-background border rounded-md" defaultValue="john@example.com" />
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">Organization</label>
                <input type="text" className="w-full px-3 py-2 bg-background border rounded-md" defaultValue="YUGMĀSTRA Research Lab" />
              </div>
            </div>
          </div>

          <div className="bg-card rounded-lg p-6 border">
            <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
              <Bell className="w-5 h-5" />
              Notifications
            </h2>
            <div className="space-y-3">
              <label className="flex items-center justify-between">
                <span className="text-sm">Email notifications for new attacks</span>
                <input type="checkbox" className="w-4 h-4" defaultChecked />
              </label>
              <label className="flex items-center justify-between">
                <span className="text-sm">Push notifications for critical alerts</span>
                <input type="checkbox" className="w-4 h-4" defaultChecked />
              </label>
              <label className="flex items-center justify-between">
                <span className="text-sm">Weekly summary reports</span>
                <input type="checkbox" className="w-4 h-4" />
              </label>
            </div>
          </div>

          <div className="bg-card rounded-lg p-6 border">
            <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
              <Shield className="w-5 h-5" />
              Training Configuration
            </h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">Population Size</label>
                <input type="number" className="w-full px-3 py-2 bg-background border rounded-md" defaultValue="10" />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">Initial Difficulty</label>
                <input type="range" min="0" max="100" defaultValue="50" className="w-full" />
                <div className="flex justify-between text-xs text-muted-foreground mt-1">
                  <span>Easy</span>
                  <span>Hard</span>
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">Learning Rate</label>
                <input type="number" step="0.0001" className="w-full px-3 py-2 bg-background border rounded-md" defaultValue="0.0003" />
              </div>
            </div>
          </div>

          <div className="bg-card rounded-lg p-6 border">
            <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
              <Database className="w-5 h-5" />
              Data & Privacy
            </h2>
            <div className="space-y-3">
              <button className="w-full px-4 py-2 bg-background border rounded-md text-left hover:bg-accent">
                Export Training Data
              </button>
              <button className="w-full px-4 py-2 bg-background border rounded-md text-left hover:bg-accent">
                Clear Cache
              </button>
              <button className="w-full px-4 py-2 bg-destructive/20 text-destructive border border-destructive/30 rounded-md text-left hover:bg-destructive/30">
                Delete All Data
              </button>
            </div>
          </div>

          <div className="flex justify-end gap-2">
            <button className="px-6 py-2 border rounded-md hover:bg-accent">Cancel</button>
            <button className="px-6 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 flex items-center gap-2">
              <Save className="w-4 h-4" />
              Save Changes
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
