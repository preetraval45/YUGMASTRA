'use client';

import { useState } from 'react';
import { Save, User, Bell, Shield, Database, Check, AlertCircle, Download, Trash2, RefreshCw } from 'lucide-react';

export default function SettingsPage() {
  const [fullName, setFullName] = useState('Preet Raval');
  const [email, setEmail] = useState('preetraval45@gmail.com');
  const [organization, setOrganization] = useState('YUGMĀSTRA Research Lab');
  const [emailNotifications, setEmailNotifications] = useState(true);
  const [pushNotifications, setPushNotifications] = useState(true);
  const [weeklyReports, setWeeklyReports] = useState(false);
  const [populationSize, setPopulationSize] = useState(10);
  const [difficulty, setDifficulty] = useState(50);
  const [learningRate, setLearningRate] = useState(0.0003);
  const [saveMessage, setSaveMessage] = useState('');
  const [showSuccess, setShowSuccess] = useState(false);
  const [showConfirmDelete, setShowConfirmDelete] = useState(false);

  const handleSave = () => {
    setSaveMessage('Saving changes...');

    setTimeout(() => {
      // Simulate saving to localStorage
      const settings = {
        fullName,
        email,
        organization,
        emailNotifications,
        pushNotifications,
        weeklyReports,
        populationSize,
        difficulty,
        learningRate,
        savedAt: new Date().toISOString()
      };

      localStorage.setItem('yugmastra_settings', JSON.stringify(settings));

      setSaveMessage('✓ Settings saved successfully!');
      setShowSuccess(true);

      setTimeout(() => {
        setShowSuccess(false);
        setSaveMessage('');
      }, 3000);
    }, 500);
  };

  const handleExportData = () => {
    const data = {
      profile: { fullName, email, organization },
      settings: {
        notifications: { emailNotifications, pushNotifications, weeklyReports },
        training: { populationSize, difficulty, learningRate }
      },
      exportDate: new Date().toISOString(),
      systemOwner: 'Preet Raval'
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `yugmastra-settings-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);

    setSaveMessage('✓ Settings exported successfully!');
    setShowSuccess(true);
    setTimeout(() => {
      setShowSuccess(false);
      setSaveMessage('');
    }, 3000);
  };

  const handleClearCache = () => {
    localStorage.removeItem('yugmastra_cache');
    setSaveMessage('✓ Cache cleared successfully!');
    setShowSuccess(true);
    setTimeout(() => {
      setShowSuccess(false);
      setSaveMessage('');
    }, 3000);
  };

  const handleDeleteData = () => {
    localStorage.clear();
    setShowConfirmDelete(false);
    setSaveMessage('✓ All data deleted successfully!');
    setShowSuccess(true);
    setTimeout(() => {
      setShowSuccess(false);
      setSaveMessage('');
    }, 3000);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900 p-8">
      <div className="container mx-auto max-w-4xl">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">Settings</h1>
          <p className="text-blue-200">Manage your account and preferences - Preet Raval</p>
        </div>

        {/* Success/Info Message */}
        {showSuccess && (
          <div className="mb-6 p-4 bg-green-500/20 border border-green-500/50 rounded-lg flex items-center gap-3 animate-in slide-in-from-top">
            <Check className="w-5 h-5 text-green-400" />
            <p className="text-green-300 font-medium">{saveMessage}</p>
          </div>
        )}

        <div className="space-y-6">
          {/* Profile Section */}
          <div className="bg-white/10 backdrop-blur-lg rounded-lg p-6 border border-white/20">
            <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
              <User className="w-5 h-5 text-blue-400" />
              Profile
            </h2>
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Full Name</label>
                  <input
                    type="text"
                    value={fullName}
                    onChange={(e) => setFullName(e.target.value)}
                    className="w-full px-3 py-2 bg-black/30 border border-white/20 rounded-md text-white focus:border-blue-500 focus:outline-none"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Email</label>
                  <input
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className="w-full px-3 py-2 bg-black/30 border border-white/20 rounded-md text-white focus:border-blue-500 focus:outline-none"
                  />
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Organization</label>
                <input
                  type="text"
                  value={organization}
                  onChange={(e) => setOrganization(e.target.value)}
                  className="w-full px-3 py-2 bg-black/30 border border-white/20 rounded-md text-white focus:border-blue-500 focus:outline-none"
                />
              </div>
              <div className="mt-4 p-4 bg-blue-500/10 border border-blue-500/30 rounded-lg">
                <p className="text-sm text-blue-300">
                  <strong>System Owner:</strong> {fullName} | <strong>Email:</strong> {email}
                </p>
                <p className="text-xs text-gray-400 mt-1">
                  This system is actively defending against Red Team AI attacks
                </p>
              </div>
            </div>
          </div>

          {/* Notifications Section */}
          <div className="bg-white/10 backdrop-blur-lg rounded-lg p-6 border border-white/20">
            <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
              <Bell className="w-5 h-5 text-yellow-400" />
              Notifications
            </h2>
            <div className="space-y-3">
              <label className="flex items-center justify-between p-3 bg-white/5 rounded-lg hover:bg-white/10 cursor-pointer transition-all">
                <div>
                  <span className="text-sm font-medium text-white">Email notifications for new attacks</span>
                  <p className="text-xs text-gray-400">Get notified when Red Team launches attacks</p>
                </div>
                <input
                  type="checkbox"
                  checked={emailNotifications}
                  onChange={(e) => setEmailNotifications(e.target.checked)}
                  className="w-5 h-5 accent-blue-500 cursor-pointer"
                />
              </label>
              <label className="flex items-center justify-between p-3 bg-white/5 rounded-lg hover:bg-white/10 cursor-pointer transition-all">
                <div>
                  <span className="text-sm font-medium text-white">Push notifications for critical alerts</span>
                  <p className="text-xs text-gray-400">Browser notifications for critical severity attacks</p>
                </div>
                <input
                  type="checkbox"
                  checked={pushNotifications}
                  onChange={(e) => setPushNotifications(e.target.checked)}
                  className="w-5 h-5 accent-blue-500 cursor-pointer"
                />
              </label>
              <label className="flex items-center justify-between p-3 bg-white/5 rounded-lg hover:bg-white/10 cursor-pointer transition-all">
                <div>
                  <span className="text-sm font-medium text-white">Weekly summary reports</span>
                  <p className="text-xs text-gray-400">Email summary of battles and metrics</p>
                </div>
                <input
                  type="checkbox"
                  checked={weeklyReports}
                  onChange={(e) => setWeeklyReports(e.target.checked)}
                  className="w-5 h-5 accent-blue-500 cursor-pointer"
                />
              </label>
            </div>
          </div>

          {/* Training Configuration */}
          <div className="bg-white/10 backdrop-blur-lg rounded-lg p-6 border border-white/20">
            <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
              <Shield className="w-5 h-5 text-purple-400" />
              Training Configuration
            </h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Population Size: <span className="text-white font-mono">{populationSize}</span>
                </label>
                <input
                  type="range"
                  min="5"
                  max="50"
                  value={populationSize}
                  onChange={(e) => setPopulationSize(Number(e.target.value))}
                  className="w-full accent-purple-500"
                />
                <div className="flex justify-between text-xs text-gray-400 mt-1">
                  <span>5 agents</span>
                  <span>50 agents</span>
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Initial Difficulty: <span className="text-white font-mono">{difficulty}%</span>
                </label>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={difficulty}
                  onChange={(e) => setDifficulty(Number(e.target.value))}
                  className="w-full accent-orange-500"
                />
                <div className="flex justify-between text-xs text-gray-400 mt-1">
                  <span>Easy</span>
                  <span>Hard</span>
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Learning Rate</label>
                <input
                  type="number"
                  step="0.0001"
                  value={learningRate}
                  onChange={(e) => setLearningRate(Number(e.target.value))}
                  className="w-full px-3 py-2 bg-black/30 border border-white/20 rounded-md text-white focus:border-purple-500 focus:outline-none font-mono"
                />
                <p className="text-xs text-gray-400 mt-1">Controls how fast AI agents learn (0.0001 - 0.001 recommended)</p>
              </div>
            </div>
          </div>

          {/* Data & Privacy */}
          <div className="bg-white/10 backdrop-blur-lg rounded-lg p-6 border border-white/20">
            <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
              <Database className="w-5 h-5 text-green-400" />
              Data & Privacy
            </h2>
            <div className="space-y-3">
              <button
                onClick={handleExportData}
                className="w-full px-4 py-3 bg-white/5 border border-white/20 rounded-md text-left hover:bg-white/10 transition-all flex items-center gap-3 text-white"
              >
                <Download className="w-5 h-5 text-green-400" />
                <div>
                  <p className="font-medium">Export Training Data</p>
                  <p className="text-xs text-gray-400">Download all settings as JSON file</p>
                </div>
              </button>
              <button
                onClick={handleClearCache}
                className="w-full px-4 py-3 bg-white/5 border border-white/20 rounded-md text-left hover:bg-white/10 transition-all flex items-center gap-3 text-white"
              >
                <RefreshCw className="w-5 h-5 text-blue-400" />
                <div>
                  <p className="font-medium">Clear Cache</p>
                  <p className="text-xs text-gray-400">Remove temporary data and reset local storage</p>
                </div>
              </button>
              <button
                onClick={() => setShowConfirmDelete(true)}
                className="w-full px-4 py-3 bg-red-500/20 border border-red-500/30 rounded-md text-left hover:bg-red-500/30 transition-all flex items-center gap-3 text-red-300"
              >
                <Trash2 className="w-5 h-5" />
                <div>
                  <p className="font-medium">Delete All Data</p>
                  <p className="text-xs text-red-400">Permanently remove all stored data</p>
                </div>
              </button>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex justify-end gap-3">
            <button
              onClick={() => window.location.reload()}
              className="px-6 py-3 border border-white/20 rounded-md text-white hover:bg-white/10 transition-all font-medium"
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              className="px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white rounded-md transition-all flex items-center gap-2 font-medium shadow-lg"
            >
              <Save className="w-5 h-5" />
              Save Changes
            </button>
          </div>
        </div>

        {/* Delete Confirmation Dialog */}
        {showConfirmDelete && (
          <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 backdrop-blur-sm animate-in fade-in">
            <div className="bg-gray-900 border border-red-500/50 rounded-lg p-6 max-w-md mx-4 animate-in zoom-in">
              <div className="flex items-center gap-3 mb-4">
                <AlertCircle className="w-8 h-8 text-red-500" />
                <h3 className="text-2xl font-bold text-white">Confirm Deletion</h3>
              </div>
              <p className="text-gray-300 mb-6">
                Are you sure you want to delete all data? This action cannot be undone.
                All settings, preferences, and cached data will be permanently removed.
              </p>
              <div className="flex gap-3">
                <button
                  onClick={() => setShowConfirmDelete(false)}
                  className="flex-1 px-4 py-2 border border-white/20 rounded-md text-white hover:bg-white/10 transition-all"
                >
                  Cancel
                </button>
                <button
                  onClick={handleDeleteData}
                  className="flex-1 px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-md transition-all font-medium"
                >
                  Delete Everything
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
