'use client';

import { useState, useEffect } from 'react';
import { Brain, Play, Pause, RotateCcw, Settings, TrendingUp, Zap, Target, Activity, Download } from 'lucide-react';

interface TrainingMetrics {
  epoch: number;
  loss: number;
  accuracy: number;
  validationLoss: number;
  validationAccuracy: number;
  learningRate: number;
}

export default function ModelTrainingPage() {
  const [isTraining, setIsTraining] = useState(false);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [totalEpochs, setTotalEpochs] = useState(100);
  const [batchSize, setBatchSize] = useState(32);
  const [learningRate, setLearningRate] = useState(0.001);
  const [optimizer, setOptimizer] = useState('adam');
  const [modelArchitecture, setModelArchitecture] = useState('transformer');

  const [metrics, setMetrics] = useState<TrainingMetrics[]>([]);
  const [liveMetrics, setLiveMetrics] = useState({
    loss: 2.45,
    accuracy: 0.65,
    valLoss: 2.52,
    valAccuracy: 0.63,
    throughput: 1250,
    eta: '00:45:32',
  });

  // Simulate training progress
  useEffect(() => {
    if (!isTraining) return;

    const interval = setInterval(() => {
      setCurrentEpoch(prev => {
        if (prev >= totalEpochs) {
          setIsTraining(false);
          return prev;
        }

        // Simulate improving metrics
        const newLoss = Math.max(0.1, 2.5 - (prev * 0.02) + (Math.random() - 0.5) * 0.1);
        const newAcc = Math.min(0.99, 0.5 + (prev * 0.004) + (Math.random() - 0.5) * 0.02);
        const newValLoss = Math.max(0.15, 2.6 - (prev * 0.018) + (Math.random() - 0.5) * 0.12);
        const newValAcc = Math.min(0.97, 0.48 + (prev * 0.0038) + (Math.random() - 0.5) * 0.025);

        setMetrics(prevMetrics => [...prevMetrics, {
          epoch: prev + 1,
          loss: newLoss,
          accuracy: newAcc,
          validationLoss: newValLoss,
          validationAccuracy: newValAcc,
          learningRate: learningRate * Math.pow(0.95, Math.floor(prev / 10)),
        }]);

        setLiveMetrics({
          loss: newLoss,
          accuracy: newAcc,
          valLoss: newValLoss,
          valAccuracy: newValAcc,
          throughput: 1200 + Math.random() * 200,
          eta: formatETA(totalEpochs - prev - 1),
        });

        return prev + 1;
      });
    }, 1000);

    return () => clearInterval(interval);
  }, [isTraining, totalEpochs, learningRate]);

  const formatETA = (remainingEpochs: number) => {
    const seconds = remainingEpochs * 2;
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const handleStartTraining = () => {
    setIsTraining(true);
    if (currentEpoch === 0) {
      setMetrics([]);
    }
  };

  const handlePauseTraining = () => {
    setIsTraining(false);
  };

  const handleResetTraining = () => {
    setIsTraining(false);
    setCurrentEpoch(0);
    setMetrics([]);
  };

  const handleExportModel = () => {
    const modelData = {
      architecture: modelArchitecture,
      hyperparameters: {
        epochs: totalEpochs,
        batchSize,
        learningRate,
        optimizer,
      },
      metrics: metrics,
      finalMetrics: liveMetrics,
      timestamp: new Date().toISOString(),
    };

    const blob = new Blob([JSON.stringify(modelData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `yugmastra-model-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen bg-background p-8 pt-32">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-4xl font-bold mb-2 flex items-center gap-3">
              <Brain className="w-8 h-8 text-purple-500" />
              Advanced Model Training
            </h1>
            <p className="text-muted-foreground">
              Train Red/Blue Team AI models with custom hyperparameters - Preet Raval's Lab
            </p>
          </div>
          <div className="flex gap-3">
            {!isTraining ? (
              <button
                onClick={handleStartTraining}
                className="px-6 py-3 bg-green-600 hover:bg-green-700 text-white rounded-lg font-semibold transition-all flex items-center gap-2"
              >
                <Play className="w-5 h-5" />
                {currentEpoch > 0 ? 'Resume Training' : 'Start Training'}
              </button>
            ) : (
              <button
                onClick={handlePauseTraining}
                className="px-6 py-3 bg-yellow-600 hover:bg-yellow-700 text-white rounded-lg font-semibold transition-all flex items-center gap-2"
              >
                <Pause className="w-5 h-5" />
                Pause Training
              </button>
            )}
            <button
              onClick={handleResetTraining}
              disabled={isTraining}
              className="px-6 py-3 bg-red-600 hover:bg-red-700 disabled:bg-red-400 disabled:cursor-not-allowed text-white rounded-lg font-semibold transition-all flex items-center gap-2"
            >
              <RotateCcw className="w-5 h-5" />
              Reset
            </button>
          </div>
        </div>

        {/* Training Progress Bar */}
        {currentEpoch > 0 && (
          <div className="bg-card rounded-lg p-4 border">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-muted-foreground">
                Epoch {currentEpoch} / {totalEpochs}
              </span>
              <span className="text-sm font-semibold">
                {((currentEpoch / totalEpochs) * 100).toFixed(1)}%
              </span>
            </div>
            <div className="h-3 bg-muted rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-purple-600 to-purple-400 transition-all duration-500"
                style={{ width: `${(currentEpoch / totalEpochs) * 100}%` }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        {/* Hyperparameters */}
        <div className="bg-card rounded-lg p-6 border">
          <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
            <Settings className="w-5 h-5 text-blue-500" />
            Hyperparameters
          </h2>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-muted-foreground mb-2">
                Model Architecture
              </label>
              <select
                value={modelArchitecture}
                onChange={(e) => setModelArchitecture(e.target.value)}
                disabled={isTraining}
                className="w-full px-3 py-2 bg-muted border rounded-md focus:border-blue-500 focus:outline-none disabled:opacity-50"
              >
                <option value="transformer">Transformer</option>
                <option value="lstm">LSTM</option>
                <option value="gru">GRU</option>
                <option value="cnn">CNN</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-muted-foreground mb-2">
                Epochs: <span className="font-mono">{totalEpochs}</span>
              </label>
              <input
                type="range"
                min="10"
                max="500"
                value={totalEpochs}
                onChange={(e) => setTotalEpochs(Number(e.target.value))}
                disabled={isTraining}
                className="w-full accent-purple-500"
              />
              <div className="flex justify-between text-xs text-muted-foreground mt-1">
                <span>10</span>
                <span>500</span>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-muted-foreground mb-2">
                Batch Size: <span className="font-mono">{batchSize}</span>
              </label>
              <input
                type="range"
                min="8"
                max="128"
                step="8"
                value={batchSize}
                onChange={(e) => setBatchSize(Number(e.target.value))}
                disabled={isTraining}
                className="w-full accent-blue-500"
              />
              <div className="flex justify-between text-xs text-muted-foreground mt-1">
                <span>8</span>
                <span>128</span>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-muted-foreground mb-2">
                Learning Rate
              </label>
              <input
                type="number"
                step="0.0001"
                value={learningRate}
                onChange={(e) => setLearningRate(Number(e.target.value))}
                disabled={isTraining}
                className="w-full px-3 py-2 bg-muted border rounded-md focus:border-blue-500 focus:outline-none font-mono disabled:opacity-50"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-muted-foreground mb-2">
                Optimizer
              </label>
              <select
                value={optimizer}
                onChange={(e) => setOptimizer(e.target.value)}
                disabled={isTraining}
                className="w-full px-3 py-2 bg-muted border rounded-md focus:border-blue-500 focus:outline-none disabled:opacity-50"
              >
                <option value="adam">Adam</option>
                <option value="sgd">SGD</option>
                <option value="rmsprop">RMSprop</option>
                <option value="adamw">AdamW</option>
              </select>
            </div>

            <button
              onClick={handleExportModel}
              disabled={metrics.length === 0}
              className="w-full px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-green-400 disabled:cursor-not-allowed text-white rounded-lg transition-all flex items-center justify-center gap-2"
            >
              <Download className="w-4 h-4" />
              Export Model
            </button>
          </div>
        </div>

        {/* Live Metrics */}
        <div className="lg:col-span-2 space-y-6">
          {/* Real-time Metrics Grid */}
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            <div className="bg-card rounded-lg p-4 border border-red-500/30">
              <div className="flex items-center gap-2 mb-2">
                <Activity className="w-4 h-4 text-red-500" />
                <h3 className="text-sm text-muted-foreground">Training Loss</h3>
              </div>
              <p className="text-2xl font-bold font-mono">{liveMetrics.loss.toFixed(4)}</p>
            </div>

            <div className="bg-card rounded-lg p-4 border border-green-500/30">
              <div className="flex items-center gap-2 mb-2">
                <Target className="w-4 h-4 text-green-500" />
                <h3 className="text-sm text-muted-foreground">Accuracy</h3>
              </div>
              <p className="text-2xl font-bold font-mono text-green-500">
                {(liveMetrics.accuracy * 100).toFixed(2)}%
              </p>
            </div>

            <div className="bg-card rounded-lg p-4 border border-yellow-500/30">
              <div className="flex items-center gap-2 mb-2">
                <Activity className="w-4 h-4 text-yellow-500" />
                <h3 className="text-sm text-muted-foreground">Val Loss</h3>
              </div>
              <p className="text-2xl font-bold font-mono">{liveMetrics.valLoss.toFixed(4)}</p>
            </div>

            <div className="bg-card rounded-lg p-4 border border-blue-500/30">
              <div className="flex items-center gap-2 mb-2">
                <Target className="w-4 h-4 text-blue-500" />
                <h3 className="text-sm text-muted-foreground">Val Accuracy</h3>
              </div>
              <p className="text-2xl font-bold font-mono text-blue-500">
                {(liveMetrics.valAccuracy * 100).toFixed(2)}%
              </p>
            </div>

            <div className="bg-card rounded-lg p-4 border border-purple-500/30">
              <div className="flex items-center gap-2 mb-2">
                <Zap className="w-4 h-4 text-purple-500" />
                <h3 className="text-sm text-muted-foreground">Throughput</h3>
              </div>
              <p className="text-2xl font-bold font-mono">{Math.round(liveMetrics.throughput)}</p>
              <p className="text-xs text-muted-foreground">samples/sec</p>
            </div>

            <div className="bg-card rounded-lg p-4 border border-orange-500/30">
              <div className="flex items-center gap-2 mb-2">
                <TrendingUp className="w-4 h-4 text-orange-500" />
                <h3 className="text-sm text-muted-foreground">ETA</h3>
              </div>
              <p className="text-2xl font-bold font-mono">{liveMetrics.eta}</p>
            </div>
          </div>

          {/* Training Charts */}
          <div className="bg-card rounded-lg p-6 border">
            <h2 className="text-xl font-bold mb-4">Training Progress</h2>
            <div className="grid grid-cols-2 gap-6">
              {/* Loss Chart */}
              <div>
                <h3 className="text-sm font-semibold mb-3 text-muted-foreground">Loss</h3>
                <div className="h-48 bg-accent/20 rounded-lg p-4 relative">
                  <div className="absolute inset-0 flex flex-col justify-between p-4 pointer-events-none">
                    {[2.5, 2.0, 1.5, 1.0, 0.5, 0].map((val) => (
                      <div key={val} className="flex items-center gap-2">
                        <span className="text-xs text-muted-foreground w-8">{val.toFixed(1)}</span>
                        <div className="flex-1 border-t border-gray-700/50" />
                      </div>
                    ))}
                  </div>
                  <div className="absolute inset-0 p-4 pl-12 flex items-end">
                    {metrics.length > 1 ? (
                      <svg className="w-full h-full" preserveAspectRatio="none">
                        <polyline
                          points={metrics.map((m, i) =>
                            `${(i / Math.max(metrics.length - 1, 1)) * 100},${100 - (m.loss / 2.5) * 100}`
                          ).join(' ')}
                          fill="none"
                          stroke="rgb(239, 68, 68)"
                          strokeWidth="2"
                        />
                        <polyline
                          points={metrics.map((m, i) =>
                            `${(i / Math.max(metrics.length - 1, 1)) * 100},${100 - (m.validationLoss / 2.5) * 100}`
                          ).join(' ')}
                          fill="none"
                          stroke="rgb(234, 179, 8)"
                          strokeWidth="2"
                          strokeDasharray="5,5"
                        />
                      </svg>
                    ) : (
                      <div className="w-full h-full flex items-center justify-center text-muted-foreground text-sm">
                        Start training to see chart
                      </div>
                    )}
                  </div>
                </div>
                <div className="flex items-center gap-4 mt-2 text-xs">
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-1 bg-red-500" />
                    <span className="text-muted-foreground">Training</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-1 bg-yellow-500 opacity-75" style={{ backgroundImage: 'repeating-linear-gradient(to right, rgb(234, 179, 8) 0, rgb(234, 179, 8) 3px, transparent 3px, transparent 6px)' }} />
                    <span className="text-muted-foreground">Validation</span>
                  </div>
                </div>
              </div>

              <div>
                <h3 className="text-sm font-semibold mb-3 text-muted-foreground">Accuracy</h3>
                <div className="h-48 bg-accent/20 rounded-lg p-4 relative">
                  <div className="absolute inset-0 flex flex-col justify-between p-4 pointer-events-none">
                    {[100, 80, 60, 40, 20, 0].map((val) => (
                      <div key={val} className="flex items-center gap-2">
                        <span className="text-xs text-muted-foreground w-8">{val}%</span>
                        <div className="flex-1 border-t border-gray-700/50" />
                      </div>
                    ))}
                  </div>
                  <div className="absolute inset-0 p-4 pl-12 flex items-end">
                    {metrics.length > 1 ? (
                      <svg className="w-full h-full" preserveAspectRatio="none">
                        <polyline
                          points={metrics.map((m, i) =>
                            `${(i / Math.max(metrics.length - 1, 1)) * 100},${100 - m.accuracy * 100}`
                          ).join(' ')}
                          fill="none"
                          stroke="rgb(34, 197, 94)"
                          strokeWidth="2"
                        />
                        <polyline
                          points={metrics.map((m, i) =>
                            `${(i / Math.max(metrics.length - 1, 1)) * 100},${100 - m.validationAccuracy * 100}`
                          ).join(' ')}
                          fill="none"
                          stroke="rgb(59, 130, 246)"
                          strokeWidth="2"
                          strokeDasharray="5,5"
                        />
                      </svg>
                    ) : (
                      <div className="w-full h-full flex items-center justify-center text-muted-foreground text-sm">
                        Start training to see chart
                      </div>
                    )}
                  </div>
                </div>
                <div className="flex items-center gap-4 mt-2 text-xs">
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-1 bg-green-500" />
                    <span className="text-muted-foreground">Training</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-1 bg-blue-500 opacity-75" style={{ backgroundImage: 'repeating-linear-gradient(to right, rgb(59, 130, 246) 0, rgb(59, 130, 246) 3px, transparent 3px, transparent 6px)' }} />
                    <span className="text-muted-foreground">Validation</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Training History */}
      <div className="bg-card rounded-lg p-6 border">
        <h2 className="text-xl font-bold mb-4">Training History</h2>
        <div className="max-h-96 overflow-y-auto">
          <table className="w-full">
            <thead className="sticky top-0 bg-card">
              <tr className="border-b">
                <th className="text-left py-2 px-3 text-sm font-semibold text-muted-foreground">Epoch</th>
                <th className="text-left py-2 px-3 text-sm font-semibold text-muted-foreground">Loss</th>
                <th className="text-left py-2 px-3 text-sm font-semibold text-muted-foreground">Accuracy</th>
                <th className="text-left py-2 px-3 text-sm font-semibold text-muted-foreground">Val Loss</th>
                <th className="text-left py-2 px-3 text-sm font-semibold text-muted-foreground">Val Acc</th>
                <th className="text-left py-2 px-3 text-sm font-semibold text-muted-foreground">LR</th>
              </tr>
            </thead>
            <tbody>
              {metrics.slice().reverse().map((metric) => (
                <tr key={metric.epoch} className="border-b hover:bg-accent/50">
                  <td className="py-2 px-3 text-sm font-mono">{metric.epoch}</td>
                  <td className="py-2 px-3 text-sm font-mono">{metric.loss.toFixed(4)}</td>
                  <td className="py-2 px-3 text-sm font-mono text-green-500">
                    {(metric.accuracy * 100).toFixed(2)}%
                  </td>
                  <td className="py-2 px-3 text-sm font-mono">{metric.validationLoss.toFixed(4)}</td>
                  <td className="py-2 px-3 text-sm font-mono text-blue-500">
                    {(metric.validationAccuracy * 100).toFixed(2)}%
                  </td>
                  <td className="py-2 px-3 text-sm font-mono text-muted-foreground">
                    {metric.learningRate.toFixed(6)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {metrics.length === 0 && (
            <div className="text-center py-12 text-muted-foreground">
              <Brain className="w-12 h-12 mx-auto mb-3 opacity-50" />
              <p>No training data yet. Start training to see metrics.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
