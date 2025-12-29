import Link from 'next/link';

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-24 bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900">
      <div className="z-10 max-w-5xl w-full items-center justify-center font-mono text-sm">
        <div className="text-center mb-12">
          <h1 className="text-6xl font-bold text-white mb-4 tracking-tight">
            YUGMĀSTRA
          </h1>
          <p className="text-2xl text-blue-200 mb-4">
            Autonomous Adversary-Defender Co-Evolution Platform
          </p>
          <div className="inline-block bg-gradient-to-r from-blue-500/20 to-purple-500/20 border border-blue-500/30 rounded-lg px-6 py-3 mb-8">
            <p className="text-lg text-white">
              <span className="font-semibold text-blue-300">System Owner:</span> Preet Raval
            </p>
            <p className="text-sm text-gray-300">
              <span className="font-semibold">Email:</span> preetraval45@gmail.com
            </p>
          </div>
          <p className="text-lg text-gray-300 max-w-3xl mx-auto mb-12">
            Where cybersecurity defenses are not engineered—they emerge through
            adversarial self-play between autonomous AI agents. Watch as Red Team AI attacks
            your system in real-time while Blue Team AI learns to defend it.
          </p>

          <div className="flex gap-4 justify-center flex-wrap">
            <Link
              href="/live-battle"
              className="bg-gradient-to-r from-red-600 to-orange-600 hover:from-red-700 hover:to-orange-700 text-white px-8 py-4 rounded-lg text-lg font-semibold transition-all transform hover:scale-105 shadow-lg"
            >
              🔥 Watch Live Battle
            </Link>
            <Link
              href="/dashboard"
              className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-4 rounded-lg text-lg font-semibold transition-colors"
            >
              Launch Dashboard
            </Link>
            <Link
              href="/evolution"
              className="bg-purple-600 hover:bg-purple-700 text-white px-8 py-4 rounded-lg text-lg font-semibold transition-colors"
            >
              View Evolution
            </Link>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-16">
          <div className="bg-white/10 backdrop-blur-lg rounded-lg p-6 border border-white/20">
            <h3 className="text-xl font-bold text-white mb-3">Red Team AI</h3>
            <p className="text-gray-300">
              Autonomous attack agent that discovers novel exploitation strategies
              through reinforcement learning and self-play.
            </p>
          </div>

          <div className="bg-white/10 backdrop-blur-lg rounded-lg p-6 border border-white/20">
            <h3 className="text-xl font-bold text-white mb-3">Blue Team AI</h3>
            <p className="text-gray-300">
              Adaptive defense system that learns detection patterns and generates
              countermeasures automatically.
            </p>
          </div>

          <div className="bg-white/10 backdrop-blur-lg rounded-lg p-6 border border-white/20">
            <h3 className="text-xl font-bold text-white mb-3">Co-Evolution</h3>
            <p className="text-gray-300">
              Multi-agent system where strategies emerge through adversarial
              competition, reaching Nash equilibrium.
            </p>
          </div>
        </div>

        <div className="mt-16 text-center">
          <h2 className="text-3xl font-bold text-white mb-6">Key Features</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-gray-300">
            <div className="bg-white/5 p-4 rounded-lg">
              <div className="text-3xl mb-2">🎯</div>
              <div className="font-semibold">Zero-Day Discovery</div>
            </div>
            <div className="bg-white/5 p-4 rounded-lg">
              <div className="text-3xl mb-2">🧠</div>
              <div className="font-semibold">Self-Play MARL</div>
            </div>
            <div className="bg-white/5 p-4 rounded-lg">
              <div className="text-3xl mb-2">📊</div>
              <div className="font-semibold">Knowledge Graph</div>
            </div>
            <div className="bg-white/5 p-4 rounded-lg">
              <div className="text-3xl mb-2">⚡</div>
              <div className="font-semibold">Real-time Analytics</div>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
