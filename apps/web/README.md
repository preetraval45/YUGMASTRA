# YUGMĀSTRA Web Application

Next.js-based web interface for YUGMĀSTRA platform.

## Features

- Real-time attack/defense dashboards
- Evolution metrics visualization
- Strategy analysis tools
- Admin controls for cyber range
- Live WebSocket updates

## Tech Stack

- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript
- **Styling**: TailwindCSS + shadcn/ui
- **State**: Zustand + React Query
- **Charts**: Recharts + D3.js
- **Real-time**: Socket.IO

## Development

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

## Project Structure

```
web/
├── app/                  # Next.js app router
│   ├── dashboard/       # Main dashboard
│   ├── evolution/       # Evolution metrics
│   ├── attacks/         # Attack analytics
│   └── defenses/        # Defense analytics
├── components/          # React components
│   ├── ui/             # shadcn/ui components
│   ├── charts/         # Custom charts
│   └── dashboard/      # Dashboard widgets
├── lib/                # Utilities
│   ├── api/           # API client
│   ├── store/         # Zustand stores
│   └── utils/         # Helper functions
└── public/            # Static assets
```

## Key Pages

- `/dashboard` - Main overview dashboard
- `/evolution` - Co-evolution metrics and graphs
- `/attacks` - Red team analytics
- `/defenses` - Blue team analytics
- `/knowledge-graph` - Interactive knowledge graph
- `/cyber-range` - Simulation control panel
