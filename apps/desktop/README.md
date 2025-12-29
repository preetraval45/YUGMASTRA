# YUGMĀSTRA Desktop App

Cross-platform desktop application for Windows, macOS, and Linux.

## Features

- Advanced analytics and visualizations
- Local cyber range control
- Multi-window support
- System tray integration
- Native OS integrations
- Offline capability

## Tech Stack

- **Framework**: Electron
- **Renderer**: React + TypeScript
- **Bundler**: Vite
- **UI**: Shared web components
- **Build**: Electron Forge

## Development

```bash
npm install
npm run dev
```

## Project Structure

```
desktop/
├── src/
│   ├── main/           # Electron main process
│   │   ├── index.ts   # Main entry point
│   │   └── ipc.ts     # IPC handlers
│   ├── renderer/       # React renderer process
│   │   ├── App.tsx    # Main app component
│   │   └── index.tsx  # Renderer entry
│   └── preload/       # Preload scripts
├── forge.config.ts    # Electron Forge config
└── package.json
```

## Build for Production

### Windows
```bash
npm run make -- --platform=win32
```

### macOS
```bash
npm run make -- --platform=darwin
```

### Linux
```bash
npm run make -- --platform=linux
```

## Distribution

Build outputs will be in the `out/make/` directory:
- Windows: `.exe` installer
- macOS: `.dmg` and `.app`
- Linux: `.deb` and `.rpm`

## Key Features

- **System Tray**: Background monitoring
- **Native Menus**: Platform-specific menus
- **Auto Updates**: Built-in update mechanism
- **Multi-Window**: Open multiple dashboards
- **Deep Links**: Open from yugmastra:// URLs
