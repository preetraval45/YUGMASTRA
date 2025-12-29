# YUGMĀSTRA Mobile App

React Native mobile application for iOS and Android.

## Features

- Real-time attack/defense notifications
- Mobile-optimized dashboards
- Push notifications for critical alerts
- Offline mode with sync
- Biometric authentication

## Tech Stack

- **Framework**: React Native (Expo)
- **Language**: TypeScript
- **Navigation**: Expo Router
- **UI**: React Native Paper
- **State**: Zustand
- **Real-time**: Socket.IO

## Development

### Prerequisites

- Node.js 20+
- Expo CLI
- iOS Simulator (Mac) or Android Studio

### Setup

```bash
npm install
npm start
```

### Run on Devices

```bash
# iOS
npm run ios

# Android
npm run android

# Web (for testing)
npm run web
```

## Project Structure

```
mobile/
├── app/                 # Expo Router pages
│   ├── (tabs)/         # Tab navigation
│   ├── dashboard/      # Dashboard screens
│   └── _layout.tsx     # Root layout
├── components/         # React Native components
├── lib/               # Utilities
│   ├── api/          # API client
│   ├── store/        # Zustand stores
│   └── utils/        # Helpers
├── assets/           # Images, fonts
└── app.json         # Expo configuration
```

## Build for Production

```bash
# Install EAS CLI
npm install -g eas-cli

# Configure project
eas build:configure

# Build for iOS
npm run build:ios

# Build for Android
npm run build:android
```

## Key Screens

- `/` - Home/Dashboard
- `/dashboard` - Main dashboard
- `/alerts` - Real-time alerts
- `/analytics` - Mobile analytics
- `/settings` - App settings
