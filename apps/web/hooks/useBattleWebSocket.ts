import { useEffect, useState, useCallback } from 'react'
import { wsClient } from '@/lib/websocket'

export interface BattleEvent {
  battleId: string
  timestamp: string
  type: 'attack' | 'defense' | 'detection' | 'evolution' | 'metrics'
  agent: 'red' | 'blue'
  data: any
}

export interface BattleMetrics {
  redScore: number
  blueScore: number
  attacksLaunched: number
  attacksBlocked: number
  detectionRate: number
  evolutionGeneration: number
  nashEquilibrium: number
}

export interface ConnectionStatus {
  connected: boolean
  reconnecting: boolean
  error: string | null
}

export function useBattleWebSocket(battleId?: string) {
  const [events, setEvents] = useState<BattleEvent[]>([])
  const [metrics, setMetrics] = useState<BattleMetrics | null>(null)
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>({
    connected: false,
    reconnecting: false,
    error: null,
  })

  // Handle connection status updates
  useEffect(() => {
    const unsubscribe = wsClient.subscribe('connection', (data) => {
      if (data.status === 'connected' || data.status === 'reconnected') {
        setConnectionStatus({
          connected: true,
          reconnecting: false,
          error: null,
        })
      } else if (data.status === 'reconnecting') {
        setConnectionStatus({
          connected: false,
          reconnecting: true,
          error: null,
        })
      } else if (data.status === 'error' || data.status === 'failed') {
        setConnectionStatus({
          connected: false,
          reconnecting: false,
          error: data.error || 'Connection failed',
        })
      } else if (data.status === 'disconnected') {
        setConnectionStatus({
          connected: false,
          reconnecting: false,
          error: null,
        })
      }
    })

    // Connect on mount
    wsClient.connect()

    return () => {
      unsubscribe()
    }
  }, [])

  // Handle battle-specific events
  useEffect(() => {
    if (!battleId) return

    const unsubscribeAttack = wsClient.subscribe('attack', (data) => {
      if (data.battleId === battleId) {
        const event: BattleEvent = {
          battleId,
          timestamp: new Date().toISOString(),
          type: 'attack',
          agent: 'red',
          data,
        }
        setEvents((prev) => [event, ...prev].slice(0, 100)) // Keep last 100 events
      }
    })

    const unsubscribeDefense = wsClient.subscribe('defense', (data) => {
      if (data.battleId === battleId) {
        const event: BattleEvent = {
          battleId,
          timestamp: new Date().toISOString(),
          type: 'defense',
          agent: 'blue',
          data,
        }
        setEvents((prev) => [event, ...prev].slice(0, 100))
      }
    })

    const unsubscribeMetrics = wsClient.subscribe('metrics', (data) => {
      if (data.battleId === battleId) {
        setMetrics(data.metrics)

        const event: BattleEvent = {
          battleId,
          timestamp: new Date().toISOString(),
          type: 'metrics',
          agent: data.leader === 'red' ? 'red' : 'blue',
          data,
        }
        setEvents((prev) => [event, ...prev].slice(0, 100))
      }
    })

    const unsubscribeUpdate = wsClient.subscribe('update', (data) => {
      if (data.battleId === battleId) {
        const event: BattleEvent = {
          battleId,
          timestamp: new Date().toISOString(),
          type: data.type || 'evolution',
          agent: data.agent || 'red',
          data,
        }
        setEvents((prev) => [event, ...prev].slice(0, 100))
      }
    })

    // Join battle room
    wsClient.send('join_battle', { battleId })

    return () => {
      // Leave battle room
      wsClient.send('leave_battle', { battleId })

      unsubscribeAttack()
      unsubscribeDefense()
      unsubscribeMetrics()
      unsubscribeUpdate()
    }
  }, [battleId])

  const startBattle = useCallback(() => {
    if (battleId) {
      wsClient.send('start_battle', { battleId })
    }
  }, [battleId])

  const stopBattle = useCallback(() => {
    if (battleId) {
      wsClient.send('stop_battle', { battleId })
    }
  }, [battleId])

  const clearEvents = useCallback(() => {
    setEvents([])
  }, [])

  return {
    events,
    metrics,
    connectionStatus,
    startBattle,
    stopBattle,
    clearEvents,
  }
}
