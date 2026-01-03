import { NextRequest, NextResponse } from 'next/server'
import { getSession } from '@/lib/auth'

const AI_ENGINE_URL = process.env.AI_ENGINE_URL || 'http://localhost:8001'

export async function POST(request: NextRequest) {
  try {
    const user = await getSession()
    if (!user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
    }

    const body = await request.json()
    const { behaviorData } = body

    // Call AI Engine behavior analysis endpoint
    const response = await fetch(`${AI_ENGINE_URL}/api/zero-day/analyze/behavior`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        behavior_data: behaviorData,
      }),
      signal: AbortSignal.timeout(30000),
    })

    if (!response.ok) {
      throw new Error(`AI Engine error: ${response.status}`)
    }

    const data = await response.json()

    return NextResponse.json({
      anomalies: data.anomalies || [],
      riskScore: data.risk_score || 0,
      indicators: data.indicators || [],
      confidence: data.confidence || 0,
      timestamp: new Date().toISOString(),
    })
  } catch (error) {
    console.error('Behavior analysis error:', error)

    return NextResponse.json({
      anomalies: [],
      riskScore: 0,
      indicators: [],
      confidence: 0,
      error: 'AI Engine unavailable',
      timestamp: new Date().toISOString(),
    })
  }
}
