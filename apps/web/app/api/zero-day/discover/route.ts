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
    const { behaviorData, analysisType = 'full' } = body

    // Call AI Engine zero-day discovery endpoint
    const response = await fetch(`${AI_ENGINE_URL}/api/zero-day/discover`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        behavior_data: behaviorData,
        analysis_type: analysisType,
        advanced: true,
      }),
      signal: AbortSignal.timeout(60000), // 60 second timeout
    })

    if (!response.ok) {
      throw new Error(`AI Engine error: ${response.status}`)
    }

    const data = await response.json()

    return NextResponse.json({
      vulnerabilities: data.vulnerabilities || [],
      confidence: data.confidence || 0,
      analysis: data.analysis || {},
      timestamp: new Date().toISOString(),
    })
  } catch (error) {
    console.error('Zero-day discovery error:', error)

    // Return mock data if AI Engine unavailable
    return NextResponse.json({
      vulnerabilities: [],
      confidence: 0,
      analysis: {},
      error: 'AI Engine unavailable - using fallback mode',
      timestamp: new Date().toISOString(),
    })
  }
}
