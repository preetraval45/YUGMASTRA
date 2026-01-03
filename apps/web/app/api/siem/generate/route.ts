import { NextRequest, NextResponse } from 'next/server'
import { getSession } from '@/lib/auth'

const AI_ENGINE_URL = process.env.AI_ENGINE_URL || 'http://localhost:8001'

export async function POST(request: NextRequest) {
  let body: any

  try {
    const user = await getSession()
    if (!user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
    }

    body = await request.json()
    const { threatDescription, format = 'sigma', severity = 'high' } = body

    if (!threatDescription) {
      return NextResponse.json({ error: 'Threat description required' }, { status: 400 })
    }

    // Call AI Engine SIEM rule generation endpoint
    const response = await fetch(`${AI_ENGINE_URL}/api/siem/generate-rule`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        threat_description: threatDescription,
        rule_format: format,
        severity,
        include_mitre: true,
      }),
      signal: AbortSignal.timeout(30000),
    })

    if (!response.ok) {
      throw new Error(`AI Engine error: ${response.status}`)
    }

    const data = await response.json()

    return NextResponse.json({
      rule: data.rule || '',
      format: data.format || format,
      severity: data.severity || severity,
      mitreTechniques: data.mitre_techniques || [],
      confidence: data.confidence || 0,
      metadata: data.metadata || {},
      timestamp: new Date().toISOString(),
    })
  } catch (error) {
    console.error('SIEM rule generation error:', error)

    const { threatDescription = 'unknown', format: reqFormat = 'sigma', severity: reqSeverity = 'high' } = body || {}

    // Return basic structure on error
    return NextResponse.json({
      rule: `# Failed to generate ${reqFormat} rule for: ${threatDescription}\n# Please try again or check AI Engine connection`,
      format: reqFormat,
      severity: reqSeverity,
      mitreTechniques: [],
      confidence: 0,
      error: 'AI Engine unavailable',
      timestamp: new Date().toISOString(),
    })
  }
}
