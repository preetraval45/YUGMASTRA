import { getSession } from '@/lib/auth'
import prisma from '@/lib/prisma'

/**
 * CYBER RANGE SECURITY CONTROLS
 * Prevents unauthorized access and abuse of real attack capabilities
 */

// Environment flags
export const CYBER_RANGE_ENABLED = process.env.CYBER_RANGE_ENABLED === 'true'
export const REAL_ATTACKS_ENABLED = process.env.REAL_ATTACKS_ENABLED === 'true'
export const REQUIRE_2FA = process.env.REQUIRE_2FA === 'true'

// Authorized users (comma-separated in env)
const AUTHORIZED_USERS = process.env.CYBER_RANGE_AUTHORIZED_USERS?.split(',') || []

// Rate limits
const MAX_ATTACKS_PER_MINUTE = 5
const MAX_ATTACKS_PER_HOUR = 50
const MAX_CONCURRENT_OPERATIONS = 3

/**
 * Verify user has access to cyber range
 */
export async function verifyCyberRangeAccess(userId: string): Promise<boolean> {
  // Check if cyber range is enabled
  if (!CYBER_RANGE_ENABLED) {
    await logSecurityEvent({
      type: 'CYBER_RANGE_DISABLED',
      userId,
      severity: 'INFO',
      message: 'Attempted to access disabled cyber range'
    })
    return false
  }

  // Check if user is authorized
  if (AUTHORIZED_USERS.length > 0 && !AUTHORIZED_USERS.includes(userId)) {
    await logSecurityEvent({
      type: 'UNAUTHORIZED_ACCESS',
      userId,
      severity: 'HIGH',
      message: 'Unauthorized user attempted cyber range access'
    })
    return false
  }

  // Check 2FA if required
  if (REQUIRE_2FA) {
    const user = await prisma.user.findUnique({
      where: { id: userId },
      select: { twoFactorEnabled: true }
    })

    if (!user?.twoFactorEnabled) {
      await logSecurityEvent({
        type: '2FA_REQUIRED',
        userId,
        severity: 'MEDIUM',
        message: '2FA not enabled for cyber range user'
      })
      return false
    }
  }

  return true
}

/**
 * Check rate limits for attacks
 */
export async function checkRateLimit(userId: string): Promise<{ allowed: boolean; reason?: string }> {
  const now = new Date()
  const oneMinuteAgo = new Date(now.getTime() - 60 * 1000)
  const oneHourAgo = new Date(now.getTime() - 60 * 60 * 1000)

  // Check attacks in last minute
  const recentAttacks = await prisma.activityLog.count({
    where: {
      userId,
      category: 'CYBER_RANGE',
      timestamp: { gte: oneMinuteAgo }
    }
  })

  if (recentAttacks >= MAX_ATTACKS_PER_MINUTE) {
    return { allowed: false, reason: 'Rate limit exceeded (per minute)' }
  }

  // Check attacks in last hour
  const hourlyAttacks = await prisma.activityLog.count({
    where: {
      userId,
      category: 'CYBER_RANGE',
      timestamp: { gte: oneHourAgo }
    }
  })

  if (hourlyAttacks >= MAX_ATTACKS_PER_HOUR) {
    return { allowed: false, reason: 'Rate limit exceeded (per hour)' }
  }

  return { allowed: true }
}

/**
 * Log cyber range activity for audit trail
 */
export async function logCyberRangeActivity(
  userId: string,
  action: string,
  details: any
) {
  await prisma.activityLog.create({
    data: {
      userId,
      action,
      details: JSON.stringify({
        ...details,
        timestamp: new Date().toISOString(),
        ipAddress: details.ipAddress,
        userAgent: details.userAgent
      }),
      category: 'CYBER_RANGE',
      timestamp: new Date()
    }
  })
}

/**
 * Log security events
 */
async function logSecurityEvent(event: {
  type: string
  userId: string
  severity: 'INFO' | 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'
  message: string
}) {
  await prisma.activityLog.create({
    data: {
      userId: event.userId,
      action: event.type,
      details: JSON.stringify({
        severity: event.severity,
        message: event.message,
        timestamp: new Date().toISOString()
      }),
      category: 'SECURITY',
      timestamp: new Date()
    }
  })

  // Alert on high/critical events
  if (event.severity === 'HIGH' || event.severity === 'CRITICAL') {
    console.error('[SECURITY ALERT]', event)
    // TODO: Send email/SMS alert
  }
}

/**
 * Validate attack target is in allowed cyber range
 */
export function validateTarget(target: string): boolean {
  const allowedTargets = [
    '172.30.0.10',  // DVWA
    '172.30.0.11',  // MySQL
    '172.30.0.12',  // FTP
    '172.30.0.20',  // Metasploitable
  ]

  return allowedTargets.includes(target)
}

/**
 * Emergency stop - kill all cyber range operations
 */
export async function emergencyStop(userId: string, reason: string) {
  await logSecurityEvent({
    type: 'EMERGENCY_STOP',
    userId,
    severity: 'CRITICAL',
    message: `Emergency stop triggered: ${reason}`
  })

  // TODO: Send Docker command to stop all containers
  console.log('[EMERGENCY STOP] Shutting down cyber range')
}
