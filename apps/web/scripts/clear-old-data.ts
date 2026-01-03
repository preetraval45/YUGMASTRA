import { PrismaClient } from '@prisma/client'

const prisma = new PrismaClient()

async function clearOldData() {
  console.log('🗑️  Clearing old data from database...\n')

  try {
    // Delete old battles (keep last 50)
    const oldBattles = await prisma.battle.findMany({
      orderBy: { createdAt: 'desc' },
      skip: 50,
      select: { id: true }
    })

    if (oldBattles.length > 0) {
      await prisma.battle.deleteMany({
        where: {
          id: { in: oldBattles.map(b => b.id) }
        }
      })
      console.log(`✅ Deleted ${oldBattles.length} old battles`)
    } else {
      console.log('✅ No old battles to delete')
    }

    // Delete old attacks (keep last 500)
    const oldAttacks = await prisma.attack.findMany({
      orderBy: { timestamp: 'desc' },
      skip: 500,
      select: { id: true }
    })

    if (oldAttacks.length > 0) {
      await prisma.attack.deleteMany({
        where: {
          id: { in: oldAttacks.map(a => a.id) }
        }
      })
      console.log(`✅ Deleted ${oldAttacks.length} old attacks`)
    } else {
      console.log('✅ No old attacks to delete')
    }

    // Delete old defenses (keep last 500)
    const oldDefenses = await prisma.defense.findMany({
      orderBy: { timestamp: 'desc' },
      skip: 500,
      select: { id: true }
    })

    if (oldDefenses.length > 0) {
      await prisma.defense.deleteMany({
        where: {
          id: { in: oldDefenses.map(d => d.id) }
        }
      })
      console.log(`✅ Deleted ${oldDefenses.length} old defenses`)
    } else {
      console.log('✅ No old defenses to delete')
    }

    // Delete old activity logs (keep last 1000)
    const oldLogs = await prisma.activityLog.findMany({
      orderBy: { timestamp: 'desc' },
      skip: 1000,
      select: { id: true }
    })

    if (oldLogs.length > 0) {
      await prisma.activityLog.deleteMany({
        where: {
          id: { in: oldLogs.map(l => l.id) }
        }
      })
      console.log(`✅ Deleted ${oldLogs.length} old activity logs`)
    } else {
      console.log('✅ No old activity logs to delete')
    }

    // Clear old knowledge graph nodes (keep last 100)
    const oldNodes = await prisma.knowledgeGraphNode.findMany({
      orderBy: { createdAt: 'desc' },
      skip: 100,
      select: { id: true }
    })

    if (oldNodes.length > 0) {
      await prisma.knowledgeGraphNode.deleteMany({
        where: {
          id: { in: oldNodes.map(n => n.id) }
        }
      })
      console.log(`✅ Deleted ${oldNodes.length} old knowledge graph nodes`)
    } else {
      console.log('✅ No old knowledge graph nodes to delete')
    }

    console.log('\n✅ Database cleanup completed!')

    // Show current stats
    const stats = await prisma.$transaction([
      prisma.battle.count(),
      prisma.attack.count(),
      prisma.defense.count(),
      prisma.activityLog.count(),
      prisma.knowledgeGraphNode.count(),
      prisma.user.count(),
    ])

    console.log('\n📊 Current Database Stats:')
    console.log(`   Battles: ${stats[0]}`)
    console.log(`   Attacks: ${stats[1]}`)
    console.log(`   Defenses: ${stats[2]}`)
    console.log(`   Activity Logs: ${stats[3]}`)
    console.log(`   Knowledge Graph Nodes: ${stats[4]}`)
    console.log(`   Users: ${stats[5]}`)

  } catch (error) {
    console.error('❌ Error clearing data:', error)
    throw error
  } finally {
    await prisma.$disconnect()
  }
}

clearOldData()
  .catch((e) => {
    console.error(e)
    process.exit(1)
  })
