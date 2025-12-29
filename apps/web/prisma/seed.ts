import { PrismaClient } from '@prisma/client';
import { PrismaPg } from '@prisma/adapter-pg';
import pg from 'pg';
import * as bcrypt from 'bcryptjs';

const { Pool } = pg;

// Create connection pool
const pool = new Pool({
  connectionString: process.env.DATABASE_URL || 'postgresql://postgres:postgressql1@localhost:5432/yugmastra?schema=public',
});

// Create Prisma adapter
const adapter = new PrismaPg(pool);

// Create Prisma client
const prisma = new PrismaClient({
  adapter,
  log: ['error', 'warn'],
});

async function main() {
  console.log('Starting database seed...');

  // Hash password for Preet Raval
  const hashedPassword = await bcrypt.hash('yugmastra2025', 12);

  // Create admin user - Preet Raval
  const preet = await prisma.user.upsert({
    where: { email: 'preetraval45@gmail.com' },
    update: {},
    create: {
      name: 'Preet Raval',
      email: 'preetraval45@gmail.com',
      password: hashedPassword,
      role: 'admin',
    },
  });

  console.log('Created admin user:', preet.email);

  // Create default settings for Preet
  await prisma.settings.upsert({
    where: { userId: preet.id },
    update: {},
    create: {
      userId: preet.id,
      notifyAttacks: true,
      notifyDefenses: true,
      notifySystem: true,
      emailNotifications: true,
      learningRate: 0.001,
      episodes: 200,
      batchSize: 64,
      theme: 'dark',
    },
  });

  console.log('Created default settings for Preet Raval');

  // Create demo analyst user
  const hashedDemoPassword = await bcrypt.hash('demo123', 12);
  const demoUser = await prisma.user.upsert({
    where: { email: 'demo@yugmastra.com' },
    update: {},
    create: {
      name: 'Demo Analyst',
      email: 'demo@yugmastra.com',
      password: hashedDemoPassword,
      role: 'analyst',
    },
  });

  console.log('Created demo analyst user:', demoUser.email);

  // Create settings for demo user
  await prisma.settings.upsert({
    where: { userId: demoUser.id },
    update: {},
    create: {
      userId: demoUser.id,
      notifyAttacks: true,
      notifyDefenses: true,
      notifySystem: true,
      emailNotifications: false,
      learningRate: 0.001,
      episodes: 100,
      batchSize: 32,
      theme: 'dark',
    },
  });

  console.log('✅ Database seeded successfully!');
  console.log('\n=== LOGIN CREDENTIALS ===');
  console.log('\nAdmin Account (Preet Raval):');
  console.log('  Email: preetraval45@gmail.com');
  console.log('  Password: yugmastra2025');
  console.log('  Role: Admin (full access)');
  console.log('\nDemo Account:');
  console.log('  Email: demo@yugmastra.com');
  console.log('  Password: demo123');
  console.log('  Role: Analyst (limited access)');
}

main()
  .catch((e) => {
    console.error('Error seeding database:', e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
