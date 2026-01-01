# Build stage
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./
COPY apps/web/package*.json ./apps/web/

# Install dependencies
RUN npm install --legacy-peer-deps || npm install --force

# Copy application source
COPY apps/web ./apps/web
COPY apps/web/prisma ./apps/web/prisma
COPY apps/web/.env ./apps/web/.env

# Build application
WORKDIR /app/apps/web
RUN npx prisma generate
# Skip static generation for API routes during build
ENV SKIP_ENV_VALIDATION=true \
    DATABASE_URL="postgresql://yugmastra:yugmastra_dev_password@postgres:5433/yugmastra?schema=public"
RUN npm run build

# Production stage
FROM node:20-alpine AS runner

WORKDIR /app

ENV NODE_ENV=production \
    PORT=3000 \
    HOSTNAME="0.0.0.0"

RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs

# Copy built files
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/apps/web/package.json ./package.json
COPY --from=builder /app/apps/web/.next ./.next
COPY --from=builder /app/apps/web/public ./public
COPY --from=builder /app/apps/web/prisma ./prisma
COPY --from=builder /app/apps/web/.env ./.env

RUN chown -R nextjs:nodejs /app

USER nextjs

EXPOSE 3000

CMD ["npm", "start"]
