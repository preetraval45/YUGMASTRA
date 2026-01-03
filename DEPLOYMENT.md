# YUGMĀSTRA - Production Deployment Guide

This guide covers deploying YUGMĀSTRA to production using a hybrid approach:
- **Frontend**: Vercel (Next.js)
- **Backend**: Railway/Render (Docker containers)
- **Database**: Supabase/Neon (PostgreSQL)

---

## 🚀 **Deployment Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                        PRODUCTION                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐      ┌─────────────────────────────┐     │
│  │   VERCEL     │      │        RAILWAY/RENDER        │     │
│  │  (Frontend)  │◄────►│      (Backend Services)      │     │
│  │              │      │                               │     │
│  │ - Next.js    │      │ ┌─────────────────────────┐ │     │
│  │ - React      │      │ │   AI Engine (Port 8001) │ │     │
│  │ - Serverless │      │ └─────────────────────────┘ │     │
│  │   Functions  │      │ ┌─────────────────────────┐ │     │
│  │              │      │ │   Ollama (Port 11434)   │ │     │
│  └──────────────┘      │ └─────────────────────────┘ │     │
│         │              │ ┌─────────────────────────┐ │     │
│         │              │ │   Redis (Port 6379)     │ │     │
│         │              │ └─────────────────────────┘ │     │
│         │              └─────────────────────────────┘     │
│         │                                                   │
│         │              ┌─────────────────────────────┐     │
│         └─────────────►│   SUPABASE/NEON (Database)  │     │
│                        │   PostgreSQL                │     │
│                        └─────────────────────────────┘     │
│                                                              │
│  Public URL: https://yugmastra.vercel.app                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 📋 **Prerequisites**

1. **Accounts Required**:
   - [Vercel Account](https://vercel.com) (Free tier available)
   - [Railway Account](https://railway.app) or [Render Account](https://render.com)
   - [Supabase Account](https://supabase.com) or [Neon Account](https://neon.tech)
   - [GitHub Account](https://github.com) (for repository)

2. **Tools Installed**:
   - Git
   - Node.js 18+
   - Vercel CLI (optional): `npm i -g vercel`
   - Railway CLI (optional): `npm i -g @railway/cli`

---

## 🗄️ **Step 1: Set Up PostgreSQL Database**

### **Option A: Supabase (Recommended)**

1. Go to [Supabase](https://supabase.com)
2. Click "New Project"
3. Fill in:
   - Name: `yugmastra`
   - Database Password: (generate strong password)
   - Region: Choose closest to you
4. Wait for database to be created
5. Go to "Settings" → "Database"
6. Copy the **Connection String** (looks like):
   ```
   postgresql://postgres:[PASSWORD]@db.[PROJECT].supabase.co:5432/postgres
   ```
7. Save this as `DATABASE_URL`

### **Option B: Neon**

1. Go to [Neon](https://neon.tech)
2. Create new project: `yugmastra`
3. Copy the connection string
4. Save as `DATABASE_URL`

---

## 🐳 **Step 2: Deploy Backend Services to Railway**

### **2.1 Create Railway Project**

1. Go to [Railway](https://railway.app)
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Connect your GitHub account
5. Select `YUGMASTRA` repository

### **2.2 Deploy AI Engine Service**

1. In Railway, click "New Service"
2. Select "Docker Image"
3. Configure:
   ```yaml
   Service Name: ai-engine
   Dockerfile Path: services/ai-engine/Dockerfile
   Port: 8001
   ```

4. Add Environment Variables:
   ```bash
   # AI Engine Configuration
   MODEL_DIR=/app/models
   DATA_DIR=/app/data
   LOG_LEVEL=INFO
   WORKERS=4

   # Ollama URL (will be internal Railway URL)
   OLLAMA_URL=http://ollama.railway.internal:11434

   # Database
   DATABASE_URL=<YOUR_SUPABASE_DATABASE_URL>

   # Redis
   REDIS_URL=redis://redis.railway.internal:6379

   # Monitoring (optional)
   SENTRY_DSN=<YOUR_SENTRY_DSN>
   ENVIRONMENT=production
   ```

5. Deploy

### **2.3 Deploy Ollama Service**

1. Click "New Service"
2. Select "Docker Image"
3. Use image: `ollama/ollama:latest`
4. Configure:
   ```yaml
   Service Name: ollama
   Port: 11434
   Memory: 4GB (minimum)
   ```

5. Environment Variables:
   ```bash
   OLLAMA_HOST=0.0.0.0:11434
   OLLAMA_KEEP_ALIVE=5m
   ```

6. After deployment, connect to service and run:
   ```bash
   railway run ollama pull llama2
   railway run ollama pull mistral
   ```

### **2.4 Deploy Redis Service**

1. Click "New Service"
2. Select "Redis" from templates
3. Configure:
   ```yaml
   Service Name: redis
   Port: 6379
   ```

4. Railway will automatically configure

### **2.5 Get Service URLs**

1. For each service, go to "Settings" → "Networking"
2. Click "Generate Domain"
3. Copy the public URLs:
   ```
   AI Engine: https://ai-engine-production-xxxx.up.railway.app
   Ollama: https://ollama-production-xxxx.up.railway.app (if needed)
   ```

---

## ▲ **Step 3: Deploy Frontend to Vercel**

### **3.1 Connect GitHub Repository**

1. Go to [Vercel Dashboard](https://vercel.com/dashboard)
2. Click "Add New" → "Project"
3. Import your GitHub repository: `YUGMASTRA`
4. Configure:
   ```yaml
   Framework Preset: Next.js
   Root Directory: apps/web
   Build Command: npm run vercel-build
   Output Directory: .next
   Install Command: npm install
   ```

### **3.2 Configure Environment Variables**

In Vercel project settings → "Environment Variables", add:

```bash
# Database
DATABASE_URL=<YOUR_SUPABASE_DATABASE_URL>

# NextAuth
NEXTAUTH_URL=https://yugmastra.vercel.app
NEXTAUTH_SECRET=<GENERATE_RANDOM_SECRET>

# JWT & Session
JWT_SECRET=<GENERATE_RANDOM_SECRET>
SESSION_SECRET=<GENERATE_RANDOM_SECRET>

# AI Engine (Railway URL)
NEXT_PUBLIC_API_URL=https://ai-engine-production-xxxx.up.railway.app
AI_ENGINE_URL=https://ai-engine-production-xxxx.up.railway.app

# OAuth (Optional - Google)
GOOGLE_CLIENT_ID=<YOUR_GOOGLE_CLIENT_ID>
GOOGLE_CLIENT_SECRET=<YOUR_GOOGLE_CLIENT_SECRET>

# Email Service (Optional)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=<YOUR_EMAIL>
SMTP_PASS=<YOUR_APP_PASSWORD>
SMTP_FROM_EMAIL=noreply@yugmastra.com

# Monitoring (Optional)
NEXT_PUBLIC_SENTRY_DSN=<YOUR_SENTRY_DSN>
SENTRY_DSN=<YOUR_SENTRY_DSN>
ENVIRONMENT=production

# Node Environment
NODE_ENV=production
```

### **3.3 Generate Secrets**

Use these commands to generate secure secrets:

```bash
# JWT_SECRET
openssl rand -hex 32

# NEXTAUTH_SECRET
openssl rand -base64 32

# SESSION_SECRET
openssl rand -hex 32
```

### **3.4 Deploy**

1. Click "Deploy"
2. Wait for build to complete (3-5 minutes)
3. Your site will be live at: `https://yugmastra.vercel.app`

---

## 🎨 **Step 4: Custom Domain (Optional)**

### **4.1 Add Custom Domain to Vercel**

1. Go to Vercel Project → "Settings" → "Domains"
2. Add your domain: `yugmastra.com`
3. Follow DNS configuration instructions
4. Wait for DNS propagation (up to 24 hours)

### **4.2 Configure DNS**

Add these DNS records to your domain provider:

```
Type    Name    Value
A       @       76.76.21.21
CNAME   www     cname.vercel-dns.com
```

---

## 🔐 **Step 5: Security Configuration**

### **5.1 Enable HTTPS (Automatic on Vercel)**
- Vercel automatically provisions SSL certificates
- All traffic is HTTPS by default

### **5.2 Configure CORS**

Update AI Engine environment variables:

```bash
CORS_ORIGIN=https://yugmastra.vercel.app,https://yugmastra.com
CORS_CREDENTIALS=true
```

### **5.3 Rate Limiting**

Already configured in nginx.conf and AI Engine

---

## 📊 **Step 6: Monitoring & Analytics**

### **6.1 Set Up Sentry**

1. Go to [Sentry.io](https://sentry.io)
2. Create project: `yugmastra`
3. Copy DSN
4. Add to Vercel environment variables:
   ```
   NEXT_PUBLIC_SENTRY_DSN=https://xxx@xxx.ingest.sentry.io/xxx
   SENTRY_DSN=https://xxx@xxx.ingest.sentry.io/xxx
   ```

### **6.2 Vercel Analytics**

1. In Vercel project → "Analytics"
2. Enable "Web Analytics"
3. Enable "Speed Insights"

---

## 🧪 **Step 7: Testing Production Deployment**

### **7.1 Health Checks**

```bash
# Frontend
curl https://yugmastra.vercel.app

# AI Engine
curl https://ai-engine-production-xxxx.up.railway.app/health

# Database Connection Test
# Login to site and try creating an account
```

### **7.2 Functional Tests**

1. ✅ **Authentication**:
   - Sign up with email
   - Login
   - Google OAuth (if configured)

2. ✅ **AI Features**:
   - Test chat with Red Team Agent
   - Test chat with Blue Team Agent
   - Try Attack Simulator

3. ✅ **Database**:
   - Create user account
   - Save data
   - Verify persistence

4. ✅ **Error Monitoring**:
   - Check Sentry for errors
   - Verify error tracking works

---

## 🚀 **Step 8: Performance Optimization**

### **8.1 Next.js Optimizations**

Already configured:
- ✅ Image optimization
- ✅ Code splitting
- ✅ Static page generation
- ✅ Incremental Static Regeneration

### **8.2 CDN & Caching**

Vercel automatically provides:
- ✅ Global CDN
- ✅ Edge caching
- ✅ Automatic compression

### **8.3 Database Optimization**

```sql
-- Add indexes for performance
CREATE INDEX idx_users_email ON "User"(email);
CREATE INDEX idx_sessions_token ON "Session"(session_token);
```

---

## 📈 **Step 9: Scaling**

### **9.1 Vercel Scaling**

Vercel automatically scales based on traffic:
- Free tier: 100GB bandwidth
- Pro tier: 1TB bandwidth
- Enterprise: Unlimited

### **9.2 Railway Scaling**

1. Go to Railway service
2. Click "Settings" → "Resources"
3. Adjust:
   - CPU: 2-8 vCPUs
   - Memory: 4-32GB
   - Replicas: 1-10

### **9.3 Database Scaling**

**Supabase**:
- Free tier: 500MB
- Pro tier: 8GB
- Team: 50GB

**Neon**:
- Free tier: 3GB
- Pro tier: Unlimited

---

## 🔄 **Step 10: CI/CD Pipeline**

### **10.1 Automatic Deployments**

Already configured:
- ✅ Push to `main` → Auto deploy to production
- ✅ Push to `dev` → Deploy to preview
- ✅ Pull requests → Deploy to preview

### **10.2 Environment Branches**

Create branches:
```bash
git checkout -b staging
git checkout -b production
```

Configure in Vercel:
- `production` branch → https://yugmastra.vercel.app
- `staging` branch → https://staging-yugmastra.vercel.app
- `dev` branch → https://dev-yugmastra.vercel.app

---

## 🆘 **Troubleshooting**

### **Issue: Build Fails on Vercel**

```bash
# Check build logs
vercel logs

# Common fixes:
1. Ensure DATABASE_URL is set
2. Run prisma generate before build
3. Check for TypeScript errors
```

### **Issue: AI Engine Can't Connect to Database**

```bash
# Test connection
railway run -- python -c "import psycopg2; print('Connected')"

# Check DATABASE_URL format
# Should be: postgresql://user:pass@host:5432/db
```

### **Issue: Ollama Models Not Loading**

```bash
# SSH into Railway
railway shell

# Pull models manually
ollama pull llama2
ollama pull mistral
```

### **Issue: CORS Errors**

Update AI Engine `main.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yugmastra.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 💰 **Cost Estimation**

### **Free Tier (Development)**
- Vercel: Free (100GB bandwidth)
- Railway: $5/month credit
- Supabase: Free (500MB database)
- **Total: ~$5/month**

### **Production Tier**
- Vercel Pro: $20/month
- Railway: $20-50/month (depends on usage)
- Supabase Pro: $25/month
- **Total: ~$65-95/month**

### **Enterprise Tier**
- Vercel Enterprise: $150+/month
- Railway: $100+/month
- Supabase Team: $599/month
- **Total: ~$850+/month**

---

## 📚 **Additional Resources**

- [Vercel Documentation](https://vercel.com/docs)
- [Railway Documentation](https://docs.railway.app)
- [Supabase Documentation](https://supabase.com/docs)
- [Next.js Deployment](https://nextjs.org/docs/deployment)

---

## ✅ **Deployment Checklist**

- [ ] PostgreSQL database created (Supabase/Neon)
- [ ] Railway project created
- [ ] AI Engine service deployed
- [ ] Ollama service deployed with models
- [ ] Redis service deployed
- [ ] Vercel project created
- [ ] Environment variables configured
- [ ] Secrets generated and added
- [ ] Domain configured (optional)
- [ ] Sentry monitoring set up
- [ ] Health checks passing
- [ ] Authentication working
- [ ] AI features working
- [ ] Database persistence verified
- [ ] Performance optimized
- [ ] CI/CD pipeline working

---

## 🎉 **Launch Checklist**

Before going live:

1. ✅ Security audit complete
2. ✅ Performance testing done
3. ✅ Error monitoring active
4. ✅ Backup system configured
5. ✅ Documentation updated
6. ✅ Support system ready
7. ✅ Analytics tracking
8. ✅ Legal pages (Privacy, Terms)

---

**Your YUGMĀSTRA deployment will be live at:**
- **Production**: https://yugmastra.vercel.app
- **API**: https://ai-engine-production.up.railway.app
- **Status**: https://yugmastra.vercel.app/health

---

*Last Updated: 2026-01-02*
