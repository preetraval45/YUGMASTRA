# YUGMĀSTRA - Vercel Deployment Quick Start

## ✅ You Already Have Vercel Postgres - Perfect!

Let's deploy YUGMĀSTRA to Vercel with your existing Postgres database.

---

## 🚀 **Quick Deployment Steps**

### **Step 1: Configure Vercel Project**

1. Go to your [Vercel Dashboard](https://vercel.com/preet-raval/yugmastra)
2. Go to "Settings" → "Environment Variables"

### **Step 2: Add Environment Variables**

Copy and paste these into Vercel (replace values with your own):

```bash
# ==================== DATABASE ====================
# Your Vercel Postgres connection string (already configured)
POSTGRES_URL="<from Vercel Postgres>"
POSTGRES_PRISMA_URL="<from Vercel Postgres>"
POSTGRES_URL_NON_POOLING="<from Vercel Postgres>"
DATABASE_URL="${POSTGRES_PRISMA_URL}"

# ==================== AUTHENTICATION ====================
# NextAuth Configuration
NEXTAUTH_URL="https://yugmastra.vercel.app"
NEXTAUTH_SECRET="<run: openssl rand -base64 32>"

# JWT & Session
JWT_SECRET="<run: openssl rand -hex 32>"
SESSION_SECRET="<run: openssl rand -hex 32>"
BCRYPT_ROUNDS="10"

# ==================== AI ENGINE ====================
# We'll deploy AI Engine to Railway (next step)
NEXT_PUBLIC_API_URL="https://yugmastra-ai.railway.app"
AI_ENGINE_URL="https://yugmastra-ai.railway.app"

# ==================== OAUTH (Optional) ====================
# Google OAuth - Get from: https://console.cloud.google.com/
GOOGLE_CLIENT_ID="<your-google-client-id>"
GOOGLE_CLIENT_SECRET="<your-google-client-secret>"

# ==================== EMAIL (Optional) ====================
# Gmail SMTP
SMTP_HOST="smtp.gmail.com"
SMTP_PORT="587"
SMTP_USER="<your-gmail-address>"
SMTP_PASS="<your-gmail-app-password>"
SMTP_FROM_NAME="YUGMASTRA"
SMTP_FROM_EMAIL="noreply@yugmastra.com"

# ==================== MONITORING (Optional) ====================
# Sentry - Get from: https://sentry.io
NEXT_PUBLIC_SENTRY_DSN="<your-sentry-dsn>"
SENTRY_DSN="<your-sentry-dsn>"
SENTRY_ORG="<your-org>"
SENTRY_PROJECT="yugmastra"
ENVIRONMENT="production"

# ==================== NODE ====================
NODE_ENV="production"
```

### **Step 3: Generate Required Secrets**

Run these commands in your terminal and copy the output:

```bash
# Generate NEXTAUTH_SECRET
openssl rand -base64 32

# Generate JWT_SECRET
openssl rand -hex 32

# Generate SESSION_SECRET
openssl rand -hex 32
```

Paste each generated secret into the corresponding Vercel environment variable.

---

## 🐳 **Deploy AI Engine to Railway**

Since Vercel can't run Docker containers, deploy AI Engine to Railway:

### **1. Create Railway Account**

1. Go to [Railway.app](https://railway.app)
2. Sign up with GitHub

### **2. Deploy AI Engine**

1. Click "New Project"
2. Select "Deploy from GitHub repo"
3. Choose your `YUGMASTRA` repository
4. Click "Add Service"
5. Configure:
   ```
   Service Name: ai-engine
   Root Directory: services/ai-engine
   Build Command: docker build -t ai-engine .
   Start Command: uvicorn main:app --host 0.0.0.0 --port 8001
   ```

### **3. Add Environment Variables to Railway**

In Railway service settings, add:

```bash
# Database (use your Vercel Postgres URL)
DATABASE_URL="<your-vercel-postgres-url>"

# Ollama (we'll use hosted Ollama or OpenAI)
OLLAMA_URL="http://ollama.railway.internal:11434"

# Or use OpenAI instead
OPENAI_API_KEY="<your-openai-api-key>"

# Monitoring
SENTRY_DSN="<your-sentry-dsn>"
ENVIRONMENT="production"
LOG_LEVEL="INFO"

# Feature Flags
ENABLE_KNOWLEDGE_GRAPH="false"
ENABLE_ZERO_DAY_DISCOVERY="false"
ENABLE_SIEM_GENERATION="false"
```

### **4. Generate Public URL**

1. In Railway service → "Settings" → "Networking"
2. Click "Generate Domain"
3. Copy the URL (e.g., `https://ai-engine-production-xxxx.up.railway.app`)
4. Update Vercel environment variables:
   ```
   NEXT_PUBLIC_API_URL="https://ai-engine-production-xxxx.up.railway.app"
   AI_ENGINE_URL="https://ai-engine-production-xxxx.up.railway.app"
   ```

---

## 🎯 **Alternative: Use OpenAI API (Simpler)**

If you don't want to deploy Ollama, use OpenAI API:

### **1. Get OpenAI API Key**

1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Create API key
3. Copy the key

### **2. Update Environment Variables**

In both Vercel and Railway:

```bash
OPENAI_API_KEY="sk-..."
USE_OPENAI="true"
```

This eliminates the need for Ollama deployment.

---

## 📦 **Deploy to Vercel**

### **Option A: Deploy via Vercel Dashboard (Recommended)**

1. Go to [Vercel Dashboard](https://vercel.com/preet-raval/yugmastra)
2. Click "Deployments"
3. Click "Deploy" or push to GitHub main branch
4. Wait 3-5 minutes for build
5. Your site is live at: `https://yugmastra.vercel.app`

### **Option B: Deploy via CLI**

```bash
# Install Vercel CLI
npm i -g vercel

# Login
vercel login

# Deploy
cd apps/web
vercel --prod
```

---

## ✅ **Verify Deployment**

### **1. Check Frontend**
Visit: https://yugmastra.vercel.app

You should see:
- ✅ Landing page loads
- ✅ Can navigate to pages
- ✅ Authentication works

### **2. Check AI Engine**
Visit: https://ai-engine-production-xxxx.up.railway.app/health

You should see:
```json
{
  "status": "healthy",
  "services": {...},
  "agents": {...},
  "agent_count": 7
}
```

### **3. Test Full Flow**

1. Sign up for account → ✅ Database working
2. Login → ✅ Authentication working
3. Go to Attack Simulator → ✅ Frontend working
4. Use AI chat → ✅ AI Engine working

---

## 🔧 **Troubleshooting**

### **Build Fails on Vercel**

```bash
# Check build logs in Vercel dashboard
# Common issues:

1. DATABASE_URL not set
   → Add POSTGRES_PRISMA_URL from Vercel Postgres

2. Prisma generate fails
   → Ensure "prisma generate" is in build script

3. TypeScript errors
   → Fix errors locally first, then push
```

### **AI Engine Connection Fails**

```bash
# Check Railway logs
# Common issues:

1. Service not started
   → Check Railway deployment logs

2. Wrong URL in Vercel
   → Ensure NEXT_PUBLIC_API_URL matches Railway URL

3. CORS errors
   → Update AI Engine CORS_ORIGIN to include Vercel URL
```

### **Database Connection Issues**

```bash
# Check Vercel Postgres
1. Go to Storage tab in Vercel
2. Verify database is active
3. Check connection string is correct
4. Test with: vercel env pull
```

---

## 🎨 **Custom Domain Setup**

### **1. Add Domain in Vercel**

1. Go to Project Settings → Domains
2. Click "Add"
3. Enter: `yugmastra.com` or your domain
4. Follow DNS setup instructions

### **2. Configure DNS**

Add these records to your domain provider:

```
Type    Name    Value
A       @       76.76.21.21
CNAME   www     cname.vercel-dns.com
```

### **3. Wait for Propagation**

- DNS propagation: 1-24 hours
- SSL certificate: Automatic (Vercel)

Your site will be available at:
- https://yugmastra.com
- https://www.yugmastra.com
- https://yugmastra.vercel.app (original URL still works)

---

## 🚀 **Go Live Checklist**

Before sharing with users:

- [ ] Environment variables configured in Vercel
- [ ] Secrets generated and added
- [ ] AI Engine deployed to Railway
- [ ] Railway URL added to Vercel env vars
- [ ] Deployment successful
- [ ] Health check passing
- [ ] Can create account
- [ ] Can login
- [ ] AI features working
- [ ] Database persisting data
- [ ] Error monitoring active (Sentry)
- [ ] Custom domain configured (optional)

---

## 📊 **Monitor Your Deployment**

### **Vercel Analytics**
- Go to: https://vercel.com/preet-raval/yugmastra/analytics
- View: Page views, performance, errors

### **Railway Logs**
- Go to: Railway dashboard → AI Engine service
- Click "Logs" to see real-time logs

### **Database Monitoring**
- Go to: Vercel dashboard → Storage → Postgres
- View: Queries, connections, performance

### **Sentry Errors**
- Go to: https://sentry.io
- View: Real-time errors, stack traces, user impact

---

## 💡 **Next Steps After Deployment**

1. **Share Your Site**:
   ```
   https://yugmastra.vercel.app
   ```

2. **Add Content**:
   - Create demo battles
   - Add sample threats
   - Configure AI agents

3. **Invite Users**:
   - Share signup link
   - Create documentation
   - Add tutorials

4. **Monitor Performance**:
   - Check analytics daily
   - Review error logs
   - Optimize slow pages

5. **Scale as Needed**:
   - Upgrade Vercel plan if needed
   - Scale Railway services
   - Add more AI models

---

## 🎉 **Your Live URLs**

Once deployed:

- **Main Site**: https://yugmastra.vercel.app
- **AI Engine**: https://yugmastra-ai.railway.app/health
- **API Docs**: https://yugmastra-ai.railway.app/docs
- **Analytics**: https://vercel.com/preet-raval/yugmastra/analytics

**Anyone can visit these URLs without logging into Vercel!** 🎊

---

## 🆘 **Need Help?**

If you encounter issues:

1. Check Vercel build logs
2. Check Railway deployment logs
3. Verify environment variables
4. Test database connection
5. Review [DEPLOYMENT.md](DEPLOYMENT.md) for detailed guide

---

*Ready to deploy? Let's go! 🚀*
