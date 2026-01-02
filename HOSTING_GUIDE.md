# Free Hosting Options for RestaurantAI

This guide covers multiple free hosting options for deploying the RestaurantAI application.

## Quick Overview

| Platform | Best For | Free Tier Limits | Database |
|----------|----------|------------------|----------|
| **Vercel** | Frontend (Next.js) | 100GB bandwidth/month | - |
| **Railway** | Backend + DB | $5 credit/month | PostgreSQL |
| **Render** | Full Stack | 750 hours/month | PostgreSQL |
| **Fly.io** | Backend | 3 shared VMs | PostgreSQL |
| **Koyeb** | Backend | 2 nano instances | - |
| **Supabase** | Database | 500MB storage | PostgreSQL |
| **Neon** | Database | 512MB storage | PostgreSQL |
| **PlanetScale** | Database | 5GB storage | MySQL |
| **Netlify** | Frontend | 100GB bandwidth | - |
| **Cyclic** | Backend | 100k requests/month | DynamoDB |
| **Deta** | Backend | Unlimited | Deta Base |
| **Adaptable.io** | Full Stack | 1 app, 256MB RAM | MongoDB |

---

## 1. Frontend Hosting

### Option A: Vercel (Recommended for Next.js)

Vercel is the creator of Next.js and offers the best experience.

**Steps:**
1. Go to [vercel.com](https://vercel.com) and sign in with GitHub
2. Click "New Project" and import your repository
3. Set **Root Directory** to `frontend`
4. Add Environment Variables:
   ```
   NEXT_PUBLIC_API_URL=https://your-backend-url.com/api/v1
   ```
5. Click Deploy

**Free Tier:**
- 100GB bandwidth/month
- Serverless functions
- Automatic SSL
- Edge network CDN

### Option B: Netlify

**Steps:**
1. Go to [netlify.com](https://netlify.com) and sign in
2. Click "Add new site" > "Import an existing project"
3. Connect your GitHub repository
4. Build settings:
   - Base directory: `frontend`
   - Build command: `npm run build`
   - Publish directory: `frontend/.next`
5. Add environment variables in Site settings

**Free Tier:**
- 100GB bandwidth/month
- 300 build minutes/month
- Serverless functions

### Option C: Cloudflare Pages

**Steps:**
1. Go to [pages.cloudflare.com](https://pages.cloudflare.com)
2. Connect your GitHub repository
3. Configure:
   - Framework preset: Next.js
   - Root directory: `frontend`
   - Build command: `npm run build`
4. Deploy

**Free Tier:**
- Unlimited bandwidth
- 500 builds/month
- Global CDN

---

## 2. Backend Hosting

### Option A: Railway (Recommended)

Railway offers a generous free tier with built-in PostgreSQL.

**Steps:**
1. Go to [railway.app](https://railway.app) and sign in
2. Click "New Project" > "Deploy from GitHub"
3. Select your repository
4. Add a PostgreSQL database (click "New" > "Database" > "PostgreSQL")
5. Set environment variables:
   ```
   DATABASE_URL=${{Postgres.DATABASE_URL}}
   SECRET_KEY=your-random-secret-key
   GROQ_API_KEY=your-groq-api-key
   CORS_ORIGINS_STR=https://your-frontend.vercel.app
   ENVIRONMENT=production
   ```
6. Deploy

**Free Tier:**
- $5 credit/month (enough for hobby projects)
- PostgreSQL included
- Automatic SSL

### Option B: Render

Render offers free web services with PostgreSQL.

**Steps:**
1. Go to [render.com](https://render.com) and sign in
2. Click "New" > "Web Service"
3. Connect your GitHub repository
4. Configure:
   - Root Directory: (leave empty)
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`
5. Add PostgreSQL database: "New" > "PostgreSQL"
6. Set environment variables

**render.yaml** (already included in repo):
```yaml
services:
  - type: web
    name: restaurant-api
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT
```

**Free Tier:**
- 750 hours/month
- Spins down after 15 min inactivity
- Free PostgreSQL (90-day limit)

### Option C: Fly.io

Fly.io offers global deployment with generous free tier.

**Steps:**
1. Install Fly CLI: `curl -L https://fly.io/install.sh | sh`
2. Login: `fly auth login`
3. Create app:
   ```bash
   cd /path/to/Restaurant_AI-1
   fly launch --name restaurant-ai-api
   ```
4. Create `fly.toml`:
   ```toml
   app = "restaurant-ai-api"
   
   [build]
     builder = "paketobuildpacks/builder:base"
   
   [env]
     PORT = "8080"
   
   [http_service]
     internal_port = 8080
     force_https = true
   
   [[services]]
     internal_port = 8080
     protocol = "tcp"
   
     [[services.ports]]
       handlers = ["http"]
       port = 80
   
     [[services.ports]]
       handlers = ["tls", "http"]
       port = 443
   ```
5. Create PostgreSQL: `fly postgres create`
6. Deploy: `fly deploy`

**Free Tier:**
- 3 shared-cpu VMs
- 3GB persistent storage
- 160GB outbound transfer

### Option D: Koyeb

Koyeb offers simple deployment with good free tier.

**Steps:**
1. Go to [koyeb.com](https://koyeb.com) and sign in
2. Click "Create App"
3. Select GitHub repository
4. Configure:
   - Builder: Dockerfile or Buildpack
   - Port: 8000
   - Run command: `uvicorn backend.app.main:app --host 0.0.0.0 --port 8000`
5. Add environment variables
6. Deploy

**Free Tier:**
- 2 nano instances
- 512MB RAM each
- Automatic SSL

### Option E: Cyclic

Cyclic offers serverless Node.js and Python hosting.

**Steps:**
1. Go to [cyclic.sh](https://cyclic.sh) and sign in
2. Connect GitHub repository
3. Configure environment variables
4. Deploy

**Free Tier:**
- 100k requests/month
- 1GB storage
- Serverless (always on)

### Option F: Deta Space

Deta offers free hosting with built-in database.

**Steps:**
1. Install Deta CLI: `curl -fsSL https://get.deta.dev/space-cli.sh | sh`
2. Login: `space login`
3. Create `Spacefile`:
   ```yaml
   v: 0
   micros:
     - name: backend
       src: backend
       engine: python3.9
       primary: true
       run: uvicorn app.main:app
   ```
4. Deploy: `space push`

**Free Tier:**
- Unlimited requests
- Free database (Deta Base)
- 10GB storage

---

## 3. Database Options

### Option A: Supabase (Recommended)

Supabase offers PostgreSQL with REST API.

**Steps:**
1. Go to [supabase.com](https://supabase.com) and create project
2. Get connection string from Settings > Database
3. Use in your backend:
   ```
   DATABASE_URL=postgresql+asyncpg://postgres:[PASSWORD]@db.[REF].supabase.co:5432/postgres
   ```

**Free Tier:**
- 500MB database
- 2GB bandwidth
- 50MB file storage

### Option B: Neon

Neon offers serverless PostgreSQL.

**Steps:**
1. Go to [neon.tech](https://neon.tech) and create account
2. Create a new project
3. Copy connection string

**Free Tier:**
- 512MB storage
- Autoscaling
- Branching

### Option C: ElephantSQL

**Steps:**
1. Go to [elephantsql.com](https://elephantsql.com)
2. Create a Tiny Turtle (free) instance
3. Copy connection URL

**Free Tier:**
- 20MB storage
- 5 concurrent connections

### Option D: CockroachDB Serverless

**Steps:**
1. Go to [cockroachlabs.com](https://cockroachlabs.com/get-started-cockroachdb/)
2. Create a Serverless cluster
3. Get connection string

**Free Tier:**
- 5GB storage
- 50M request units/month

---

## 4. Recommended Free Stack

For the best free hosting experience, use:

| Component | Platform | Why |
|-----------|----------|-----|
| Frontend | **Vercel** | Best Next.js support, fast CDN |
| Backend | **Railway** or **Render** | Easy deployment, free PostgreSQL |
| Database | **Supabase** or **Neon** | Generous free tier, PostgreSQL |
| File Storage | **Cloudflare R2** | 10GB free |
| AI/LLM | **Groq** | Fast inference, free tier |

---

## 5. Environment Variables Reference

### Backend (.env)
```env
# Database
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/db

# Security
SECRET_KEY=your-random-256-bit-secret
ENVIRONMENT=production

# AI
GROQ_API_KEY=your-groq-api-key

# CORS (comma-separated frontend URLs)
CORS_ORIGINS_STR=https://your-app.vercel.app,http://localhost:3000
```

### Frontend (.env.local)
```env
NEXT_PUBLIC_API_URL=https://your-backend.railway.app/api/v1
NEXT_PUBLIC_SUPABASE_URL=https://xxx.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key
```

---

## 6. Deployment Checklist

- [ ] Push code to GitHub
- [ ] Deploy database (Supabase/Neon)
- [ ] Run migrations
- [ ] Deploy backend (Railway/Render)
- [ ] Set backend environment variables
- [ ] Deploy frontend (Vercel)
- [ ] Set frontend environment variables
- [ ] Update CORS origins in backend
- [ ] Test all endpoints
- [ ] Set up custom domain (optional)

---

## 7. Troubleshooting

### Backend won't start
- Check logs for errors
- Verify DATABASE_URL is correct
- Ensure all required env vars are set

### CORS errors
- Add frontend URL to CORS_ORIGINS_STR
- Redeploy backend after changes

### Database connection fails
- Check if database is running
- Verify connection string format
- Try pooler connection (port 6543 for Supabase)

### Slow cold starts (Render)
- Free tier sleeps after 15 min
- Consider upgrading or using Railway

---

## 8. Cost Comparison

| Stack | Monthly Cost | Limits |
|-------|--------------|--------|
| Vercel + Railway + Supabase | $0 | Best for hobby |
| Vercel + Render + Neon | $0 | Good alternative |
| Vercel + Fly.io + CockroachDB | $0 | Global deployment |
| Netlify + Koyeb + ElephantSQL | $0 | Lightweight option |

All these stacks are 100% free for hobby projects and portfolios.
