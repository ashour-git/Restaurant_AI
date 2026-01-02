# Restaurant AI - Deployment Guide

Complete deployment documentation for the Restaurant AI SaaS platform.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Prerequisites](#prerequisites)
3. [Environment Setup](#environment-setup)
4. [Local Development](#local-development)
5. [Docker Deployment](#docker-deployment)
6. [Production Deployment](#production-deployment)
7. [Authentication](#authentication)
8. [API Reference](#api-reference)
9. [ML Services](#ml-services)
10. [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/ashour-git/Restaurant_AI.git
cd Restaurant_AI

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example backend/.env
# Edit backend/.env with your API keys

# Initialize database
cd backend
python -c "from app.core.database import init_db; import asyncio; asyncio.run(init_db())"
python seed_data.py

# Run the server
python -m uvicorn app.main:app --host 127.0.0.1 --port 5000 --reload
```

---

## Prerequisites

### System Requirements

| Component | Minimum | Recommended |
| --------- | ------- | ----------- |
| Python    | 3.11+   | 3.13        |
| Node.js   | 18+     | 20+         |
| RAM       | 4GB     | 8GB+        |
| Storage   | 2GB     | 5GB+        |

### Required Software

- **Python 3.11+** - Backend API and ML services
- **Node.js 18+** - Frontend build tools
- **Git** - Version control
- **Docker** (optional) - Containerized deployment

### API Keys Required

| Service  | Purpose               | Get Key From             |
| -------- | --------------------- | ------------------------ |
| Groq API | LLM for NLP assistant | https://console.groq.com |

---

## Environment Setup

### 1. Create Environment File

Copy the example environment file:

```bash
cp .env.example backend/.env
```

### 2. Configure Variables

Edit `backend/.env`:

```env
# Application Settings
SECRET_KEY=your-super-secret-key-min-32-chars
ENVIRONMENT=development
DEBUG=true

# Database (SQLite for development)
DATABASE_URL=sqlite+aiosqlite:///./restaurant.db

# JWT Settings
ACCESS_TOKEN_EXPIRE_MINUTES=1440
REFRESH_TOKEN_EXPIRE_DAYS=7

# AI/ML Configuration
GROQ_API_KEY=your-groq-api-key
```

### 3. Generate Secret Key

```python
import secrets
print(secrets.token_urlsafe(32))
```

---

## Local Development

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv ../.venv
../.venv\Scripts\activate  # Windows
source ../.venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r ../requirements.txt

# Initialize database
python -c "from app.core.database import init_db; import asyncio; asyncio.run(init_db())"

# Seed with sample data (optional)
python seed_data.py

# Run development server
python -m uvicorn app.main:app --host 127.0.0.1 --port 5000 --reload
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

### Access Points

| Service      | URL                          | Description            |
| ------------ | ---------------------------- | ---------------------- |
| Backend API  | http://127.0.0.1:5000        | FastAPI server         |
| API Docs     | http://127.0.0.1:5000/docs   | Interactive Swagger UI |
| Frontend     | http://localhost:3000        | Next.js application    |
| Health Check | http://127.0.0.1:5000/health | Server status          |

---

## Docker Deployment

### Using Docker Compose

```bash
# Build and start all services
docker-compose up --build

# Run in background
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Services

| Service  | Port | Description            |
| -------- | ---- | ---------------------- |
| backend  | 5000 | FastAPI backend        |
| frontend | 3000 | Next.js frontend       |
| ml       | 5001 | ML services (optional) |

### Building Individual Images

```bash
# Backend
docker build -f Dockerfile.backend -t restaurant-api:latest .

# Frontend
docker build -f Dockerfile.frontend -t restaurant-frontend:latest .

# ML Services
docker build -f Dockerfile.ml -t restaurant-ml:latest .
```

---

## Production Deployment

### Vercel + Render/Railway Deployment (Recommended)

This project is optimized for deploying the **frontend on Vercel** and the **backend on Render or Railway**.

#### Step 1: Deploy Backend to Render

1. **Create a Render Account**: Go to [render.com](https://render.com)

2. **New Web Service**:

   - Connect your GitHub repository
   - Select "Web Service"
   - Choose Python as the environment

3. **Configure Build Settings**:

   ```
   Build Command: pip install -r requirements.txt
   Start Command: uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT
   ```

4. **Set Environment Variables**:

   ```
   SECRET_KEY=<generate-with-secrets.token_urlsafe(32)>
   ENVIRONMENT=production
   DEBUG=false
   GROQ_API_KEY=<your-groq-api-key>
   CORS_ORIGINS_STR=https://your-app.vercel.app
   ```

5. **Add PostgreSQL** (optional but recommended):

   - Create a PostgreSQL database in Render
   - Copy the Internal Database URL to `DATABASE_URL`

6. **Deploy**: Render will build and deploy automatically

7. **Copy your API URL**: e.g., `https://restaurant-api.onrender.com`

#### Step 2: Deploy Frontend to Vercel

1. **Create a Vercel Account**: Go to [vercel.com](https://vercel.com)

2. **Import Project**:

   - Click "New Project"
   - Import from GitHub
   - Select your repository

3. **Configure Project**:

   ```
   Framework Preset: Next.js
   Root Directory: frontend
   Build Command: npm run build
   Output Directory: .next
   ```

4. **Set Environment Variables**:

   ```
   NEXT_PUBLIC_API_URL=https://your-backend.onrender.com/api/v1
   ```

5. **Deploy**: Click "Deploy" and wait for completion

6. **Update Backend CORS**: Add your Vercel URL to the backend's `CORS_ORIGINS_STR`

#### Step 3: Verify Deployment

1. Visit your Vercel URL
2. Check the dashboard loads
3. Test authentication
4. Verify API connections

---

### Alternative: Railway Deployment

1. **Create a Railway Account**: Go to [railway.app](https://railway.app)

2. **New Project from GitHub**:

   - Connect repository
   - Railway auto-detects Python

3. **Add PostgreSQL**:

   - Click "New" → "Database" → "PostgreSQL"
   - Railway provides `DATABASE_URL` automatically

4. **Set Variables**:

   ```
   SECRET_KEY=<your-secret-key>
   ENVIRONMENT=production
   GROQ_API_KEY=<your-key>
   CORS_ORIGINS_STR=https://your-app.vercel.app
   ```

5. **Generate Domain**:
   - Settings → Networking → Generate Domain

---

### Pre-deployment Checklist

- [ ] Set `ENVIRONMENT=production`
- [ ] Set `DEBUG=false`
- [ ] Generate strong `SECRET_KEY` (32+ chars)
- [ ] Configure production database (PostgreSQL recommended)
- [ ] Set up HTTPS/SSL certificates
- [ ] Configure CORS for your domain
- [ ] Set up monitoring and logging
- [ ] Configure backup strategy

### Production Environment Variables

```env
# Production Settings
SECRET_KEY=<strong-random-key-32-chars-min>
ENVIRONMENT=production
DEBUG=false

# PostgreSQL (recommended for production)
DATABASE_URL=postgresql+asyncpg://user:password@host:5432/restaurant_db

# Security
ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com

# JWT (adjust for security requirements)
ACCESS_TOKEN_EXPIRE_MINUTES=60
REFRESH_TOKEN_EXPIRE_DAYS=7

# AI/ML
GROQ_API_KEY=<your-groq-api-key>
```

### Running with Gunicorn (Production)

```bash
pip install gunicorn

gunicorn app.main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:5000 \
    --access-logfile - \
    --error-logfile -
```

### Nginx Configuration

```nginx
upstream restaurant_api {
    server 127.0.0.1:5000;
}

server {
    listen 80;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://restaurant_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

## Authentication

### Overview

The API uses JWT (JSON Web Token) based authentication with the following endpoints:

| Endpoint                | Method | Description          |
| ----------------------- | ------ | -------------------- |
| `/api/v1/auth/register` | POST   | Register new user    |
| `/api/v1/auth/login`    | POST   | Login and get token  |
| `/api/v1/auth/me`       | GET    | Get current user     |
| `/api/v1/auth/refresh`  | POST   | Refresh access token |

### User Registration

```bash
curl -X POST "http://127.0.0.1:5000/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "SecurePass123!",
    "first_name": "John",
    "last_name": "Doe"
  }'
```

Response:

```json
{
  "id": 1,
  "email": "user@example.com",
  "first_name": "John",
  "last_name": "Doe",
  "role": "staff",
  "is_active": true
}
```

### User Login

```bash
curl -X POST "http://127.0.0.1:5000/api/v1/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=user@example.com&password=SecurePass123!"
```

Response:

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 86400
}
```

### Using the Token

Include the token in the Authorization header for protected endpoints:

```bash
curl -X GET "http://127.0.0.1:5000/api/v1/auth/me" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

### Role-Based Access Control

| Role      | Permissions                              |
| --------- | ---------------------------------------- |
| `staff`   | Basic operations, view data              |
| `manager` | Staff permissions + edit menu, inventory |
| `admin`   | Full access, user management             |

---

## API Reference

### Core Endpoints

| Category  | Base Path           | Description               |
| --------- | ------------------- | ------------------------- |
| Menu      | `/api/v1/menu`      | Menu items and categories |
| Orders    | `/api/v1/orders`    | Order management          |
| Customers | `/api/v1/customers` | Customer management       |
| Inventory | `/api/v1/inventory` | Stock management          |
| Analytics | `/api/v1/analytics` | Sales analytics           |
| ML        | `/api/v1/ml`        | Machine learning features |

### Full API Documentation

Visit `/docs` on your running server for interactive Swagger documentation.

---

## ML Services

### Available Features

| Feature            | Endpoint                 | Description           |
| ------------------ | ------------------------ | --------------------- |
| Demand Forecasting | `/api/v1/ml/forecast`    | Predict future demand |
| Recommendations    | `/api/v1/ml/recommend`   | Item recommendations  |
| NLP Assistant      | `/api/v1/ml/chat`        | AI chat assistant     |
| Menu Search        | `/api/v1/ml/menu-search` | Semantic menu search  |

### Demand Forecasting

```bash
curl -X POST "http://127.0.0.1:5000/api/v1/ml/forecast" \
  -H "Content-Type: application/json" \
  -d '{"days_ahead": 7, "item_id": 1}'
```

### Recommendations

```bash
curl -X POST "http://127.0.0.1:5000/api/v1/ml/recommend" \
  -H "Content-Type: application/json" \
  -d '{"item_ids": [1, 2, 3], "top_k": 5, "strategy": "hybrid"}'
```

### Chat with AI Assistant

```bash
curl -X POST "http://127.0.0.1:5000/api/v1/ml/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "What vegetarian options do you have?", "use_rag": true}'
```

---

## Troubleshooting

### Common Issues

#### Port Already in Use

```powershell
# Windows - Find and kill process on port 5000
netstat -ano | findstr :5000
taskkill /F /PID <PID>
```

```bash
# Linux/Mac
lsof -i :5000
kill -9 <PID>
```

#### Database Connection Issues

```bash
# Reset database
cd backend
rm restaurant.db
python -c "from app.core.database import init_db; import asyncio; asyncio.run(init_db())"
python seed_data.py
```

#### ML Models Not Loading

Ensure you have:

1. Valid Groq API key in `.env`
2. Internet connection (for downloading sentence-transformers)
3. Sufficient disk space for model cache

#### bcrypt Issues

If you see bcrypt errors, ensure you're using a compatible version:

```bash
pip install bcrypt==4.0.1
```

### Logs

```bash
# View server logs
python -m uvicorn app.main:app --log-level debug

# Docker logs
docker-compose logs -f backend
```

### Health Check

```bash
curl http://127.0.0.1:5000/health
```

Expected response:

```json
{
  "status": "healthy",
  "app": "Restaurant SaaS API",
  "version": "1.0.0"
}
```

---

## Support

- **Repository**: https://github.com/ashour-git/Restaurant_AI
- **Issues**: https://github.com/ashour-git/Restaurant_AI/issues

---

## License

MIT License - See LICENSE file for details.
