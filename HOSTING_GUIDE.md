# Hosting Guide: Supabase & Railway

This guide explains how to host your **Smart Restaurant** application using **Supabase** for the database and **Railway** for the backend/frontend services.

## 1. Database Setup (Supabase)

You have already configured the project to use Supabase.

### Verification

Ensure your `backend/.env` file contains the correct connection string:

```env
DATABASE_URL=postgresql+asyncpg://postgres:[PASSWORD]@db.[PROJECT_REF].supabase.co:5432/postgres
```

_Note: If your password contains special characters, they must be URL-encoded._

### Troubleshooting Connection

If you encounter `getaddrinfo failed` or DNS errors:

1. Verify your **Project Reference** (the part after `db.` and before `.supabase.co`).
2. Check if your project is "Paused" in the Supabase Dashboard.
3. Try using the **Transaction Pooler** connection string (port 6543) if direct connection (port 5432) fails.
   - Go to Supabase Dashboard -> Settings -> Database -> Connection String -> URI -> Mode: Transaction.
   - Update `DATABASE_URL` in `backend/.env`.

### Running Migrations

Once the connection is verified, apply the database schema:

```powershell
# In the root directory
$env:PYTHONPATH = "backend"
.\.venv\Scripts\python.exe -m alembic upgrade head
```

## 2. Backend Hosting (Render - Free Tier)

Render offers a free tier for Web Services that is perfect for this project.

### Steps:

1.  **Create a Render Account** at [render.com](https://render.com).
2.  **New +** -> **Web Service**.
3.  **Connect GitHub** and select your repository.
4.  **Configuration**:
    - **Name**: `restaurant-api` (or similar)
    - **Region**: Choose closest to you (e.g., Oregon, Frankfurt)
    - **Branch**: `main`
    - **Root Directory**: `.` (Leave empty)
    - **Runtime**: `Python 3`
    - **Build Command**: `pip install -r requirements.txt`
    - **Start Command**: `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`
    - **Instance Type**: `Free`
5.  **Environment Variables** (Advanced):
    - Add all variables from `backend/.env`:
      - `DATABASE_URL`: Your Supabase connection string.
      - `SECRET_KEY`: A random secure string.
      - `GROQ_API_KEY`: Your Groq API key.
      - `CORS_ORIGINS_STR`: `https://your-frontend.vercel.app,http://localhost:3000`
      - `ENVIRONMENT`: `production`
      - `PYTHON_VERSION`: `3.11.9` (Optional, but recommended)

### Note on Free Tier

Render's free tier spins down after 15 minutes of inactivity. The first request might take 30-60 seconds to load. For a production app, consider upgrading to the Starter plan ($7/mo).

## 3. Frontend Hosting (Vercel or Railway)

Vercel is recommended for Next.js.

### Vercel (Recommended)

1. Import the project in Vercel.
2. Set **Root Directory** to `frontend`.
3. Add Environment Variables from `frontend/.env.local`:
   - `NEXT_PUBLIC_API_URL`: Your deployed Railway Backend URL (e.g., `https://smart-restaurant-production.up.railway.app/api/v1`).
   - `NEXT_PUBLIC_SUPABASE_URL`
   - `NEXT_PUBLIC_SUPABASE_ANON_KEY`
4. Deploy.

### Railway

1. Add a new service in your Railway project.
2. Select the same repo.
3. Set **Root Directory** to `frontend`.
4. Add Environment Variables.
5. Railway will detect Next.js and build it.

## 4. Connecting Frontend to Backend

1. Once Backend is deployed on Railway, copy its URL.
2. Update `NEXT_PUBLIC_API_URL` in your Frontend deployment (Vercel/Railway) to point to the production backend.
3. Update `CORS_ORIGINS` in Backend variables to include your Frontend URL (e.g., `https://your-frontend.vercel.app`).
