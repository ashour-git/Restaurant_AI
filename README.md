# RestAI 🍽️

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.124+-green.svg)](https://fastapi.tiangolo.com/)
[![Next.js 15](https://img.shields.io/badge/Next.js-15+-black.svg)](https://nextjs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready AI-powered Restaurant Management SaaS platform built with FastAPI, Next.js, and Machine Learning.

## Features

### Core Features

- **POS System**: Intuitive point-of-sale interface with cart management
- **Menu Management**: Full CRUD for categories, subcategories, and menu items
- **Order Management**: Real-time order tracking with status updates
- **Customer Management**: Customer profiles with loyalty tiers
- **Inventory Tracking**: Stock management with low-stock alerts
- **Analytics Dashboard**: Sales reports, trends, and insights

### AI-Powered Features

- **Demand Forecasting**: Time-series prediction using LightGBM with Optuna hyperparameter tuning
- **Smart Recommendations**: Hybrid recommender (co-occurrence + content-based)
- **AI Assistant**: Groq Llama 3.3 70B powered chatbot with semantic RAG (sentence-transformers)

### Security Features

- **JWT Authentication**: Secure token-based authentication
- **Role-Based Access Control**: Admin, Manager, Staff roles
- **Password Hashing**: BCrypt password hashing
- **Protected API Routes**: Secure endpoints with proper authorization

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+

### 1. Clone & Setup

```bash
git clone https://github.com/ashour-git/Restaurant_AI.git
cd Restaurant_AI

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy and edit the backend .env file
cp backend/.env.example backend/.env
# Edit with your GROQ_API_KEY and other settings
```

### 3. Seed Database (Optional)

```bash
cd backend
python seed_data.py
```

### 4. Start Backend

```bash
# From project root
python run.py

# Or with custom port
python run.py --port 8000
```

The backend will be available at:

- **API**: http://localhost:5000
- **Docs**: http://localhost:5000/docs
- **Health**: http://localhost:5000/health

### 5. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

The frontend will be available at:

- **App**: http://localhost:3000

## Authentication

### Register a New User

```bash
curl -X POST http://localhost:5000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "admin@restaurant.com", "password": "securepass123", "first_name": "Admin", "last_name": "User", "role": "admin"}'
```

### Login

```bash
curl -X POST http://localhost:5000/api/v1/auth/login \
  -d "username=admin@restaurant.com&password=securepass123"
```

### Use Protected Endpoints

```bash
curl http://localhost:5000/api/v1/auth/me \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

## Deployment

For detailed deployment instructions, see **[DEPLOYMENT.md](DEPLOYMENT.md)** which covers:

- Local development setup
- Docker deployment
- Production deployment with Gunicorn/Nginx
- Environment configuration
- Troubleshooting guide

## Architecture

```
Restaurant_AI/
├── backend/                 # FastAPI Backend
│   ├── app/
│   │   ├── api/            # API routes (auth, menu, orders, etc.)
│   │   ├── core/           # Config, database, security
│   │   ├── models/         # SQLAlchemy models
│   │   └── schemas/        # Pydantic schemas
│   └── seed_data.py        # Database seeder
├── frontend/               # Next.js 16 Frontend
│   └── src/
│       ├── app/            # App router pages
│       ├── components/     # React components
│       └── lib/            # API client
├── ml/                     # Machine Learning
│   ├── pipelines/          # ML pipelines
│   │   ├── demand_forecasting.py
│   │   ├── enhanced_forecasting.py  # Optuna optimization
│   │   ├── recommender.py
│   │   └── nlp_assistant.py         # Semantic RAG
│   └── utils/              # Data utilities
├── data/                   # Datasets
├── tests/                  # Test suite (162 tests)
└── run.py                  # Single entry point
```

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Run migrations
alembic upgrade head

# Start server
uvicorn app.main:app --reload
```

2. **Frontend Setup**

   ```bash
   cd frontend
   npm install
   npm run dev
   ```

3. **Generate Sample Data**
   ```bash
   python data/generate_datasets.py
   ```

## API Endpoints

### Menu

- `GET /api/v1/menu/categories` - List categories
- `GET /api/v1/menu/items` - List menu items
- `POST /api/v1/menu/items` - Create menu item
- `PUT /api/v1/menu/items/{id}` - Update menu item

### Orders

- `GET /api/v1/orders` - List orders
- `POST /api/v1/orders` - Create order
- `PATCH /api/v1/orders/{id}/status` - Update order status

### Analytics

- `GET /api/v1/analytics/dashboard` - Dashboard metrics
- `GET /api/v1/analytics/sales` - Sales report
- `GET /api/v1/analytics/top-items` - Top selling items

### ML Features

- `POST /api/v1/ml/forecast` - Demand forecast
- `POST /api/v1/ml/recommend` - Item recommendations
- `POST /api/v1/ml/chat` - AI assistant chat
- `GET /api/v1/ml/menu-search` - Semantic menu search

## ML Pipelines

### Demand Forecasting

Uses LightGBM with time-series features:

- Lag features (7, 14, 21, 28 days)
- Rolling statistics (7, 14, 30-day windows)
- Calendar features (day of week, month, holidays)

### Hybrid Recommender

Combines two approaches:

1. **Co-occurrence**: Items frequently ordered together
2. **Content-based**: Semantic similarity using embeddings

### NLP Assistant

- **RAG Architecture**: Retrieves relevant menu context
- **Claude Integration**: Natural language understanding
- **Knowledge Base**: Menu items, policies, operational data

## Configuration

Key environment variables:

```bash
# Database
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/restaurant_db

# Redis
REDIS_URL=redis://localhost:6379/0

# API Keys
ANTHROPIC_API_KEY=your-api-key-here

# Security
SECRET_KEY=your-secret-key
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

## Data Schema

### Core Tables

- `categories` / `subcategories` - Menu organization
- `menu_items` - Dishes with prices, descriptions
- `orders` / `order_items` - Transaction records
- `customers` / `loyalty_tiers` - Customer data
- `inventory_items` / `inventory_transactions` - Stock management

### ML Tables

- `demand_forecasts` - Predicted demand
- `menu_item_embeddings` - Vector embeddings (pgvector)

## Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=app --cov-report=html

# Specific test file
pytest tests/test_orders.py
```

## Deployment

### Production Docker Build

```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Environment Checklist

- [ ] Set strong `SECRET_KEY`
- [ ] Configure production `DATABASE_URL`
- [ ] Set `ENVIRONMENT=production`
- [ ] Enable HTTPS/SSL
- [ ] Configure CORS origins
- [ ] Set up monitoring (Prometheus/Grafana)

## Tech Stack

**Backend**

- FastAPI (async Python web framework)
- SQLAlchemy 2.0 (async ORM)
- Alembic (migrations)
- Pydantic (validation)

**Frontend**

- Next.js 14 (React framework)
- TailwindCSS (styling)
- React Query (data fetching)
- Zustand (state management)

**Database**

- PostgreSQL + pgvector
- Redis (caching)

**ML/AI**

- LightGBM (forecasting)
- scikit-learn (preprocessing)
- sentence-transformers (embeddings)
- Anthropic Claude (NLP)

**DevOps**

- Docker & Docker Compose
- GitHub Actions (CI/CD)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

Built for the restaurant industry
#   R e s t a u r a n t _ A I 
 
 
