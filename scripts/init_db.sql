-- Restaurant SaaS Database Initialization Script
-- This script runs when PostgreSQL container starts for the first time
-- Enable pgvector extension for embeddings
CREATE EXTENSION IF NOT EXISTS vector;
-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
-- Create enum types
DO $$ BEGIN CREATE TYPE order_status AS ENUM (
    'pending',
    'preparing',
    'ready',
    'served',
    'cancelled',
    'completed'
);
EXCEPTION
WHEN duplicate_object THEN null;
END $$;
DO $$ BEGIN CREATE TYPE order_type AS ENUM ('dine_in', 'takeout', 'delivery');
EXCEPTION
WHEN duplicate_object THEN null;
END $$;
DO $$ BEGIN CREATE TYPE payment_method AS ENUM (
    'cash',
    'credit_card',
    'debit_card',
    'mobile_payment',
    'gift_card'
);
EXCEPTION
WHEN duplicate_object THEN null;
END $$;
DO $$ BEGIN CREATE TYPE transaction_type AS ENUM (
    'purchase',
    'sale',
    'adjustment',
    'waste',
    'return'
);
EXCEPTION
WHEN duplicate_object THEN null;
END $$;
-- Grant privileges
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO restaurant;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO restaurant;
-- Log successful initialization
DO $$ BEGIN RAISE NOTICE 'Database initialized successfully with pgvector extension';
END $$;
