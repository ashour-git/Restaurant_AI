"""
Restaurant AI Assistant powered by Groq (Llama 3.3).

This module provides an intelligent conversational assistant for restaurant
operations, featuring:
- Menu knowledge and recommendations
- Order assistance
- Inventory insights
- Business analytics Q&A

Uses Groq's ultra-fast inference with Llama 3.3 70B model.
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from groq import Groq

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Data Loading Utilities
# =============================================================================


def get_data_path() -> Path:
    """Get the path to the data directory."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        data_path = current / "data" / "raw"
        if data_path.exists():
            return data_path
        current = current.parent
    return Path("data/raw")


def load_menu_items() -> pd.DataFrame:
    """Load menu items from CSV."""
    try:
        data_path = get_data_path()
        df = pd.read_csv(data_path / "menu_items.csv")
        if "name" in df.columns and "item_name" not in df.columns:
            df = df.rename(columns={"name": "item_name"})
        return df
    except Exception as e:
        logger.warning(f"Could not load menu items: {e}")
        return pd.DataFrame()


def load_inventory() -> pd.DataFrame:
    """Load inventory from CSV."""
    try:
        data_path = get_data_path()
        return pd.read_csv(data_path / "inventory.csv")
    except Exception as e:
        logger.warning(f"Could not load inventory: {e}")
        return pd.DataFrame()


def load_transactions() -> pd.DataFrame:
    """Load transactions from CSV."""
    try:
        data_path = get_data_path()
        return pd.read_csv(data_path / "transactions.csv")
    except Exception as e:
        logger.warning(f"Could not load transactions: {e}")
        return pd.DataFrame()


# =============================================================================
# Vector Store for RAG
# =============================================================================


@dataclass
class Document:
    """A document with content and metadata."""

    content: str
    metadata: dict = field(default_factory=dict)
    embedding: np.ndarray | None = None


class SimpleVectorStore:
    """
    Vector store with optional sentence-transformers for semantic search.

    Falls back to TF-IDF if sentence-transformers is not available.
    """

    def __init__(self, use_semantic: bool = True):
        """
        Initialize the vector store.

        Args:
            use_semantic: Whether to use sentence-transformers (if available).
        """
        self.documents: list[Document] = []
        self.vectorizer = None
        self.embeddings = None
        self.use_semantic = use_semantic
        self.model = None

        # Try to load sentence-transformers for better embeddings
        if use_semantic:
            try:
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("Using sentence-transformers for semantic embeddings")
            except ImportError:
                logger.info("sentence-transformers not available, using TF-IDF")
                self.use_semantic = False

    def add_documents(self, docs: list[dict]) -> None:
        """Add documents to the store."""
        for doc in docs:
            self.documents.append(Document(content=doc.get("content", ""), metadata=doc))

        if not self.documents:
            return

        texts = [d.content for d in self.documents]

        if self.use_semantic and self.model is not None:
            # Use sentence-transformers for semantic embeddings
            self.embeddings = self.model.encode(texts, convert_to_numpy=True)
            logger.info(f"Added {len(docs)} documents with semantic embeddings")
        else:
            # Fallback to TF-IDF
            from sklearn.feature_extraction.text import TfidfVectorizer

            self.vectorizer = TfidfVectorizer(
                max_features=512, stop_words="english", ngram_range=(1, 2)
            )
            self.embeddings = self.vectorizer.fit_transform(texts)
            logger.info(f"Added {len(docs)} documents with TF-IDF embeddings")

    def search(self, query: str, top_k: int = 5) -> list[Document]:
        """Search for similar documents."""
        if not self.documents or self.embeddings is None:
            return []

        if self.use_semantic and self.model is not None:
            # Semantic search with cosine similarity
            query_embedding = self.model.encode([query], convert_to_numpy=True)[0]
            # Normalize for cosine similarity
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            doc_norms = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            similarities = doc_norms @ query_norm
        else:
            # TF-IDF search
            if self.vectorizer is None:
                return []
            query_vec = self.vectorizer.transform([query])
            similarities = (self.embeddings @ query_vec.T).toarray().flatten()

        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.documents[i] for i in top_indices if similarities[i] > 0]


# =============================================================================
# Knowledge Base
# =============================================================================


class RestaurantKnowledgeBase:
    """
    Knowledge base for restaurant information.

    Indexes menu items, inventory, and business data for RAG.
    """

    def __init__(self):
        self.vector_store = SimpleVectorStore()
        self.menu_df: pd.DataFrame | None = None
        self.inventory_df: pd.DataFrame | None = None
        self.transactions_df: pd.DataFrame | None = None

    def load_data(self) -> None:
        """Load all restaurant data into the knowledge base."""
        documents = []

        # Load menu items
        self.menu_df = load_menu_items()
        if not self.menu_df.empty:
            for _, row in self.menu_df.iterrows():
                # Build dietary info
                dietary = []
                if row.get("is_vegetarian"):
                    dietary.append("vegetarian")
                if row.get("is_vegan"):
                    dietary.append("vegan")
                if row.get("is_gluten_free"):
                    dietary.append("gluten-free")

                content = f"""
Menu Item: {row.get('item_name', 'Unknown')}
Category: {row.get('category', 'N/A')}
Subcategory: {row.get('subcategory', 'N/A')}
Description: {row.get('description', 'N/A')}
Price: ${row.get('price', 0):.2f}
Dietary: {', '.join(dietary) if dietary else 'None'}
Prep Time: {row.get('prep_time_minutes', 'N/A')} minutes
Calories: {row.get('calories', 'N/A')}
""".strip()

                documents.append(
                    {
                        "type": "menu_item",
                        "item_id": row.get("item_id"),
                        "item_name": row.get("item_name"),
                        "category": row.get("category"),
                        "price": row.get("price"),
                        "content": content,
                    }
                )

            logger.info(f"Loaded {len(self.menu_df)} menu items")

        # Load inventory
        self.inventory_df = load_inventory()
        if not self.inventory_df.empty:
            for _, row in self.inventory_df.iterrows():
                content = f"""
Inventory Item: {row.get('ingredient_name', 'Unknown')}
Category: {row.get('category', 'N/A')}
Current Stock: {row.get('quantity_on_hand', 0)} {row.get('unit', 'units')}
Reorder Level: {row.get('reorder_level', 'N/A')}
Unit Cost: ${row.get('unit_cost', 0):.2f}
Supplier: {row.get('supplier_id', 'N/A')}
""".strip()

                documents.append({"type": "inventory", "content": content})

            logger.info(f"Loaded {len(self.inventory_df)} inventory items")

        # Add to vector store
        if documents:
            self.vector_store.add_documents(documents)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search the knowledge base."""
        docs = self.vector_store.search(query, top_k)
        return [d.metadata for d in docs]

    def get_menu_summary(self) -> str:
        """Get a summary of the menu."""
        if self.menu_df is None or self.menu_df.empty:
            return "No menu data available."

        categories = self.menu_df["category"].value_counts().to_dict()
        avg_price = self.menu_df["price"].mean()

        summary = f"Menu has {len(self.menu_df)} items across {len(categories)} categories.\n"
        summary += f"Average price: ${avg_price:.2f}\n"
        summary += "Categories: " + ", ".join(
            f"{cat} ({count})" for cat, count in categories.items()
        )
        return summary

    def get_inventory_alerts(self) -> str:
        """Get inventory alerts for low stock items."""
        if self.inventory_df is None or self.inventory_df.empty:
            return "No inventory data available."

        # Check for low stock
        low_stock = self.inventory_df[
            self.inventory_df["quantity_on_hand"] < self.inventory_df["reorder_level"]
        ]

        if low_stock.empty:
            return "All inventory items are well-stocked."

        alerts = [f"Low Stock Alerts ({len(low_stock)} items):"]
        for _, row in low_stock.head(5).iterrows():
            alerts.append(
                f"  - {row['ingredient_name']}: {row['quantity_on_hand']} "
                f"(reorder at {row['reorder_level']})"
            )

        return "\n".join(alerts)

    def sync_live_data(self, menu_items: list[dict], orders: list[dict] | None = None,
                       customers: list[dict] | None = None, inventory: list[dict] | None = None) -> int:
        """
        Sync live database data into the knowledge base.
        
        This enables RAG with real-time restaurant data from the database.
        
        Args:
            menu_items: List of menu item dicts from database
            orders: Optional list of recent orders
            customers: Optional list of customers
            inventory: Optional list of inventory items
            
        Returns:
            Number of documents indexed
        """
        documents = []
        
        # Index menu items from live database
        for item in menu_items:
            dietary = []
            if item.get("is_vegetarian"):
                dietary.append("vegetarian")
            if item.get("is_vegan"):
                dietary.append("vegan")
            if item.get("is_gluten_free"):
                dietary.append("gluten-free")
            
            content = f"""
Menu Item: {item.get('name', 'Unknown')}
Category: {item.get('category', 'N/A')}
Description: {item.get('description', 'N/A')}
Price: ${item.get('price', 0):.2f}
Dietary: {', '.join(dietary) if dietary else 'None'}
Active: {'Yes' if item.get('is_active', True) else 'No'}
""".strip()
            
            documents.append({
                "type": "menu_item",
                "item_id": item.get("id"),
                "item_name": item.get("name"),
                "category": item.get("category"),
                "price": item.get("price"),
                "description": item.get("description"),
                "is_vegetarian": item.get("is_vegetarian", False),
                "is_vegan": item.get("is_vegan", False),
                "is_gluten_free": item.get("is_gluten_free", False),
                "content": content,
            })
        
        # Update internal DataFrame for fallback queries
        if menu_items:
            self.menu_df = pd.DataFrame(menu_items)
            if "name" in self.menu_df.columns and "item_name" not in self.menu_df.columns:
                self.menu_df["item_name"] = self.menu_df["name"]
        
        # Index recent orders for analytics context
        if orders:
            order_summary = f"""Recent Order Activity:
- Total recent orders: {len(orders)}
- Latest orders: {', '.join([f"#{o.get('order_number', o.get('id', 'N/A'))}" for o in orders[:5]])}
"""
            documents.append({
                "type": "order_analytics",
                "content": order_summary,
            })
        
        # Index customer data for insights
        if customers:
            customer_summary = f"""Customer Base:
- Total customers: {len(customers)}
- Active customers with profiles for personalized recommendations
"""
            documents.append({
                "type": "customer_analytics",
                "content": customer_summary,
            })
        
        # Index inventory
        if inventory:
            for inv in inventory:
                content = f"""
Inventory Item: {inv.get('ingredient_name', inv.get('name', 'Unknown'))}
Category: {inv.get('category', 'N/A')}
Current Stock: {inv.get('quantity_on_hand', inv.get('quantity', 0))} {inv.get('unit', 'units')}
Reorder Level: {inv.get('reorder_level', 'N/A')}
""".strip()
                documents.append({"type": "inventory", "content": content})
            
            self.inventory_df = pd.DataFrame(inventory)
        
        # Rebuild vector store with new documents
        if documents:
            self.vector_store = SimpleVectorStore()
            self.vector_store.add_documents(documents)
            logger.info(f"Synced {len(documents)} live documents to knowledge base")
        
        return len(documents)


# =============================================================================
# Groq-Powered Restaurant Assistant
# =============================================================================

SYSTEM_PROMPT = """You are RestAI, an advanced AI assistant for a restaurant management system called "Smart Restaurant SaaS".

You have REAL-TIME access to the restaurant's database including:
- **Live Menu Data**: All current menu items with prices, descriptions, dietary info
- **Recent Orders**: Latest order activity and trends
- **Customer Insights**: Customer preferences and analytics  
- **Inventory Status**: Stock levels and alerts

Your capabilities include:
1. **Menu Expert**: Answer questions about menu items, prices, ingredients, dietary options (vegetarian, vegan, gluten-free)
2. **Smart Recommendations**: Suggest dishes based on preferences, dietary needs, pairings, or mood
3. **Inventory Intelligence**: Provide real-time stock levels and low-stock alerts
4. **Business Analytics**: Answer questions about sales trends, popular items, peak hours
5. **Order Assistance**: Help with order-related queries, modifications, upselling suggestions
6. **Customer Intelligence**: Insights about customer preferences and dining patterns

Response Guidelines:
- Be helpful, friendly, and professional - you represent the restaurant
- Provide specific, actionable information using REAL data from the knowledge base
- Always use the context provided - it contains live restaurant data
- Format prices with $ and two decimal places (e.g., $12.99)
- Use bullet points, emojis, and clear formatting for readability
- For recommendations, always suggest 3-5 specific items from the menu
- When suggesting items, include the price and key attributes
- If asked about something not in your data, say so clearly and offer alternatives
- Keep responses concise but comprehensive

IMPORTANT: The menu data you receive is REAL and CURRENT from the restaurant's database. Use it to give accurate, personalized recommendations."""


@dataclass
class Message:
    """A chat message."""

    role: str  # "user", "assistant", or "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)


class RestaurantAssistant:
    """
    AI-powered restaurant assistant using Groq's Llama 3.3.

    Features:
    - Conversational interface
    - RAG with restaurant knowledge base
    - Fast inference via Groq

    Example:
        >>> assistant = RestaurantAssistant()
        >>> assistant.initialize()
        >>> response = assistant.chat("What vegetarian options do you have?")
        >>> print(response)
    """

    def __init__(self, api_key: str | None = None, model: str = "llama-3.3-70b-versatile"):
        """
        Initialize the restaurant assistant.

        Args:
            api_key: Groq API key. Falls back to GROQ_API_KEY env var.
            model: Groq model to use.
        """
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.model = model
        self.client: Groq | None = None
        self.knowledge_base = RestaurantKnowledgeBase()
        self.conversation_history: list[Message] = []
        self.initialized = False

    def initialize(self) -> None:
        """Initialize the assistant with data and API client."""
        if self.initialized:
            return

        # Initialize Groq client
        if self.api_key:
            self.client = Groq(api_key=self.api_key)
            logger.info("Groq client initialized")
        else:
            logger.warning("No Groq API key found - assistant will run in limited mode")

        # Load knowledge base
        self.knowledge_base.load_data()
        self.initialized = True
        logger.info("Restaurant assistant initialized successfully")

    def _build_context(self, query: str) -> str:
        """Build context from knowledge base for the query."""
        # Search for relevant documents
        results = self.knowledge_base.search(query, top_k=5)

        if not results:
            return ""

        context_parts = ["Relevant Information:"]
        for doc in results:
            if "content" in doc:
                context_parts.append(doc["content"])

        return "\n\n".join(context_parts)

    def _get_system_context(self) -> str:
        """Get additional system context."""
        parts = [SYSTEM_PROMPT]

        # Add menu summary
        menu_summary = self.knowledge_base.get_menu_summary()
        parts.append(f"\n\nCurrent Menu Overview:\n{menu_summary}")

        # Add inventory alerts if any
        inventory_alerts = self.knowledge_base.get_inventory_alerts()
        if "Low Stock" in inventory_alerts:
            parts.append(f"\n\n{inventory_alerts}")

        return "\n".join(parts)

    def chat(
        self,
        message: str,
        include_context: bool = True,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """
        Chat with the assistant.

        Args:
            message: User's message.
            include_context: Whether to include RAG context.
            temperature: Response creativity (0-1).
            max_tokens: Maximum response length.

        Returns:
            Assistant's response.

        Example:
            >>> response = assistant.chat("What's your best steak?")
            >>> print(response)
        """
        if not self.initialized:
            self.initialize()

        # Add user message to history
        self.conversation_history.append(Message(role="user", content=message))

        # Build context if needed
        context = ""
        if include_context:
            context = self._build_context(message)

        # If no API key, provide a helpful response
        if not self.client:
            response = self._fallback_response(message, context)
            self.conversation_history.append(Message(role="assistant", content=response))
            return response

        # Build messages for Groq
        messages = [{"role": "system", "content": self._get_system_context()}]

        # Add context if available
        if context:
            messages.append(
                {"role": "system", "content": f"Use this context to answer:\n\n{context}"}
            )

        # Add conversation history (last 10 messages)
        for msg in self.conversation_history[-10:]:
            messages.append({"role": msg.role, "content": msg.content})

        try:
            # Call Groq API
            completion = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens
            )

            response = completion.choices[0].message.content

        except Exception as e:
            logger.error(f"Groq API error: {e}")
            response = f"I apologize, but I encountered an error. Please try again. (Error: {str(e)[:100]})"

        # Add to history
        self.conversation_history.append(Message(role="assistant", content=response))

        return response

    def _fallback_response(self, message: str, context: str) -> str:
        """Provide a smart response using RAG context when API is not available."""
        message_lower = message.lower()
        df = self.knowledge_base.menu_df

        # Spicy/flavor queries - CHECK FIRST before recommendations
        if any(word in message_lower for word in ["spicy", "hot wing", "buffalo", "jalapeño", "cajun", "sriracha", "heat"]):
            if df is not None and not df.empty:
                # Search for spicy items in name or description
                name_col = "item_name" if "item_name" in df.columns else "name"
                spicy_mask = (
                    df["description"].str.lower().str.contains("spic|buffalo|cajun|pepper|jalap|hot sauce|chili|wing", na=False, regex=True) |
                    df[name_col].str.lower().str.contains("buffalo|cajun|spic|hot|wing", na=False, regex=True)
                )
                spicy_items = df[spicy_mask]
                if not spicy_items.empty:
                    response = "🌶️ **Spicy Menu Items:**\n\n"
                    for _, row in spicy_items.head(5).iterrows():
                        price = f"${row.get('price', 0):.2f}" if pd.notna(row.get('price')) else ""
                        desc = row.get('description', '')[:70] if pd.notna(row.get('description')) else ''
                        response += f"• **{row.get(name_col, 'Item')}** {price}\n  {desc}\n\n"
                    response += "🔥 *Love the heat? These items will spice things up!*"
                    return response.strip()
                # If no spicy items found, suggest flavor-forward options
                return (
                    "🌶️ **Looking for something with kick?**\n\n"
                    "While we don't have explicitly 'spicy' items on our current menu, "
                    "I'd recommend:\n\n"
                    "• **Philly Cheesesteak** - Sliced beef with peppers and onions\n"
                    "• Ask your server about adding hot sauce or jalapeños!\n\n"
                    "🍳 *Would you like to see our full menu instead?*"
                )

        # Recommendation queries - prioritize menu items over inventory
        if any(word in message_lower for word in ["recommend", "suggest", "try", "best", "popular", "favorite", "what's good"]):
            # Always use menu data directly for recommendations (not RAG which might return inventory)
            if df is not None and not df.empty:
                name_col = "item_name" if "item_name" in df.columns else "name"
                # Get featured/popular items
                if "is_featured" in df.columns:
                    featured = df[df["is_featured"] == True]
                    if featured.empty:
                        featured = df.sample(min(5, len(df)))
                else:
                    featured = df.sample(min(5, len(df)))
                items = featured.head(5)
                response = "🌟 **Today's Top Recommendations:**\n\n"
                for _, row in items.iterrows():
                    price = f"${row.get('price', 0):.2f}" if pd.notna(row.get('price')) else ""
                    desc = row.get('description', '')[:70] if pd.notna(row.get('description')) else ''
                    category = row.get('category', '') if pd.notna(row.get('category')) else ''
                    response += f"• **{row.get(name_col, 'Item')}** {price}\n  _{category}_ - {desc}\n\n"
                response += "💡 *Ask about specific dietary needs or cuisine preferences!*"
                return response.strip()

        # Menu/food queries  
        if any(word in message_lower for word in ["menu", "item", "food", "dish", "eat", "what do you have", "show me"]):
            if context:
                return f"📋 **From Our Menu:**\n\n{context}\n\n💡 *Ask about specific categories like appetizers, mains, or desserts!*"
            return self.knowledge_base.get_menu_summary()

        # Vegetarian/dietary queries
        if any(word in message_lower for word in ["vegetarian", "vegan", "gluten", "healthy", "diet"]):
            if df is not None and not df.empty:
                if "vegetarian" in message_lower and "is_vegetarian" in df.columns:
                    items = df[df["is_vegetarian"] == True]
                    if not items.empty:
                        response = "🥗 **Vegetarian Options:**\n\n"
                        for _, row in items.head(6).iterrows():
                            price = f"${row.get('price', 0):.2f}" if pd.notna(row.get('price')) else ""
                            response += f"• **{row.get('item_name', 'Item')}** {price}\n"
                        return response
                if "vegan" in message_lower and "is_vegan" in df.columns:
                    items = df[df["is_vegan"] == True]
                    if not items.empty:
                        response = "🌱 **Vegan Options:**\n\n"
                        for _, row in items.head(6).iterrows():
                            price = f"${row.get('price', 0):.2f}" if pd.notna(row.get('price')) else ""
                            response += f"• **{row.get('item_name', 'Item')}** {price}\n"
                        return response
                if "gluten" in message_lower and "is_gluten_free" in df.columns:
                    items = df[df["is_gluten_free"] == True]
                    if not items.empty:
                        response = "🌾 **Gluten-Free Options:**\n\n"
                        for _, row in items.head(6).iterrows():
                            price = f"${row.get('price', 0):.2f}" if pd.notna(row.get('price')) else ""
                            response += f"• **{row.get('item_name', 'Item')}** {price}\n"
                        return response
            if context:
                return f"🥗 **Dietary Options:**\n\n{context}"

        # Dessert queries
        if any(word in message_lower for word in ["dessert", "sweet", "cake", "pie", "ice cream", "chocolate"]):
            if context:
                return f"🍰 **Dessert Menu:**\n\n{context}\n\n🍨 *Perfect way to end your meal!*"
            if df is not None and not df.empty:
                desserts = df[df["category"].str.lower().str.contains("dessert", na=False)] if "category" in df.columns else pd.DataFrame()
                if not desserts.empty:
                    response = "🍰 **Our Desserts:**\n\n"
                    for _, row in desserts.head(5).iterrows():
                        price = f"${row.get('price', 0):.2f}" if pd.notna(row.get('price')) else ""
                        response += f"• **{row.get('item_name', 'Item')}** {price}\n  {row.get('description', '')[:60]}\n\n"
                    return response.strip()

        # Price queries
        if any(word in message_lower for word in ["price", "cost", "cheap", "expensive", "affordable", "budget"]):
            if df is not None and not df.empty and "price" in df.columns:
                avg = df["price"].mean()
                min_price = df["price"].min()
                max_price = df["price"].max()
                cheapest = df.nsmallest(3, "price")
                response = f"💰 **Pricing Overview:**\n\n"
                response += f"• Average price: **${avg:.2f}**\n"
                response += f"• Price range: **${min_price:.2f}** - **${max_price:.2f}**\n\n"
                response += "**Budget-Friendly Options:**\n"
                for _, row in cheapest.iterrows():
                    response += f"• {row.get('item_name', 'Item')} - ${row['price']:.2f}\n"
                return response

        # Inventory queries
        if any(word in message_lower for word in ["inventory", "stock", "supply", "ingredient"]):
            return self.knowledge_base.get_inventory_alerts()

        # Greeting
        if any(word in message_lower for word in ["hello", "hi", "hey", "good morning", "good evening"]):
            return (
                "👋 **Hello! Welcome to Smart Restaurant!**\n\n"
                "I'm your AI assistant and I can help you with:\n"
                "• 🍽️ Menu recommendations\n"
                "• 🥗 Dietary options (vegetarian, vegan, gluten-free)\n"
                "• 🌶️ Spicy or mild preferences\n"
                "• 💰 Pricing information\n"
                "• 📦 Inventory insights\n\n"
                "What would you like to know?"
            )

        # If we have RAG context, use it for any query
        if context:
            return f"📋 **Here's what I found:**\n\n{context}\n\n💡 *Need more specific info? Just ask!*"

        # Default response
        return (
            "🤖 **I'm the Smart Restaurant Assistant!**\n\n"
            "I can help you with:\n"
            "• 🍽️ Menu items and recommendations\n"
            "• 🥗 Dietary options (vegetarian, vegan, gluten-free)\n"
            "• 🌶️ Spicy food suggestions\n"
            "• 🍰 Desserts and drinks\n"
            "• 💰 Pricing information\n"
            "• 📦 Inventory and stock information\n\n"
            "**Try asking:**\n"
            "• *\"Recommend something spicy\"*\n"
            "• *\"What vegetarian options do you have?\"*\n"
            "• *\"Show me desserts\"*\n"
            "• *\"What's popular today?\"*"
        )

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []

    @property
    def document_store(self) -> RestaurantKnowledgeBase:
        """Get the document store/knowledge base."""
        return self.knowledge_base
    
    def sync_knowledge(self, menu_items: list[dict], orders: list[dict] | None = None,
                       customers: list[dict] | None = None, inventory: list[dict] | None = None) -> int:
        """
        Sync live database data into the assistant's knowledge base.
        
        Args:
            menu_items: List of menu item dicts from database
            orders: Optional list of recent orders
            customers: Optional list of customers  
            inventory: Optional list of inventory items
            
        Returns:
            Number of documents indexed
        """
        return self.knowledge_base.sync_live_data(menu_items, orders, customers, inventory)

    def get_recommendations(
        self,
        preferences: list[str] | None = None,
        dietary: list[str] | None = None,
        max_price: float | None = None,
        count: int = 5,
    ) -> list[dict]:
        """
        Get menu recommendations based on criteria.

        Args:
            preferences: Food preferences (e.g., ["spicy", "seafood"])
            dietary: Dietary restrictions (e.g., ["vegetarian"])
            max_price: Maximum price filter
            count: Number of recommendations

        Returns:
            List of recommended items.
        """
        if self.knowledge_base.menu_df is None or self.knowledge_base.menu_df.empty:
            return []

        df = self.knowledge_base.menu_df.copy()

        # Apply dietary filters
        if dietary:
            for diet in dietary:
                if diet.lower() == "vegetarian" and "is_vegetarian" in df.columns:
                    df = df[df["is_vegetarian"] == True]
                elif diet.lower() == "vegan" and "is_vegan" in df.columns:
                    df = df[df["is_vegan"] == True]
                elif diet.lower() == "gluten-free" and "is_gluten_free" in df.columns:
                    df = df[df["is_gluten_free"] == True]

        # Apply price filter
        if max_price and "price" in df.columns:
            df = df[df["price"] <= max_price]

        # Apply preference search
        if preferences and "description" in df.columns:
            mask = (
                df["description"]
                .str.lower()
                .apply(lambda x: any(p.lower() in str(x) for p in preferences))
            )
            if mask.any():
                df = df[mask]

        # Return top items
        recommendations = []
        for _, row in df.head(count).iterrows():
            recommendations.append(
                {
                    "item_id": row.get("item_id"),
                    "name": row.get("item_name"),
                    "category": row.get("category"),
                    "price": row.get("price"),
                    "description": row.get("description"),
                }
            )

        return recommendations


# =============================================================================
# Factory Function
# =============================================================================

_assistant_instance: RestaurantAssistant | None = None


def create_assistant(api_key: str | None = None) -> RestaurantAssistant:
    """
    Create or get the restaurant assistant instance.

    Args:
        api_key: Optional Groq API key.

    Returns:
        Initialized RestaurantAssistant.

    Example:
        >>> assistant = create_assistant()
        >>> response = assistant.chat("Show me appetizers")
    """
    global _assistant_instance

    if _assistant_instance is None:
        _assistant_instance = RestaurantAssistant(api_key=api_key)
        _assistant_instance.initialize()

    return _assistant_instance


# =============================================================================
# CLI for Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Smart Restaurant AI Assistant")
    print("=" * 60)
    print("Powered by Groq (Llama 3.3 70B)")
    print("Type 'quit' to exit, 'clear' to reset conversation\n")

    assistant = create_assistant()

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "quit":
                print("\nGoodbye!")
                break

            if user_input.lower() == "clear":
                assistant.clear_history()
                print("Conversation cleared.")
                continue

            response = assistant.chat(user_input)
            print(f"\nAssistant: {response}")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
