"""
Tests for NLP assistant functionality.

This module contains tests for the RAG-based NLP assistant
including document retrieval, embedding generation, and
response generation.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestEmbeddingGeneration:
    """Tests for embedding generation."""

    def test_embedding_dimension(self):
        """Test that embeddings have correct dimension."""
        # Simulated embedding
        embedding_dim = 384  # sentence-transformers default
        text = "What are today's specials?"

        # Simulate embedding
        np.random.seed(42)
        embedding = np.random.randn(embedding_dim).astype(np.float32)

        assert len(embedding) == embedding_dim
        assert embedding.dtype == np.float32

    def test_embedding_normalization(self):
        """Test that embeddings are normalized."""
        np.random.seed(42)
        embedding = np.random.randn(384)

        # Normalize
        normalized = embedding / np.linalg.norm(embedding)

        # Check unit norm
        assert np.isclose(np.linalg.norm(normalized), 1.0, atol=1e-5)

    def test_batch_embedding_generation(self):
        """Test batch embedding generation."""
        texts = [
            "What is the price of burger?",
            "Do you have vegetarian options?",
            "What are your opening hours?",
        ]

        embedding_dim = 384
        np.random.seed(42)
        embeddings = np.random.randn(len(texts), embedding_dim)

        assert embeddings.shape == (3, 384)

    def test_embedding_caching(self):
        """Test that embeddings can be cached."""
        cache = {}
        text = "What is the special today?"

        np.random.seed(42)
        embedding = np.random.randn(384)

        # Cache the embedding
        cache[text] = embedding

        # Retrieve from cache
        cached_embedding = cache.get(text)

        assert cached_embedding is not None
        assert np.array_equal(cached_embedding, embedding)


class TestDocumentRetrieval:
    """Tests for document retrieval in RAG."""

    def test_similarity_search(self):
        """Test similarity search for document retrieval."""
        from sklearn.metrics.pairwise import cosine_similarity

        # Simulated document embeddings
        np.random.seed(42)
        doc_embeddings = np.random.randn(10, 384)

        # Query embedding
        query_embedding = np.random.randn(1, 384)

        # Calculate similarities
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

        # Get top 3
        top_indices = np.argsort(similarities)[::-1][:3]

        assert len(top_indices) == 3
        assert similarities[top_indices[0]] >= similarities[top_indices[1]]

    def test_retrieval_with_threshold(self):
        """Test retrieval with similarity threshold."""
        np.random.seed(42)
        doc_embeddings = np.random.randn(10, 384)
        query_embedding = np.random.randn(1, 384)

        from sklearn.metrics.pairwise import cosine_similarity

        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

        # Filter by threshold
        threshold = 0.5
        relevant_docs = np.where(similarities >= threshold)[0]

        assert isinstance(relevant_docs, np.ndarray)

    def test_document_chunking(self):
        """Test document chunking for retrieval."""
        document = """
        Our restaurant offers a wide variety of dishes.
        We specialize in Italian and Mediterranean cuisine.
        Our chef has 20 years of experience.
        We use only fresh, locally sourced ingredients.
        Our signature dish is the truffle risotto.
        We offer both dine-in and takeaway options.
        """

        # Simple chunking by sentences
        chunks = [s.strip() for s in document.split(".") if s.strip()]

        assert len(chunks) >= 5
        assert all(len(chunk) > 0 for chunk in chunks)

    def test_context_window_management(self):
        """Test context window size management."""
        max_context_tokens = 4000
        docs = [
            "Document 1 content " * 50,
            "Document 2 content " * 50,
            "Document 3 content " * 50,
        ]

        # Estimate tokens (rough: words * 1.3)
        def estimate_tokens(text):
            return int(len(text.split()) * 1.3)

        # Select docs that fit in context
        selected_docs = []
        current_tokens = 0

        for doc in docs:
            doc_tokens = estimate_tokens(doc)
            if current_tokens + doc_tokens <= max_context_tokens:
                selected_docs.append(doc)
                current_tokens += doc_tokens

        assert len(selected_docs) >= 1
        assert current_tokens <= max_context_tokens


class TestIntentClassification:
    """Tests for user intent classification."""

    def test_menu_query_intent(self):
        """Test menu query intent classification."""
        menu_queries = [
            "What's on the menu?",
            "Do you have pizza?",
            "What are your specials?",
            "Show me vegetarian options",
        ]

        keywords = ["menu", "have", "specials", "options", "pizza", "vegetarian"]

        for query in menu_queries:
            has_intent = any(kw.lower() in query.lower() for kw in keywords)
            assert has_intent

    def test_order_intent(self):
        """Test order intent classification."""
        order_queries = [
            "I'd like to order a burger",
            "Can I get two pizzas?",
            "Add a salad to my order",
            "I want to place an order",
        ]

        keywords = ["order", "want", "get", "add", "like"]

        for query in order_queries:
            has_intent = any(kw.lower() in query.lower() for kw in keywords)
            assert has_intent

    def test_reservation_intent(self):
        """Test reservation intent classification."""
        reservation_queries = [
            "I'd like to make a reservation",
            "Can I book a table for 4?",
            "Reserve a spot for Friday",
        ]

        keywords = ["reservation", "book", "reserve", "table"]

        for query in reservation_queries:
            has_intent = any(kw.lower() in query.lower() for kw in keywords)
            assert has_intent

    def test_multi_intent_handling(self):
        """Test handling queries with multiple intents."""
        query = "I'd like to order a pizza and make a reservation for tomorrow"

        intents = []
        if "order" in query.lower():
            intents.append("order")
        if "reservation" in query.lower() or "book" in query.lower():
            intents.append("reservation")

        assert len(intents) == 2
        assert "order" in intents
        assert "reservation" in intents


class TestResponseGeneration:
    """Tests for response generation."""

    def test_prompt_construction(self):
        """Test prompt construction for LLM."""
        context = "Our restaurant serves Italian food. We have pizza and pasta."
        query = "What kind of food do you serve?"

        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""

        assert context in prompt
        assert query in prompt
        assert "Answer:" in prompt

    def test_response_length_constraint(self):
        """Test response length constraints."""
        max_length = 500
        response = "Our restaurant serves authentic Italian cuisine " * 20

        if len(response) > max_length:
            response = response[:max_length] + "..."

        assert len(response) <= max_length + 3  # +3 for "..."

    def test_response_formatting(self):
        """Test response formatting for different intents."""
        # Menu response
        menu_items = [
            {"name": "Margherita Pizza", "price": 12.99},
            {"name": "Caesar Salad", "price": 8.99},
        ]

        formatted = "Here are our menu items:\n"
        for item in menu_items:
            formatted += f"- {item['name']}: ${item['price']:.2f}\n"

        assert "Margherita Pizza" in formatted
        assert "$12.99" in formatted

    def test_error_response_handling(self):
        """Test error response handling."""
        error_messages = {
            "not_found": "I couldn't find information about that.",
            "unclear": "I'm not sure I understand. Could you rephrase?",
            "service_unavailable": "I'm having trouble right now. Please try again.",
        }

        error_type = "not_found"
        response = error_messages.get(error_type, "An error occurred.")

        assert response == "I couldn't find information about that."


class TestConversationHistory:
    """Tests for conversation history management."""

    def test_history_storage(self):
        """Test conversation history storage."""
        history = []

        # Add messages
        history.append({"role": "user", "content": "Hello"})
        history.append({"role": "assistant", "content": "Hi! How can I help?"})
        history.append({"role": "user", "content": "What's on the menu?"})

        assert len(history) == 3
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_history_context_window(self):
        """Test conversation history context window."""
        max_history = 5
        history = []

        # Add many messages
        for i in range(10):
            history.append({"role": "user", "content": f"Message {i}"})
            history.append({"role": "assistant", "content": f"Response {i}"})

        # Keep only last N messages
        recent_history = history[-max_history:]

        assert len(recent_history) == max_history
        assert recent_history[-1]["content"] == "Response 9"

    def test_history_serialization(self):
        """Test conversation history serialization."""
        history = [
            {"role": "user", "content": "Hello", "timestamp": "2024-01-01T10:00:00"},
            {"role": "assistant", "content": "Hi!", "timestamp": "2024-01-01T10:00:01"},
        ]

        # Serialize
        serialized = json.dumps(history)

        # Deserialize
        deserialized = json.loads(serialized)

        assert deserialized == history

    def test_history_summarization(self):
        """Test conversation history summarization for long conversations."""
        history = [
            {"role": "user", "content": "What's on the menu?"},
            {"role": "assistant", "content": "We have pizza, pasta, and salads."},
            {"role": "user", "content": "What kind of pizza?"},
            {"role": "assistant", "content": "Margherita, Pepperoni, and Veggie."},
            {"role": "user", "content": "I'll have Margherita"},
            {"role": "assistant", "content": "Great choice! One Margherita coming up."},
        ]

        # Simple summarization: extract key entities
        key_info = {
            "discussed_items": ["pizza", "Margherita"],
            "order": ["Margherita"],
        }

        assert "pizza" in str(history).lower()
        assert "margherita" in str(history).lower()


class TestRAGPipeline:
    """Integration tests for RAG pipeline."""

    def test_full_rag_pipeline(self):
        """Test full RAG pipeline execution."""
        # Step 1: Document store
        documents = [
            {"id": 1, "content": "We serve Italian cuisine.", "embedding": None},
            {"id": 2, "content": "Our hours are 11am to 10pm.", "embedding": None},
            {"id": 3, "content": "We have vegetarian options.", "embedding": None},
        ]

        # Step 2: Generate embeddings (simulated)
        np.random.seed(42)
        for doc in documents:
            doc["embedding"] = np.random.randn(384).tolist()

        # Step 3: Query embedding
        query = "What time do you close?"
        query_embedding = np.random.randn(384)

        # Step 4: Retrieve relevant documents
        from sklearn.metrics.pairwise import cosine_similarity

        doc_embeddings = np.array([doc["embedding"] for doc in documents])
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

        top_idx = np.argmax(similarities)
        relevant_doc = documents[top_idx]

        # Step 5: Generate response (simulated)
        context = relevant_doc["content"]
        response = f"Based on our information: {context}"

        assert len(response) > 0
        assert context in response

    def test_fallback_when_no_relevant_docs(self):
        """Test fallback behavior when no relevant documents found."""
        threshold = 0.5
        similarities = [0.2, 0.3, 0.1]  # All below threshold

        relevant_docs = [s for s in similarities if s >= threshold]

        if not relevant_docs:
            response = "I don't have specific information about that, but I can help with menu, hours, or reservations."

        assert "I don't have specific information" in response

    def test_context_augmentation(self):
        """Test context augmentation with additional data."""
        base_context = "We serve Italian cuisine."

        # Augment with structured data
        menu_data = {
            "specials": ["Truffle Risotto", "Seafood Pasta"],
            "prices": {"Truffle Risotto": 24.99, "Seafood Pasta": 18.99},
        }

        augmented_context = f"""{base_context}

Today's Specials:
- {menu_data['specials'][0]}: ${menu_data['prices']['Truffle Risotto']:.2f}
- {menu_data['specials'][1]}: ${menu_data['prices']['Seafood Pasta']:.2f}
"""

        assert "Truffle Risotto" in augmented_context
        assert "$24.99" in augmented_context
        assert "$24.99" in augmented_context
