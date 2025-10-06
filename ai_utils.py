import os
import httpx
import asyncio
import re
import numpy as np
from transformers import pipeline
from httpx import HTTPStatusError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize sentiment analysis with multiple lightweight alternatives
def create_optimized_sentiment_pipeline():
    """Create optimized sentiment pipeline - tries multiple lightweight alternatives"""
    
    # Option 1: ONNX Runtime (Lightest - ~50MB)
    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer, pipeline
        
        print("🚀 Trying ONNX Runtime (lightest option)...")
        model_name = "microsoft/DialoGPT-medium"  # Much smaller ONNX model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        onnx_model = ORTModelForSequenceClassification.from_pretrained(model_name, from_tf=False)
        return pipeline("sentiment-analysis", model=onnx_model, tokenizer=tokenizer)
    except Exception as e1:
        print(f"⚠️ ONNX failed: {e1}")
        
        # Option 2: CPU-only TensorFlow (Medium weight)
        try:
            print("🔄 Trying TensorFlow backend...")
            # Use a smaller TF model
            tf_model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            return pipeline("sentiment-analysis", model=tf_model, framework="tf", device=-1)
        except Exception as e2:
            print(f"⚠️ TensorFlow failed: {e2}")
            
            # Option 3: Rule-based (Ultra light - no models)
            print("🪶 Using rule-based sentiment analysis (no AI models)")
            return create_rule_based_analyzer()

def create_rule_based_analyzer():
    """Ultra-lightweight rule-based sentiment analysis"""
    
    positive_words = {
        'excellent', 'amazing', 'wonderful', 'fantastic', 'great', 'good', 'awesome',
        'outstanding', 'perfect', 'brilliant', 'superb', 'delicious', 'tasty', 'fresh',
        'clean', 'friendly', 'helpful', 'fast', 'love', 'recommend', 'satisfied', 'happy'
    }
    
    negative_words = {
        'terrible', 'awful', 'horrible', 'bad', 'worst', 'disgusting', 'slow', 'dirty',
        'rude', 'unfriendly', 'cold', 'stale', 'expensive', 'disappointing', 'frustrated',
        'angry', 'hate', 'poor', 'broken', 'delayed', 'wrong', 'problem'
    }
    
    def analyze(text):
        words = text.lower().split()
        pos_score = sum(1 for word in words if word in positive_words)
        neg_score = sum(1 for word in words if word in negative_words)
        
        if pos_score > neg_score:
            return [{"label": "POSITIVE", "score": 0.8}]
        elif neg_score > pos_score:
            return [{"label": "NEGATIVE", "score": 0.8}]
        else:
            return [{"label": "NEUTRAL", "score": 0.6}]
    
    return analyze

# Initialize with the best available option
sentiment_pipeline = create_optimized_sentiment_pipeline()

# Read OpenRouter API key and endpoint from environment
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Configuration from environment
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "20"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.1"))


def analyze_sentiment(text):
    """
    Enhanced sentiment classification that can detect NEUTRAL sentiment.
    Uses confidence scores and neutral keywords to classify sentiment properly.
    """
    text_lower = text.lower()
    
    # Enhanced topic classification with proper priority order
    # App/Technology issues should be checked first as they're specific
    if any(keyword in text_lower for keyword in ["app", "application", "website", "site", "crash", "crashed", "crashing", "bug", "error", "login", "logout", "system", "technical", "technology", "software", "platform", "interface"]):
        topic = "app"
    elif any(keyword in text_lower for keyword in ["service", "staff", "wait", "server", "waiter", "employee", "team", "help", "assist", "support", "rude", "friendly", "polite"]):
        topic = "service"
    elif any(keyword in text_lower for keyword in ["cleanliness", "clean", "hygiene", "dirty", "mess", "tidy", "sanitary", "spotless", "filthy"]):
        topic = "cleanliness" 
    elif any(keyword in text_lower for keyword in ["food", "meal", "dish", "taste", "flavor", "cook", "delicious", "bland", "fresh", "stale", "hot", "cold", "quality"]):
        topic = "food"
    elif any(keyword in text_lower for keyword in ["price", "cost", "expensive", "cheap", "value", "money", "worth", "affordable", "overpriced", "deal", "discount"]):
        topic = "value"
    elif any(keyword in text_lower for keyword in ["delivery", "deliver", "pickup", "order", "late", "missing", "wrong", "time", "fast", "slow"]):
        topic = "delivery"
    elif any(keyword in text_lower for keyword in ["location", "parking", "area", "building", "address", "far", "close", "convenient"]):
        topic = "location"
    else:
        topic = "other"

    # Get sentiment with confidence score
    sentiment_result = sentiment_pipeline(text)[0]
    sentiment_label = sentiment_result['label']
    confidence = sentiment_result['score']
    
    # Neutral keywords that indicate neutral sentiment
    neutral_keywords = [
        "okay", "ok", "average", "decent", "fine", "standard", "typical", "normal",
        "mediocre", "fair", "acceptable", "reasonable", "nothing special", 
        "not bad", "not great", "so-so", "meh", "alright", "adequate"
    ]
    
    # Mixed sentiment indicators
    mixed_indicators = [
        "but", "however", "although", "some good", "some bad", "mixed", "partly"
    ]
    
    # Check for neutral indicators
    has_neutral_keywords = any(keyword in text_lower for keyword in neutral_keywords)
    has_mixed_indicators = any(indicator in text_lower for indicator in mixed_indicators)
    low_confidence = confidence < 0.7  # Low confidence suggests neutral
    
    # Determine final sentiment
    if has_neutral_keywords or (low_confidence and has_mixed_indicators):
        final_sentiment = "NEUTRAL"
    elif confidence < 0.6:  # Very low confidence, likely neutral
        final_sentiment = "NEUTRAL"
    else:
        # Use the model's prediction for high confidence cases
        final_sentiment = sentiment_label
    
    return final_sentiment, topic


async def suggest_reply_async(review_text, sentiment, topic):
    """
    Generate a polite and empathetic customer service reply using OpenRouter API.
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    json_data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a helpful customer service assistant."},
            {
                "role": "user",
                "content": (
                    "You are a customer service AI assistant. "
                    "Write a polite, empathetic reply that acknowledges the customer's issue, "
                    "apologizes sincerely, and offers assurance of improvements.\n\n"
                    f"Customer review: \"{review_text}\"\n\nReply:"
                ),
            },
        ],
        "max_tokens": 150,
        "temperature": 0.7,
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(OPENROUTER_API_URL, headers=headers, json=json_data)
            response.raise_for_status()
        except HTTPStatusError as e:
            # Fallback reply for HTTP errors including rate limit
            fallback_reply = (
                "Sorry, our AI service is currently at capacity. "
                "We value your feedback and will respond as soon as possible."
            )
            reasoning = f"Fallback reply due to OpenRouter API error: {str(e)}"
            return fallback_reply, reasoning

        result = response.json()

    reply = result["choices"][0]["message"]["content"].strip()

    # Redact emails and phones for security
    reply = re.sub(r'\S+@\S+', '[redacted email]', reply)
    reply = re.sub(r'\b\d{10,}\b', '[redacted phone]', reply)

    reasoning = f"Based on detected sentiment '{sentiment}' and topic '{topic}', reply generated by OpenRouter GPT."

    return reply, reasoning


def find_similar_reviews(query: str, reviews: List[Any], top_k: int = None) -> List[Dict[str, Any]]:
    """
    Find similar reviews using TF-IDF vectors and cosine similarity.
    
    Args:
        query: Search query string
        reviews: List of review objects or dictionaries
        top_k: Number of top similar reviews to return (default: from environment)
    
    Returns:
        List of top_k most similar reviews with similarity scores
    """
    if not reviews or not query.strip():
        return []
    
    # Use environment variable for top_k if not provided
    if top_k is None:
        top_k = MAX_SEARCH_RESULTS
    
    # Convert review objects to dictionaries and extract text
    review_dicts = []
    review_texts = []
    
    for review in reviews:
        if hasattr(review, 'dict'):
            review_dict = review.dict()
        elif hasattr(review, 'model_dump'):
            review_dict = review.model_dump()
        else:
            review_dict = review
        
        review_text = review_dict.get('text', '')
        if review_text.strip():  # Only include reviews with non-empty text
            review_dicts.append(review_dict)
            review_texts.append(review_text)
    
    if not review_texts:
        return []
    
    # Create corpus with query + all review texts
    corpus = [query] + review_texts
    
    # Initialize TF-IDF Vectorizer with preprocessing
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        max_features=1000,
        ngram_range=(1, 2),  # Include unigrams and bigrams
        min_df=1,  # Minimum document frequency
        max_df=0.95,  # Maximum document frequency (ignore too common words)
        strip_accents='unicode'
    )
    
    try:
        # Fit and transform the corpus
        tfidf_matrix = vectorizer.fit_transform(corpus)
        
        # Query vector is the first row (index 0)
        query_vector = tfidf_matrix[0]
        
        # Review vectors are from index 1 onwards
        review_vectors = tfidf_matrix[1:]
        
        # Calculate cosine similarity between query and all reviews
        similarities = cosine_similarity(query_vector, review_vectors).flatten()
        
        # Create results with similarity scores
        scored_reviews = []
        for i, similarity_score in enumerate(similarities):
            if similarity_score >= SIMILARITY_THRESHOLD:  # Use environment threshold
                review_with_score = review_dicts[i].copy()
                review_with_score['similarity_score'] = float(similarity_score)
                scored_reviews.append(review_with_score)
        
        # Sort by similarity score (descending) and return top_k
        scored_reviews.sort(key=lambda x: x['similarity_score'], reverse=True)
        return scored_reviews[:top_k]
        
    except Exception as e:
        # Fallback to simple word matching if TF-IDF fails
        print(f"TF-IDF search failed, using fallback: {e}")
        return _fallback_similarity_search(query, review_dicts, top_k)


def _fallback_similarity_search(query: str, review_dicts: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """
    Fallback similarity search using simple word overlap.
    Used when TF-IDF approach fails.
    """
    query_words = set(query.lower().split())
    scored_reviews = []
    
    for review_dict in review_dicts:
        review_text = review_dict.get('text', '').lower()
        review_words = set(review_text.split())
        
        # Calculate Jaccard similarity
        intersection = query_words.intersection(review_words)
        union = query_words.union(review_words)
        
        if len(union) > 0:
            similarity_score = len(intersection) / len(union)
            if similarity_score > 0:
                review_with_score = review_dict.copy()
                review_with_score['similarity_score'] = similarity_score
                scored_reviews.append(review_with_score)
    
    scored_reviews.sort(key=lambda x: x['similarity_score'], reverse=True)
    return scored_reviews[:top_k]
