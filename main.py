from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from database import get_db, RealDB  # Your async DB wrapper class
from ai_utils import analyze_sentiment, suggest_reply_async, find_similar_reviews

app = FastAPI(title="Multi-location Reviews AI")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "http://localhost:5173",  # Vite default port
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = "your-api-key"  # replace with env or config

def check_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

class ReviewIn(BaseModel):
    id: Optional[int] = None
    location: str
    rating: int
    text: str
    date: str

class ReplyOut(BaseModel):
    sentiment: str
    topic: str
    reply: str
    reasoning: str

@app.on_event("startup")
async def startup():
    db_instance: RealDB = get_db()
    await db_instance.db.connect()

@app.on_event("shutdown")
async def shutdown():
    db_instance: RealDB = get_db()
    await db_instance.db.disconnect()

@app.post("/ingest", dependencies=[Depends(check_api_key)])
async def ingest(reviews: List[ReviewIn], db: RealDB = Depends(get_db)):
    for r in reviews:
        await db.add_review(r)
    return {"ingested": len(reviews)}

@app.get("/reviews")
async def get_reviews(location: Optional[str] = None, sentiment: Optional[str] = None, q: Optional[str] = None, page: int = 1, pagesize: int = 20, db: RealDB = Depends(get_db)):
    """
    Get reviews with AI-analyzed sentiment and topic included.
    
    Args:
        location: Filter by location (optional)
        sentiment: Filter by sentiment (optional) - values: POSITIVE, NEGATIVE, NEUTRAL
        q: Text search query (optional)
        page: Page number for pagination (default: 1)
        pagesize: Number of reviews per page (default: 20)
    
    Returns:
        List of reviews with sentiment and topic analysis
    """
    reviews = await db.query_reviews(location, sentiment, q, page, pagesize)
    
    # Add AI-analyzed sentiment and topic to each review
    enriched_reviews = []
    for review in reviews:
        # Convert review to dict
        review_dict = review.model_dump()
        
        # Add AI analysis
        ai_sentiment, ai_topic = analyze_sentiment(review.text)
        review_dict['ai_sentiment'] = ai_sentiment
        review_dict['ai_topic'] = ai_topic
        
        enriched_reviews.append(review_dict)
    
    return enriched_reviews

@app.get("/reviews/{id}")
async def get_review(id: int, db: RealDB = Depends(get_db)):
    review = await db.get_review(id)
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    return review

@app.post("/reviews/{id}/suggest-reply", response_model=ReplyOut)
async def post_reply(id: int, db: RealDB = Depends(get_db)):
    review = await db.get_review(id)
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")

    sentiment, topic = analyze_sentiment(review.text)
    reply, reasoning = await suggest_reply_async(review.text, sentiment, topic)

    return ReplyOut(sentiment=sentiment, topic=topic, reply=reply, reasoning=reasoning)


@app.get("/analytics")
async def analytics(db: RealDB = Depends(get_db)):
    """
    Get comprehensive analytics data for dashboard including:
    - Overview metrics (total reviews, average rating, positive rate)
    - Sentiment distribution (positive, neutral, negative counts)
    - Topic distribution (service, quality, price, delivery, etc.)
    - Location breakdown
    """
    analytics_data = await db.get_analytics()
    return analytics_data

@app.get("/search")
async def search(q: str, k: int = 5, db: RealDB = Depends(get_db)):
    """
    Search for similar reviews using TF-IDF vectors and cosine similarity.
    
    Args:
        q: Search query string
        k: Number of top similar reviews to return (default: 5)
    
    Returns:
        List of top-k most similar reviews with similarity scores
    """
    items = await db.get_all_reviews()
    return find_similar_reviews(q, items, top_k=k)

@app.get("/health")
async def health():
    return {"status": "ok"}


