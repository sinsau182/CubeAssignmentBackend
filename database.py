from typing import List, Optional
from pydantic import BaseModel
import sqlalchemy
from sqlalchemy import Table, Column, Integer, String, MetaData
from databases import Database

DATABASE_URL = "sqlite:///./test.db"

# Pydantic model (same as before)
class Review(BaseModel):
    id: int
    location: str
    rating: int
    text: str
    date: str

# SQLAlchemy Table definition
metadata = MetaData()

reviews_table = Table(
    "reviews",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("location", String),
    Column("rating", Integer),
    Column("text", String),
    Column("date", String),
)

# Database connection object
database = Database(DATABASE_URL)

# Create tables on startup (run once)
engine = sqlalchemy.create_engine(DATABASE_URL)
metadata.create_all(engine)

# DB access class replacing FakeDB
class RealDB:
    def __init__(self, db: Database):
        self.db = db

    async def add_review(self, review: Review):
        query = reviews_table.insert().values(**review.model_dump())
        await self.db.execute(query)

    async def query_reviews(self, loc=None, sent=None, q=None, page=1, pagesize=20) -> List[Review]:
        """
        Query reviews with optional filtering by location, sentiment, and text search.
        Sentiment filtering is done post-query using AI analysis since we don't store sentiment in DB.
        """
        query = reviews_table.select()
        if loc:
            query = query.where(reviews_table.c.location == loc)
        if q:
            query = query.where(reviews_table.c.text.ilike(f"%{q}%"))
        
        # Get all matching reviews first (without pagination for sentiment filtering)
        all_rows = await self.db.fetch_all(query)
        all_reviews = [Review(**row) for row in all_rows]
        
        # Filter by sentiment using AI analysis if sentiment filter is provided
        if sent:
            from ai_utils import analyze_sentiment
            filtered_reviews = []
            for review in all_reviews:
                ai_sentiment, _ = analyze_sentiment(review.text)
                if ai_sentiment.upper() == sent.upper():
                    filtered_reviews.append(review)
            all_reviews = filtered_reviews
        
        # Apply pagination after filtering
        start_idx = (page - 1) * pagesize
        end_idx = start_idx + pagesize
        return all_reviews[start_idx:end_idx]

    async def get_review(self, id: int) -> Optional[Review]:
        query = reviews_table.select().where(reviews_table.c.id == id)
        row = await self.db.fetch_one(query)
        return Review(**row) if row else None

    async def get_all_reviews(self) -> List[Review]:
        query = reviews_table.select()
        rows = await self.db.fetch_all(query)
        return [Review(**row) for row in rows]

    async def get_analytics(self):
        """
        Get comprehensive analytics data for dashboard:
        - Total reviews, average rating, positive rate
        - Sentiment distribution, topic distribution
        """
        from ai_utils import analyze_sentiment
        
        # Get all reviews for analysis
        reviews = await self.get_all_reviews()
        
        if not reviews:
            return {
                "total_reviews": 0,
                "average_rating": 0,
                "positive_rate": 0,
                "sentiment_distribution": {"Positive": 0, "Neutral": 0, "Negative": 0},
                "topic_distribution": {"Service": 0, "App": 0, "Delivery": 0, "Quality": 0, "Price": 0, "Other": 0}
            }
        
        # Calculate basic metrics
        total_reviews = len(reviews)
        total_rating = sum(review.rating for review in reviews)
        average_rating = round(total_rating / total_reviews, 1)
        
        # Analyze sentiment and topics for each review
        sentiment_counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
        topic_counts = {"Service": 0, "App": 0, "Delivery": 0, "Quality": 0, "Price": 0, "Other": 0}
        positive_count = 0
        
        for review in reviews:
            # Get AI-based sentiment and topic analysis
            sentiment, topic = analyze_sentiment(review.text)
            
            # Map sentiment labels to dashboard categories
            if sentiment in ["POSITIVE", "HIGHLY_POSITIVE"]:
                sentiment_counts["Positive"] += 1
                positive_count += 1
            elif sentiment in ["NEGATIVE", "STRONGLY_NEGATIVE"]:
                sentiment_counts["Negative"] += 1
            else:
                sentiment_counts["Neutral"] += 1
            
            # Map topics to dashboard categories
            topic_mapping = {
                "service": "Service",
                "app": "App",
                "delivery": "Delivery", 
                "cleanliness": "Quality",
                "food": "Quality", 
                "ambiance": "Quality",
                "value": "Price",
                "location": "Other",  # Physical location is now "Other"
                "other": "Other",
                "general": "Other"
            }
            
            dashboard_topic = topic_mapping.get(topic.lower(), "Other")
            topic_counts[dashboard_topic] += 1
        
        # Calculate positive rate
        positive_rate = round((positive_count / total_reviews) * 100)
        
        return {
            "total_reviews": total_reviews,
            "average_rating": average_rating,
            "positive_rate": positive_rate,
            "negative_count": total_reviews - positive_count,
            "sentiment_distribution": sentiment_counts,
            "topic_distribution": topic_counts,
            "location_breakdown": await self._get_location_breakdown()
        }
    
    async def _get_location_breakdown(self):
        """Get review counts by location"""
        query = sqlalchemy.text("SELECT location, COUNT(*) as count FROM reviews GROUP BY location")
        rows = await self.db.fetch_all(query)
        return {row['location']: row['count'] for row in rows}

# Dependency for FastAPI
def get_db():
    return RealDB(database)
