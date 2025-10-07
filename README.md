text
# Restaurant Review Analysis API

FastAPI backend for analyzing restaurant reviews with sentiment analysis, topic classification, and AI-powered responses.

## Features
- Sentiment analysis with DistilBERT (PyTorch)
- Topic classification (food, service, cleanliness, etc.)
- AI-generated replies using OpenRouter API
- TF-IDF based smart similarity search
- Analytics dashboard with sentiment and topic insights
- Full CRUD REST API for reviews

## Technology Stack
- FastAPI (Python 3.8+)
- SQLite + SQLAlchemy ORM
- PyTorch, Transformers (DistilBERT), scikit-learn
- OpenRouter (GPT-3.5-turbo)
- Ubuntu EC2, Uvicorn ASGI server

## Setup Instructions

### Prerequisites
- Python 3.8+, pip or conda, Git

### 1. Clone Repo
git clone https://github.com/sinsau182/CubeAssignmentBackend.git
cd CubeAssignmentBackend

text

### 2. Create Virtual Env
python -m venv myenv
source myenv/bin/activate # Linux/Mac

or on Windows
myenv\Scripts\activate

text

### 3. Install Dependencies
For CPU:
pip install -r requirements.txt

### 4. Configure Environment
Create `.env` with keys like:
DATABASE_URL=sqlite:///./test.db
OPENROUTER_API_KEY=your-openrouter-api-key
API_KEY=your-internal-api-key
HOST=0.0.0.0
PORT=8000
SENTIMENT_MODEL=distilbert-base-uncased-finetuned-sst-2-english
AI_DEVICE=cpu

## Running the App

### Development
uvicorn main:app --reload --host 0.0.0.0 --port 8000

## API Endpoints
| Method | Endpoint      | Description             |
|--------|---------------|-------------------------|
| GET    | `/`           | Health check            |
| GET    | `/reviews`    | Paginated reviews       |
| POST   | `/reviews`    | Create a review         |
| GET    | `/reviews/{id}` | Get a review by ID      |
| POST   | `/reviews/{id}/suggest-reply` | Generating reply for a review by ID |
| GET    | `/analytics`  | Sentiment analytics     |
| GET    | `/search`     | Similar reviews search  |

## Deployment Notes

### EC2 Setup (Ubuntu 20.04 LTS)
- Instance: t3.medium
- Open ports: 80, 443, 22
sudo apt update && sudo apt upgrade -y
sudo apt install python3 python3-pip python3-venv git -y

git clone https://github.com/sinsau182/CubeAssignmentBackend.git
cd CubeAssignmentBackend

python3 -m venv myenv
source myenv/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

cp .env

Edit .env for production
uvicorn main:app --host 0.0.0.0 --port 8000

### Process Management with PM2 (Optional)
npm install -g pm2
pm2 start "uvicorn main:app --host 0.0.0.0 --port 8000" --name "CopilotBackend"

## Performance & Security
- CPU-only PyTorch for smaller deploy
- Model loaded once, reused across requests
- Async DB operations
- CORS, API key, input sanitization and SQL injection prevention

## Future Improvements
- Automated with user Review taking platform to give reply in run time
- Background task queue (Celery)
- Real-time analytics UI
- Multi-language support
- Redis caching
- Monitoring and logging enhancements

## Contact
- GitHub: [@sinsau182](https://github.com/sinsau182)
- Email: singhsaurav182001@gmail.com

---
Demo project showcasing FastAPI, ML, and modern Python backends.