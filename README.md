# Heart Disease Risk Predictor

A modern FastAPI web app and Telegram bot for AI-powered cardiovascular risk assessment using probabilistic graphical models.

## Project Structure

```
pgmbot/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── bot.py
│   ├── models/
│   │   └── heart_disease_model.pkl
│   └── templates/
│       └── index.html
├── requirements.txt
├── README.md
└── .env.example
```

## Setup

1. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
2. Run the web app:
   ```sh
   uvicorn app.main:app --reload
   ```
3. Run the Telegram bot:
   ```sh
   python app/bot.py
   ```

## Environment Variables
Copy `.env.example` to `.env` and fill in your secrets (e.g., Telegram bot token).

