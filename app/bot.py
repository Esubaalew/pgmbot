import os
import pickle
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters,
)
from pgmpy.inference import VariableElimination

load_dotenv()

# Load your saved model
with open("models/heart_disease_model.pkl", "rb") as f:
    model = pickle.load(f)

inference = VariableElimination(model)

# Corrected value_map with exact capitalization matching model states
value_map = {
    "ca": {
        "zero": "0", "0": "0",
        "one": "1", "1": "1",
        "two": "2", "2": "2",
        "three": "3", "3": "3"
    },
    "thalach": {
        "low": "Low",
        "medium": "Medium",
        "high": "High"
    },
    "trestbps": {
        "low": "Low",
        "normal": "Normal",
        "high": "High"
    },
    "slope": {
        "up": "Upsloping",
        "flat": "Flat",
        "down": "Downsloping"
    },
    "thal": {
        "normal": "Normal",
        "fixed-defect": "Fixed-Defect",
        "reversible-defect": "Reversable-Defect"
    },
    "cp": {
        "typical-angina": "Typical-Angina",
        "asymptomatic": "Asymptomatic",
        "atypical-angina": "Atypical-Angina",
        "non-anginal-pain": "Non-Anginal-Pain",
    },
    "sex": {
        "male": "Male",
        "female": "Female"
    },
    "age": {
        "young": "Young",
        "middle-aged": "Middle-Aged",
        "senior": "Senior",
        "very-senior": "Very-Senior"
    },
    "exang": {
        "yes": "Yes",
        "no": "No"
    },
}

def format_prob(prob, variable):
    return {state: f"{prob.values[i]:.2%}" for i, state in enumerate(prob.state_names[variable])}

def normalize_input(key, val):
    key = key.strip().lower()
    val = val.strip().lower()
    if key in value_map and val in value_map[key]:
        return key, value_map[key][val]
    return key, val

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Welcome to the Heart Disease Bot!\n"
        "Send your info like this:\n"
        "sex=Male, age=Middle-Aged, cp=Typical-Angina, thalach=High, ca=2\n"
        "You can mix upper/lowercase and say zero, one, etc."
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_input = update.message.text
        evidence = {}
        model_nodes = [
            'age', 'trestbps', 'cp', 'exang', 'thalach', 'thal', 'sex', 'ca', 'slope'
        ]
        for pair in user_input.split(","):
            if "=" not in pair:
                await update.message.reply_text("⚠️ Format error: use key=value pairs separated by commas.")
                return
            key, val = pair.strip().split("=")
            key, val = normalize_input(key, val)
            if key in model_nodes:
                evidence[key] = val

        result = inference.query(variables=["target"], evidence=evidence, joint=False)
        probs = format_prob(result["target"], "target")
        response = "\n".join(f"{k}: {v}" for k, v in probs.items())
        await update.message.reply_text(f"📊 Estimated Heart Disease Risk:\n{response}")
    except Exception as e:
        await update.message.reply_text(f"⚠️ Error: {e}")

def main():
    TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    if not TOKEN:
        print("Error: TELEGRAM_BOT_TOKEN not set in .env file.")
        return
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    app.run_polling()

if __name__ == "__main__":
    main()
