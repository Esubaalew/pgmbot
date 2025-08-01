import os
import pickle
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pgmpy.inference import VariableElimination
import json

# Load model once on startup
with open("models/heart_disease_model.pkl", "rb") as f:
    model = pickle.load(f)

inference = VariableElimination(model)

# Enhanced value mapping with descriptions
value_map = {
    "ca": {
        "zero": "0", "0": "0", "one": "1", "1": "1",
        "two": "2", "2": "2", "three": "3", "3": "3"
    },
    "thalach": {"low": "Low", "medium": "Medium", "high": "High"},
    "trestbps": {"low": "Low", "normal": "Normal", "high": "High"},
    "chol": {"low": "Low", "normal": "Normal", "high": "High"},
    "oldpeak": {"low": "Low", "medium": "Medium", "high": "High"},
    "slope": {"up": "Upsloping", "flat": "Flat", "down": "Downsloping"},
    "thal": {"normal": "Normal", "fixed-defect": "Fixed-Defect", "reversible-defect": "Reversible-Defect"},
    "cp": {
        "typical-angina": "Typical-Angina",
        "asymptomatic": "Asymptomatic",
        "atypical-angina": "Atypical-Angina",
        "non-anginal-pain": "Non-Anginal-Pain",
    },
    "sex": {"male": "Male", "female": "Female"},
    "age": {"young": "Young", "middle-aged": "Middle-Aged", "senior": "Senior", "very-senior": "Very-Senior"},
    "exang": {"yes": "Yes", "no": "No"},
    "fbs": {"true": "True", "false": "False"},
    "restecg": {"normal": "Normal", "stt-abnormality": "ST-T Abnormality",
                "lv-hypertrophy": "Left Ventricular Hypertrophy"}
}

# Field descriptions for educational purposes
field_descriptions = {
    "sex": "Biological sex affects heart disease risk patterns due to hormonal and physiological differences.",
    "age": "Age is a primary risk factor - cardiovascular risk increases significantly with age.",
    "cp": "Chest pain type is crucial for diagnosis. Asymptomatic cases often indicate silent heart disease.",
    "trestbps": "Resting blood pressure above 140/90 mmHg indicates hypertension, a major risk factor.",
    "chol": "Total cholesterol levels above 240 mg/dl significantly increase cardiovascular risk.",
    "fbs": "Fasting blood sugar >120 mg/dl may indicate diabetes, doubling heart disease risk.",
    "restecg": "ECG abnormalities can reveal underlying cardiac conditions even when asymptomatic.",
    "thalach": "Maximum heart rate during exercise testing indicates cardiovascular fitness.",
    "exang": "Exercise-induced chest pain suggests coronary artery disease and reduced blood flow.",
    "oldpeak": "ST depression during exercise indicates ischemia and potential coronary blockage.",
    "slope": "ST segment slope pattern provides insights into coronary artery health.",
    "ca": "Number of major vessels with significant blockage detected via cardiac catheterization.",
    "thal": "Thalassemia test results indicating blood flow patterns to the heart muscle."
}


def normalize_input(key, val):
    key = key.strip().lower()
    val = val.strip().lower()
    if key in value_map and val in value_map[key]:
        return key, value_map[key][val]
    return key, val


def format_prob(prob, variable):
    return {state: f"{prob.values[i]:.2%}" for i, state in enumerate(prob.state_names[variable])}


def get_risk_interpretation(probs):
    """Provide clinical interpretation of results"""
    if 'Heart-Disease' in probs:
        risk_percent = float(probs['Heart-Disease'].strip('%'))
        if risk_percent >= 70:
            return {
                "level": "High Risk",
                "color": "#FF3B30",
                "recommendation": "Immediate medical consultation recommended. Consider comprehensive cardiac evaluation.",
                "icon": "exclamation-triangle"
            }
        elif risk_percent >= 40:
            return {
                "level": "Moderate Risk",
                "color": "#FF9500",
                "recommendation": "Schedule appointment with healthcare provider. Lifestyle modifications advised.",
                "icon": "exclamation-circle"
            }
        else:
            return {
                "level": "Low Risk",
                "color": "#34C759",
                "recommendation": "Continue healthy lifestyle. Regular check-ups recommended.",
                "icon": "check-circle"
            }
    return {"level": "Unknown", "color": "#8E8E93", "recommendation": "Unable to assess risk.",
            "icon": "question-circle"}


app = FastAPI(title="Heart Disease Risk Predictor", description="AI-Powered Cardiovascular Risk Assessment")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": None,
        "field_descriptions": field_descriptions
    })


@app.post("/", response_class=HTMLResponse)
async def predict(
        request: Request,
        sex: str = Form(...),
        age: str = Form(...),
        cp: str = Form(...),
        trestbps: str = Form(...),
        chol: str = Form(...),
        fbs: str = Form(...),
        restecg: str = Form(...),
        thalach: str = Form(...),
        exang: str = Form(...),
        oldpeak: str = Form(...),
        slope: str = Form(...),
        ca: str = Form(...),
        thal: str = Form(...)
):
    try:
        # Model nodes for filtering
        model_nodes = ['age', 'trestbps', 'cp', 'exang', 'thalach', 'thal', 'sex', 'ca', 'slope']

        evidence = {}
        form_data = {
            "sex": sex, "age": age, "cp": cp, "trestbps": trestbps,
            "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
            "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
        }

        for k, v in form_data.items():
            nk, nv = normalize_input(k, v)
            if nk in model_nodes:
                evidence[nk] = nv

        # Run inference
        result = inference.query(variables=["target"], evidence=evidence, joint=False)
        probs = format_prob(result["target"], "target")
        interpretation = get_risk_interpretation(probs)

        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": probs,
            "evidence": evidence,
            "interpretation": interpretation,
            "field_descriptions": field_descriptions,
            "form_data": form_data
        })

    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": str(e),
            "result": None,
            "field_descriptions": field_descriptions
        })


@app.get("/api/model-info")
async def get_model_info():
    """API endpoint for model information"""
    try:
        nodes_info = {}
        for node in model.nodes():
            cpd = model.get_cpds(node)
            if cpd:
                nodes_info[node] = {
                    "states": cpd.state_names[node],
                    "description": field_descriptions.get(node, "No description available")
                }

        return {
            "model_nodes": list(model.nodes()),
            "model_edges": list(model.edges()),
            "nodes_info": nodes_info,
            "total_parameters": sum(cpd.cardinality.prod() for cpd in model.get_cpds())
        }
    except Exception as e:
        return {"error": str(e)}
