import os
import pandas as pd
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from flask_cors import CORS
from flask_caching import Cache
from tenacity import retry, stop_after_attempt, wait_exponential
from risk import ImprovedAllDiseaseRiskPredictionModel, load_data
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')

# Configure caching
cache = Cache(config={'CACHE_TYPE': 'SimpleCache'})
cache.init_app(app)

# Constants
MODEL_PATH = 'disease_risk_model.joblib'
OLLAMA_MODEL = "deepseek-r1:1.5b"
OLLAMA_TIMEOUT = 45  # seconds

def initialize_model():
    """Initialize or load the disease risk prediction model"""
    if os.path.exists(MODEL_PATH):
        print("Loading existing model...")
        return ImprovedAllDiseaseRiskPredictionModel.load_model(MODEL_PATH)
    else:
        print("Training new model...")
        model = ImprovedAllDiseaseRiskPredictionModel(n_splits=5)
        file_path = "dataset/data.csv"
        data = load_data(file_path)
        X_processed, y = model.preprocess_data(data)
        model.build_models()
        model.train_and_evaluate_models(X_processed, y)
        model.save_model(MODEL_PATH)
        return model

# Global model instance
global_model = initialize_model()

def get_bmi_category(bmi):
    """Categorize BMI value"""
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal weight"
    elif 25 <= bmi < 30:
        return "Overweight"
    return "Obese"

def get_bp_category(systolic, diastolic):
    """Categorize blood pressure"""
    if systolic < 120 and diastolic < 80:
        return "Normal"
    elif 120 <= systolic < 130 and diastolic < 80:
        return "Elevated"
    elif 130 <= systolic < 140 or 80 <= diastolic < 90:
        return "Stage 1 Hypertension"
    return "Stage 2 Hypertension"

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_with_retry(llm, prompt):
    """Retry wrapper for LLM generation"""
    return llm.invoke(prompt, timeout=OLLAMA_TIMEOUT)

def check_llm_health(llm):
    """Check if the LLM service is responsive"""
    try:
        test_response = llm.invoke("Respond with 'OK' if ready.", timeout=5)
        return "ok" in test_response.lower()
    except:
        return False

def initialize_llm():
    """Initialize the Ollama LLM with DeepSeek-R1"""
    return Ollama(
        model=OLLAMA_MODEL,
        temperature=0.7,
        top_p=0.9,
        repeat_penalty=1.1,
        num_ctx=2048
    )

def format_predictions(predictions):
    """Format predictions for the LLM prompt"""
    return "\n".join(
        f"- {disease.replace('_', ' ')}: {pred['risk_level']} risk (probability: {pred['probability']*100:.1f}%)"
        for disease, pred in predictions.items()
    )

def parse_report_response(response):
    """Parse the LLM response into structured sections"""
    sections = {
        "risk_summary": "",
        "detailed_recommendations": "",
        "prevention_plan": "",
        "treatment_options": "",
        "lifestyle_modifications": "",
        "monitoring_plan": ""
    }
    
    current_section = None
    for line in response.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Check for section headers
        if "risk summary" in line.lower():
            current_section = "risk_summary"
            continue
        elif "detailed recommendations" in line.lower():
            current_section = "detailed_recommendations"
            continue
        elif "prevention plan" in line.lower():
            current_section = "prevention_plan"
            continue
        elif "treatment options" in line.lower():
            current_section = "treatment_options"
            continue
        elif "lifestyle modifications" in line.lower():
            current_section = "lifestyle_modifications"
            continue
        elif "monitoring plan" in line.lower():
            current_section = "monitoring_plan"
            continue
            
        # Add content to current section
        if current_section and current_section in sections:
            if sections[current_section]:
                sections[current_section] += "\n" + line
            else:
                sections[current_section] = line
                
    return sections

@app.route('/')
def home():
    """Render the main input form"""
    return render_template('index.html')

@app.route('/get-report-data', methods=['GET'])
def get_report_data():
    """Endpoint to retrieve stored report data"""
    predictions = session.get('prediction_data', {})
    patient_data = session.get('form_data', {})
    return jsonify({
        'predictionData': predictions,
        'formData': patient_data
    })

@app.route('/report')
def report():
    """Render the report page if data exists"""
    if not session.get('prediction_data') or not session.get('form_data'):
        return redirect(url_for('home'))
    return render_template('report.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 415
    
    try:
        input_data = request.get_json()
        new_patient_data = pd.DataFrame(input_data, index=[0])
        new_patient_processed = global_model.preprocess_data(new_patient_data, is_training=False)
        risk_predictions = global_model.predict_risks(new_patient_processed)
        
        formatted_predictions = {
            disease: {
                'probability': float(prediction['probability'][0]),
                'risk_level': prediction['risk_level'][0]
            }
            for disease, prediction in risk_predictions.items()
        }
        
        session['prediction_data'] = formatted_predictions
        session['form_data'] = input_data
        
        return jsonify(formatted_predictions)
        
    except KeyError as e:
        return jsonify({'error': f'Missing key in input data: {str(e)}'}), 400
    except ValueError as e:
        return jsonify({'error': f'Invalid value in input data: {str(e)}'}), 400
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

@app.route('/generate-report', methods=['POST'])
@cache.memoize(timeout=3600)
def generate_report():
    """Generate comprehensive health report using LLM"""
    try:
        predictions = session.get('prediction_data', {})
        patient_data = session.get('form_data', {})
        
        if not predictions or not patient_data:
            return jsonify({"error": "No prediction data available"}), 400

        llm = initialize_llm()
        
        if not check_llm_health(llm):
            return jsonify({"error": "AI model service unavailable"}), 503

        try:
            report = generate_comprehensive_report(llm, predictions, patient_data)
            return jsonify(report)
        except Exception as e:
            app.logger.error(f"Report generation failed: {str(e)}")
            return jsonify({"error": "Report generation failed"}), 500
            
    except Exception as e:
        app.logger.error(f"Unexpected error in generate-report: {str(e)}")
        return jsonify({"error": str(e)}), 500

def generate_comprehensive_report(llm, predictions, patient_data):
    """Generate a comprehensive health report using the LLM"""
    prompt_template = ChatPromptTemplate.from_template(
        """### Medical Expert System Report Generation ###
        
        **Patient Profile**:
        - Age: {age}
        - Gender: {gender}
        - BMI: {bmi} ({bmi_category})
        - Blood Pressure: {bp_systolic}/{bp_diastolic} mmHg ({bp_category})
        
        **Identified Health Risks**:
        {predictions}
        
        **Report Structure Requirements**:
        1. **<u>Risk Summary</u>**: Concise overview of primary health concerns
        2. **<u>Detailed Recommendations</u>**: Actionable, evidence-based advice for each risk
        3. **<u>Prevention Plan</u>**: Specific prevention strategies
        4. **<u>Treatment Options</u>**: Both clinical and lifestyle interventions
        5. **<u>Lifestyle Modifications</u>**: Practical daily habit changes
        6. **<u>Monitoring Plan</u>**: Follow-up schedule and metrics to track
        
        **Formatting Guidelines**:
        - Use clear section headers with **<u>bold and underlined text</u>**
        - Present information in bullet points
        - Prioritize recommendations by risk level
        - Include citations to medical guidelines where appropriate
        - Maintain professional yet accessible language
        - Limit each section to 5-7 key points
        
        **Example Format**:
        **<u>Risk Summary</u>**
        - The patient shows elevated risk for cardiovascular disease (45% probability)
        - Moderate risk for type 2 diabetes (32% probability)
        
        **<u>Detailed Recommendations</u>**
        - For cardiovascular risk: Increase physical activity to 150 minutes per week
        - Monitor blood pressure weekly and maintain below 130/80 mmHg
        """
    )
    
    formatted_predictions = format_predictions(predictions)
    bmi = patient_data.get('BMI', 0)
    systolic = patient_data.get('Systolic_BP', 0)
    diastolic = patient_data.get('Diastolic_BP', 0)
    
    prompt = prompt_template.format_messages(
        age=patient_data.get('Age', 'unknown'),
        gender=patient_data.get('Gender', 'unknown'),
        bmi=patient_data.get('BMI', 'unknown'),
        bmi_category=get_bmi_category(bmi),
        bp_systolic=systolic,
        bp_diastolic=diastolic,
        bp_category=get_bp_category(systolic, diastolic),
        predictions=formatted_predictions
    )
    
    try:
        response = generate_with_retry(llm, prompt)
        return parse_report_response(response)
    except TimeoutError:
        app.logger.error("Ollama model timed out")
        return {
            "risk_summary": "Report generation timed out. Please try again.",
            "detailed_recommendations": "",
            "prevention_plan": "",
            "treatment_options": "",
            "lifestyle_modifications": "",
            "monitoring_plan": ""
        }
    except Exception as e:
        app.logger.error(f"Error in generate_comprehensive_report: {str(e)}")
        return {
            "risk_summary": "Error generating recommendations. Please try again.",
            "detailed_recommendations": "",
            "prevention_plan": "",
            "treatment_options": "",
            "lifestyle_modifications": "",
            "monitoring_plan": ""
        }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)