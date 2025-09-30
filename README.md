Llama 3 70B SOAP Note Generation API
This project implements a high-performance web API using FastAPI to leverage the power of the Meta-Llama-3-70B-Instruct model for medical transcription and structured SOAP Note (Subjective, Objective, Assessment, Plan) generation.
The application loads the massive 70B parameter model locally using 4-bit quantization via the transformers and bitsandbytes libraries for efficient, in-house inference.
⚠️ Critical Hardware Requirement Disclaimer
Running the Llama 3 70B model, even in 4-bit quantized mode, requires extremely powerful hardware.
| Requirement | Specification |
|---|---|
| GPU | NVIDIA A100 or H100 (multiple units may be required) |
| VRAM | Minimum 80GB (or more, depending on system overhead) |
| CPU RAM | At least 128GB recommended |




🚀 Getting Started
Prerequisites
 * Python 3.8+
 * GPU Drivers for NVIDIA CUDA and PyTorch compatibility.
 * Hugging Face Account & Token: Llama 3 is a gated model. You must accept the license on the Hugging Face model page and obtain a Read token.
Installation
 * Clone the repository (or save main.py):
   # Assuming you have the main.py file

 * Create and activate a virtual environment:
   python -m venv venv
source venv/bin/activate  # On Linux/macOS
# or venv\Scripts\activate  # On Windows

 * Install dependencies:
   This command installs FastAPI, the ASGI server, and the necessary LLM libraries including bitsandbytes for 4-bit quantization, accelerate, and torch.
   pip install fastapi uvicorn pydantic torch transformers accelerate bitsandbytes

▶️ Running the API
 * Set the Hugging Face Token:
   You must set your Hugging Face Access Token as an environment variable for the application to authenticate and download the Llama 3 model weights.
   # On Linux/macOS
export HUGGING_FACE_HUB_TOKEN="YOUR_HF_TOKEN_HERE"

# On Windows (PowerShell)
$env:HUGGING_FACE_HUB_TOKEN="YOUR_HF_TOKEN_HERE"

 * Start the FastAPI Server:
   The model loading is handled during the startup phase (lifespan function). This step will take several minutes as the 70B model is initialized and loaded onto the GPU.
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload

📝 API Usage
The API exposes a single POST endpoint for transcription.
Endpoint: /soap-generation
Method: POST
Request Body
| Field | Type | Description | Required |
|---|---|---|---|
| patient_summary | string | The raw clinical narrative or transcription text to be converted into a SOAP note. | Yes |
Example Request (using cURL)
Use the following cURL command to send a request to the API.
curl -X POST "[http://127.0.0.1:8000/soap-generation](http://127.0.0.1:8000/soap-generation)" \
-H "Content-Type: application/json" \
-d '{
  "patient_summary": "45 y/o male presents with a headache described as severe, pulsating, 8/10, starting 24 hours ago. Denies fever or neck stiffness. Vital signs are normal. Patient is alert, oriented, and photophobic. Diagnosis is classic migraine. Plan includes Sumatriptan prescription and follow-up in one week with a headache diary."
}'

Example Response
The model will process the text and return a structured SOAP note:
{
  "soap_note": "Subjective (S):\n- 45 y/o male c/o severe, pulsating headache (8/10) onset 24 hours ago.\n- Denies fever or neck stiffness. Known history of migraines.\n\nObjective (O):\n- Vitals stable. Alert and oriented x3.\n- Photophobia noted.\n- Neurological exam within normal limits (WNL).\n\nAssessment (A):\n- Classic Migraine, acute.\n\nPlan (P):\n- Rx: Sumatriptan 50mg, take 1 tablet at onset.\n- Advised to keep a detailed headache diary.\n- Follow up in 1 week, or sooner if symptoms escalate."
}

