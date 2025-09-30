import os
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, Body
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# --- Configuration and Constants ---

# IMPORTANT: Attempting to load this model (70B parameters) locally
# requires extreme hardware (e.g., multiple NVIDIA A100 GPUs with massive VRAM).
# This code will likely fail in standard environments.
LLAMA_MODEL_ID = "meta-llama/Meta-Llama-3-70B-Instruct" 

# Set up the quantization config for 4-bit loading (most memory efficient)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    # This setting is required for Llama models
    bnb_4bit_use_double_quant=True, 
)

# Global storage for the model and tokenizer
# These will be loaded once at application startup
class LLMState(object):
    """Simple container for the model and tokenizer."""
    def __init__(self):
        self.model = None
        self.tokenizer = None

llm_state = LLMState()

def load_llama_model():
    """
    Attempts to load the Llama 3 70B model using 4-bit quantization.
    Requires significant GPU VRAM and acceptance of Meta's license agreement
    on Hugging Face (requiring HUGGING_FACE_HUB_TOKEN to be set).
    """
    try:
        print(f"--- Attempting to load {LLAMA_MODEL_ID} (70B, 4-bit quantized) ---")
        
        # Ensure the Hugging Face token is available for gated models like Llama
        if not os.getenv("HUGGING_FACE_HUB_TOKEN"):
             print("WARNING: HUGGING_FACE_HUB_TOKEN environment variable is not set. Loading a gated model will fail.")
        
        llm_state.model = AutoModelForCausalLM.from_pretrained(
            LLAMA_MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto", # Distributes the model across available devices/GPUs
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        llm_state.tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_ID)
        
        # Ensure the pad token is set for batched inference, though typically not needed here
        if llm_state.tokenizer.pad_token is None:
            llm_state.tokenizer.pad_token = llm_state.tokenizer.eos_token
            
        print("Model and Tokenizer loaded successfully (or ready for use).")
        
    except Exception as e:
        # NOTE: This block is expected to be hit in environments without the necessary hardware.
        error_msg = f"Failed to load the large LLM: {e}. " \
                    "This is often due to insufficient GPU VRAM for the 70B model. " \
                    "Proceeding with a mock model for dependency resolution."
        print(f"CRITICAL ERROR: {error_msg}")
        
        # To prevent immediate application crash, we could mock the model here 
        # for a functional demonstration, but we prioritize alerting the user.
        # For this demonstration, we let the startup fail if the model can't be loaded,
        # as the user insisted on this loading mechanism.
        raise RuntimeError(error_msg)


# --- FastAPI Lifespan and Dependency Management ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager to handle startup and shutdown events.
    The model is loaded here once before the application starts accepting requests.
    """
    load_llama_model()
    yield
    # Clean up resources on shutdown
    llm_state.model = None
    llm_state.tokenizer = None
    print("Application shutdown complete.")

app = FastAPI(
    title="Llama 3 70B SOAP Note Generator",
    description="An API to generate structured medical SOAP notes using the locally loaded Llama 3 70B model (requires powerful GPU hardware).",
    lifespan=lifespan
)

def get_llm_components():
    """
    Dependency injector for accessing the loaded model and tokenizer in routes.
    """
    if llm_state.model is None or llm_state.tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Check server logs for hardware requirement errors."
        )
    return llm_state.model, llm_state.tokenizer


# --- API Models ---

class SOAPNoteRequest(BaseModel):
    """Input structure for the medical transcription text."""
    patient_summary: str = Field(
        ...,
        description="Raw transcription or text describing the patient's visit and findings."
    )

class SOAPNoteResponse(BaseModel):
    """Output structure for the generated SOAP note."""
    soap_note: str = Field(
        ...,
        description="The structured SOAP note (Subjective, Objective, Assessment, Plan)."
    )


# --- Endpoint Definition ---

@app.post(
    "/soap-generation",
    response_model=SOAPNoteResponse,
    summary="Generate SOAP Note from Text"
)
def generate_soap_note(
    patient_summary: str = Body(..., embed=True, description="The raw medical text input."),
    llm_deps: tuple = Depends(get_llm_components)
):
    """
    Processes the raw patient summary and uses the loaded Llama 3 70B model
    to generate a structured SOAP note (S, O, A, P).
    """
    model, tokenizer = llm_deps

    # 1. Define the system and user messages following Llama 3's chat format
    system_prompt = (
        "You are a professional medical transcriber and assistant. "
        "Your task is to organize the provided clinical narrative into a clear, "
        "structured SOAP note format. The output MUST contain these four "
        "sections: Subjective (S), Objective (O), Assessment (A), and Plan (P). "
        "Ensure the generated note is concise and clinically accurate based on the input."
    )
    user_prompt = f"Clinical Narrative: \"{patient_summary}\""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # 2. Apply the chat template and tokenize
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True, # Important for Llama 3 instruction models
        return_tensors="pt"
    ).to(model.device) # Move tensor to the correct device (GPU/CPU)

    # 3. Generate the response
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id, # Use EOS as PAD token
        )

    # 4. Decode and format the response
    # The output includes the prompt, so we decode and remove the prompt part
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Simple post-processing to isolate the generated text part
    # We look for the last assistant response (which contains the SOAP note)
    # The Llama 3 format is complex, but the final output should be the note itself.
    try:
        # A robust way to clean Llama 3 output: find the start of the final assistant response
        # which is usually after the last 'user' and 'assistant' markers
        # The prompt itself contains the system and user message. We want the text AFTER it.
        prompt_length = len(tokenizer.decode(input_ids[0], skip_special_tokens=True))
        soap_note_content = generated_text[prompt_length:].strip()
        
        # For Llama 3, the final response often begins immediately.
        # Simple extraction based on the expected format
        if "Assessment (A):" not in soap_note_content and "S:" in soap_note_content:
            return SOAPNoteResponse(soap_note=soap_note_content)
        elif generated_text.count("assistant") > 1:
            # If the model repeats the conversation structure, get the last block
            last_assistant_index = generated_text.rfind("assistant\n")
            if last_assistant_index != -1:
                 soap_note_content = generated_text[last_assistant_index + len("assistant\n"):].strip()
                 
        # Fallback to the simplest trimmed version
        if not soap_note_content:
            soap_note_content = generated_text[generated_text.rfind(user_prompt) + len(user_prompt):].strip()
            
        return SOAPNoteResponse(soap_note=soap_note_content)
        
    except Exception:
        # If parsing fails, return the full generated text for inspection
        return SOAPNoteResponse(soap_note=generated_text)

