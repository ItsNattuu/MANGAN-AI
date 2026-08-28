from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException
)

from fastapi.middleware.cors import (
    CORSMiddleware
)

import tempfile
import os

from pipeline.orchestrator import (
    run_pipeline
)


app = FastAPI(
    title="MANGAN-AI API",
    version="1.0.0"
)


# ==========================================
# CORS
# ==========================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173"
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]
)


# ==========================================
# Health check
# ==========================================

@app.get("/")
def root():

    return {
        "status": "online",
        "system": "MANGAN-AI"
    }


# ==========================================
# Full pipeline
# ==========================================

@app.post("/analyze")
async def analyze(

    file: UploadFile = File(...),

):

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No file supplied."
        )


    suffix = os.path.splitext(
        file.filename
    )[1]

    # --------------------------------------
    # Temporary file
    # --------------------------------------

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp:

        contents = await file.read()

        temp.write(contents)

        temp_path = temp.name


    try:

        result = run_pipeline(
            image_path=temp_path,

            request=(
                "Analyze this area for "
                "manganese exploration and "
                "identify the highest-priority "
                "targets."
            )
        )

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        if os.path.exists(temp_path):

            os.remove(temp_path)
