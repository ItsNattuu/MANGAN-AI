import os
import json

from dotenv import load_dotenv
from google import genai


load_dotenv()

API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

if not API_KEY:

    raise RuntimeError(
        "GEMINI_API_KEY is missing."
    )


client = genai.Client(
    api_key=API_KEY
)


def generate_ai_report(
    user_request,
    layer1_summary,
    layer2_results,
    layer3_results
):

    prompt = f"""

You are MANGAN-AI.

You are an AI decision-support system
for manganese exploration.

IMPORTANT:

Satellite-derived results indicate
PROSPECTIVITY, not confirmed underground
ore reserves.

You must never claim that satellite
imagery proves the exact presence or
quantity of manganese underground.

USER REQUEST:

{user_request}


LAYER 1 SUMMARY:

{json.dumps(
    layer1_summary,
    indent=2
)}


LAYER 2 RESULTS:

{json.dumps(
    layer2_results[:20],
    indent=2
)}


LAYER 3 RESULTS:

{json.dumps(
    layer3_results[:20],
    indent=2
)}


Return:

1. Best exploration target
2. Top 5 targets
3. Mn prospectivity
4. Mining feasibility
5. Main reasons for ranking
6. Environmental/restricted-area warnings
7. Recommended next action
8. Field-validation requirements

Do not invent measurements.

"""


    response = client.models.generate_content(

        model="gemini-3.1-flash",

        contents=prompt
    )

    return response.text
