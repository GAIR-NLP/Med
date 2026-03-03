import json
import logging
import random
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from verifier import Verifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# define the payload class
class Payload(BaseModel):
    data_source: str  # data source

    query: str  # query (without chat format)
    prompt: str  # prompt (apply chat format)
    answer: str  # answer (not formatted)
    solution: str  # solution (formatted)
    response: str  # response from model

    reward_verifier: str  # reward verifier
    reward_verifier_parm: str  # reward verifier parm
    fmt_ratio: Optional[float] = None  # TODO: merge format_ratio into parm
    len_ratio: Optional[float] = None


def check_payload(payload_dict):
    # Validate required fields
    required_fields = [
        "data_source", "query", "prompt", "answer", "solution", "response", "reward_verifier", "reward_verifier_parm"
    ]
    for field in required_fields:
        if field not in payload_dict:
            raise HTTPException(status_code=400, detail=f"Missing required field: {field}")


app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
async def root():
    return {"message": "Reward Judge Server"}


@app.post("/judge")
async def judge_reward(payload: Payload):
    # Check payload
    check_payload(payload.dict())

    # Initialize default result
    result = {
        "rewards": {
            "format_reward": 0.0,
            "accuracy_reward": 0.0,
            "final_reward": 0.0,
        }
    }

    # <----------- 0. define the ratio of each reward ------------>
    acc_ratio = 1.0
    fmt_ratio = 0.1

    try:
        if payload.reward_verifier in Verifier.list_verifiers():
            verifier_cls = Verifier.get(payload.reward_verifier)
        else:
            raise HTTPException(status_code=404, detail=f"Invalid reward verifier: {payload.reward_verifier}")

        # Create verifier instance
        verifier_parm = json.loads(payload.reward_verifier_parm)
        verifier = verifier_cls(**verifier_parm)

        # <----------- 1. format reward ------------>
        fmt_score = verifier.verify_format(payload.response)
        # <----------- 2. accuracy reward ------------>
        acc_score_gathered = verifier.verify_accuracy(payload.response, payload.solution)
        if isinstance(acc_score_gathered, dict):
            # if the accuracy score is a dict, we need to get the final score
            acc_score = acc_score_gathered['final_score']

            # to log the score of each metric
            for socre_key, socre_value in acc_score_gathered.items():
                if socre_key != 'final_score':
                    result['rewards'][f'{socre_key}_reward'] = socre_value
        else:
            # if the accuracy score is not a dict, we can directly use the score
            acc_score = acc_score_gathered

        # <----------- 3. length reward ------------>
        # TODO: add length reward if needed
        # length_score = verifier.verify_length(payload.response, payload.answer)
        length_score = 0

        # <----------- 4. assign reward into return result------------>
        result['rewards']['format_reward'] = float(fmt_score)
        result['rewards']['accuracy_reward'] = float(acc_score)

        # <----------- 5. calcualte the final reward into return result------------>
        if payload.fmt_ratio is not None:
            # Use optimal fmt_ratio
            fmt_ratio = payload.fmt_ratio
        else:
            logger.warning(f"Format ratio is not provided, using default {fmt_ratio}")

        total_reward_weight = acc_ratio + fmt_ratio
        result['rewards']['final_reward'] = \
            acc_score * (acc_ratio / total_reward_weight) + \
            fmt_score * (fmt_ratio / total_reward_weight)

        if random.random() <= 0.2:
            # 20% chance to print the reward info
            print_dict = {
                'reward_verifier': payload.reward_verifier,
                'reward_verifier_parm': payload.reward_verifier_parm,
                'prediction': payload.response,
                'solution': payload.solution,
                'format_score': fmt_score,
                'accuracy_score': acc_score,
                'final_reward': result['rewards']['final_reward'],
            }
            if isinstance(acc_score_gathered, dict):
                for key, value in acc_score_gathered.items():
                    if key != 'final_score':
                        print_dict[key] = value
            logger.info(json.dumps(print_dict, ensure_ascii=False))

    except Exception as e:
        # Log the error but return a valid result structure
        import traceback
        print(f"Error during verification: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        # return in default: 0 acc, 0 format, 0 reflection, 0 final

    return result


if __name__ == "__main__":
    # Run the server with Uvicorn
    uvicorn.run(
        "reward_serving_fastapi:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=32  # For load balancing
    )
