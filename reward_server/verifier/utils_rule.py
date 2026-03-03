import logging

from latex2sympy2_extended import NormalizationConfig
import math_verify

from .helper import tag_count_reward
from .main import BaseVerifier, Verifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@Verifier.register(name="mathverify")
class MathVerifyVerifier(BaseVerifier):

    def verify_format(self, predict_str: str) -> float:
        try:
            return tag_count_reward(predict_str, with_answer=False)
        except Exception as e:
            logger.error(f"Error during format reward in mathverify: {str(e)}")
            return 0.0

    def verify_accuracy(self, predict_str: str, solution: str) -> float:
        """
        Reward function that checks if the completion is the same as the ground truth.
        referenced this part of the code from 
        https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py
        """

        # extract match
        if predict_str.lower().strip() == solution.lower().strip():
            return 1.0

        gold_parsed = math_verify.parse(
            solution,
            extraction_config=[
                math_verify.LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        boxed="all",
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )

        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = math_verify.parse(
                predict_str,
                extraction_config=[
                    math_verify.LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            if len(answer_parsed) == 0:
                answer_parsed = math_verify.parse(
                    predict_str,
                    extraction_mode="first_match",
                )

            # Compute binary rewards if verifiable, `None` otherwise to skip this example
            try:
                reward = float(math_verify.verify(gold_parsed, answer_parsed))
            except Exception as e:
                logger.error(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
                reward = 0.
        else:
            # If the gold solution is not parseable, we assign `None` to skip this example
            #TODO: @yema the return should be none, and we need mask these "None" samples
            reward = 0.
            logger.error(f"Failed to parse gold solution: {solution}")

        return reward
