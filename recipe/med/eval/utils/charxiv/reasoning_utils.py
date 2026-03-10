from copy import deepcopy

from recipe.med.eval.utils.charxiv.constants import (REASONING_GRADING_INST,
                                                    REASONING_GRADING_PREFIX)


def get_number_instruction(answer):
    base = answer.split(".")
    whole, decimal = base[0], None if len(base) == 1 else base[1]
    # check if it contains decimal places
    if whole is not None and decimal is None:
        inst = "* Your final answer must be an exact integer."
    elif whole is not None and decimal is not None:
        num_decimal = len(decimal)
        inst = (
            f"* Your final answer must be a number with {num_decimal} decimal places."
        )
    else:
        raise ValueError(f"Invalid answer: {answer}")
    return inst


def build_reasoning_grading_queries(input, resp):
    queries = {}
    for _, data in input.items():
        figure_id = str(data["figure_id"])
        # question without instruction, response
        query, response = resp[figure_id]["raw_question"], resp[figure_id]["response"]
        # get query for answer type (inst_category), then
        # populate the query with the question, ground truth, and response
        grading_query = REASONING_GRADING_PREFIX + deepcopy(
            REASONING_GRADING_INST[data["inst_category"]]
        ).replace("<|question|>", query).replace(
            "<|ground_truth|>", data["answer"]
        ).replace(
            "<|response|>", response
        )
        query = {
            "figure_id": figure_id,
            "grading_query": grading_query,
        }
        queries[figure_id] = query
    return queries
