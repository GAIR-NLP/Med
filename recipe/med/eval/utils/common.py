PROMPT_JUDGE = """You are given a question and its standard answer, your task is to judge if a model-generated solution is correct.
You first need to extract the final choice made in the model-generated solution, and then judge if it matches the standard answer.
The <image> placeholder indicates an image in the question, but you do not need its content.

For multiple choice questions, note that the model-generated solution may not directly give the option letter, in this case, you need to judge from the solution's content to find out which option it has chosen.
For floating-point answers, the model-generated final answer needs to be close enough (5% relative error) to the standard answer to be considered correct.
Note that math expressions can have multiple equivalent forms, e.g., 0.5km and 500m are equivalent.
Other short-text answers can be considered equivalent if they are semantically equivalent under the context of the question.

Give a score of 1 if the model-generated solution matches the standard answer, and 0 otherwise.

[[Begin of Question]]
{question}
[[End of Question]]

[[Begin of Standard Answer]]
{answer}
[[End of Standard Answer]]

[[Begin of Model-Generated Solution]]
{response}
[[End of Model-Generated Solution]]


Your output should begin with a concise analysis, mentioning what answer the Model-Generated Solution has finally given.
Finally give an integer score in the following json format:
```json
{{
    "score": ## an integer, 0 or 1 ##
}}
```
"""

PROMPT_JUDGE_MODERATE = """You are given a question and its standard answer, your task is to judge if a model-generated solution is correct with moderate criteria.
You first need to extract the final choice made in the model-generated solution, and then judge if it matches the standard answer.
The <image> placeholder indicates an image in the question, but you do not need its content.

MODERATE EVALUATION CRITERIA:
- For multiple choice questions: The model must state the option letter clearly. Only allow inference if the content unambiguously points to exactly one option.
- For numerical answers: The model-generated answer must be within 1.5% relative error of the standard answer.
- For exact text answers: The extracted answer should match the standard answer closely (case-insensitive, but only trivial grammatical variations allowed, e.g., "the red line" vs "red line").
- For mathematical expressions: Basic algebraic equivalence and simple unit conversions are accepted (e.g., 0.5km = 500m), but complex transformations are not.
- The final answer must be clearly stated and easily extractable from the response.

Give a score of 1 if the model-generated solution matches the standard answer under these moderate criteria, and 0 otherwise.

[[Begin of Question]]
{question}
[[End of Question]]

[[Begin of Standard Answer]]
{answer}
[[End of Standard Answer]]

[[Begin of Model-Generated Solution]]
{response}
[[End of Model-Generated Solution]]


Your output should begin with a concise analysis, mentioning what answer the Model-Generated Solution has finally given.
Finally give an integer score in the following json format:
```json
{{
    "score": ## an integer, 0 or 1 ##
}}
```
"""

PROMPT_JUDGE_STRICT = """You are given a question and its standard answer, your task is to judge if a model-generated solution is correct with strict criteria.
You first need to extract the final choice made in the model-generated solution, and then judge if it matches the standard answer.
The <image> placeholder indicates an image in the question, but you do not need its content.

STRICT EVALUATION CRITERIA:
- For multiple choice questions: The model must explicitly state the correct option letter (A, B, C, D, etc.). Inferring from content is NOT allowed.
- For numerical answers: The model-generated answer must be within 1% relative error of the standard answer (not 5%).
- For exact text answers: The extracted answer must match the standard answer exactly (case-insensitive, but exact wording required).
- For mathematical expressions: Only basic algebraic equivalence is accepted (e.g., 2+3 = 5, but not complex unit conversions).
- The final answer must be clearly stated and easily extractable from the response.

Give a score of 1 ONLY if the model-generated solution exactly matches the standard answer under these strict criteria, and 0 otherwise.

[[Begin of Question]]
{question}
[[End of Question]]

[[Begin of Standard Answer]]
{answer}
[[End of Standard Answer]]

[[Begin of Model-Generated Solution]]
{response}
[[End of Model-Generated Solution]]


Your output should begin with a concise analysis, mentioning what answer the Model-Generated Solution has finally given.
Finally give an integer score in the following json format:
```json
{{
    "score": ## an integer, 0 or 1 ##
}}
```
"""
