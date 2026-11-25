"""Frozen inference prompts: interpolate the question only."""
PROMPTS = {
    'P0': 'Answer the following question based only on the video.\nQuestion: {question}\nGive a brief and direct answer.',
    'P1': 'Answer the question using only evidence visible in the video.\nIf the evidence is sufficient, answer directly.\nIf the evidence supports only a plausible guess, state the guess with clear uncertainty.\nIf the video does not provide enough evidence, answer: "Cannot be determined from the video."\nQuestion: {question}',
}
