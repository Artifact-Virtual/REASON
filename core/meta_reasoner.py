
def evaluate_meta_reasoning(hypotheses):
    return "Logically sound" if all('undefined' not in h for h in hypotheses) else "Issues detected"
