
def generate_abductive_hypotheses(data):
    return [f"y = x * {data[i+1] - data[i]}" for i in range(len(data)-1)]
