
def find_analogies(hypothesis, known_theorems):
    """Return analogies between hypothesis and known theorems."""
    return [t["name"] for t in known_theorems if t.get("structure") in str(hypothesis)]
