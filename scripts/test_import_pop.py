import traceback

try:
    import cenop.agents.population as pop
    print('imported')
except Exception:
    traceback.print_exc()
