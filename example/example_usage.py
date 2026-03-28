if True:
    import tiny_rag_ai
    tiny_rag_ai.index("./docs")
    print("DOCUMENTS - INDEXED")
    print("The Answer:",tiny_rag_ai.chat("Can we choose Mars over Proxima B Centuri in a Civilization ending event like sun turning into a blackholey?.", use_case="question assistant on the given topics"))