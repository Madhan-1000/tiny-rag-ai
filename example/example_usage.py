if True:
    import tiny_rag_ai
    tiny_rag_ai.index("./docs")
    print("DOCUMENTS - INDEXED")
    print("The Answer:",tiny_rag_ai.chat("Can we choose Mars over Proxima B Centuri as first planet?.", use_case="question assistant on the given topics",k=7))
    print("The Answer:",tiny_rag_ai.chat("Is there life possible on Proxima and stuff?.", use_case="question assistant on the given topics",k=7))
    print("The Answer:",tiny_rag_ai.chat("What equipement will we carry out to proxima b if we have a potential worm hole to it for a 2 days?.", use_case="question assistant on the given topics",k=7))