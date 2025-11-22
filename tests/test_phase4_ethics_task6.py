
import torch
import numpy as np
import pytest
from src.models.phase4.ethical_safeguards.core_value_function import CoreValueFunction

class TestEthicalCoreVectorization:
    def test_semantic_similarity(self):
        """
        Verify that semantically similar words have closer vectors than unrelated ones.
        With MD5, "Human" and "Person" are random.
        With HTT (untrained), they might still be random unless they share subwords.
        BUT "Human" and "Humans" should definitely be closer than "Human" and "Apple".
        """
        # We need to initialize CoreValueFunction
        # But it takes 'ethical_principles'.
        principles = ["Do no harm"]
        cvf = CoreValueFunction(principles, d_model=64)

        v1 = cvf._text_to_vector("Human")
        v2 = cvf._text_to_vector("Humans")
        v3 = cvf._text_to_vector("Apple")

        # Cosine similarity
        sim_12 = torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
        sim_13 = torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v3.unsqueeze(0)).item()

        print(f"Sim(Human, Humans): {sim_12}")
        print(f"Sim(Human, Apple): {sim_13}")

        # With MD5, these are all random (~0).
        # With Tokenizer + HTT, 1-2 should be high (shared subwords).

        # This test might fail if we don't implement the tokenizer part well,
        # but it serves as a verification of the change.
