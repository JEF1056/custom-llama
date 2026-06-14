"""Reproducible, resumable speculative-decoding + context/parallel config sweep
for the custom-llama llama-server (qwopus3.6-27b).

Edits config/models.ini in place, force-recreates the container, and measures
steady-state decode throughput at mid/long contexts for text and code.

Entry point: ``python -m scripts.spec_sweep`` (see __main__.py).
"""
