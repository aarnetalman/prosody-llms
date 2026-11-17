"""Prompting hosted LLMs to mark prominent words .

* ``prompts``: instruction variants, output formats and few-shot examples;
* ``parse``: robust parsing of answers back to one label per corpus word;
* ``client``, ``cache``: OpenAI-compatible chat client, response cache and cost log;
* ``local``: many-shot prompting of open models with constrained decoding;
* ``run``: dry runs, budgeted runs and outputs; ``costs``: spending so far.
"""
