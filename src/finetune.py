import os
import openai
import config

openai.api_key = config.OPEN_AI_KEY

openai.FineTuningJob.list(limit=10)