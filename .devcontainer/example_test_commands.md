# Example Test Commands

# Test Hugging Face API (ensure .env has HF_TOKEN={HF_TOKEN} and .env.vault has credential HF_TOKEN=<API_KEY>)
curl https://huggingface.co/api/whoami-v2 -H "Authorization: Bearer HF_TOKEN"

# Test Anthropic API
curl https://api.anthropic.com/v1/messages \
  --header "x-api-key: ANTHROPIC_AUTH_TOKEN" \
  --header "anthropic-version: 2023-06-01" \
  --header "content-type: application/json" \
  --data '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello, world!"}]
  }'
