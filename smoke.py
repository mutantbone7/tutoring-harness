from anthropic import Anthropic

client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment
msg = client.messages.create(
    model="claude-fable-5-1",
    max_tokens=20,
    messages=[{"role": "user", "content": "Reply with the single word OK."}],
)
print(msg.content[0].text)
print(msg.model)



