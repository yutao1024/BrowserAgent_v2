import openai
def ask_openai(prompt):
    openai.api_key = "sk-5TOLjHJSn7uyRj2gXZLxYsRe9vxmr8N9XWK2lQHalvgXiBoc"
    openai.base_url="https://open.xiaojingai.com/v1/"
    try:
        response=openai.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[
                {"role":"user",
                "content":prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return str(e)

print(ask_openai("What is the capital of France?"))