import gradio as gr

# import asyncio
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerHTTP

server = MCPServerHTTP(url="http://localhost:8000/sse")
system_prompt = """
    Tu es un assistant juridique. Tu réponds au questions de l'utilisateur en français. 
    L'utilisateur n'est pas spécialiste du droit. Tu réponds le plus précisément possible, en utilisant des termes compréhensibles.
    La première de ta réponse donne une réponse immédiate et synthétique. Dans un deuxième temps, tu peux donner des détails supplémentaires.
    Appuie toi sur les codes en vigueurs. Cite les articles de références en fin de réponse.
"""
agent = Agent(
    "mistral:mistral-small-latest", mcp_servers=[server], system_prompt=system_prompt
)


async def answer(query):
    async with agent.run_mcp_servers():
        result = await agent.run(query)
    return result.data


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            query = gr.Textbox(label="query")
            btn = gr.Button("Poser la question au LLM")
        output = gr.Markdown(label="answer")
    btn.click(answer, inputs=query, outputs=output)


if __name__ == "__main__":
    demo.launch()
#     loop = asyncio.new_event_loop()
#     loop.run_until_complete(main())
