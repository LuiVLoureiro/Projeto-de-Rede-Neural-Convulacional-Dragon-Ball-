import openai

def get_dragon_ball_fluid_mechanics_questions(api_key):
    """
    Gera três questões de mecânica dos fluidos baseadas na história canônica de Dragon Ball.

    Args:
        api_key (str): Chave de API do OpenAI.

    Returns:
        list: Uma lista contendo as três questões geradas.
    """
    # Configure a chave da API
    openai.api_key = "SUA_CHAVE_OPENAI_AQUI"

    # Envie a requisição para o modelo GPT-3.5-turbo
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "Você é um especialista em mecânica dos fluidos e fã da franquia Dragon Ball."
                    "Crie três questões técnicas e criativas relacionadas à mecânica dos fluidos,"
                    "utilizando a história canônica de Dragon Ball como cenário temático."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Quero três questões de mecânica dos fluidos baseadas em eventos ou cenários canônicos de Dragon Ball."
                    "Use exemplos como o vôo dos Saiyajins, a energia das esferas do dragão e o Kamehameha."
                ),
            },
        ],
        max_tokens=500,
        temperature=0.7,
    )

    # Extraia as respostas geradas
    generated_text = response["choices"][0]["message"]["content"]

    # Divida as questões em uma lista, se possível
    questions = generated_text.strip().split("\n")

    return questions

# Insira sua chave da API OpenAI aqui
api_key = "sua-chave-api-aqui"

# Obtenha as questões
dragon_ball_questions = get_dragon_ball_fluid_mechanics_questions(api_key)

# Exiba as questões
print("\nQUESTÕES DE MECÂNICA DOS FLUIDOS COM TEMA DE DRAGON BALL:\n")
for i, question in enumerate(dragon_ball_questions, 1):
    print(f"{i}. {question}")