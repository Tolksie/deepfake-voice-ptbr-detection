from openai import OpenAI
import os

client = OpenAI()

phrases = [
    "Pesquisa é uma coisa que muda toda hora",
    "No total serão chamados vinte e seis mil candidatos",
    "O numero de convocados por vaga é de doze candidatos",
    "Atualmente esse abatimento é limitado a setenta porcento dos gastos",
    "Sandra Regina Machado acho que ela enfim criou juízo",
    "Eles estão colocando armadilhas nas fazendas onde já ocorreram os ataques",
    "Dessas somente umas trezentas e vinte foram inauguradas em território americano",
    "No total sete mísseis foram disparados contra o encrave",
    "Em Florianópolis foi registrado dois graus celsius na manhã de domingo",
    "As situações ditas embaraçosas são resolvidas com os dados",
    "Itamar tem razão de estar exultante como nunca desde que virou presidente",
    "A mãe de todas as reformas é a reforma política",
    "Conseguiram eliminar áreas supérfluas ou que antes eram desperdiçadas",
    "Uma lata de leite em pó integral vale o ingresso",
    "A maioria dos passageiros do barco naufragado eram de crianças"
]

voices = [
    "alloy", "echo", "fable", "onyx", "nova",
    "shimmer", "coral", "verse", "ballad", "ash",
    "sage", "marin", "cedar", "alloy", "echo"
]

save_dir = "data/fake"
os.makedirs(save_dir, exist_ok=True)

speaker_id = 1

for voice in voices:
    print(f"gerando speaker {speaker_id:03d} com voz {voice}...")

    speaker_folder = f"{save_dir}/speaker_{speaker_id:03d}"
    os.makedirs(speaker_folder, exist_ok=True)

    for i, text in enumerate(phrases):
        result = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice=voice,
            input=text
        )

        with open(f"{speaker_folder}/fake_{speaker_id:03d}-{i:03d}.wav", "wb") as f:
            f.write(result.read())

    speaker_id += 1
