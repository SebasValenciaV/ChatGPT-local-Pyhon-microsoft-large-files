from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def truncate_history(history_ids, max_tokens, tokenizer):
    """
    Recorta el historial de conversación para que no exceda de `max_tokens`.
    Se asume que history_ids es un tensor 2D del tamaño [1, tokens].
    """
    if history_ids.shape[-1] > max_tokens:
        # Mantener solo los últimos 'max_tokens' tokens
        history_ids = history_ids[:, -max_tokens:]
    return history_ids

def main():
    # Establecer dispositivo: GPU si está disponible, caso contrario CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Cargar el tokenizador y el modelo DialoGPT-Large
    print("Cargando modelo y tokenizador...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large").to(device)
    
    print("Inicia la conversación (escribe 'salir' para terminar):")
    chat_history_ids = None  # Aquí se acumula el historial de la conversación
    step = 0

    # Definir el límite máximo de tokens en el historial para evitar exceder la longitud del contexto
    max_history_tokens = 1024  # Puedes ajustar este valor según las capacidades del modelo

    while True:
        try:
            user_input = input("Usuario: ").strip()
            if user_input.lower() == "salir":
                print("Terminando la conversación.")
                break

            # Codificar la entrada del usuario agregándole el token de fin de secuencia
            new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt").to(device)

            # Si es el primer paso, no hay historial previo
            if step == 0 or chat_history_ids is None:
                bot_input_ids = new_user_input_ids
            else:
                # Acumular el historial: concatenar historial previo con la nueva entrada
                bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
                # Truncar el historial si supera el límite de tokens
                bot_input_ids = truncate_history(bot_input_ids, max_history_tokens, tokenizer)

            # Generar la respuesta usando beam search para una salida más determinista y coherente
            with torch.no_grad():
                chat_history_ids = model.generate(
                    bot_input_ids,
                    max_length=512,           # Longitud máxima de la secuencia generada
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False,          # Generación determinista
                    num_beams=3,              # Uso de beam search para mejorar la coherencia
                    repetition_penalty=1.2,
                    early_stopping=True
                )

            # Extraer solo la respuesta generada en el turno actual
            output_ids = chat_history_ids[:, bot_input_ids.shape[-1]:]
            response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print("Asistente:", response)
            step += 1

        except Exception as e:
            print("Ocurrió un error:", e)
            continue

if __name__ == "__main__":
    main()
