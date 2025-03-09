from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def main():
    # Cargar el tokenizador y el modelo DialoGPT-Large
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")
    
    print("Inicia la conversación (escribe 'salir' para terminar):")
    chat_history_ids = None  # Aquí se acumula el historial de la conversación
    step = 0

    while True:
        user_input = input("Usuario: ").strip()
        if user_input.lower() == "salir":
            break

        # Codificar la entrada del usuario con el token de fin de secuencia
        new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
        
        # Concatenar con el historial previo si ya existe
        if step == 0:
            bot_input_ids = new_user_input_ids
        else:
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
        
        # Generar la respuesta del asistente
        chat_history_ids = model.generate(
            bot_input_ids,
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
    temperature=0.6,  # Reducir aleatoriedad
    top_k=100,  # Considerar más opciones
    top_p=0.95  # Ajustar filtro de probabilidad
        )
     

        
        # Extraer la respuesta generada para este turno
        response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        print("Asistente:", response)
        step += 1

if __name__ == "__main__":
    
    main()
