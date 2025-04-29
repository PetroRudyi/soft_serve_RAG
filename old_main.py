import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
from PIL import Image, ImageTk
from scripts.AI_bot import load_vdb, user_response
from config import Config


def AI_responce(api_key: str, query: str, model: str, temperature: int):
    text_collection, image_collection = load_vdb()
    print('Vector DBs loaded')

    text, images = user_response(api_key, query, text_collection, image_collection, model, temperature)
    print(images)
    opened_images = [Image.open(image) for image in images]
    return text, opened_images


def on_submit():
    api_key = api_key_entry.get().strip()
    query = query_entry.get().strip()
    temperature = temp_entry.get().strip()
    model = model_selector.get()

    if not api_key or not query:
        messagebox.showerror("Error", "API KEY and Query can not be empty!")
        return

    if len(query) > Config.max_query_length:
        messagebox.showerror("Error", f"Query has a limit of {Config.max_query_length} letters!")
        return

    try:
        temp_val = float(temperature)
        if not (0 <= temp_val <= 100):
            raise ValueError("Temperature has to be integer between 0 and 100")
    except ValueError:
        messagebox.showerror("Error", "Temperature has to be integer between 0 and 100!")
        return

    try:
        text, images = AI_responce(api_key, query, model, int(temperature))

        text_output.delete(1.0, tk.END)
        text_output.insert(tk.END, text)

        img_resized_1 = images[0].resize((400, 300))
        img_tk_1 = ImageTk.PhotoImage(img_resized_1)
        image_label_1.config(image=img_tk_1)
        image_label_1.image = img_tk_1
        img_resized_2 = images[1].resize((400, 300))
        img_tk_2 = ImageTk.PhotoImage(img_resized_2)
        image_label_2.config(image=img_tk_2)
        image_label_2.image = img_tk_2


    except Exception as e:
        messagebox.showerror("Error", str(e))


# ---------- UI ----------

root = tk.Tk()
root.title("Multimodal RAG System")
root.geometry("1024x768")
root.configure(bg="#f5f5f5")

tk.Label(root, text="API KEY:", bg="#f5f5f5").pack(pady=(10, 0))
api_key_entry = tk.Entry(root, width=50)
api_key_entry.pack()

tk.Label(root, text="Query:", bg="#f5f5f5").pack(pady=(10, 0))
query_entry = tk.Entry(root, width=50)
query_entry.pack()

control_frame = tk.Frame(root, bg="#f5f5f5")
control_frame.pack(pady=10)

tk.Label(control_frame, text="Temperature (0-100):", bg="#f5f5f5").grid(row=0, column=0, padx=5)
temp_entry = tk.Entry(control_frame, width=5)
temp_entry.insert(0, Config.default_temperature)
temp_entry.grid(row=0, column=1)

tk.Label(control_frame, text="Model:", bg="#f5f5f5").grid(row=0, column=2, padx=10)
model_selector = ttk.Combobox(control_frame, values=Config_new.available_LLM_models, width=20)
model_selector.set(Config_new.default_llm_model)
model_selector.grid(row=0, column=3)

submit_btn = tk.Button(root, text="Send", command=on_submit, bg="#4CAF50", fg="white")
submit_btn.pack(pady=5)

result_frame = tk.Frame(root)
result_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

# Frame to hold both images side-by-side
top_frame = tk.Frame(result_frame)
top_frame.pack(pady=5)

image_label_1 = tk.Label(top_frame)
image_label_1.pack(side=tk.LEFT, padx=10)

image_label_2 = tk.Label(top_frame)
image_label_2.pack(side=tk.LEFT, padx=10)

# Text output below the image row
text_output = scrolledtext.ScrolledText(result_frame, wrap=tk.WORD, width=80, height=15)
text_output.pack(fill=tk.BOTH, expand=True, pady=5)

root.mainloop()
