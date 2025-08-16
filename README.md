# Fake vs Real Text Classification

This project detects **fake vs real text pairs** using Electra.

Hackathon: [Fake or Real: The Impostor Hunt in Texts](https://www.kaggle.com/competitions/fake-or-real-the-impostor-hunt/overview)

Part of the Secure Your AI series of competitions by the European Space Agency

## Pipeline
1. **Data Loading** → Read paired text files + labels  
2. **Preprocessing** → Chunk long texts into overlapping windows  
3. **EDA** → Quick length comparison between real and fake texts  
4. **Dataset Conversion** → Convert into HuggingFace `Dataset`  
5. **Model Training** → Fine-tune Electra with HuggingFace `Trainer`

## Run Training
```bash
python main.py
