# LangChain RAG - Guida Completa

Un progetto educativo completo per costruire un sistema **RAG (Retrieval-Augmented Generation)** utilizzando LangChain, con confronti tra diverse tecnologie e strategie.

## 📋 Indice

- [Cos'è RAG?](#cosè-rag)
- [Caratteristiche](#caratteristiche)
- [Prerequisiti](#prerequisiti)
- [Installazione](#installazione)
- [Configurazione](#configurazione)
- [Utilizzo](#utilizzo)
- [Contenuto del Notebook](#contenuto-del-notebook)
- [Confronti Tecnologici](#confronti-tecnologici)
- [Troubleshooting](#troubleshooting)
- [Risorse](#risorse)

## 🤔 Cos'è RAG?

**RAG (Retrieval-Augmented Generation)** è una tecnica potente che combina i punti di forza dei Large Language Models (LLM) con il recupero di informazioni da una base di conoscenza esterna.

### Funzionamento

```
Query Utente → Embedding → Vector Search → Documenti Recuperati → LLM → Risposta
                 ↓                           ↓
          Vector Store ← Embeddings ← Chunks ← Documenti
```

RAG migliora le risposte degli LLM in tre passi:
1. **Recupero**: trova documenti rilevanti da una knowledge base
2. **Augmentation**: arricchisce il prompt con il contesto recuperato
3. **Generazione**: produce risposte informate basate su LLM + documenti

## ✨ Caratteristiche

- 📚 **Caricamento documenti** da web, PDF, testi
- ✂️ **Strategie di chunking** configurabili
- 🔄 **Confronto embeddings**: OpenAI vs HuggingFace
- 🔍 **Confronto retrieval**: Similarity Search vs MMR
- 🤖 **Chain RAG complete** end-to-end
- 🏷️ **Metadata filtering** per ricerche avanzate
- 📊 **Source attribution** per trasparenza
- 💡 **Best practices** e pitfalls comuni

## 🔧 Prerequisiti

- **Python 3.8+** (testato con Python 3.14)
- **API Key OpenAI** (obbligatoria) - [Ottienila qui](https://platform.openai.com/api-keys)
- **API Key HuggingFace** (opzionale) - Per embeddings locali non serve
- **4GB+ RAM** - Per i modelli sentence-transformers

## 🚀 Installazione

### 1. Clona o scarica il progetto

```bash
cd /percorso/della/cartella
```

### 2. Crea un ambiente virtuale Python

```bash
# Crea l'ambiente virtuale
python3 -m venv venv

# Attiva l'ambiente virtuale
# Su macOS/Linux:
source venv/bin/activate

# Su Windows:
# venv\Scripts\activate
```

### 3. Installa le dipendenze

```bash
pip install -r requirements.txt
```

Questo installerà tutte le librerie necessarie:
- **LangChain** e moduli correlati
- **OpenAI** client API
- **FAISS** per similarity search
- **Sentence Transformers** per embeddings locali
- **Jupyter** per eseguire il notebook

### 4. Verifica l'installazione

```bash
python -c "import langchain; import openai; import faiss; print('✓ Installazione completata!')"
```

## ⚙️ Configurazione

### 1. Configura le API Keys

Crea un file `.env` nella directory del progetto:

```bash
# Crea il file .env
touch .env
```

Aggiungi le tue chiavi API:

```env
# Obbligatorio - Ottieni da https://platform.openai.com/api-keys
OPENAI_API_KEY=sk-proj-...

# Opzionale - Solo se vuoi usare API HuggingFace (non necessario per embeddings locali)
HUGGINGFACE_API_KEY=hf_...
```

⚠️ **IMPORTANTE**: Non condividere mai il file `.env` o le tue API keys! Il file è già escluso da git tramite `.gitignore`.

### 2. Verifica la configurazione

Esegui il notebook e verifica la cella di test API key - vedrai:
```
✓ API key is VALID! Connection successful.
```

## 📓 Utilizzo

### Avvia Jupyter Notebook

```bash
# Assicurati che l'ambiente virtuale sia attivo
source venv/bin/activate

# Avvia Jupyter
jupyter notebook
```

Il browser si aprirà automaticamente. Apri `langchain_rag_complete.ipynb` e segui il notebook cella per cella.

### Esecuzione Rapida

1. **Run All**: Kernel → Restart & Run All
2. Attendi il completamento (2-5 minuti al primo avvio)
3. Esplora i risultati e sperimenta con le query

## 📚 Contenuto del Notebook

Il notebook è organizzato in sezioni progressive:

### 1. Setup e Installazione
- Installazione dipendenze
- Configurazione API keys
- Test connessione OpenAI

### 2. Document Loading
- WebBaseLoader per documentazione online
- Metadata personalizzati
- Gestione multi-source

### 3. Text Splitting
- RecursiveCharacterTextSplitter
- Confronto strategie (1000/200 vs 500/100)
- Best practices per chunk size

### 4. Embeddings
**Confronto completo OpenAI vs HuggingFace:**
- Dimensioni vettori (1536 vs 384)
- Performance (tempo, qualità)
- Costi e privacy

### 5. Vector Stores
- Creazione FAISS vector stores
- Indexing e similarity search
- Testing con entrambi gli embeddings

### 6. Retrieval Strategies
**Confronto Similarity vs MMR:**
- Similarity: massima rilevanza
- MMR: bilanciamento rilevanza/diversità
- Parametri e configurazione

### 7. RAG Chains
- Costruzione chain completa
- LLM initialization (GPT-4o-mini)
- Prompt engineering
- Document combination

### 8. Evaluation
- Test query multiple
- Confronto risultati tra strategie
- Source attribution

### 9. Advanced Features
- Metadata filtering
- Custom retrievers
- Production tips

### 10. Best Practices
- Common pitfalls da evitare
- Performance optimization
- Security considerations

## 🔄 Confronti Tecnologici

### OpenAI vs HuggingFace Embeddings

| Caratteristica | OpenAI | HuggingFace |
|----------------|--------|-------------|
| **Qualità** | ⭐⭐⭐⭐⭐ Eccellente | ⭐⭐⭐⭐ Molto buona |
| **Costo** | 💰 Pay-per-use | 🆓 Gratis |
| **Velocità** | ⚡ Veloce (API) | 🐢 Più lento (locale) |
| **Privacy** | ☁️ Dati su cloud | 🔒 Dati locali |
| **Dimensione** | 1536d | 384d |
| **Setup** | API key | Download modello |

**Raccomandazione:**
- **Produzione/Qualità**: OpenAI
- **Sviluppo/Privacy**: HuggingFace

### Similarity Search vs MMR

| Caratteristica | Similarity | MMR |
|----------------|------------|-----|
| **Rilevanza** | ⭐⭐⭐⭐⭐ Massima | ⭐⭐⭐⭐ Alta |
| **Diversità** | ⭐⭐ Bassa | ⭐⭐⭐⭐⭐ Alta |
| **Velocità** | ⚡ Veloce | 🐢 Più lento |
| **Ridondanza** | 📝 Possibile | ✅ Minimizzata |
| **Use case** | Query specifiche | Esplorazione topic |

**Raccomandazione:**
- **Query precise**: Similarity
- **Overview/Diversità**: MMR

## 🐛 Troubleshooting

### Errore: "API key is INVALID"

```bash
# Verifica che la chiave sia corretta nel file .env
cat .env | grep OPENAI_API_KEY

# Ricarica il kernel Jupyter dopo aver modificato .env
# Kernel → Restart
```

### Errore: "ipykernel not found"

```bash
pip install ipykernel
python -m ipykernel install --user --name=venv
```

### ModuleNotFoundError

```bash
# Reinstalla le dipendenze
pip install -r requirements.txt --upgrade
```

### Download lento HuggingFace models

Il primo download del modello `sentence-transformers/all-MiniLM-L6-v2` può richiedere 1-2 minuti. È normale e avviene solo la prima volta.

### FAISS import error

```bash
# Su Mac con Apple Silicon, potrebbe servire:
pip uninstall faiss-cpu
pip install faiss-cpu --no-cache-dir
```

### Memory errors

Se hai meno di 4GB RAM disponibili:
- Riduci `chunk_size` a 500
- Usa meno documenti in `urls`
- Riduci `k` nei retriever a 2-3

## 📖 Risorse

### Documentazione Ufficiale
- [LangChain Docs](https://python.langchain.com/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [FAISS GitHub](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)

### Tutorial e Guide
- [LangChain RAG Tutorial](https://python.langchain.com/docs/use_cases/question_answering/)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [RAG Best Practices](https://python.langchain.com/docs/use_cases/question_answering/sources)

### Paper e Research
- [RAG Paper (Lewis et al.)](https://arxiv.org/abs/2005.11401)
- [Sentence-BERT](https://arxiv.org/abs/1908.10084)

## 🤝 Contributi

Suggerimenti per migliorare questo progetto:
1. Testare con diversi tipi di documenti (PDF, CSV, etc.)
2. Aggiungere metriche di valutazione automatiche
3. Implementare conversational memory
4. Confrontare altri modelli di embeddings
5. Ottimizzare per dataset più grandi

## 📝 Licenza

Progetto educativo - libero utilizzo per scopi di apprendimento.

## 🙏 Ringraziamenti

- **LangChain** per il framework eccellente
- **OpenAI** per GPT e embeddings API
- **HuggingFace** per modelli open source
- **FAISS** per similarity search efficiente

---

**Buon apprendimento! 🚀**

Per domande o problemi, consulta la sezione [Troubleshooting](#troubleshooting) o la documentazione ufficiale.
